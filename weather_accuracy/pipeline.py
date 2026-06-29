"""Pipeline orchestration.

Two jobs run on a daily schedule:

  run_forecast_pull()   each morning  -- capture same-day high/low from every
                                         source for every airport
  run_verification()    next morning  -- fetch yesterday's observed high/low,
                                         join to the forecasts, score them

build_scoreboard() turns the verification table into the rolling accuracy
ranking that the dashboard reads.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from . import config, observations, sources, store


# --------------------------------------------------------------------------
# Job 1: morning forecast pull
# --------------------------------------------------------------------------
def run_forecast_pull(issue_ts: datetime | None = None) -> pd.DataFrame:
    """Capture each source's same-day high/low for each airport."""
    issue_ts = issue_ts or datetime.now(timezone.utc)
    active = sources.all_sources()
    print(f"Forecast pull @ {issue_ts:%Y-%m-%d %H:%M UTC} "
          f"| {len(config.AIRPORTS)} airports x {len(active)} sources")

    rows: list[dict] = []
    for airport in config.AIRPORTS:
        local_date = issue_ts.astimezone(ZoneInfo(airport.tz)).date()
        for src in active:
            result = src.fetch(airport, local_date)
            if not result:
                print(f"  - {airport.icao:5} {src.name:18} no data")
                continue
            rows.append({
                "issue_ts": issue_ts.replace(microsecond=0),
                "source": src.name,
                "icao": airport.icao,
                "target_date": local_date.isoformat(),
                "lead_days": 0,                      # same-day forecast
                "fcst_tmax_f": _round1(result.get("tmax_f")),
                "fcst_tmin_f": _round1(result.get("tmin_f")),
            })
            print(f"  + {airport.icao:5} {src.name:18} "
                  f"hi={result.get('tmax_f')} lo={result.get('tmin_f')}")

    df = pd.DataFrame(rows)
    if not df.empty:
        store.append_forecasts(df)
    print(f"Captured {len(df)} forecast rows.")
    return df


# --------------------------------------------------------------------------
# Job 2: verification (run the day after the target date)
# --------------------------------------------------------------------------
def run_verification(target_date: date | None = None, now: datetime | None = None) -> pd.DataFrame:
    """Fetch observed high/low for target_date and score the forecasts.

    Observations are only pulled for a day that has fully ended in the airport's
    local timezone (plus a finalize buffer for late-arriving readings). Any
    airport whose `target_date` is not yet complete is skipped, so the NWS
    accuracy data is never pulled before the following day.
    """
    now = now or datetime.now(timezone.utc)
    target_date = target_date or (now.date() - timedelta(days=1))
    target = target_date.isoformat()
    print(f"Verification for {target}")

    # 1. Pull ground truth and upsert the observations table.
    obs_rows: list[dict] = []
    obs_by_icao: dict[str, dict] = {}
    for airport in config.AIRPORTS:
        ready_at = _day_complete_utc(target_date, airport.tz)
        if now < ready_at:
            print(f"  . {airport.icao:5} day not complete until "
                  f"{ready_at:%Y-%m-%d %H:%M UTC} -- skipping")
            continue
        actual = observations.fetch_actuals(airport, target_date)
        if not actual:
            print(f"  - {airport.icao:5} no observations")
            continue
        provisional = actual["n_obs"] < 12  # sparse day -> treat as provisional
        obs_by_icao[airport.icao] = actual
        obs_rows.append({
            "icao": airport.icao,
            "target_date": target,
            "actual_tmax_f": _round1(actual["tmax_f"]),
            "actual_tmin_f": _round1(actual["tmin_f"]),
            "n_obs": actual["n_obs"],
            "provisional": provisional,
        })
        print(f"  + {airport.icao:5} hi={actual['tmax_f']:.1f} lo={actual['tmin_f']:.1f} "
              f"(n={actual['n_obs']}{', provisional' if provisional else ''})")
    if obs_rows:
        store.upsert_observations(pd.DataFrame(obs_rows))

    # 2. Join the day's forecasts to the observations and compute errors.
    forecasts = store.read_forecasts()
    if forecasts.empty:
        print("  no forecasts on file to verify.")
        return pd.DataFrame()
    todays = forecasts[forecasts["target_date"] == target]

    ver_rows: list[dict] = []
    for _, f in todays.iterrows():
        actual = obs_by_icao.get(f["icao"])
        if not actual:
            continue
        a_hi, a_lo = actual["tmax_f"], actual["tmin_f"]
        ver_rows.append({
            "source": f["source"],
            "icao": f["icao"],
            "target_date": target,
            "fcst_tmax_f": f["fcst_tmax_f"],
            "fcst_tmin_f": f["fcst_tmin_f"],
            "actual_tmax_f": _round1(a_hi),
            "actual_tmin_f": _round1(a_lo),
            "err_tmax": _err(f["fcst_tmax_f"], a_hi),
            "err_tmin": _err(f["fcst_tmin_f"], a_lo),
            "abs_err_tmax": _abs_err(f["fcst_tmax_f"], a_hi),
            "abs_err_tmin": _abs_err(f["fcst_tmin_f"], a_lo),
        })
    ver = pd.DataFrame(ver_rows)
    if not ver.empty:
        store.upsert_verification(ver)
    print(f"Scored {len(ver)} forecast/observation pairs.")
    return ver


# --------------------------------------------------------------------------
# Scoreboard: rolling accuracy ranking
# --------------------------------------------------------------------------
def build_scoreboard(by_airport: bool = False) -> pd.DataFrame:
    """Rolling MAE / bias / RMSE / hit-rate per source over each window."""
    ver = store.read_verification()
    if ver.empty:
        return pd.DataFrame()
    ver = ver.copy()
    ver["target_date"] = pd.to_datetime(ver["target_date"])
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    group_cols = ["source", "icao"] if by_airport else ["source"]

    out: list[pd.DataFrame] = []
    for window in config.SCOREBOARD_WINDOWS:
        scope = ver if window is None else ver[ver["target_date"] >= today - pd.Timedelta(days=window)]
        if scope.empty:
            continue
        agg = scope.groupby(group_cols).apply(_metrics, include_groups=False).reset_index()
        agg["window_days"] = "all" if window is None else window
        out.append(agg)

    board = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if not board.empty:
        # Rank each window by combined high+low mean absolute error.
        board["mae_combined"] = (board["mae_tmax"] + board["mae_tmin"]) / 2
        board = board.sort_values(["window_days", "mae_combined"]).reset_index(drop=True)
    return board


def _metrics(g: pd.DataFrame) -> pd.Series:
    tol = config.HIT_TOLERANCE_F
    return pd.Series({
        "n": len(g),
        "mae_tmax": g["abs_err_tmax"].mean(),
        "mae_tmin": g["abs_err_tmin"].mean(),
        "bias_tmax": g["err_tmax"].mean(),
        "bias_tmin": g["err_tmin"].mean(),
        "rmse_tmax": math.sqrt((g["err_tmax"] ** 2).mean()),
        "rmse_tmin": math.sqrt((g["err_tmin"] ** 2).mean()),
        "hit_rate_tmax": (g["abs_err_tmax"] <= tol).mean(),
        "hit_rate_tmin": (g["abs_err_tmin"] <= tol).mean(),
    })


# --------------------------------------------------------------------------
# Small numeric / timing helpers
# --------------------------------------------------------------------------
def _day_complete_utc(target_date: date, tz: str) -> datetime:
    """UTC instant after which `target_date` is finished for this timezone.

    The local day ends at the following local midnight; we add a finalize
    buffer to allow late-arriving observations to post before we score.
    """
    end_local = datetime.combine(target_date + timedelta(days=1), time.min, tzinfo=ZoneInfo(tz))
    end_utc = end_local.astimezone(timezone.utc)
    return end_utc + timedelta(hours=config.OBSERVATION_FINALIZE_BUFFER_HOURS)


def _round1(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 1)


def _err(fcst, actual):
    return None if fcst is None or actual is None else round(float(fcst) - float(actual), 1)


def _abs_err(fcst, actual):
    e = _err(fcst, actual)
    return None if e is None else abs(e)
