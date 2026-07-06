"""Statistical analysis of the forecasts.

Two things live here:

1. forecast_spread()  -- per airport, per day: how tightly the sources agree
   (mean, standard deviation, and min-max range of the high/low forecasts).

2. build_consensus_rows()  -- an accuracy-weighted consensus forecast. Each
   source is weighted by its recent accuracy (weight = 1 / MAE, so a source
   with half the error gets double the weight), computed per airport with a
   fallback to the source's overall record and then to equal weights. Weights
   use only accuracy known at forecast time, so there is no look-ahead: the
   consensus is stored as its own source and scored day-to-day like the rest.

Two consensus sources are produced so you can tell whether weighting helps:
   consensus_mean  -- plain equal-weight average of the sources
   consensus_wtd   -- the accuracy-weighted average
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import config, store

CONSENSUS_MEAN = "consensus_mean"
CONSENSUS_WTD = "consensus_wtd"
CONSENSUS_SOURCES = (CONSENSUS_MEAN, CONSENSUS_WTD)


def is_consensus(source) -> bool:
    return str(source).startswith("consensus")


# --------------------------------------------------------------------------
# 1. Forecast spread (agreement) among sources
# --------------------------------------------------------------------------
def forecast_spread(day_fc: pd.DataFrame) -> pd.DataFrame:
    """Per-airport dispersion of the real source forecasts for a single day."""
    real = day_fc[~day_fc["source"].map(is_consensus)]
    rows = []
    for icao, g in real.groupby("icao"):
        his = g["fcst_tmax_f"].dropna()
        los = g["fcst_tmin_f"].dropna()
        rows.append({
            "icao": icao,
            "n": int(len(g)),
            "high_mean": his.mean() if len(his) else None,
            "high_std": his.std(ddof=0) if len(his) else None,
            "high_min": his.min() if len(his) else None,
            "high_max": his.max() if len(his) else None,
            "high_range": (his.max() - his.min()) if len(his) else None,
            "low_mean": los.mean() if len(los) else None,
            "low_std": los.std(ddof=0) if len(los) else None,
            "low_min": los.min() if len(los) else None,
            "low_max": los.max() if len(los) else None,
            "low_range": (los.max() - los.min()) if len(los) else None,
        })
    return pd.DataFrame(rows)


def spread_label(high_std) -> str:
    if high_std is None or pd.isna(high_std):
        return "—"
    if high_std <= config.SPREAD_TIGHT_F:
        return "tight"
    if high_std <= config.SPREAD_WIDE_F:
        return "moderate"
    return "wide"


# --------------------------------------------------------------------------
# 2. Accuracy-weighted consensus
# --------------------------------------------------------------------------
def source_error_tables(now: datetime | None = None, window_days: int | None = None):
    """Return (per_airport, per_source) MAE tables over the trailing window.

    Computed from the verification table, excluding consensus rows. Used to
    weight the real sources when building the consensus.
    """
    ver = store.read_verification()
    empty = (pd.DataFrame(), pd.DataFrame())
    if ver.empty:
        return empty
    ver = ver[~ver["source"].map(is_consensus)].copy()
    if ver.empty:
        return empty
    now = now or datetime.now(timezone.utc)
    window_days = window_days or config.CONSENSUS_WINDOW_DAYS
    ver["target_date"] = pd.to_datetime(ver["target_date"])
    cutoff = pd.Timestamp(now.date()) - pd.Timedelta(days=window_days)
    ver = ver[ver["target_date"] >= cutoff]
    if ver.empty:
        return empty

    per_air = (ver.groupby(["source", "icao"])
                  .agg(n=("abs_err_tmax", "size"),
                       mae_high=("abs_err_tmax", "mean"),
                       mae_low=("abs_err_tmin", "mean"))
                  .reset_index())
    per_src = (ver.groupby("source")
                  .agg(n=("abs_err_tmax", "size"),
                       mae_high=("abs_err_tmax", "mean"),
                       mae_low=("abs_err_tmin", "mean"))
                  .reset_index())
    return per_air, per_src


def _weight(mae) -> float:
    if mae is None or pd.isna(mae):
        return 0.0
    base = max(float(mae), 0.0) + config.CONSENSUS_MAE_FLOOR
    return 1.0 / (base ** config.CONSENSUS_WEIGHT_POWER)


def _lookup_mae(source, icao, field, per_air, per_src):
    """MAE for (source, airport, field), with per-airport -> global fallback."""
    if not per_air.empty:
        hit = per_air[(per_air["source"] == source) & (per_air["icao"] == icao)]
        if not hit.empty and int(hit.iloc[0]["n"]) >= config.CONSENSUS_MIN_SAMPLES:
            return hit.iloc[0][field]
    if not per_src.empty:
        hit = per_src[per_src["source"] == source]
        if not hit.empty:
            return hit.iloc[0][field]
    return None  # no history -> caller falls back to equal weights


def _combine(pairs, weights):
    """Weighted mean of (value) list given matching weights; equal if all zero."""
    vals = [v for v in pairs if v is not None and not pd.isna(v)]
    if not vals:
        return None
    ws = [w for v, w in zip(pairs, weights) if v is not None and not pd.isna(v)]
    total = sum(ws)
    if total <= 0:  # no usable weights yet -> equal weighting
        return sum(vals) / len(vals)
    return sum(v * w for v, w in zip(vals, ws)) / total


def build_consensus_rows(day_rows: pd.DataFrame, issue_ts: datetime,
                         now: datetime | None = None) -> pd.DataFrame:
    """Build consensus_mean and consensus_wtd forecast rows from one day's real
    source forecasts. `day_rows` must contain only real sources."""
    if day_rows.empty:
        return pd.DataFrame()
    per_air, per_src = source_error_tables(now=now)

    out = []
    for (icao, target_date), g in day_rows.groupby(["icao", "target_date"]):
        srcs = list(g["source"])
        highs = list(g["fcst_tmax_f"])
        lows = list(g["fcst_tmin_f"])

        # Equal-weight mean.
        mean_hi = _combine(highs, [1.0] * len(highs))
        mean_lo = _combine(lows, [1.0] * len(lows))

        # Accuracy weights (per field).
        w_hi = [_weight(_lookup_mae(s, icao, "mae_high", per_air, per_src)) for s in srcs]
        w_lo = [_weight(_lookup_mae(s, icao, "mae_low", per_air, per_src)) for s in srcs]
        wtd_hi = _combine(highs, w_hi)
        wtd_lo = _combine(lows, w_lo)

        base = {"issue_ts": issue_ts.replace(microsecond=0), "icao": icao,
                "target_date": target_date, "lead_days": 0}
        out.append({**base, "source": CONSENSUS_MEAN,
                    "fcst_tmax_f": _r(mean_hi), "fcst_tmin_f": _r(mean_lo)})
        out.append({**base, "source": CONSENSUS_WTD,
                    "fcst_tmax_f": _r(wtd_hi), "fcst_tmin_f": _r(wtd_lo)})
    return pd.DataFrame(out)


def _r(x):
    return None if x is None or pd.isna(x) else round(float(x), 1)
