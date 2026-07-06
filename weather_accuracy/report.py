"""Self-contained HTML report for the standalone (non-Databricks) setup.

Reads the stored tables and renders a single .html file you can open in any
browser by double-clicking it -- no server, no internet, no dependencies
beyond what the pipeline already uses. Regenerated each day by the scheduled
run so it always shows the latest standings.
"""
from __future__ import annotations

import html
import os
from datetime import datetime, timezone

import pandas as pd

from . import analysis, config, pipeline, store


# --------------------------------------------------------------------------
# Color scale for mean-absolute-error cells (degrees F). Lower = better.
# --------------------------------------------------------------------------
def _mae_color(mae: float | None) -> str:
    if mae is None or pd.isna(mae):
        return "#f3f3f1"
    if mae <= 2:
        return "#cfe8d6"   # green
    if mae <= 4:
        return "#eef0c9"   # yellow-green
    if mae <= 6:
        return "#f7e3c0"   # amber
    if mae <= 9:
        return "#f6d2bd"   # orange
    return "#f2c0c0"        # red


def _fmt(x, nd=1):
    if x is None or pd.isna(x):
        return "&mdash;"
    return f"{float(x):.{nd}f}"


def _pct(x):
    return "&mdash;" if x is None or pd.isna(x) else f"{float(x) * 100:.0f}%"


# --------------------------------------------------------------------------
# HTML building blocks
# --------------------------------------------------------------------------
def _pick_window(board: pd.DataFrame, preferred="30") -> tuple[pd.DataFrame, str]:
    """Use the preferred rolling window if it has data, else fall back."""
    if board.empty:
        return board, preferred
    for w in (preferred, "90", "7", "all"):
        sub = board[board["window"].astype(str) == w] if "window" in board else \
              board[board["window_days"].astype(str) == w]
        if not sub.empty:
            return sub, w
    return board, preferred


def build_html(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    fc = store.read_forecasts()
    ver = store.read_verification()
    board = pipeline.build_scoreboard(by_airport=False)
    if not board.empty:
        board = board.rename(columns={"window_days": "window"})
        board["window"] = board["window"].astype(str)

    src_series = fc["source"] if not fc.empty else (ver["source"] if not ver.empty else None)
    n_sources = int(src_series.nunique()) if src_series is not None else 0
    if ver.empty:
        days, span = 0, "&mdash;"
    else:
        dates = pd.to_datetime(ver["target_date"])
        days = dates.dt.date.nunique()
        span = f"{dates.min():%b %d, %Y} &ndash; {dates.max():%b %d, %Y}"

    parts: list[str] = [_HEAD]
    parts.append(f"""
      <h1>Airport forecast accuracy</h1>
      <p class="meta">Generated {now:%b %d, %Y %H:%M UTC} &middot;
        {len(config.AIRPORTS)} airports &middot; {n_sources} sources &middot;
        {days} day{'s' if days != 1 else ''} of verified accuracy data ({span})</p>
    """)

    if fc.empty and ver.empty:
        parts.append('<div class="empty">No data yet. Run '
                     '<code>python -m weather_accuracy daily</code> once to capture '
                     "today's forecasts; accuracy scores follow the next day.</div>")
        parts.append(_FOOT)
        return "".join(parts)

    # The daily forecasts themselves -- shown as soon as any have been captured.
    if not fc.empty:
        parts.append(_forecasts_section(fc))
        parts.append(_spread_section(fc))

    if ver.empty:
        parts.append('<div class="empty">Accuracy scores will appear here once the '
                     "first day completes and is verified (tomorrow). Today's "
                     "forecasts are shown above.</div>")
    else:
        parts.append(_leaderboard_section(board))
        parts.append(_yesterday_section(ver))
        parts.append(_heatmap_section())
    parts.append(_FOOT)
    return "".join(parts)


def _forecasts_section(fc: pd.DataFrame) -> str:
    fc = fc.copy()
    latest = fc["target_date"].max()
    day = fc[fc["target_date"] == latest]
    label = pd.to_datetime(latest).strftime("%b %d, %Y")

    real = day[~day["source"].map(analysis.is_consensus)]
    sources = list(dict.fromkeys(real["source"]))          # appearance order
    present = set(real["icao"])
    order = [a.icao for a in config.AIRPORTS if a.icao in present]
    hi = real.set_index(["icao", "source"])["fcst_tmax_f"].to_dict()
    lo = real.set_index(["icao", "source"])["fcst_tmin_f"].to_dict()
    # Stored consensus values (fall back to computed mean if disabled/absent).
    cmean_hi = day[day["source"] == analysis.CONSENSUS_MEAN].set_index("icao")["fcst_tmax_f"].to_dict()
    cmean_lo = day[day["source"] == analysis.CONSENSUS_MEAN].set_index("icao")["fcst_tmin_f"].to_dict()
    cwtd_hi = day[day["source"] == analysis.CONSENSUS_WTD].set_index("icao")["fcst_tmax_f"].to_dict()
    cwtd_lo = day[day["source"] == analysis.CONSENSUS_WTD].set_index("icao")["fcst_tmin_f"].to_dict()
    has_cons = bool(cwtd_hi)

    head = "".join(f"<th>{html.escape(s)}</th>" for s in sources)
    cons_head = ("<th>Mean</th><th>Wtd&#9733;</th>" if has_cons else "<th>Average</th>")
    body = []
    for icao in order:
        his = [hi.get((icao, s)) for s in sources]
        los = [lo.get((icao, s)) for s in sources]
        cells = "".join(f"<td class='hl'>{_hilo(h, l)}</td>" for h, l in zip(his, los))
        if has_cons:
            cons = (f"<td class='hl cons'>{_hilo(cmean_hi.get(icao), cmean_lo.get(icao))}</td>"
                    f"<td class='hl cons wtd'>{_hilo(cwtd_hi.get(icao), cwtd_lo.get(icao))}</td>")
        else:
            vh = [x for x in his if x is not None and not pd.isna(x)]
            vl = [x for x in los if x is not None and not pd.isna(x)]
            cons = (f"<td class='hl cons'>"
                    f"{_hilo(sum(vh)/len(vh) if vh else None, sum(vl)/len(vl) if vl else None)}</td>")
        body.append(f"<tr><td class='src'>{icao}</td>{cells}{cons}</tr>")

    note = ("Last two columns are the equal-weight mean and the "
            "accuracy-weighted consensus (&#9733;). "
            if has_cons else "The last column is the average across sources. ")
    return f"""
      <h2>Daily forecasts <span class="sub">(high / low &deg;F for {label})</span></h2>
      <p class="note">What each source is predicting for every airport. {note}</p>
      <div class="scroll">
      <table class="fcst">
        <thead><tr><th>Airport</th>{head}{cons_head}</tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table></div>"""


def _spread_section(fc: pd.DataFrame) -> str:
    fc = fc.copy()
    latest = fc["target_date"].max()
    day = fc[fc["target_date"] == latest]
    label = pd.to_datetime(latest).strftime("%b %d, %Y")
    sp = analysis.forecast_spread(day)
    if sp.empty:
        return ""
    order = {a.icao: i for i, a in enumerate(config.AIRPORTS)}
    sp = sp.sort_values("icao", key=lambda s: s.map(lambda x: order.get(x, 999)))

    rows = []
    for _, r in sp.iterrows():
        lab = analysis.spread_label(r["high_std"])
        color = {"tight": "#cfe8d6", "moderate": "#f7e3c0", "wide": "#f2c0c0"}.get(lab, "#f3f3f1")
        rng_hi = (f"{r['high_min']:.0f}&ndash;{r['high_max']:.0f}"
                  if pd.notna(r["high_min"]) else "&mdash;")
        rng_lo = (f"{r['low_min']:.0f}&ndash;{r['low_max']:.0f}"
                  if pd.notna(r["low_min"]) else "&mdash;")
        rows.append(f"""
          <tr>
            <td class="src">{r['icao']}</td>
            <td class="num">{_fmt(r['high_std'])}</td>
            <td class="num dim">{rng_hi}</td>
            <td class="num">{_fmt(r['low_std'])}</td>
            <td class="num dim">{rng_lo}</td>
            <td class="hl"><span class="lean" style="background:{color}">{lab}</span></td>
          </tr>""")
    return f"""
      <h2>Forecast agreement <span class="sub">(spread across sources, {label})</span></h2>
      <p class="note">How tightly the sources agree, by airport. &sigma; is the standard
        deviation of the {sp['n'].max()} source forecasts (&deg;F) &mdash; smaller means
        stronger agreement and a more trustworthy consensus; wider spread signals
        greater uncertainty for that day.</p>
      <table class="board">
        <thead><tr>
          <th>Airport</th><th>High &sigma;</th><th>High range</th>
          <th>Low &sigma;</th><th>Low range</th><th>Agreement</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>"""


def _hilo(hi, lo) -> str:
    h_na = hi is None or pd.isna(hi)
    l_na = lo is None or pd.isna(lo)
    if h_na and l_na:
        return "&mdash;"
    h = "&middot;" if h_na else f"{float(hi):.0f}"
    l = "&middot;" if l_na else f"{float(lo):.0f}"
    return f"{h}&deg; / {l}&deg;"


def _leaderboard_section(board: pd.DataFrame) -> str:
    sub, window = _pick_window(board, "30")
    sub = sub.sort_values("mae_combined")
    label = "all-time" if window == "all" else f"last {window} days"
    max_mae = max(sub["mae_combined"].max(), 0.1)

    rows = []
    for rank, (_, r) in enumerate(sub.iterrows(), start=1):
        bar = 100 * float(r["mae_combined"]) / max_mae
        bias = float(r["bias_high"]) if "bias_high" in r else float(r["bias_tmax"])
        lean = "warm" if bias > 1 else "cold" if bias < -1 else "even"
        mae_hi = r.get("mae_high", r.get("mae_tmax"))
        mae_lo = r.get("mae_low", r.get("mae_tmin"))
        hit = r.get("hit_rate_high", r.get("hit_rate_tmax"))
        cons = analysis.is_consensus(r["source"])
        name = html.escape(str(r["source"]))
        if cons:
            name = f"&#9733; {name}"
        rows.append(f"""
          <tr class="{'consrow' if cons else ''}">
            <td class="rank">{rank}</td>
            <td class="src">{name}</td>
            <td class="bar"><div class="track"><div class="fill"
                style="width:{bar:.0f}%"></div></div><span>{_fmt(r['mae_combined'])}&deg;</span></td>
            <td class="num">{_fmt(mae_hi)}</td>
            <td class="num">{_fmt(mae_lo)}</td>
            <td class="num">{'+' if bias >= 0 else ''}{_fmt(bias)} <span class="lean {lean}">{lean}</span></td>
            <td class="num">{_pct(hit)}</td>
            <td class="num dim">{int(r['n'])}</td>
          </tr>""")

    return f"""
      <h2>Accuracy leaderboard <span class="sub">({label})</span></h2>
      <p class="note">Ranked by combined high + low mean absolute error &mdash; lower is more accurate.
        Bias shows whether a source runs systematically warm or cold. Hit rate = share of
        high-temp forecasts within &plusmn;{config.HIT_TOLERANCE_F:.0f}&deg;F.</p>
      <table class="board">
        <thead><tr>
          <th>#</th><th>Source</th><th>Combined error</th>
          <th>MAE high</th><th>MAE low</th><th>Bias (high)</th>
          <th>Hit rate</th><th>n</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>"""


def _yesterday_section(ver: pd.DataFrame) -> str:
    v = ver.copy()
    v["target_date"] = pd.to_datetime(v["target_date"])
    last = v["target_date"].max()
    day = v[v["target_date"] == last]
    agg = (day.groupby("source")["abs_err_tmax"].mean()
              .sort_values().reset_index())
    rows = "".join(
        f"<tr><td class='src'>{html.escape(str(s))}</td>"
        f"<td class='num' style='background:{_mae_color(m)}'>{_fmt(m)}&deg;</td></tr>"
        for s, m in zip(agg["source"], agg["abs_err_tmax"])
    )
    return f"""
      <h2>Most recent verified day <span class="sub">({last:%b %d, %Y})</span></h2>
      <p class="note">Average high-temperature miss across all airports that day.</p>
      <table class="mini">
        <thead><tr><th>Source</th><th>Avg high error</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>"""


def _heatmap_section() -> str:
    by_air = pipeline.build_scoreboard(by_airport=True)
    if by_air.empty:
        return ""
    by_air = by_air.rename(columns={"window_days": "window"})
    by_air["window"] = by_air["window"].astype(str)
    sub, window = _pick_window(by_air, "30")
    label = "all-time" if window == "all" else f"last {window} days"

    pivot = sub.pivot_table(index="icao", columns="source",
                            values="mae_tmax", aggfunc="mean")
    # Preserve the configured airport order.
    order = [a.icao for a in config.AIRPORTS if a.icao in pivot.index]
    pivot = pivot.reindex(order)
    sources = list(pivot.columns)

    head = "".join(f"<th>{html.escape(s)}</th>" for s in sources)
    body = []
    for icao, row in pivot.iterrows():
        cells = "".join(
            f"<td class='num' style='background:{_mae_color(row[s])}'>{_fmt(row[s])}</td>"
            for s in sources
        )
        body.append(f"<tr><td class='src'>{icao}</td>{cells}</tr>")

    return f"""
      <h2>Accuracy by airport <span class="sub">({label})</span></h2>
      <p class="note">High-temperature mean absolute error (&deg;F) per airport and source.
        Greener is more accurate; redder is less.</p>
      <div class="scroll">
      <table class="heat">
        <thead><tr><th>Airport</th>{head}</tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table></div>"""


def write_report(path: str | None = None, now: datetime | None = None) -> str:
    path = path or os.path.join(config.LOCAL_DATA_DIR, "report.html")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build_html(now))
    print(f"Report written to {os.path.abspath(path)}")
    return path


# --------------------------------------------------------------------------
# Static head/foot (inline CSS so the file is fully self-contained)
# --------------------------------------------------------------------------
_HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Airport forecast accuracy</title>
<style>
  :root { color-scheme: light; }
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         max-width: 1000px; margin: 0 auto; padding: 32px 20px 64px;
         color: #1c1c1a; background: #fbfbf9; line-height: 1.5; }
  h1 { font-size: 26px; font-weight: 600; margin: 0 0 4px; }
  h2 { font-size: 18px; font-weight: 600; margin: 36px 0 4px; }
  .sub { font-weight: 400; color: #76756f; font-size: 15px; }
  .meta { color: #76756f; font-size: 14px; margin: 0 0 8px; }
  .note { color: #76756f; font-size: 13px; margin: 2px 0 12px; }
  .empty { background: #fff; border: 1px solid #e6e5df; border-radius: 10px;
           padding: 24px; color: #555; }
  table { border-collapse: collapse; width: 100%; background: #fff;
          border: 1px solid #e6e5df; border-radius: 10px; overflow: hidden; font-size: 14px; }
  th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #efeee9; }
  thead th { background: #f4f3ee; font-weight: 600; font-size: 12px;
             text-transform: uppercase; letter-spacing: .03em; color: #57564f; }
  tbody tr:last-child td { border-bottom: none; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .hl { text-align: center; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .hl.cons { background: #f6f5f0; font-weight: 600; }
  .hl.cons.wtd { background: #e7edf5; }
  .fcst td.src { position: sticky; left: 0; background: #fff; }
  .dim { color: #9a988f; }
  .rank { color: #9a988f; width: 28px; }
  .src { font-weight: 500; }
  .board .bar { width: 38%; }
  .track { display: inline-block; width: calc(100% - 56px); height: 8px;
           background: #efeee9; border-radius: 5px; vertical-align: middle; margin-right: 8px; }
  .fill { height: 8px; background: #5b7fb0; border-radius: 5px; }
  .bar span { font-variant-numeric: tabular-nums; color: #57564f; }
  .lean { font-size: 11px; padding: 1px 6px; border-radius: 10px; margin-left: 4px; }
  .lean.warm { background: #f6d2bd; color: #8a3b1c; }
  .lean.cold { background: #cfe0f0; color: #1d4e7a; }
  .lean.even { background: #e9e8e2; color: #6b6a63; }
  .consrow { background: #f3f6fb; }
  .consrow .src { font-weight: 600; }
  .mini { max-width: 360px; }
  .scroll { overflow-x: auto; }
  .heat td.src { position: sticky; left: 0; background: #fff; }
</style></head><body>"""

_FOOT = """
  <p class="meta" style="margin-top:40px">Ground truth: National Weather Service
  observations at each airport. High-temperature accuracy is the headline metric;
  same-day low forecasts are partly settled by morning and read artificially well.</p>
  </body></html>"""
