"""Ground-truth observations from the National Weather Service.

The official daily max/min ultimately comes from the local climate report, but
the fully automated, no-key route is to pull the airport's hourly observations
for the local calendar day and take the max and min. That is what we do here.

We return:
    {"tmax_f": float, "tmin_f": float, "n_obs": int}   (or None)

`n_obs` lets the pipeline flag days with sparse data (station outages), which
should be treated as provisional rather than authoritative.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from . import config
from .sources import _c_to_f, _get_json


def fetch_actuals(airport: "config.Airport", local_date: date) -> Optional[dict]:
    """Observed high/low for `local_date` at the airport, in Fahrenheit."""
    tz = ZoneInfo(airport.tz)
    # Local midnight-to-midnight window, expressed in UTC for the API query.
    start_local = datetime.combine(local_date, time.min, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)

    headers = {"User-Agent": config.NWS_USER_AGENT, "Accept": "application/geo+json"}
    data = _get_json(
        f"https://api.weather.gov/stations/{airport.station}/observations",
        params={
            "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        headers=headers,
    )
    if not data:
        return None

    temps_f: list[float] = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        temp = props.get("temperature", {}) or {}
        value = temp.get("value")  # NWS reports observation temps in Celsius
        if value is None:
            continue
        # Guard the boundary: keep only observations inside the local day.
        ts = props.get("timestamp")
        if ts:
            obs_local = datetime.fromisoformat(ts).astimezone(tz).date()
            if obs_local != local_date:
                continue
        f = _c_to_f(value)
        if f is not None:
            temps_f.append(f)

    if not temps_f:
        return None
    return {"tmax_f": max(temps_f), "tmin_f": min(temps_f), "n_obs": len(temps_f)}
