"""Forecast source adapters.

Each adapter fetches the same-day high/low for one airport from one provider
and normalizes it to a common shape:

    {"tmax_f": float, "tmin_f": float}   (or None if unavailable)

Temperatures are always returned in degrees Fahrenheit. "Same day" means the
airport's local calendar day (the date passed in as `local_date`).

Keyless sources (NWS, Open-Meteo) are always enabled. The freemium sources
(OpenWeather, Tomorrow.io, Visual Crossing) enable themselves only when their
API key is configured in config.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import requests

from . import config


# --------------------------------------------------------------------------
# HTTP helper with retries + polite throttling
# --------------------------------------------------------------------------
def _get_json(url: str, *, params: dict | None = None, headers: dict | None = None) -> Optional[dict]:
    last_err: Exception | None = None
    for attempt in range(1, config.RETRIES + 1):
        try:
            resp = requests.get(
                url, params=params, headers=headers, timeout=config.REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            time.sleep(config.THROTTLE_SECONDS)
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - we genuinely want to retry anything transient
            last_err = exc
            if attempt < config.RETRIES:
                time.sleep(config.RETRY_BACKOFF * attempt)
    print(f"  ! request failed after {config.RETRIES} tries: {url} ({last_err})")
    return None


def _c_to_f(celsius: Optional[float]) -> Optional[float]:
    return None if celsius is None else celsius * 9.0 / 5.0 + 32.0


# --------------------------------------------------------------------------
# National Weather Service (official forecast)
# --------------------------------------------------------------------------
# The /points endpoint resolves a lat/lon to the office forecast URL, which we
# cache per airport to avoid re-resolving every run.
_NWS_FORECAST_URL_CACHE: dict[str, str] = {}


def _nws_forecast_url(airport: "config.Airport") -> Optional[str]:
    if airport.icao in _NWS_FORECAST_URL_CACHE:
        return _NWS_FORECAST_URL_CACHE[airport.icao]
    headers = {"User-Agent": config.NWS_USER_AGENT, "Accept": "application/geo+json"}
    points = _get_json(
        f"https://api.weather.gov/points/{airport.lat:.4f},{airport.lon:.4f}",
        headers=headers,
    )
    if not points:
        return None
    url = points.get("properties", {}).get("forecast")
    if url:
        _NWS_FORECAST_URL_CACHE[airport.icao] = url
    return url


def fetch_nws(airport: "config.Airport", local_date: date) -> Optional[dict]:
    url = _nws_forecast_url(airport)
    if not url:
        return None
    headers = {"User-Agent": config.NWS_USER_AGENT, "Accept": "application/geo+json"}
    data = _get_json(url, headers=headers)
    if not data:
        return None
    target = local_date.isoformat()
    tmax = tmin = None
    # NWS periods alternate daytime (high) / night (low); temperatures are
    # already Fahrenheit by default. Match periods whose start date is today.
    for period in data.get("properties", {}).get("periods", []):
        start = period.get("startTime", "")[:10]
        if start != target:
            continue
        temp = period.get("temperature")
        if period.get("isDaytime"):
            tmax = temp if tmax is None else max(tmax, temp)
        else:
            tmin = temp if tmin is None else min(tmin, temp)
    if tmax is None and tmin is None:
        return None
    return {"tmax_f": tmax, "tmin_f": tmin}


# --------------------------------------------------------------------------
# Open-Meteo (keyless) -- one adapter per underlying model
# --------------------------------------------------------------------------
def fetch_open_meteo(airport: "config.Airport", local_date: date, model: str) -> Optional[dict]:
    data = _get_json(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": f"{airport.lat:.4f}",
            "longitude": f"{airport.lon:.4f}",
            "daily": "temperature_2m_max,temperature_2m_min",
            "temperature_unit": "fahrenheit",
            "timezone": "auto",
            "forecast_days": 1,
            "models": model,
        },
    )
    if not data:
        return None
    daily = data.get("daily", {})
    days = daily.get("time", [])
    if not days:
        return None
    # With timezone=auto + forecast_days=1, index 0 is the local "today".
    try:
        idx = days.index(local_date.isoformat())
    except ValueError:
        idx = 0
    highs = daily.get("temperature_2m_max", [])
    lows = daily.get("temperature_2m_min", [])
    if idx >= len(highs) or idx >= len(lows):
        return None
    return {"tmax_f": highs[idx], "tmin_f": lows[idx]}


# Open-Meteo model identifiers exposed as separate sources. The "_seamless"
# variants stitch each provider's global + regional resolutions into one series.
# Add or remove rows here to change which models are tracked. (Open-Meteo's free
# tier is non-commercial use only.)
OPEN_METEO_MODELS: list[tuple[str, str]] = [
    ("om_best", "best_match"),           # Open-Meteo's blended best-available forecast
    ("om_ecmwf", "ecmwf_ifs025"),        # ECMWF IFS (European)
    ("om_gfs", "gfs_seamless"),          # NOAA GFS/HRRR (US)
    ("om_icon", "icon_seamless"),        # DWD ICON (German)
    ("om_gem", "gem_seamless"),          # Environment Canada GEM
    ("om_ukmo", "ukmo_seamless"),        # UK Met Office
    ("om_meteofrance", "meteofrance_seamless"),  # Meteo-France ARPEGE/AROME
    ("om_jma", "jma_seamless"),          # Japan Meteorological Agency
]


# --------------------------------------------------------------------------
# MET Norway (Yr) -- keyless, independent provider
# --------------------------------------------------------------------------
# api.met.no requires a descriptive, identifying User-Agent (it rejects generic
# ones) -- we reuse the NWS user-agent, so set your contact in config.py.
def fetch_metno(airport: "config.Airport", local_date: date) -> Optional[dict]:
    data = _get_json(
        "https://api.met.no/weatherapi/locationforecast/2.0/compact",
        params={"lat": f"{airport.lat:.4f}", "lon": f"{airport.lon:.4f}"},
        headers={"User-Agent": config.NWS_USER_AGENT},
    )
    if not data:
        return None
    tz = ZoneInfo(airport.tz)
    temps_c: list[float] = []
    for entry in data.get("properties", {}).get("timeseries", []):
        ts = entry.get("time")
        if not ts:
            continue
        try:
            when = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(tz)
        except ValueError:
            continue
        if when.date() != local_date:
            continue
        t = (entry.get("data", {}).get("instant", {})
                  .get("details", {}).get("air_temperature"))
        if t is not None:
            temps_c.append(t)  # Celsius
    if not temps_c:
        return None
    return {"tmax_f": _c_to_f(max(temps_c)), "tmin_f": _c_to_f(min(temps_c))}


# --------------------------------------------------------------------------
# OpenWeather (One Call 3.0) -- optional, needs OPENWEATHER_API_KEY
# --------------------------------------------------------------------------
def fetch_openweather(airport: "config.Airport", local_date: date) -> Optional[dict]:
    data = _get_json(
        "https://api.openweathermap.org/data/3.0/onecall",
        params={
            "lat": f"{airport.lat:.4f}",
            "lon": f"{airport.lon:.4f}",
            "units": "imperial",
            "exclude": "current,minutely,hourly,alerts",
            "appid": config.OPENWEATHER_API_KEY,
        },
    )
    if not data:
        return None
    for day in data.get("daily", []):
        # dt is unix seconds at local noon; compare date in airport tz.
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo

        day_local = datetime.fromtimestamp(day["dt"], tz=timezone.utc).astimezone(
            ZoneInfo(airport.tz)
        ).date()
        if day_local == local_date:
            temp = day.get("temp", {})
            return {"tmax_f": temp.get("max"), "tmin_f": temp.get("min")}
    return None


# --------------------------------------------------------------------------
# Tomorrow.io -- optional, needs TOMORROWIO_API_KEY
# --------------------------------------------------------------------------
def fetch_tomorrowio(airport: "config.Airport", local_date: date) -> Optional[dict]:
    data = _get_json(
        "https://api.tomorrow.io/v4/weather/forecast",
        params={
            "location": f"{airport.lat:.4f},{airport.lon:.4f}",
            "timesteps": "1d",
            "units": "imperial",
            "apikey": config.TOMORROWIO_API_KEY,
        },
    )
    if not data:
        return None
    for day in data.get("timelines", {}).get("daily", []):
        if day.get("time", "")[:10] == local_date.isoformat():
            vals = day.get("values", {})
            return {"tmax_f": vals.get("temperatureMax"), "tmin_f": vals.get("temperatureMin")}
    return None


# --------------------------------------------------------------------------
# Visual Crossing -- optional, needs VISUALCROSSING_API_KEY
# --------------------------------------------------------------------------
def fetch_visualcrossing(airport: "config.Airport", local_date: date) -> Optional[dict]:
    data = _get_json(
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/"
        f"timeline/{airport.lat:.4f},{airport.lon:.4f}/{local_date.isoformat()}",
        params={
            "unitGroup": "us",
            "include": "days",
            "key": config.VISUALCROSSING_API_KEY,
            "elements": "datetime,tempmax,tempmin",
        },
    )
    if not data:
        return None
    for day in data.get("days", []):
        if day.get("datetime") == local_date.isoformat():
            return {"tmax_f": day.get("tempmax"), "tmin_f": day.get("tempmin")}
    return None


# --------------------------------------------------------------------------
# Source registry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    name: str
    fetch: Callable[["config.Airport", date], Optional[dict]]
    enabled: bool


def all_sources() -> list[Source]:
    # Keyless: NWS, all Open-Meteo models, and MET Norway.
    sources: list[Source] = [Source("nws", fetch_nws, True)]
    for name, model in OPEN_METEO_MODELS:
        sources.append(Source(name, (lambda a, d, m=model: fetch_open_meteo(a, d, m)), True))
    sources.append(Source("metno", fetch_metno, True))
    # Optional freemium brands -- enabled only when their key is set.
    sources += [
        Source("openweather", fetch_openweather, bool(config.OPENWEATHER_API_KEY)),
        Source("tomorrowio", fetch_tomorrowio, bool(config.TOMORROWIO_API_KEY)),
        Source("visualcrossing", fetch_visualcrossing, bool(config.VISUALCROSSING_API_KEY)),
    ]
    return [s for s in sources if s.enabled]
