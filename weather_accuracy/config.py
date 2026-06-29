"""Configuration: airports, source toggles, and runtime settings.

Everything you'd normally want to change lives here. Add or remove airports
by editing AIRPORTS. Turn paid/freemium sources on by setting the matching
environment variable (the source enables itself automatically when its key
is present).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Airports
# --------------------------------------------------------------------------
# `station` is the NWS observation station id. For nearly every major US
# airport this is just the ICAO identifier (K + 3-letter IATA). `tz` is the
# IANA timezone, used to define the local calendar day for verification.
@dataclass(frozen=True)
class Airport:
    icao: str
    name: str
    lat: float
    lon: float
    tz: str
    station: str  # NWS observation station id

    @property
    def iata(self) -> str:
        return self.icao[1:] if self.icao.startswith("K") else self.icao


AIRPORTS: list[Airport] = [
    Airport("KATL", "Atlanta Hartsfield-Jackson", 33.6407, -84.4277, "America/New_York", "KATL"),
    Airport("KMIA", "Miami", 25.7959, -80.2870, "America/New_York", "KMIA"),
    Airport("KDCA", "Washington Reagan National", 38.8512, -77.0402, "America/New_York", "KDCA"),
    Airport("KHOU", "Houston Hobby", 29.6454, -95.2789, "America/Chicago", "KHOU"),
    Airport("KBOS", "Boston Logan", 42.3656, -71.0096, "America/New_York", "KBOS"),
    Airport("KMSY", "New Orleans Louis Armstrong", 29.9934, -90.2580, "America/Chicago", "KMSY"),
    Airport("KOKC", "Oklahoma City Will Rogers", 35.3931, -97.6007, "America/Chicago", "KOKC"),
    Airport("KAUS", "Austin-Bergstrom", 30.1945, -97.6699, "America/Chicago", "KAUS"),
    Airport("KDEN", "Denver", 39.8561, -104.6737, "America/Denver", "KDEN"),
    Airport("KSFO", "San Francisco", 37.6213, -122.3790, "America/Los_Angeles", "KSFO"),
    Airport("KPHL", "Philadelphia", 39.8729, -75.2437, "America/New_York", "KPHL"),
    Airport("KLAX", "Los Angeles", 33.9416, -118.4085, "America/Los_Angeles", "KLAX"),
    Airport("KLAS", "Las Vegas Harry Reid", 36.0840, -115.1537, "America/Los_Angeles", "KLAS"),
    Airport("KSEA", "Seattle-Tacoma", 47.4502, -122.3088, "America/Los_Angeles", "KSEA"),
    Airport("KMSP", "Minneapolis-St. Paul", 44.8848, -93.2223, "America/Chicago", "KMSP"),
    Airport("KMDW", "Chicago Midway", 41.7868, -87.7522, "America/Chicago", "KMDW"),
    Airport("KPHX", "Phoenix Sky Harbor", 33.4342, -112.0116, "America/Phoenix", "KPHX"),
    Airport("KDFW", "Dallas-Fort Worth", 32.8998, -97.0403, "America/Chicago", "KDFW"),
    Airport("KSAT", "San Antonio", 29.5337, -98.4698, "America/Chicago", "KSAT"),
]


# --------------------------------------------------------------------------
# API keys for optional (freemium) sources.
# Leave unset to run on the free, keyless sources only (NWS + Open-Meteo).
# --------------------------------------------------------------------------
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "").strip()
TOMORROWIO_API_KEY = os.environ.get("TOMORROWIO_API_KEY", "").strip()
VISUALCROSSING_API_KEY = os.environ.get("VISUALCROSSING_API_KEY", "").strip()

# Required by the NWS API. Set to something identifying you / your org.
# https://www.weather.gov/documentation/services-web-api
NWS_USER_AGENT = os.environ.get(
    "NWS_USER_AGENT", "weather-accuracy-report (contact: you@example.com)"
)


# --------------------------------------------------------------------------
# Runtime settings
# --------------------------------------------------------------------------
# Networking
REQUEST_TIMEOUT = 20          # seconds per HTTP request
RETRIES = 3                   # retry attempts on transient failures
RETRY_BACKOFF = 2.0           # seconds, multiplied by attempt number
THROTTLE_SECONDS = 0.3        # polite pause between calls to the same host

# "Hit rate" tolerance: a forecast counts as a hit if it is within this many
# degrees Fahrenheit of the observed value.
HIT_TOLERANCE_F = 2.0

# Hours to wait after a local day ends before its NWS observations are
# considered final. The verification job only scores a day once it is fully
# over in the airport's local timezone plus this buffer, so accuracy data is
# never pulled before the following day.
OBSERVATION_FINALIZE_BUFFER_HOURS = float(os.environ.get("WX_FINALIZE_BUFFER_HOURS", "3"))

# Rolling windows (in days) reported on the scoreboard. None == all-time.
SCOREBOARD_WINDOWS = [7, 30, 90, None]

# Storage location for the local-Parquet fallback (ignored on Databricks,
# which uses Delta tables instead -- see store.py).
LOCAL_DATA_DIR = os.environ.get("WX_DATA_DIR", "./wx_data")

# Delta table names used when running on Databricks.
DELTA_CATALOG = os.environ.get("WX_DELTA_CATALOG", "main")
DELTA_SCHEMA = os.environ.get("WX_DELTA_SCHEMA", "weather_accuracy")
