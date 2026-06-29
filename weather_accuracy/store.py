"""Persistence layer.

Runs on two backends with the same interface:

* Databricks  -> Delta tables (when a `spark` session is available)
* anywhere    -> local Parquet files under config.LOCAL_DATA_DIR

The pipeline code never needs to know which one is active.

Tables
------
forecasts     append-only      keys: issue_ts, source, icao, target_date
observations  upsert           keys: icao, target_date
verification  upsert (derived) keys: source, icao, target_date
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from . import config

_FORECASTS = "forecasts"
_OBSERVATIONS = "observations"
_VERIFICATION = "verification"


def _get_spark():
    """Return the active Spark session on Databricks, else None."""
    try:
        from pyspark.sql import SparkSession  # type: ignore

        return SparkSession.getActiveSession()
    except Exception:  # noqa: BLE001
        return None


_SPARK = _get_spark()
ON_DATABRICKS = _SPARK is not None


# ==========================================================================
# Local Parquet backend
# ==========================================================================
def _path(table: str) -> str:
    os.makedirs(config.LOCAL_DATA_DIR, exist_ok=True)
    return os.path.join(config.LOCAL_DATA_DIR, f"{table}.parquet")


def _local_read(table: str) -> pd.DataFrame:
    path = _path(table)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def _local_append(table: str, df: pd.DataFrame, keys: list[str]) -> None:
    existing = _local_read(table)
    combined = pd.concat([existing, df], ignore_index=True) if not existing.empty else df
    combined = combined.drop_duplicates(subset=keys, keep="last")
    combined.to_parquet(_path(table), index=False)


def _local_upsert(table: str, df: pd.DataFrame, keys: list[str]) -> None:
    existing = _local_read(table)
    if not existing.empty:
        merged = df.set_index(keys)
        keep = existing.set_index(keys)
        keep = keep[~keep.index.isin(merged.index)]
        combined = pd.concat([keep.reset_index(), df], ignore_index=True)
    else:
        combined = df
    combined.to_parquet(_path(table), index=False)


# ==========================================================================
# Delta backend (Databricks)
# ==========================================================================
def _fqn(table: str) -> str:
    return f"{config.DELTA_CATALOG}.{config.DELTA_SCHEMA}.{table}"


def _delta_ensure_schema() -> None:
    _SPARK.sql(f"CREATE SCHEMA IF NOT EXISTS {config.DELTA_CATALOG}.{config.DELTA_SCHEMA}")


def _delta_read(table: str) -> pd.DataFrame:
    try:
        return _SPARK.table(_fqn(table)).toPandas()
    except Exception:  # noqa: BLE001 - table not created yet
        return pd.DataFrame()


def _delta_append(table: str, df: pd.DataFrame, keys: list[str]) -> None:
    _delta_ensure_schema()
    sdf = _SPARK.createDataFrame(df)
    sdf.write.format("delta").mode("append").saveAsTable(_fqn(table))


def _delta_upsert(table: str, df: pd.DataFrame, keys: list[str]) -> None:
    _delta_ensure_schema()
    from delta.tables import DeltaTable  # type: ignore

    sdf = _SPARK.createDataFrame(df)
    if not _SPARK.catalog.tableExists(_fqn(table)):
        sdf.write.format("delta").saveAsTable(_fqn(table))
        return
    tgt = DeltaTable.forName(_SPARK, _fqn(table))
    cond = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    (tgt.alias("t").merge(sdf.alias("s"), cond)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute())


# ==========================================================================
# Public interface
# ==========================================================================
def append_forecasts(df: pd.DataFrame) -> None:
    keys = ["issue_ts", "source", "icao", "target_date"]
    (_delta_append if ON_DATABRICKS else _local_append)(_FORECASTS, df, keys)


def upsert_observations(df: pd.DataFrame) -> None:
    keys = ["icao", "target_date"]
    (_delta_upsert if ON_DATABRICKS else _local_upsert)(_OBSERVATIONS, df, keys)


def upsert_verification(df: pd.DataFrame) -> None:
    keys = ["source", "icao", "target_date"]
    (_delta_upsert if ON_DATABRICKS else _local_upsert)(_VERIFICATION, df, keys)


def read_forecasts() -> pd.DataFrame:
    return _delta_read(_FORECASTS) if ON_DATABRICKS else _local_read(_FORECASTS)


def read_observations() -> pd.DataFrame:
    return _delta_read(_OBSERVATIONS) if ON_DATABRICKS else _local_read(_OBSERVATIONS)


def read_verification() -> pd.DataFrame:
    return _delta_read(_VERIFICATION) if ON_DATABRICKS else _local_read(_VERIFICATION)
