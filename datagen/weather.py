#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Outdoor air temperature time series for the Verbier DHN site, used to drive
the demand and supply-temperature schedules (`schedules.py`) and, via
`KusudaSoil`, the ground/surface temperature seen by all pipes (including
aerial ones, modeled at `depth=0` -- see `network_builder.py`).

Primary source: hourly station data via `meteostat`, spatially interpolated
to the site coordinates (same approach as the reference notebook). Results
are cached to disk so re-running a simulation over the same period doesn't
re-hit the network. This series also feeds `KusudaSoil`, which supplies the
ground/surface temperature used by all pipes (including aerial ones, which
are modeled at `depth=0` -- see `network_builder.py`).

If `meteostat` is unavailable or the request fails (offline environment,
provider outage), a deterministic synthetic Alpine winter-climate proxy is
used instead, so the pipeline never silently produces a flat/constant
weather assumption without saying so.

`pydhn.soils.KusudaSoil` derives its seasonal (annual) ground-temperature
model from a *whole number of exactly-24h days* of air temperature, and
indexes into it by an absolute "hours since day 0" `ts_id` -- which is
unrelated to, and must not be confused with, the simulation's own (possibly
non-hourly) per-step index. `prepare_weather()` therefore always fetches
full calendar year(s) of hourly data covering the run, and returns both the
full hourly series (for `KusudaSoil`) and the resampled/sliced series at the
simulation's own frequency (for the demand/supply-temperature schedules).
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# Verbier, Switzerland (~1500 m a.s.l.)
SITE_LATITUDE = 46.1
SITE_LONGITUDE = 7.23
SITE_ELEVATION_M = 1500


def _cache_path(cache_dir, start, end, tag):
    key = f"{tag}_{start}_{end}_{SITE_LATITUDE}_{SITE_LONGITUDE}_{SITE_ELEVATION_M}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return Path(cache_dir) / f"weather_{digest}.csv"


def _fetch_hourly_from_meteostat(start, end):
    import meteostat as ms

    point = ms.Point(SITE_LATITUDE, SITE_LONGITUDE, SITE_ELEVATION_M)
    stations = ms.stations.nearby(point, limit=4)
    ts = ms.hourly(stations, pd.Timestamp(start), pd.Timestamp(end))
    df = ms.interpolate(ts, point).fetch()
    t_air = df["temp"].astype(float)
    if t_air.isna().all():
        raise ValueError("meteostat returned no usable temperature data")

    full_index = pd.date_range(start=start, end=end, freq="1h")
    t_air = t_air.reindex(full_index).interpolate(limit_direction="both")
    return t_air


def _synthetic_alpine_temperature(index, seed=12345):
    """
    Deterministic fallback outdoor air temperature series for a mid-latitude
    alpine resort at ~1500 m, used only if live weather data can't be
    fetched. Combines an annual sinusoid (mean ~2 degC, amplitude ~9 degC,
    coldest ~mid-Jan), a daily sinusoid (amplitude ~3 degC, coldest shortly
    before sunrise), and smoothed noise for day-to-day variability. This is
    a coarse climate proxy, not a weather forecast/reanalysis -- it exists
    only so the pipeline remains runnable offline.
    """
    day_of_year = index.dayofyear.values + index.hour.values / 24.0
    annual = 2.0 - 9.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
    hour = index.hour.values + index.minute.values / 60.0
    daily = -3.0 * np.cos(2 * np.pi * (hour - 5) / 24.0)

    rng = np.random.default_rng(seed)
    raw_noise = rng.normal(0, 1.5, size=len(index))
    noise = pd.Series(raw_noise).ewm(span=12).mean().values

    t_air = annual + daily + noise
    return pd.Series(t_air, index=index, name="temp")


def _get_full_years_hourly(year_start, year_end, cache_dir=None, allow_live=True):
    """Hourly air temperature from Jan 1 00:00 of `year_start` through Dec 31
    23:00 of `year_end`, guaranteed to have a length that's an exact
    multiple of 24 (required by KusudaSoil)."""
    start = f"{year_start}-01-01 00:00"
    end = f"{year_end}-12-31 23:00"
    index = pd.date_range(start=start, end=end, freq="1h")

    cache_file = _cache_path(cache_dir, start, end, "hourly") if cache_dir else None
    if cache_file is not None and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"[weather] loaded cached hourly series from {cache_file}")
        return df["temp"].reindex(index).interpolate(limit_direction="both")

    t_air, source = None, None
    if allow_live:
        try:
            t_air = _fetch_hourly_from_meteostat(start, end)
            source = "meteostat (live station data)"
        except Exception as exc:  # noqa: BLE001 - any failure -> fallback
            print(f"[weather] live fetch failed ({exc!r}); using synthetic fallback")

    if t_air is None:
        t_air = _synthetic_alpine_temperature(index)
        source = "synthetic Alpine climate proxy (offline fallback)"

    print(
        f"[weather] source: {source}; {year_start}-{year_end}; "
        f"range {t_air.min():.1f}..{t_air.max():.1f} degC"
    )

    assert len(t_air) % 24 == 0, "hourly weather series must span whole days"

    if cache_file is not None:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        t_air.rename("temp").to_frame().to_csv(cache_file)

    return t_air


def prepare_weather(start, end, freq="1h", cache_dir=None, allow_live=True):
    """
    Returns
    -------
    t_air_sim : pd.Series
        Outdoor air temperature (degC) at the simulation's own `freq`, over
        [start, end]. Drives demand/supply-temperature schedules and aerial
        pipes.
    t_air_hourly_full : np.ndarray
        Hourly air temperature covering full calendar year(s) from Jan 1 of
        `start`'s year through Dec 31 of `end`'s year. Feeds `KusudaSoil`.
    year_start : pd.Timestamp
        Jan 1 00:00 of `start`'s year -- the reference origin for the
        absolute hour-of-year `ts_id` passed to the solver each step (see
        `simulate.py`).
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    year_start = pd.Timestamp(year=start.year, month=1, day=1)

    t_air_hourly_full = _get_full_years_hourly(
        start.year, end.year, cache_dir=cache_dir, allow_live=allow_live
    )

    sim_index = pd.date_range(start=start, end=end, freq=freq)
    t_air_sim = (
        t_air_hourly_full.reindex(t_air_hourly_full.index.union(sim_index))
        .interpolate(limit_direction="both")
        .reindex(sim_index)
    )
    t_air_sim.name = "temp"

    return t_air_sim, t_air_hourly_full.values, year_start
