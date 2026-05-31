import argparse
import calendar
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


SITES = {
    # name used by run_solarv4.py -> (lat, lon, start_date, end_date)
    "guowang_site1_50MW": (36.1, 103.8, "2019-01-01", "2020-12-31"),
    "guowang_site2_130MW": (36.6, 101.8, "2019-01-01", "2020-12-31"),
    "guowang_site3_30MW": (32.1, 118.8, "2019-01-01", "2020-12-31"),
    "guowang_site4_130MW": (36.7, 117.0, "2019-01-01", "2020-12-31"),
    "guowang_site5_110MW": (36.7, 117.0, "2019-01-01", "2020-12-31"),
    "guowang_site6_35MW": (25.0, 102.7, "2019-01-01", "2020-12-31"),
    "guowang_site7_30MW": (26.6, 106.7, "2019-01-01", "2020-12-31"),
    "guowang_site8_30MW": (34.3, 108.9, "2019-01-01", "2020-12-31"),
    "skippd_stanford": (37.427, -122.174, "2017-01-01", "2017-12-31"),
}


VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "surface_solar_radiation_downwards",
    "total_cloud_cover",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]


def month_iter(start_date, end_date):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cur = pd.Timestamp(start.year, start.month, 1)
    while cur <= end:
        last_day = calendar.monthrange(cur.year, cur.month)[1]
        m_start = max(start, pd.Timestamp(cur.year, cur.month, 1))
        m_end = min(end, pd.Timestamp(cur.year, cur.month, last_day))
        yield cur.year, cur.month, list(range(m_start.day, m_end.day + 1))
        cur = cur + pd.offsets.MonthBegin(1)


def request_month(client, site_name, lat, lon, year, month, days, cache_dir, box_size):
    north = lat + box_size
    south = lat - box_size
    west = lon - box_size
    east = lon + box_size
    target = cache_dir / f"{site_name}_{year}{month:02d}.nc"
    if target.exists() and target.stat().st_size > 0:
        print(f"[skip] {target.name}")
        return target

    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in days],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [north, west, south, east],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    print(f"[download] {site_name} {year}-{month:02d} -> {target}")
    client.retrieve("reanalysis-era5-single-levels", request, str(target))
    return target


def _coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"Cannot find coordinate among {candidates}; got {list(ds.coords)}")


def _var_name(ds, candidates):
    for name in candidates:
        if name in ds:
            return name
    raise KeyError(f"Cannot find variable among {candidates}; got {list(ds.data_vars)}")


def relative_humidity_from_t_dewpoint(temp_k, dew_k):
    temp_c = temp_k - 273.15
    dew_c = dew_k - 273.15
    es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    e = 6.112 * np.exp((17.67 * dew_c) / (dew_c + 243.5))
    return np.clip(100.0 * e / es, 0.0, 100.0)


def nc_to_dataframe(nc_files, lat, lon):
    import xarray as xr
