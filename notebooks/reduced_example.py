# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from psychrolib import GetTWetBulbFromRelHum, SI, SetUnitSystem
from src.dayofyear import dayofyear_checker

absolute_zero = -273.15
SetUnitSystem(SI)

# %%
# Get RiceAtlas data
from src.RiceAtlas import ra

ra

# TODO, give some more explanation of what this dataset is

# %%
# Reduce scope of RiceAtlas data for speed
ra = ra[ra.COUNTRY == "Vietnam"]

# %%
# Get input climate data
# Use openDAP to access directly from ESGF
from src.get_data.esgf_opendap import get_openDAP_urls

CMIP6_table = "Amon"  # could use 'day' but would be slower
CMIP6_model = "UKESM1-0-LL"
CMIP6_variables = ["tas", "tasmax", "hurs", "ps"]
CMIP6_search = {
    "project": "CMIP6",
    "source_id": "UKESM1-0-LL",
    "experiment_id": "historical",
    "variable": "tas",
    "frequency": "mon",
    "variant_label": "r1i1p1f2",
    "data_node": "esgf-data3.ceda.ac.uk",
}

openDAP_urls = {}
for var in CMIP6_variables:
    CMIP6_search["variable"] = var
    openDAP_urls[var] = get_openDAP_urls(CMIP6_search)

print(openDAP_urls)

# %%
# Open using xarray as openDAP
# For monthly data, this is about 900 MB
ds = xr.open_mfdataset(
    openDAP_urls.values(), join="exact", combine="by_coords", use_cftime=True
)
ds

# %%
# Reduce scope of climate data for speed
# Based on the geographic limits of the RiceAtlas data we have selected
min_lon, min_lat, max_lon, max_lat = ra.total_bounds
stepsize_lat = ds.lat.values[1] - ds.lat.values[0]
stepsize_lon = ds.lon.values[1] - ds.lon.values[0]

ds = ds.where(
    (ds.lat >= min_lat - stepsize_lat)
    & (ds.lat <= max_lat + stepsize_lat)
    & (ds.lon >= min_lon - stepsize_lon)
    & (ds.lon <= max_lon + stepsize_lon),
    drop=True,
)

# %%
# Add in date auxillaries
# This is because direct access via cftime dummy is slow.
ds["dayofyear"] = ds.time.dt.dayofyear

# %%
# Specify WBT calculation
# This is still a delayed computation
ds["wbt_max"] = xr.apply_ufunc(
    GetTWetBulbFromRelHum,
    ds["tasmax"] + absolute_zero,
    ds["hurs"] / 100,
    ds["ps"],
    dask="parallelized",
    output_dtypes=[float],
)
ds["wbt_max"]
# The full calculation uses mean and max...

# %%
# Spatially subset climate gridded data according to RiceAtlas
# RiceAtlas is in WGS 84, so I think it's fine to use the lat/lon numbers directly
ra_lons = xr.DataArray(
    ra.centroid.x.values, dims="HASC", coords={"HASC": ra.HASC.values}
)
ra_lats = xr.DataArray(
    ra.centroid.y.values, dims="HASC", coords={"HASC": ra.HASC.values}
)
ds_locations = ds.interp(lon=ra_lons, lat=ra_lats, method="nearest")

# %%
# Temporally subset, according to dayofyear
# Create an xr.DataArray containing dates which meet the criteria for each region
doy_mask = xr.DataArray(
    np.array(
        [
            dayofyear_checker(
                ra[f"HARV_ST{season}"].astype(int).values,
                ra[f"HARV_END{season}"].astype(int).values,
                ds_locations.dayofyear.values,
            )
            for season in (1, 2, 3)
        ]
    ),
    dims=["seasonid", "HASC", "time"],
    coords={"HASC": ra.HASC, "seasonid": [1, 2, 3], "time": ds_locations.time},
)

# %%
ds_locations_seasons = ds_locations.where(doy_mask)

# %%
ds_locations_seasons_annual = ds_locations_seasons.groupby("time.year").mean()

# %%
weights = xr.DataArray(
    ra[["P_S1", "P_S2", "P_S3"]].values,
    dims=["HASC", "seasonid"],
    coords={"HASC": ra.HASC.values, "seasonid": [1, 2, 3]},
)
wbt_weighted_annual = (
    ds_locations_seasons["wbt_max"].groupby("time.year").mean() * weights
).sum(("HASC")) / weights.sum("HASC")

wbt_weighted_locationwise = (
    ds_locations_seasons["wbt_max"].mean("time") * weights
).sum(("seasonid")) / weights.sum("seasonid")

# %%
wbt_weighted = wbt_weighted.compute()

# %%
wbt_weighted.plot.hist()
plt.show()

# %%
wbt_weighted.to_dataset(name="wbt_max").plot.scatter("year", "wbt_max")

# %%
ra["result"] = wbt_weighted_locationwise.values
ra.plot("result", legend=True)
plt.show()

# %%
# Long term trends...
# %%
periods_starts = np.array(range(1850, 2016 - 20, 20))
periods_ends = periods_starts + 20
