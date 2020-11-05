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
from scipy import stats
import matplotlib.pyplot as plt
from psychrolib import GetTWetBulbFromRelHum, SI, SetUnitSystem
from src.dayofyear import dayofyear_checker
from src.Labour import labour_sahu

absolute_zero = -273.15
SetUnitSystem(SI)

# Silence warning about dividing by 0 or nan.
np.seterr(divide='ignore', invalid='ignore')

# %%
# Get RiceAtlas data
from src.RiceAtlas import ra

ra

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
    #"data_node": "esgf-data3.ceda.ac.uk",
}

openDAP_urls = {}
for var in CMIP6_variables:
    CMIP6_search["variable"] = var
    openDAP_urls[var] = get_openDAP_urls(CMIP6_search)

print(openDAP_urls)

# %%
# Open using xarray as openDAP
ds = xr.open_mfdataset(
    openDAP_urls.values(), join="exact", combine="by_coords", use_cftime=True
)
ds

# %%
# Calculate global mean surface air temperature
# Even if you decide to do the rest of the analysis with daily data, I think
# there is no reason not to use monthly data for this.
# Note that ds gets sub-setted further down
weights = np.cos(np.deg2rad(ds.lat))
gsat = (
    ds["tas"].weighted(weights).mean(("lat", "lon")).resample(time="Y").mean().compute()
)
gsat_reference = gsat.sel(time=slice("1850", "1900")).mean("time")
gsat_change = (gsat - gsat_reference).groupby("time.year").first()

# %%
# Temperatures are in kelvin by default - I want them in Celsius
ds["tas"] = ds["tas"] + absolute_zero
ds["tasmax"] = ds["tasmax"] + absolute_zero

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
# Calculate the heat stress indices.
# This is a delayed computation
for WBGT, WBT, Ta in (
    ("wbgt_max", "wbt_max", "tasmax"),
    ("wbgt_mean", "wbt_mean", "tas"),
):
    # Specify WBT calculation, using psychrolib.
    ds[WBT] = xr.apply_ufunc(
        GetTWetBulbFromRelHum,
        ds[Ta],
        ds["hurs"] / 100,
        ds["ps"],
        dask="parallelized",
        output_dtypes=[float],
    )

    # Calculate WBGT, assuming the black globe temperature is approximated by the
    # air temperature. This will be approximately true in the shade.
    ds[WBGT] = ds[WBT] * 0.7 + ds[Ta] * 0.3
ds["wbgt_mid"] = (ds["wbgt_max"] + ds["wbgt_mean"]) / 2
ds

# %%
# Calculate the labour effect.
# This is a delayed computation.
for WBGT, labour in (
    ("wbgt_max", "labour_sahu_max"),
    ("wbgt_mean", "labour_sahu_mean"),
    ("wbgt_mid", "labour_sahu_mid"),
):
    ds[labour] = xr.apply_ufunc(
        labour_sahu, ds[WBGT], dask="parallelized", output_dtypes=[float]
    )
ds

# %%
# Apply 4+4+4 weighting to labour effect, to approximate sub-daily variation.
ds["labour_sahu_444"] = (
    ds["labour_sahu_max"] + ds["labour_sahu_mean"] + ds["labour_sahu_mid"]
) / 3

# %%
# Spatially subset climate gridded data according to RiceAtlas
# RiceAtlas is in WGS 84, so I think it's fine to use the lat/lon numbers directly
ra_lons = xr.DataArray(
    ra.centroid.x.values, dims="HASC", coords={"HASC": ra.HASC.values}
)
ra_lats = xr.DataArray(
    ra.centroid.y.values, dims="HASC", coords={"HASC": ra.HASC.values}
)
# 'nearest' method of interpolation is fast
ds_locations = ds.interp(lon=ra_lons, lat=ra_lats, method="nearest")

# %%
# Temporally subset, according to dayofyear.
# HARV_ST1 is the start day of the first cropping season.
# HARV_END1 is the end day of the first cropping season.
# Create an xr.DataArray containing dates which meet the criteria for each region.
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
ds_locations_seasons = ds_locations.where(doy_mask)
ds_locations_seasons

# %%
# Downsample to yearly mean.
# Because the harvest seasons have been selected, this will give annual results
# for each harvest season. That is, more than one season per year. In locations
# without multiple harvest seasons, the result will be 0.
ds_locations_seasons_annual = ds_locations_seasons.groupby("time.year").mean()

# %%
# Weight the result according to rice harvest weight.
# P_S1 means the production of the first cropping season.
weights = xr.DataArray(
    ra[["P_S1", "P_S2", "P_S3"]].values,
    dims=["HASC", "seasonid"],
    coords={"HASC": ra.HASC.values, "seasonid": [1, 2, 3]},
)
weights

# %%
# Do a weighted average summing over locations to get a single annual value for
# each cropping season. Invoke computation.
ds_weighted_annual = (
    (ds_locations_seasons.groupby("time.year").mean() * weights).sum(("HASC"))
    / weights.sum("HASC")
).compute()
ds_weighted_annual

# %%
# Plot the effect against year.
ds_weighted_annual.plot.scatter("year", "labour_sahu_444")

# %%
# Do a weighted average summing over time and cropping season, to get a single
# value for each location. Invoke computation.
ds_weighted_locationwise_s1 = (
    (ds_locations_seasons.mean("time")).sel(seasonid=1).compute()
)

ds_weighted_locationwise_s1

# %%
# Plot a map of the results.
ra["labour_sahu_average"] = ds_weighted_locationwise_s1["labour_sahu_444"].values
ra.plot("labour_sahu_average", legend=True)
plt.show()

# %%
# Long term trends...
trend_window = 20


def year_ranges_masking(data, trend_window):
    year_min = data.year.min().item()
    year_max = data.year.max().item()
    period_starts = np.array(list(range(year_min, year_max, trend_window)))
    period_middles = period_starts + trend_window / 2
    mask = xr.DataArray(
        [(data.year >= year) & (data.year > year + 20) for year in period_starts],
        dims=("period", "year",),
        coords={"year": data.year, "period": period_middles},
    )
    return data.where(mask)


ds_locations_seasons_periods = (
    year_ranges_masking(ds_locations_seasons_annual, trend_window)
    .mean("year")
    .compute()
)
ds_locations_seasons_periods

# %%
# Draw the time series of a couple of locations
# I didn't choose the locations systematically, this is just for illustration.
for HASC, data in (
    ds_locations_seasons_periods.sel(seasonid=1).isel(HASC=[0, 5, 10]).groupby("HASC")
):
    data.plot.scatter(
        "period", "labour_sahu_444", label=ra.set_index("HASC").loc[HASC].SUB_REGION
    )
plt.legend(loc="best")
plt.ylabel("Labour effect (%)")
plt.show()

# %%
# How does this look plotted against GSAT?
x = (
    year_ranges_masking(gsat_change, trend_window).mean("year").dropna("period")
)
y = (
    ds_locations_seasons_periods.sel(seasonid=1)
    .isel(HASC=0)["labour_sahu_444"]
    .dropna("period")
)
plt.scatter(x, y, label=ra.iloc[1].SUB_REGION)
lr = stats.linregress(x, y)
plt.plot(
    x.values, x.values * lr.slope + lr.intercept, label="Fit"
)
plt.xlabel("GSAT ($\degree C$)")
plt.ylabel("Labour effect (%)")
plt.legend(loc="best")
print(lr)
plt.show()

# %%
# What if we look at annual instead of long term values?
x = ds_locations_seasons_annual.sel(seasonid=1).isel(HASC=0)["labour_sahu_444"]
y = gsat_change
plt.scatter(x, y, label=ra.iloc[1].SUB_REGION)
lr = stats.linregress(x, y)
plt.plot(x, x * lr.slope + lr.intercept, label="Fit")
plt.xlabel("GSAT ($\degree C$)")
plt.ylabel("Labour effect (%)")
plt.legend(loc="best")
print(lr)
plt.show()
# The trend is much less clear using annual data than long term averages.

# %%
# Independently fit lines in each location
def fit_parallel(X, Y):
    A = np.apply_along_axis(
        lambda y: stats.linregress(X, y),
        -1,
        Y,
    )
    return A
def fit_parallel_wrapper(x, y, dim='time'):
    result = xr.apply_ufunc(
        fit_parallel,
        x, y,
        input_core_dims = [[dim], [dim]],
        output_core_dims = [['linregress']],
        dask='forbidden', output_dtypes=[float]
    )
    result = result.assign_coords({'linregress': ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']})
    return result
x = ( year_ranges_masking(gsat_change, trend_window).mean("year").dropna("period"))
y = ( ds_locations_seasons_periods["labour_sahu_444"].sel(period=x.period))
ds_parallel_fit = fit_parallel_wrapper( x.load(), y.load(), 'period')
ds_parallel_fit.sel(linregress='slope').plot.hist()
plt.xlabel('Long-term hazard gradient (%/C)')
plt.show()

# %%
plt.scatter(ds_parallel_fit.sel(linregress='slope'), ds_parallel_fit.sel(linregress='pvalue'),)
plt.show()

# %%
# Plot the gradient of the worst affected season in each location
ra['gradient_max_season'] = ds_parallel_fit.max('seasonid').sel(linregress='slope')
ra.plot('gradient_max_season', legend=True)
plt.show()

# %%
# Plot the proportion of the harvest in each location that is exposed to a significant gradient
is_exposed = ds_parallel_fit.sel(linregress='pvalue') < 0.01
weight_exposed = weights.where(is_exposed).sum('seasonid')
ra['weight_exposed'] = weight_exposed / weights.sum('seasonid')
ra.plot('weight_exposed', legend=True)
plt.show()

# %%
# Calculate the proportion of the total harvest that is exposed.
exposed_percent = (weight_exposed.sum() / weights.sum()).item()*100
print(f"{exposed_percent:0.1f}% is exposed")

# %%
# In what months is production exposed?
# Unfortunately not in order
print("exposed:", np.unique(ra[['HMO_PK1', 'HMO_PK3', 'HMO_PK3']][is_exposed.values].astype(str), return_counts=True))
print("not exposed:", np.unique(ra[['HMO_PK1', 'HMO_PK3', 'HMO_PK3']][~is_exposed.values].astype(str), return_counts=True))




# TODO
# Non-centroid selection of grid cells
# Global line fit
# Arbitrary fit
# Reference for the 0.7 * WBT + 0.3 * T_a number for shade WBGT
# Reference for the 4+4+4 weighting.
# More explanation of RiceAtlas
# Correction for 360 day calendar
# Turn longer notes into markdown.
# Give project more structure - copy elements from cookiecutter project
