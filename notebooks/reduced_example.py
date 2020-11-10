# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Climate risk to rice labour
# Rice is an important part of the global food supply. It provides half
# of calories consumed in Asia. Frequently, rice is harvested by hand.
# This can be strenuous work, often in hot and humid conditions. Hot and
# humid conditions are projected to become more common due to global
# warming.
#
# In this notebook, a global climate model is combined with crop
# calendars in order to identify locations in which workers engaged in
# rice harvest may already be affected by hot and humid weather, and the
# extent to which this will increase with global warming.
#
# This is intended as an illustration of a more complete analysis, which
# you can follow and reproduce on your own computer.
# For this reason, we use only one climate model, and only one future
# pathway for emissions.

# %%
# Imports

import warnings
import xarray as xr
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from psychrolib import GetTWetBulbFromHumRatio, SI, SetUnitSystem

from src.dayofyear import dayofyear_checker
from src.Labour import labour_sahu

absolute_zero = -273.15
SetUnitSystem(SI)

# Silence warning about dividing by 0 or nan.
np.seterr(divide="ignore", invalid="ignore")
warnings.filterwarnings("once", ".*No gridpoint belongs to any region.*")
warnings.filterwarnings("once", ".*Geometry is in a geographic CRS.*")
warnings.filterwarnings("once", ".*invalid value .*")
warnings.filterwarnings("once", ".*All-NaN slice.*")
warnings.filterwarnings("once", ".*invalid value encountered in.*")

# %% [markdown]
# ## RiceAtlas
#
# "...a spatial database on the seasonal distribution of the world’s rice
# production. It consists of data on rice planting and harvesting dates by
# growing season and estimates of monthly production for all rice-producing
# countries. Sources used for planting and harvesting dates include global and
# regional databases, national publications, online reports, and expert
# knowledge."
#
# [Laborte, A. G. et al., 2017](https://www.nature.com/articles/sdata201774)
#
# Details of loading the RiceAtlas data are handled by [../src/RiceAtlas.py](../src/RiceAtlas.py)

# %%
from src.RiceAtlas import ra

ra

# %%
# Reduce scope of RiceAtlas data
ra = ra[ra.CONTINENT == "Asia"]


# %% [markdown]
# RiceAtlas has one row per location. The locations have geometries
# provided. Some are much larger that others.
#
# Within each row, there is information for multiple cropping seasons.
# For example, the columns 'P_S1', 'P_S2', and 'P_S3', give production
# in the first, second and third cropping season.
# The number of columns does not change between rows, and there are
# always entries for 3 cropping seasons. If there are not 3 croppping
# seasons in a location, then the production will be 0 for that season.

# %%
ra[["COUNTRY", "REGION", "SUB_REGION", "P_S1", "P_S2", "P_S3"]].sample(10)


# %% [markdown]
# ## CMIP6
# Using the latest generation of climate models, we look for trends in
# local changes in WBGT against global climate change.
#
# In this example, we only use one model.
#
# * 'tas' = mean near-surface air temperature (Kelvin)
# * 'tasmax' = max near-surface air temperature (Kelvin)
# * 'huss' = humidity ratio (dimensionless)
# * 'ps' = near-surface pressure (Pa)
#
#
# See references re. CMIP6:
# * [Eyring, V. et al., 2016](https://doi.org/10.5194/gmd-9-1937-2016)
# * [O’Neill, B. C. et al., 2016](https://doi.org/10.5194/gmd-9-3461-2016)


# %% [markdown]
# Use openDAP to access directly from ESGF.
# This requires you to supply an ESGF login.
# See [ESGF User Guide](https://esgf.github.io/esgf-user-support/user_guide.html)

# %%
from src.get_data.esgf_opendap import get_openDAP_urls

CMIP6_variables = ["tas", "tasmax", "huss", "ps"]
CMIP6_experiments = ["historical", "ssp245"]
CMIP6_search = {
    "project": "CMIP6",
    "source_id": "UKESM1-0-LL",
    "experiment_id": "historical",
    "variable": "tas",
    "frequency": "mon",
    "variant_label": "r1i1p1f2",
    "data_node": "esgf-data3.ceda.ac.uk",
}

openDAP_urls =[] 
for experiment in CMIP6_experiments:
    CMIP6_search["experiment_id"] = experiment
    for var in CMIP6_variables:
        CMIP6_search["variable"] = var
        openDAP_urls.append( get_openDAP_urls(CMIP6_search))

print(openDAP_urls)

# %%
# Open using xarray as openDAP.
# If this fails, you might try changing the data_node in the query.
ds = xr.open_mfdataset(
    openDAP_urls, join="exact", combine="by_coords", use_cftime=True
)
ds = ds.drop('height')
ds

# %% [markdown]
# ## Climate Change
# Calculate global mean surface air temperature, i.e. global climate change.
#
# Even if you decide to do the rest of the analysis with daily data, I think
# there is no reason not to use monthly data for this.
#
# Note that ds gets mutated further down in this script, so be careful
# running these cells out of order..

# %%
weights = np.cos(np.deg2rad(ds.lat))  # To account area in spherical coordinates.
gsat = (
    ds["tas"].weighted(weights).mean(("lat", "lon")).resample(time="Y").mean().compute()
)
gsat_reference = gsat.sel(time=slice("1850", "1900")).mean("time")
gsat_change = (gsat - gsat_reference).groupby("time.year").first()

gsat_change.attrs['long_name'] = 'Global mean surface air temperature change'
gsat_change.attrs['short_name'] = 'GSAT change'
gsat_change.attrs['units'] = 'C'

gsat_change.plot()

# %%
# Temperatures are in kelvin by default - I want them in Celsius
ds["tas"] = ds["tas"] + absolute_zero
ds["tasmax"] = ds["tasmax"] + absolute_zero

# %%
# Reduce scope of climate data for speed.
# Based on the geographic limits of the RiceAtlas data we have selected.
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
ds

# %%
# Add in date auxillaries
# This is because direct access via cftime dummy is slow.
ds["dayofyear"] = ds.time.dt.dayofyear
# If the calendar is only 360 days long (a common assumption in climate
# models), then apply a correction to make it more like the 365 day
# calendar.  This is not the most accurate method of correction, but it
# is fast and easy to understand.
if ds.dayofyear.max() == 360:
    for day in (73, 145, 218, 291, 364):
        ds["dayofyear"][ds.dayofyear >= day] = ds.dayofyear[ds.dayofyear >= day] + 1


# %% [markdown]
# ## Heat stress index
#
# Many studies focussed on the risk of occupational heat stress use
# wet-bulb globe temperature (WBGT), which is a heat-stress index
# defined by ISO 7243. WBGT is intended to combine all the factors that
# affect the human experience of heat, namely air temperature, radiant
# temperature, humidity, and air velocity. As performing work
# generates heat, in a high WBGT environment labour must be reduced in
# order to maintain a safe body temperature.
#
# We use the [psychrolib](https://github.com/psychrometrics/psychrolib)
# software library, which implements formulae from the ASHRAE handbook, to
# calculate wet bulb temperature.
#
# We neglect irradiance by assuming that the black globe temperature is
# approximated by the air temperature. This will be approximately true in the
# shade.
# WBGT in sunny weather will be underestimated, but we are focussed on long
# term trends, and trends in irradiance are not so clear as those in air
# temperature and humidity.
#
# See references:
# * [Parsons, K., 2006](https://doi.org/10.2486/indhealth.44.368)
# * [Parsons, K., 2013](https://doi.org/10.2486/indhealth.2012-0165)
# * [Lemke, B. & Kjellstrom, T.](https://doi.org/10.2486/indhealth.ms1352)

# %%
# This is a delayed computation, so will return quickly.
for WBGT, WBT, Ta in (
    ("wbgt_max", "wbt_max", "tasmax"),
    ("wbgt_mean", "wbt_mean", "tas"),
):
    # Specify WBT calculation, using psychrolib.
    ds[WBT] = xr.apply_ufunc(
        GetTWetBulbFromHumRatio,
        ds[Ta],
        ds["huss"],
        ds["ps"],
        dask="parallelized",
        output_dtypes=[float],
    )

    # Calculate WBGT, assuming the black globe temperature is approximated by the
    # air temperature. This will be approximately true in the shade.
    ds[WBGT] = ds[WBT] * 0.7 + ds[Ta] * 0.3
ds["wbgt_mid"] = (ds["wbgt_max"] + ds["wbgt_mean"]) / 2
ds

# %% [markdown]
# ## Labour effect
#
# Sahu et al observed a 5% per °C WBGT decrease in the labour capacity of
# labourers harvesting rice between 23-33 °C WBGT. Rate of collection was
# measured in 124 workers in groups of 10-18, and WBGT was measured in-situ, at
# a single location in India.
# We adopt this for our labour impact metric, and assume that this is
# representative of manual rice harvest labour.
#
# [Sahu, S. et al., 2013](https://doi.org/10.2486/indhealth.2013-0006)
#
# See also [Gosling, S. N., Zaherpour, J. & Ibarreta, D.](http://doi.org/10.2760/07911)
#
# Other labour impact functions are included in [../src/Labour.py](../src/Labour.py), so you
# could explore how the choice of labour impact function affects the results,
# and even define your own.
#
# The '4+4+4' assumption means that air temperature in the working day is
# assumed to be close to the maxmimum for 4 hours, the mean for 4 hours, and
# half-way between for 4 hours. This is a reasonably good approximation.
# This assumption comes from
# [Kjellstrom, T. et al., 2018](https://doi.org/10.1007/s00484-017-1407-0)
#


# %%
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

# %% [markdown]
# ## Combining the data

# %%
# Spatially subset climate gridded data according to RiceAtlas
# RiceAtlas is in WGS 84, so I think it's fine to use the lat/lon numbers directly
# Using centroid only. Very fast, but less accurate for large regions.
ra_lons = xr.DataArray(
    ra.centroid.x.values, dims="HASC", coords={"HASC": ra.HASC.values}
)
ra_lats = xr.DataArray(
    ra.centroid.y.values, dims="HASC", coords={"HASC": ra.HASC.values}
)
ds_locations = ds.interp(lon=ra_lons, lat=ra_lats, method="nearest")
ds_locations

# %% [markdown]
# Temporally subset, according to dayofyear.
#
# HARV_ST1 is the start day of the first cropping season.
# HARV_END1 is the end day of the first cropping season.
#
# Create an xr.DataArray containing dates which meet the criteria for each region.
# The logic for this is imported from [../src/dayofyear.py].

# %%
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

# %% [markdown]
# Downsample to yearly mean.
#
# Because the harvest seasons have been selected, this will give annual results
# for each harvest season. That is, more than one season per year. In locations
# without multiple harvest seasons, the result will be 0.

# %%
ds_locations_seasons_annual = ds_locations_seasons.groupby("time.year").mean()
ds_locations_seasons_annual

# %% [markdown]
# Weight the result according to rice harvest weight.
# P_S1 means the production of the first cropping season.

# %%
weights = xr.DataArray(
    ra[["P_S1", "P_S2", "P_S3"]].values,
    dims=["HASC", "seasonid"],
    coords={"HASC": ra.HASC.values, "seasonid": [1, 2, 3]},
)
weights

# %% [markdown]
# Do a weighted average summing over locations to get a single annual value for
# each cropping season.

# %%
# This is a somewhat slow cell, as computation is triggered.
ds_weighted_annual = (
    (ds_locations_seasons.groupby("time.year").mean() * weights).sum(("HASC"))
    / weights.sum("HASC")
).compute()
ds_weighted_annual

# %%
# Plot the labour effect against year.
ds_weighted_annual.plot.scatter("year", "labour_sahu_444")
plt.ylabel("Labour impact (%)")

# %% [markdown]
# There is clearly a long term trend. The data fall into two groups, as we haven't averaged over cropping seasons.
#
# Let's examine the long term trends in the labour effect.

# %%
# This is a somewhat slow cell, as computation is triggered.
trend_window = 20  # how many years to average over to specify 'long-term'


def year_ranges_masking(data: xr.Dataset, trend_window: int) -> xr.Dataset:
    """Select data in a certain year range."""
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

# %% [markdown]
# Draw the time series of a couple of locations
# I didn't choose the locations systematically, this is just for illustration.
# We should see that some locations have a clear trend, and others don't.

# %%
for HASC, data in (
    ds_locations_seasons_periods.sel(seasonid=1).isel(HASC=[0, 5, 10]).groupby("HASC")
):
    data.plot.scatter(
        "period", "labour_sahu_444", label=ra.set_index("HASC").loc[HASC].SUB_REGION
    )
plt.legend(loc="best")
plt.ylabel("Labour effect (%)")
plt.ylim(bottom=0)

# %% [markdown]
# Is the long-term trend dominated by global changes in surface air temperature?
# If so, we would expect there to be a correlation, so let's plot the local
# changes against global warming, in long-term averages.

# %%
x = year_ranges_masking(gsat_change, trend_window).mean("year").dropna("period")
y = (
    ds_locations_seasons_periods.sel(seasonid=1)
    .isel(HASC=0)["labour_sahu_444"]
    .dropna("period")
)
plt.scatter(x, y, label=ra.iloc[1].SUB_REGION)
lr = stats.linregress(x, y)
plt.plot(x.values, x.values * lr.slope + lr.intercept, label="Fit")
plt.xlabel("GSAT ($\degree C$)")
plt.ylabel("Labour effect (%)")
plt.legend(loc="best")
print(lr)

# %% [markdown]
# There is a clear correlation (at least in the cropping season-location that I
# selected for this example!).
#
# Does this explain all of the year-to-year variation? No, and we wouldn't expect it to.
# When we plot annual values instead of long-term averages. (This is at a single location again.)

# %%
# This is a somewhat slow cell, as computation is triggered.
x = ds_locations_seasons_annual.sel(seasonid=1).isel(HASC=0)["labour_sahu_444"]
y = gsat_change
plt.scatter(x, y, label=ra.iloc[1].SUB_REGION)
lr = stats.linregress(x, y)
plt.plot(x, x * lr.slope + lr.intercept, label="Fit")
plt.xlabel("GSAT ($\degree C$)")
plt.ylabel("Labour effect (%)")
plt.legend(loc="best")
print(lr)

# %% [markdown]
# The trend is not just present in a few individual locations.
# If we sum the effect across the whole region, weighting by rice production, we see a clear trend.

# %%
x = year_ranges_masking(gsat_change, trend_window).mean("year").dropna("period")
y = ds_locations_seasons_periods["labour_sahu_444"].sel(period=x.period)
y_weighted = (y * weights).sum(("seasonid", "HASC")) / weights.sum()
lr = stats.linregress(x, y_weighted)
plt.scatter(x, y_weighted)
plt.plot(x, x * lr.slope + lr.intercept)
plt.xlabel("GSAT ($\degree C$)")
plt.ylabel("Labour impact %")


# %% [markdown]
# The trend isn't present in all cropping location-seasons.
# So, let's independently fit lines in each location-season, and see where the
# trend with respect to global warming is significant.

# %%
def fit_parallel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Do a linear fit along the last axis."""
    A = np.apply_along_axis(lambda y: stats.linregress(X, y), -1, Y,)
    return A


def fit_parallel_wrapper(
    x: xr.DataArray, y: xr.DataArray, dim: str = "time"
) -> xr.DataArray:
    """Do a linear fit along an axis, independently in other dimensions.

    Different outputs of the fit are arranged along a new dimension, to get
    around the limitation that xr.apply_ufunc can only take a single variable
    output.
    """
    result = xr.apply_ufunc(
        fit_parallel,
        x,
        y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[["linregress"]],
        dask="forbidden",
        output_dtypes=[float],
    )
    result = result.assign_coords(
        {"linregress": ["slope", "intercept", "rvalue", "pvalue", "stderr"]}
    )
    return result


x = year_ranges_masking(gsat_change, trend_window).mean("year").dropna("period")
y = ds_locations_seasons_periods["labour_sahu_444"].sel(period=x.period)
ds_parallel_fit = fit_parallel_wrapper(x.load(), y.load(), "period")
ds_parallel_fit

# %%
plt.hist(
    ds_parallel_fit.sel(linregress="slope").values.reshape(-1),
    bins=np.linspace(-1, 10, 23),
    weights=weights.values.reshape(-1),
)
plt.xlabel("Long-term hazard gradient (%/C)")
plt.ylabel("Harvest affected (tonnes)")

# %% [markdown]
# There is a lot of variation in this 'hazard gradient'.
#
# Note that due to format of the data, locations which do not have multiple
# cropping seasons will be present, and that empty parts of the table will have
# a gradient of exactly 0.
#
# Many of the gradients close to 0 will be non-significant, which becomes clear
# when we examine the p-values of the fit:

# %%
plt.scatter(
    ds_parallel_fit.sel(linregress="slope"), ds_parallel_fit.sel(linregress="pvalue"),
)
plt.yscale("log")
plt.xlabel("Slope (%/C)")
plt.ylabel("pvalue")


# %% [markdown]
# What time of year the rice is harvested is important. Harvests that occur in
# November-January tend to be less affected.
# Therefore, averaging over cropping seasons will sometimes obscure that
# harvests in (e.g.) August are severely affected.
#
# (Note that, because the labour affect function I've used has a lower
# threshold, cropping season-locations that start off relatively cool will have
# a poor goodness of fit, as the trend will have an 'elbow'.)
#
# What is the hazard gradient in the worst affected cropping season in each
# location?

# %%
# Plot the gradient of the worst affected season in each location
ra["gradient_max_season"] = ds_parallel_fit.max("seasonid").sel(linregress="slope")


def map_plot(ra, variable, label, z_min, z_max, nbins):
    bins = np.round(np.linspace(z_min, z_max, nbins), 1)
    norm = mpl.colors.BoundaryNorm(boundaries=bins, ncolors=256)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    min_lon, min_lat, max_lon, max_lat = ra.total_bounds
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="gray")
    ax.add_feature(cfeature.BORDERS, edgecolor="whitesmoke")
    ax.add_feature(cfeature.OCEAN, color="lightgray")
    return ra.plot(
        column=variable,
        legend=True,
        ax=ax,
        # norm=norm,
        legend_kwds={"label": label, "orientation": "horizontal", "boundaries": bins,},
    )


map_plot(
    ra,
    "gradient_max_season",
    "Hazard gradient in worst affected season (%/C)",
    0,
    10,
    11,
)

# %% [markdown]
# What proportion of the harvest in each location is exposed to a significant gradient?
# Assuming this means p<0.01.
#

# %%
is_exposed = ds_parallel_fit.sel(linregress="pvalue") < 0.01
weight_exposed = weights.where(is_exposed).sum("seasonid")
ra["weight_exposed"] = weight_exposed / weights.sum("seasonid") * 100
map_plot(
    ra,
    "weight_exposed",
    "Proportion of production exposed to significant hazard gradient (%)",
    0,
    100,
    11,
)

# %%
# Calculate the proportion of the total harvest that is exposed.
exposed_percent = (weight_exposed.sum() / weights.sum()).item() * 100
print(f"{exposed_percent:0.1f}% of the examined harvest is exposed")

# %%
# In what months is production exposed?
months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

exposed_months = np.unique(
    ra[["HMO_PK1", "HMO_PK3", "HMO_PK3"]][is_exposed.values].astype(str),
    return_counts=True,
)
exposed_months = dict(zip(exposed_months[0], exposed_months[1]))
non_exposed_months = np.unique(
    ra[["HMO_PK1", "HMO_PK3", "HMO_PK3"]][~is_exposed.values].astype(str),
    return_counts=True,
)
non_exposed_months = dict(zip(non_exposed_months[0], non_exposed_months[1]))


print("Month\tExposed\tNon-exposed\t(number of cropping season-locations)")
for month in months:
    print(month, "\t", exposed_months[month], "\t", non_exposed_months[month])


# %%
# TODO
# Give project more structure - copy elements from cookiecutter project
# Add a future projection.

# %%
plt.show()
