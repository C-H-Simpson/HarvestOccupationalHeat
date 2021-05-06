# -*- coding: utf-6 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: riceheat
#     language: python
#     name: riceheat
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
# Use this cell if the conda environment is not already set up
# #!. ../env.sh riceheat ../environment.yml >> env_build_log.txt

# %%
# Imports

# Relative import workaround
import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import warnings
import xarray as xr
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import psychrolib as psl
import regionmask
from intake import open_catalog
from collections import defaultdict
import calendar
from collections import defaultdict
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from pathlib import Path
import calendar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask
import seaborn as sns
import xarray as xr

from src import Labour

absolute_zero = -273.15
psl.SetUnitSystem(psl.SI)
plt.style.use("seaborn-colorblind")

# Dictionary for translating month names
month_dict = defaultdict(lambda: -1)
for i in range(1, 13):
    month_dict[calendar.month_abbr[i]] = i
month_dict[None] = -1

# Silence warning about dividing by 0 or nan.
np.seterr(divide="ignore", invalid="ignore")
warnings.filterwarnings("once", ".*No gridpoint belongs to any region.*")
warnings.filterwarnings("once", ".*Geometry is in a geographic CRS.*")
warnings.filterwarnings("once", ".*invalid value .*")
warnings.filterwarnings("once", ".*All-NaN slice.*")
warnings.filterwarnings("once", ".*invalid value encountered in.*")

# %%
from dask.distributed import Client, LocalCluster

cluster = LocalCluster()
client = Client(cluster)
client

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
# Details of loading the RiceAtlas data are handled by
# [../src/RiceAtlas.py](../src/RiceAtlas.py)

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
# We can use the RiceAtlas data to spatiall subset the climate data.
# However, the method below assumes that that the latitude and longitude are
# defined with consistent conventions
# between the climate data and the RiceAtlas data, so be careful.

# %%
# Define region to spatially subset


def asia_only(da):
    minx, miny, maxx, maxy = ra[ra.CONTINENT == "Asia"].total_bounds
    return (
        da.where(da.lat > miny, drop=True)
        .where(da.lat < maxy, drop=True)
        .where(da.lon > minx, drop=True)
        .where(da.lon < maxx, drop=True)
    )


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
# Use intake to access pangeo catalogue, get data from GCS.

# %%
cat = open_catalog(
    "https://raw.githubusercontent.com/pangeo-data/pangeo-datastore/master/intake-catalogs/master.yaml"
)

# %%
# TODO Explain why it isn't simple to extend to lots of models.
# TODO Make it simple to switch to 3-hourly data.
CMIP6_variables = ["tas", "tasmax", "huss", "ps"]
CMIP6_experiments = ["historical", "ssp585"]
CMIP6_search = {
    "source_id": "UKESM1-0-LL",
    "experiment_id": CMIP6_experiments,
    "variable_id": CMIP6_variables,
    "table_id": "Amon",
    "grid_label": "gn",
    "member_id": "r1i1p1f2",
}

cat_return = cat.climate.cmip6_gcs.search(**CMIP6_search)

ds = (
    xr.combine_by_coords(
        cat_return.to_dataset_dict(zarr_kwargs={"consolidated": True}).values(),
        combine_attrs="drop",
    )
    .drop("height")
    .squeeze()
)
ds


# %%
# Regardless of whether we are using monthly/daily/3-hourly data for the analysis,
# we will want monthly surface air temperature for calculating GSAT.
CMIP6_search = {
    "source_id": "UKESM1-0-LL",
    "experiment_id": CMIP6_experiments,
    "variable_id": "tas",
    "table_id": "Amon",
    "grid_label": "gn",
    "member_id": "r1i1p1f2",
}
cat_return = cat.climate.cmip6_gcs.search(**CMIP6_search)
ds_tas = (
    xr.combine_by_coords(
        cat_return.to_dataset_dict(zarr_kwargs={"consolidated": True}).values(),
        combine_attrs="drop",
    )
    .drop("height")
    .squeeze()
)
ds_tas


# %%
# Cell area and land fraction grids are the same for all scenarios, so we
# should take what is available.
# Cell area
CMIP6_search = {
    "source_id": "UKESM1-0-LL",
    "variable_id": "areacella",
    "grid_label": "gn",
}
cat_return = cat.climate.cmip6_gcs.search(**CMIP6_search)
ds_areacella = xr.combine_by_coords(
    cat_return.to_dataset_dict(zarr_kwargs={"consolidated": True}).values(),
    combine_attrs="drop",
).squeeze()
ds_areacella

# %%
# Land fraction
CMIP6_search = {
    "source_id": "UKESM1-0-LL",
    "variable_id": "sftlf",
    "grid_label": "gn",
}
cat_return = cat.climate.cmip6_gcs.search(**CMIP6_search)
ds_sftlf = xr.combine_by_coords(
    cat_return.to_dataset_dict(zarr_kwargs={"consolidated": True}).values(),
    combine_attrs="drop",
).squeeze()
ds_sftlf

# %% [markdown]
# ## Climate Change
# Calculate global mean surface air temperature, i.e. global climate change.
#
# Generally global warming is defined either in terms of near surface air
# temperatures, or surface temperatures.
# They do not give exactly the same result.

# %%
gsat = (
    ds_tas.tas.weighted(ds_areacella.areacella)
    .mean(("lat", "lon"))
    .resample(time="Y")
    .mean()
    .compute()
)
gsat_reference = gsat.sel(time=slice("1850", "1900")).mean("time")
gsat_change = (gsat - gsat_reference).groupby("time.year").first()

gsat_change.attrs = ds_tas.tas.attrs
gsat_change.attrs["long_name"] = "Global mean surface air temperature change"
gsat_change.attrs["short_name"] = "GSAT change"
gsat_change.attrs["units"] = "degC"

gsat_change.plot()
plt.show()

# %%
# Temperatures are in kelvin by default - I want them in Celsius
for var in ds:
    if "units" not in ds[var].attrs:
        continue
    elif ds[var].attrs["units"] == "K":
        print(f"Changing units of {var} K->degC")
        attrs = ds[var].attrs
        attrs["units"] = "degC"
        ds[var] = ds[var] + absolute_zero
        ds[var].attrs = attrs

# %%
# Because of compression, huss will sometimes have small negative values, which is not valid.
# It should be zero bounded.
ds["huss"] = ds.huss.where(ds.huss > 0, 0)

# %%
# Reduce scope of climate data for speed.
# Based on the geographic limits of the RiceAtlas data we have selected.
# And selecting only land.
valid_gridcells = ds_sftlf.sftlf.pipe(asia_only).pipe(lambda _da: _da > 0)
valid_gridcells = valid_gridcells.drop("type").drop("member_id")
ds = ds.pipe(asia_only).where(valid_gridcells)
valid_gridcells.plot()
plt.show()

# Add in date auxillaries
# This is because direct access via cftime dummy is slow.
ds["dayofyear"] = ds.time.dt.dayofyear
ds["year"] = ds.time.dt.year

# %% [markdown]
# ## Heat stress index
#
# Many studies focussed on the risk of occupational heat stress use
# wet-bulb globe temperature (WBGT), which is a heat-stress index
# defined by ISO 7243. WBGT is intended to combine all the factors that
# affect the human experience of heat, namely air temperature, radiant
# temperature, humidity, and air velocity. As plabourorming work
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
# TODO make this dimensional.
for WBGT, WBT, Ta in (
    ("wbgt_max", "wbt_max", "tasmax"),
    ("wbgt_mean", "wbt_mean", "tas"),
):
    # Specify WBT calculation, using psychrolib.
    ds[WBT] = xr.apply_ufunc(
        psl.GetTWetBulbFromHumRatio,
        ds[Ta],
        ds["huss"],
        ds["ps"],
        dask="parallelized",
        output_dtypes=[float],
    )
    # TODO add in Gaspar based BGT. Actually this probably isn't necessary.

    # Calculate WBGT, assuming the black globe temperature is approximated by the
    # air temperature. This will be approximately true in the shade.
    ds[WBGT] = ds[WBT] * 0.7 + ds[Ta] * 0.3
    ds[WBGT].attrs = {
        "units": "degC",
        "long_name": "Wet-bulb globe temperature",
        "short_name": WBGT,
    }
ds["wbgt_mid"] = (ds["wbgt_max"] + ds["wbgt_mean"]) / 2
# TODO it will be worth saving and reloading this data.

# %%
# Check the data make sense
# It should be a cone.
ds.sel(lat=10.0634, lon=105.5943, method='nearest').plot.scatter('tasmax', 'wbt_max')
plt.show()

# %%
# Check the data make sense
# This is West Bengal, it should get above 24 C regularly.
ds['wbgt_max'].sel(lat=10.0634, lon=105.5943, method='nearest').plot()
plt.show()


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
# You can see these different assumptions plotted below.
# TODO add other labour functions as dimension.

# %%
fig, ax = plt.subplots()
x = np.linspace(22, 40)
for loss, name in [
    ("labour_sahu", "Sahu et al."),
    ("labour_dunne", "Dunne et al."),
    ("labour_hothaps_high", "HOTHAPS high-intensity"),
]:
    ax.plot(x, Labour.__dict__[loss](x), label=name)

ax.set_ylabel("Labour impact (%)")
ax.set_xlabel("WBGT ($\degree C$)")
plt.tight_layout()
plt.legend()
plt.show()

# %% [markdown]
#
# The '4+4+4' assumption means that air temperature in the working day is
# assumed to be close to the maxmimum for 4 hours, the mean for 4 hours, and
# half-way between for 4 hours. This is a reasonably good approximation.
# This assumption comes from
# [Kjellstrom, T. et al., 2018](https://doi.org/10.1007/s00484-017-1407-0)
# but also gets used in several other papers including Orlov et al and Watts et al
# TODO fix those references.
#

# %%
# This is a delayed computation.
labour_ds_list = []
for labour_func_name in ("labour_sahu", "labour_dunne", "labour_hothaps_high"):
    labour_ds_list.append([])
    for WBGT_stat in (
        "max",
        "mean",
        "mid",
    ):
        WBGT = f"wbgt_{WBGT_stat}"
        func = Labour.__dict__[labour_func_name]

        labour_ds_list[-1].append(
            xr.apply_ufunc(
                func, ds[WBGT], dask="parallelized", output_dtypes=[float]
            ).assign_coords({"labour_func": labour_func_name, "wbgt_stat": WBGT_stat})
        )
        # print("A", func(ds[WBGT].isel(time=-6)).max().compute()) # Just checking
        # print("B", labour_ds_list[-1][-1].isel(time=-6).max().compute()) # Just checking

ds_labourloss = (
    xr.combine_nested(
        labour_ds_list,
        concat_dim=[
            "labour_func",
            "wbgt_stat",
        ],
    )
    .mean("wbgt_stat")
    .to_dataset(name="labour")
)
ds_labourloss.labour.attrs = {'long_name': 'Labour lost due to heat stress', 'short_name': 'lost_labour', 'units': '%'}

# %%
# Plot to check there is valid data
ds_labourloss.labour.isel(time=-6, labour_func=1).plot.hist()
plt.show()

# %%
# Invoke computation of the monthly labour loss.
# TODO it probably would make sense to save and reload this.
# This will be about 800MB in the stock configuration of this notebook.
ds_labourloss.to_netcdf('data/ds_labourloss.nc', encoding={'labour': {'dtype':'float32'}})

# %%
ds_labourloss = xr.open_dataset('data/ds_labourloss.nc')

# %% [markdown]
# Calculate correlations between annual GSAT and labour loss for each gridcell for each month.
# A key hypothesis of this work is that these are highly correlated.

# %%
def new_linregress(x, y):
    """Wrapper around scipy linregress to use in apply_ufunc"""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return np.array([slope, intercept, r_value, p_value, std_err])


def fit_parallel_wrapper(
    x: xr.DataArray, y: xr.DataArray, dim: str = "time"
) -> xr.DataArray:
    """Do a linear fit along an axis, independently in other dimensions.
    Different outputs of the fit are arranged along a new dimension, to get
    around the limitation that xr.apply_ufunc can only take a single variable
    output.
    """
    # return a new DataArray
    result = xr.apply_ufunc(
        new_linregress,
        x,
        y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[["linregress"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"],
        output_sizes={"parameter": 5},
    )
    result = result.assign_coords(
        {"linregress": ["slope", "intercept", "rvalue", "pvalue", "stderr"]}
    )
    return result


# %%
# Fit independently for each month and gridcell.
# Note that the data array needs to contain monthly data by this point, even if
# you started with daily or subdaily data.
ds_monthly_trends = ds_labourloss.labour.groupby("time.month").apply(
    lambda x: fit_parallel_wrapper(
        gsat_change, x.groupby("time.year").mean().sel(year=gsat_change.year), "year"
    )
)
ds_monthly_trends

# %%
# Save and reload
# This will be about 8 MB
ds_monthly_trends.to_dataset(name='fit').to_netcdf(Path("data") / "ds_monthly_trends.nc", encoding={'fit': {'dtype':'float32'}})

# %%
ds_monthly_trends = xr.open_dataset(Path("data") / "ds_monthly_trends.nc")

# %%
ds_monthly_trends.fit.sel(labour_func="labour_sahu", linregress="slope").max("month").plot()
plt.show()

# %%
# Fit for each gridcell with the whole year average.
ds_yearly_trends = (
    ds_labourloss.labour.groupby("time.year")
    .mean()
    .pipe(
        lambda x: fit_parallel_wrapper(
            gsat_change, x.sel(year=gsat_change.year), "year"
        )
    )
)
ds_yearly_trends

# %%
# Save and reload
# This will be about 8 MB
ds_yearly_trends.to_dataset(name='fit').to_netcdf(Path("data") / "ds_yearly_trends.nc", encoding={'fit': {'dtype':'float32'}})

# %%
ds_yearly_trends = xr.open_dataset(Path("data") / "ds_yearly_trends.nc")

# %%
ds_yearly_trends.fit.sel(labour_func="labour_sahu", linregress="slope").plot()
plt.show()


# %% [markdown]
# ## Combining the data
# TODO make the method more like the supplementary material plots.
# TODO make all the supplementary material-style plots.
# TODO improve the prose, read aloud.

# %%
# Use the polygons from the RiceAtlas shapefile to get subsets of the data.
# Use regionmask to start with
regions = regionmask.from_geopandas(ra, names="HASC")
mask = regions.mask(ds.lon, ds.lat)
# Turn regions into dimensions instead
mask_regions = xr.concat([(mask == r.number) for r in regions], dim="region")
mask_regions["region"] = ra.HASC.values

all_masks = []
all_weights = []
monthly_weights = []
peak_months = []
for i_region, region in ra.iterrows():
    months = [month_dict[m] for m in region[[f"HMO_PK{i}" for i in (1, 2, 3)]]]
    weights = region[[f"P_S{i}" for i in (1, 2, 3)]].astype(float).values
    single_mask = mask == regions[region.HASC].number
    if not single_mask.any():
        single_mask = valid_gridcells.sel(
            lon=[region.geometry.centroid.x],
            lat=[region.geometry.centroid.y],
            method="nearest",
        )

    single_mask = (
        single_mask.assign_coords({"HASC": region.HASC})
        .expand_dims(("HASC"))
        .to_dataset(name="mask")
    )
    all_masks.append(single_mask)

    all_weights.append(
        xr.DataArray(
            weights.reshape(-1, 1),
            dims=("season", "HASC"),
            coords={
                "season": np.array([1, 2, 3]),
                "HASC": np.array(region.HASC).reshape(1),
            },
        )
    )

    peak_months.append(
        xr.DataArray(
            np.array(months).reshape(-1, 1),
            dims=("season", "HASC"),
            coords={
                "season": np.array([1, 2, 3]),
                "HASC": np.array(region.HASC).reshape(1),
            },
        )
    )

    monthly_weights.append(
        xr.DataArray(
            np.array(
                [region[f"P_{calendar.month_abbr[m]}"] for m in range(1, 13)]
            ).reshape(12, 1),
            dims=("month", "HASC"),
            coords={
                "month": np.array(range(1, 13)),
                "HASC": np.array(region.HASC).reshape(1),
            },
        )
    )

ds_mask = xr.concat(all_masks, dim="HASC")
ds_mask["mask"] = ds_mask["mask"] > 0
da_weights_seasonal = xr.concat(all_weights, dim="HASC")
da_weights_monthly = xr.concat(monthly_weights, dim="HASC")
da_peak_months = xr.concat(peak_months, dim="HASC")

# %%
# Latitude and monthly plots

#damage_function = "WBGT_erf_sahu_fm"
use_seasonal_weights = True  # If False, there will be more small points


        # %%
        # Get rid of non-significant gradients
        ds.sel(linregress="slope")[damage_function] = ds.sel(linregress="slope")[
            damage_function
        ].where(ds.sel(linregress="rvalue")[damage_function] > 0.5, 0)

        # %%
        # Gridcell valid if it has any non nan points in any month.
        # These should generally be all the land gridcells
        # Because I masked out non-land in a previous step.
        valid_gridcells = (
            ds[damage_function]
            .pipe(np.isnan)
            .pipe(np.logical_not)
            .sel(linregress="slope")
            .squeeze()
            .any("month")
        )
        for remove_coordinate in ("type", "height", "linregress"):
            if remove_coordinate in valid_gridcells.coords:
                valid_gridcells = valid_gridcells.drop(remove_coordinate)

        # Can we do the regionwise aggregation quickly?
        # Use regionmask to start with
        regions = regionmask.from_geopandas(ra, names="HASC")
        mask = regions.mask(ds.lon, ds.lat)
        # Turn regions into dimensions instead
        mask_regions = xr.concat([(mask == r.number) for r in regions], dim="region")
        mask_regions["region"] = ra.HASC.values

        all_masks = []
        all_weights = []
        monthly_weights = []
        peak_months = []
        for i_region, region in ra.iterrows():
            months = [month_dict[m] for m in region[[f"HMO_PK{i}" for i in (1, 2, 3)]]]
            weights = region[[f"P_S{i}" for i in (1, 2, 3)]].astype(float).values
            single_mask = mask == regions[region.HASC].number
            if not single_mask.any():
                single_mask = valid_gridcells.sel(
                    lon=[region.geometry.centroid.x],
                    lat=[region.geometry.centroid.y],
                    method="nearest",
                )

            single_mask = (
                single_mask.assign_coords({"HASC": region.HASC})
                .expand_dims(("HASC"))
                .to_dataset(name="mask")
            )
            all_masks.append(single_mask)

            all_weights.append(
                xr.DataArray(
                    weights.reshape(-1, 1),
                    dims=("season", "HASC"),
                    coords={
                        "season": np.array([1, 2, 3]),
                        "HASC": np.array(region.HASC).reshape(1),
                    },
                )
            )

            peak_months.append(
                xr.DataArray(
                    np.array(months).reshape(-1,1),
                    dims=("season", "HASC"),
                    coords={
                        "season": np.array([1, 2, 3]),
                        "HASC": np.array(region.HASC).reshape(1),
                    },
                )
            )

            monthly_weights.append(
                xr.DataArray(
                    np.array(
                        [region[f"P_{calendar.month_abbr[m]}"] for m in range(1, 13)]
                    ).reshape(12, 1),
                    dims=("month", "HASC"),
                    coords={
                        "month": np.array(range(1, 13)),
                        "HASC": np.array(region.HASC).reshape(1),
                    },
                )
            )

        ds_mask = xr.concat(all_masks, dim="HASC")
        ds_mask["mask"] = ds_mask["mask"] > 0
        da_weights_seasonal = xr.concat(all_weights, dim="HASC")
        da_weights_monthly = xr.concat(monthly_weights, dim="HASC")
        da_peak_months = xr.concat(peak_months, dim="HASC")

        # Data processing
        # average trend by HASC, by month or season
        if use_seasonal_weights:
            ds_HASC = (
                ds.where(ds_mask.mask)
                .where(ds.month==da_peak_months)
                .weighted(da_weights_seasonal)
                .mean(("lon", "lat", "month"))
                .drop("height")
            )
            ds_HASC["weight"] = da_weights_seasonal
            ds_HASC["month"] = da_peak_months.where(lambda x:x>0)
            seasonal_variable = 'season'
            assert('season' in ds_HASC.dims)
        else: # use monthly weights
            ds_HASC = (
                ds.where(ds_mask.mask)
                .where(ds.month==da_peak_months)
                .weighted(da_weights_seasonal)
                .mean(("lon", "lat"))
                .drop("height")
            )
            ds_HASC["weight"] = da_weights_monthly
            seasonal_variable = 'month'
            assert('month' in ds_HASC.dims)

        ds_HASC["lat_"] = xr.DataArray(
            ra.centroid.y, dims="HASC", coords={"HASC": ra.HASC}
        )

        # Plotting
        lookup = defaultdict(lambda: "All others")
        lookup["CN"] = "China"
        lookup["IN"] = "India"
        lookup["ID"] = "Indonesia"
        # lookup['PH'] = 'Philippines'
        # for code in ('KH', 'MM', 'MY', 'TH', 'VN', 'BN', 'PH'):
        # lookup[code] = 'Southeast Asia'
        ds_HASC["country_label"] = (
            "HASC",
            [lookup[r] for r in ds_HASC.HASC.str[0:2].values],
        )

        ds_HASC["COUNTRY"] = ds_HASC.HASC.str[0:2]
        ds_HASC["REGION"] = ds_HASC.HASC.str[3:5]

        # Convert to pandas dataframe
        df = ds_HASC.sel(linregress="slope").to_dataframe().reset_index()
        df["file"] = f
        df_list.append(df)

    df = pd.concat(df_list)

    df["group_region"] = np.where(
        np.in1d(df.COUNTRY, ["IN", "CN"]), df.REGION, df.COUNTRY
    )

    # Do weighted aggregations to make the plot less busy
    df["weight_x_damage"] = df.weight * df[damage_function]
    # df = df.groupby(["group_region", "month"]).aggregate( # inconsistent with other plots
    # df = df.groupby(['country_label', 'month']).aggregate( # too tidy
    df = df.groupby(["COUNTRY", "REGION", "month"]).aggregate(  # messy but consistent
        {
            "weight": "sum",
            "weight_x_damage": "sum",
            "lat_": "mean",
            "country_label": "first",
        }
    )
    df["mean_damage"] = df["weight_x_damage"] / df["weight"]
    df = df.reset_index().sort_values("country_label")

    # Exclude tiny points
    # df = df.where(df.weight > df.weight.max() / 30)

    # Make canvas for plot against lat
    fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))

    # Plot with seaborn
    # Scatterplot by latitude
    sns.scatterplot(
        x=df.lat_,
        y=df["mean_damage"],
        size=df.weight,
        hue=df.country_label,
        # hue=m,
        sizes=(0, 300),
        size_norm=Normalize(0, df.weight.max()),
        alpha=0.8,
        ax=ax1,
        legend=False,
    )

    # Use a legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            label="All others",
            marker="o",
            color="none",
            markerfacecolor=sns.color_palette()[0],
            markeredgecolor="none",
        ),
        Line2D(
            [0],
            [0],
            label="China",
            marker="o",
            color="none",
            markerfacecolor=sns.color_palette()[1],
            markeredgecolor="none",
        ),
        Line2D(
            [0],
            [0],
            label="India",
            marker="o",
            color="none",
            markerfacecolor=sns.color_palette()[2],
            markeredgecolor="none",
        ),
        Line2D(
            [0],
            [0],
            label="Indonesia",
            marker="o",
            color="none",
            markerfacecolor=sns.color_palette()[3],
            markeredgecolor="none",
        ),
        # Line2D([0], [0], label='All Asia', color='k', markeredgecolor='none',),
    ]
    # ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # ax1.set_xticks(ticks=range(1,13))
    # ax1.set_xticklabels(labels=[calendar.month_abbr[i][0] for i in range(1,13)])
    ax1.set_xlabel("Latitude (deg N)", fontsize=8)
    ax1.set_ylabel("Hazard gradient (%/C)", fontsize=8)
    plt.tight_layout()

    output_dir = Path("subdaily/figures/latitude_trend") / damage_function
    output_dir.mkdir(exist_ok=True, parents=True)
    df[['COUNTRY', 'REGION', 'month', 'lat_', 'weight', 'mean_damage']].to_csv(output_dir / 'monthly_breakdown.csv')
    fig1.savefig(output_dir / "lat_breakdown.pdf")
    fig1.savefig(output_dir / "lat_breakdown.png")

    # Make canvas for plot against month
    fig2, ax1 = plt.subplots(figsize=(3.5, 3.5))

    # Plot with seaborn
    # Scatterplot by month
    sns.scatterplot(
        x=df.month,
        y=df["mean_damage"],
        size=df.weight,
        hue=df.country_label,
        # hue=m,
        sizes=(0, 300),
        size_norm=Normalize(0, df.weight.max()),
        alpha=0.8,
        ax=ax1,
        legend=False,
    )

    # Use a legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            label="All others",
            marker="o",
            color="none",
            markerfacecolor=sns.color_palette()[0],
            markeredgecolor="none",
        ),
        Line2D(
            [0],
            [0],
            label="China",
            marker="o",
            color="none",
            markerfacecolor=sns.color_palette()[1],
            markeredgecolor="none",
        ),
        Line2D(
            [0],
            [0],
            label="India",
            marker="o",
            color="none",
            markerfacecolor=sns.color_palette()[2],
            markeredgecolor="none",
        ),
        Line2D(
            [0],
            [0],
            label="Indonesia",
            marker="o",
            color="none",
            markerfacecolor=sns.color_palette()[3],
            markeredgecolor="none",
        ),
        Line2D([0], [0], label="All Asia", color="k", markeredgecolor="none",),
    ]
    # ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)

    ax1.set_xticks(ticks=range(1, 13))
    ax1.set_xticklabels(labels=[calendar.month_abbr[i][0] for i in range(1, 13)])
    ax1.set_xlabel("Month", fontsize=8)
    ax1.set_ylabel("Hazard gradient (%/C)", fontsize=8)
    plt.tight_layout()

    output_dir = Path("subdaily/figures/monthly_trend") / damage_function
    output_dir.mkdir(exist_ok=True, parents=True)
    fig2.savefig(output_dir / "monthly_breakdown.pdf")
    fig2.savefig(output_dir / "monthly_breakdown.png")
    # plt.show()
    # plt.close()

# %%
plt.show() # TODO remove this
