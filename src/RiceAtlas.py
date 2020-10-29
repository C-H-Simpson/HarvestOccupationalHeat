"""Load the RiceAtlas database. Fixing a few issues along the way.

The main issues are with sub-region names and 'HASC' names.
I also add the true shape area.

Laborte, Alice G.; Gutierrez, Mary Anne; Balanza, Jane Girly; Saito, Kazuki;
Zwart, Sander J.; Boschetti, Mirco; Murty, MVR; Villano, Lorena; Aunario,
Jorrel Khalil; Reinke, Russell; Koo, Jawoo; Hijmans, Robert J.; Nelson, Andrew,
2017, "RiceAtlas, a spatial database of global rice calendars and production",
https://doi.org/10.7910/DVN/JE6R2R, Harvard Dataverse, V3
"""
import geopandas as gpd
from pathlib import Path
import numpy as np
from mysettings.storage_settings import storage

# %%
# Load RiceAtlas
ra_path = big_storage / "RiceProduction_v1.shp"
if not ra_path.exists():
    raise FileNotFoundError('Download IRRI RiceAtlas from  https://doi.org/10.7910/DVN/JE6R2R')

ra = gpd.read_file(ra_path)

# Drop countries not in Asia
ra = ra[ra.CONTINENT == "Asia"]

# %%
# Give HASC-style codes to entities that are special cases.
# A few regions have no HASC code
# A few regions have non-unique HASC codes

# If HASC is missing, assign the ISO instead
for i in ra[[HASC is None for HASC in ra.HASC.values]].index:
    # Check the ISO is unique
    if (ra.loc[i].ISO is not None) and (np.count_nonzero(ra.ISO == ra.loc[i].ISO) == 1):
        ra.loc[i, 'HASC'] = ra.loc[i, 'ISO']
    else:
        print(ra.loc[i][['ISO', 'HASC', 'COUNTRY', 'REGION', 'SUB_REGION']])
        raise NotImplementedError()

# %%
# REGION name = COUNTRY name
# in cases where there is no REGION name
for i in ra[[REGION is None for REGION in ra.REGION.values]].index:
    # Check the ISO is unique
    ra.loc[i, 'REGION'] = ra.loc[i, 'COUNTRY']

# REGION name = COUNTRY name
# in cases where there is no REGION name
for i in ra[[SUB_REGION is None for SUB_REGION in ra.SUB_REGION.values]].index:
    # Check the ISO is unique
    ra.loc[i, 'SUB_REGION'] = ra.loc[i, 'REGION']

# %%
# Cases where the HASC is not unique
counts = ra.HASC.value_counts() > 1
non_unique_HASC = counts.index[counts]
for HASC in non_unique_HASC:
    a = 0
    for i in ra[ra.HASC == HASC].index:
        a += 1
        ra.loc[i, 'HASC'] += f'.{a}'

# %%
# Add the true area
import pyproj    
import shapely
import shapely.ops as ops
from shapely.geometry.polygon import Polygon
from functools import partial


geom_area = [
    ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat_1=geom.bounds[1],
                lat_2=geom.bounds[3])),
        geom).area
    for geom in ra.geometry]
ra['Area_True'] = geom_area
