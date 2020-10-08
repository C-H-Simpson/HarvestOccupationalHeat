"""
Find files and set up calculations for WBGT in CMIP6.
"""
import argparse
import baspy as bp
import warnings
from mysettings.geography import Asia_latlon_slice
from mysettings.storage_settings import storage_big
from src.get_data.esgf_opendap import get_openDAP_urls
from src.WBGT import Calculate_WBGT

parser = argparse.ArgumentParser(
    description="Calculate WBGT from temperature and humidity data"
)
parser.add_argument("Model", type=str)
parser.add_argument("experiment_id", type=str)
parser.add_argument("member_id", type=str, default="r1i1p1f2")
args = parser.parse_args(["UKESM1-0-LL", "historical", "r1i1p1f2"])
print(args)

DS = Calculate_WBGT.DatasetSpec()
DS.input_variables = {
    "max_temperature": "tasmax",
    "mean_temperature": "tas",
    "relative_humidity": "hurs",
}

search = {
    "variable_id": "tas",  # iterate this in this script
    "table_id": "day",
    "member_id": args.member_id,
    "activity_id": "CMIP",  # CMIP if experiment_id=='historical', ScenarioMIP if scenario
    "experiment_id": args.experiment_id,  # iterate this using argparse
    "source_id": args.Model,  # iterate this using argparse
}

if args.experiment_id == "historical":
    search["activity_id"] = "CMIP"
else:
    search["activity_id"] = "ScenarioMIP"

# Find the basic variables
for var_name, var_short_name in DS.input_variables.items():
    print(var_name)
    search["variable_id"] = var_short_name
    DS.input_path_dict[var_name] = get_openDAP_urls(search)

# Find pressure
# A bit tricky, will sometimes end up with sea level pressure instead.
# Strictly, one should adjust for altitude.
pressure = None
for var in ["ps", "psl"]:
    print(var)
    pressure = var
    search["variable_id"] = var
    try:
        urls = get_openDAP_urls(search)
    except ValueError:
        print("couldn't find", var)
        continue
    DS.input_path_dict["pressure"] = urls
    DS.input_variables["pressure"] = var
    break
else:
    raise ValueError("Could not find valid pressure data")

DS.output_directory = storage_big / args.Model / args.experiment_id / args.member_id

DS.output_directory.mkdir(exist_ok=True, parents=True)
print(DS.output_directory)

# specify some things about the dataset
DS.input_has_tdewpoint = False
DS.input_has_vapor_pressure = False
DS.input_has_pressure = True if pressure else False
DS.input_has_relhum = True
DS.input_in_kelvin = True

# define any slicing that will be applied to the data
DS.slicing = Asia_latlon_slice

DS.run_computation()
