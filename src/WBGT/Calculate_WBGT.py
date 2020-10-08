"""
   Given some temperature and humidity data,
    get the wet-bulb globe temperature.
"""
import warnings
import xarray as xr
import psychrolib as psy
from pathlib import Path

psy.SetUnitSystem(psy.SI)
STANDARD_PRESSURE = 101325.


class DatasetSpec(object):
    """ Calculate WBGT using psychrolib from source datasets.
    See settings in __init__ and examples provided.
    Invoke with run_computation. Results will be saved to disk, not returned.


    This maybe didn't need class structure"""

    def __init__(self):
        self.input_path_dict = {}
        self.input_variables = {}
        self.output_directory = "/gws/nopw/j04/bas_climate/users/champs/CMIP6/"
        self.input_has_vapor_pressure = False
        self.input_has_relhum = False
        self.input_has_tdewpoint = False
        self.input_has_pressure = False
        self.input_in_kelvin = True
        self.converted_to_celsius = False
        self.slicing = {}
        self.rename = {}

    def load_data(self):
        """ I load xr.DataArray objects into a dictionary, as loading them into
        a dataset doesn't always work this is because the mean and max values
        daily values are not defined on the same time grid for e.g. in ERA5.
        """
        self.ds_input = {}

        for variable in self.input_variables:
            print("loading", self.input_path_dict[variable])
            self.ds_input[variable] = xr.open_mfdataset(
                self.input_path_dict[variable],
                combine='by_coords',
                use_cftime=True,
                )[self.input_variables[variable]]
            # rename the coords if necessary
            self.ds_input[variable] = \
                self.ds_input[variable].rename(self.rename)

            # did the load fall back on cftime?
            #if self.ds_input[variable].time.dtype == 'O':
                #raise ValueError('failed to decode dates')

            # slice out irrelvant data points
            self.ds_input[variable] = \
                self.ds_input[variable].sel(**(self.slicing))

        if self.input_has_pressure:
            if 'pressure' in self.ds_input:
                print("using correct surface pressure")
                self.pressure = self.ds_input['pressure']
            else:
                raise ValueError(
                    'input_has_pressure==True, but file not found')
        else:
            warnings.warn("Using standard surface pressure")
            self.pressure = STANDARD_PRESSURE

    #######################################################
    def check_alignment(self):
        """
         Check alignment of min, max, mean - in some cases, esp. ERA5 daily
         data, the time axes are off by 30 min, due to the difference in bounds.
       """
        try:
            xr.align(
                self.ds_input['mean_temperature'],
                self.ds_input['max_temperature'], join="exact")
        except ValueError:
            warnings.warn(
              "overriding coordinates, as self.da_vap does not exactly match self.da_max_temperature")
            self.ds_input['mean_temperature'], \
                self.ds_input['max_temperature'] = \
                xr.align(
                self.ds_input['mean_temperature'],
                self.ds_input['max_temperature'], join="override")

    #######################################################
    def convert_to_celsius(self):
        if self.converted_to_celsius:
            warnings.warn("already converted to celsius")
            return

        if self.input_in_kelvin:
            print("converting kelvin to celsius")

            # self.ds_input can contain things that aren't temperatures
            for key in (
                'mean_temperature',
                'max_temperature',
                'dewpoint_temperature'
            ):
                # self.ds_input does not necessarily have dewpoint_temperature
                if key not in self.ds_input:
                    continue

                self.ds_input[key] = xr.apply_ufunc(
                        psy.GetTCelsiusFromTKelvin,
                        self.ds_input[key],
                        dask='parallelized', output_dtypes=[float],
                        keep_attrs=True
                    )
                self.ds_input[key].attrs['units'] = 'degC' # metadata
        self.converted_to_celsius = True

    #######################################################
    def get_humratio(self):
        print("getting humidity ratio")
        self.da_humratio = None

        # if it has vapor pressure, use that
        if self.input_has_vapor_pressure:
            self.da_humratio = xr.apply_ufunc(
                psy.GetHumRatioFromVapPres,
                self.ds_input["vapor_pressure"]*100,
                self.pressure,
                dask='parallelized', output_dtypes=[float],
                keep_attrs=True
            )

        elif self.input_has_relhum:
            self.da_humratio = xr.apply_ufunc(
                psy.GetHumRatioFromRelHum,
                self.ds_input['mean_temperature'],
                # this line assumes that the relative humidity
                # is defined over the range 0-100%, rather than 0-1
                # which is true of all CMIP6 data
                (self.ds_input['relative_humidity'] / 100.).clip(0, 1),
                self.pressure,
                dask='parallelized', output_dtypes=[float],
                keep_attrs=True
            )

        elif self.input_has_tdewpoint:
            self.da_humratio = xr.apply_ufunc(
                psy.GetHumRatioFromTDewPoint,
                self.ds_input['dewpoint_temperature'],
                self.pressure,
                dask='parallelized', output_dtypes=[float],
                keep_attrs=True
            )

        else:
            raise ValueError(
                'dataset has neither vap, rel hum, nor dewpoint temp')

        # Metadata
        self.da_humratio = self.da_humratio.rename("HumRatio")
        self.da_humratio.attrs['units'] = ''
        self.da_humratio.attrs['Conventions'] = ''  # probably not compliant
        self.da_humratio.attrs['standard_name'] = 'humidity_ratio'
        self.da_humratio.attrs['long_name'] = 'humidity ratio'
        self.da_humratio.attrs['Calculation'] =\
                'Calculated using psychrolib, and WBGT=WBT*0.7+T_a*0.3 assumption'

        # Save and reload, so that computation is not invoked repeatedly
        humratio_fname = self.save(self.da_humratio)
        self.da_humratio = xr.open_dataset(humratio_fname, use_cftime=True)

    #######################################################
    def calculate_WBGT(self):
        print("getting WBGT")
        def wbt(temperature, humratio):
            return xr.apply_ufunc(psy.GetTWetBulbFromHumRatio, temperature,
                                  humratio, self.pressure, dask='parallelized',
                                  output_dtypes=[float], keep_attrs=True)

        def wbgt(temperature, humratio):
            # here we are using KFL's indoor assumption
            result = 0.67 * wbt(temperature, humratio) + 0.33 * temperature

            # Metadata
            result.attrs['units'] = 'degC'
            result.attrs['Conventions'] = ''
            result.attrs['short_name'] = 'WBGT'
            result.attrs['long_name'] = 'wet-bulb globe temperature'
            return result

        self.da_WBGTmean = wbgt(
            self.ds_input['mean_temperature'], self.da_humratio['HumRatio'])

        self.da_WBGTmax = wbgt(
            self.ds_input['max_temperature'], self.da_humratio['HumRatio'])

        self.da_Tmid = (
            self.ds_input['max_temperature'] +
            self.ds_input['mean_temperature']
        ) / 2.
        self.da_WBGTmid = wbgt(self.da_Tmid, self.da_humratio['HumRatio'])

        #######################################################
        # Update metadata
        self.da_WBGTmean = self.da_WBGTmean.rename('WBGTmean')
        self.da_WBGTmax = self.da_WBGTmax.rename('WBGTmax')
        self.da_WBGTmid = self.da_WBGTmid.rename('WBGTmid')

        self.save(self.da_WBGTmean)
        self.save(self.da_WBGTmax)
        self.save(self.da_WBGTmid)

    def save(self, ds):
        """Function to save xarray data in a directory"""
        # Add in individual fields for calendar components access via the
        # datetime accessor is slow, so it is convenient to have these in later
        # steps.
        # This is just for convenience.
        name = ds.name
        ds = ds.to_dataset(name=name)
        ds['dayofyear'] = ds.time.dt.dayofyear
        ds['day'] = ds.time.dt.dayofyear
        ds['month'] = ds.time.dt.month
        ds['year'] = ds.time.dt.year

        # Adjust 360 day calendar if appropriate
        if ds.dayofyear.max() == 360:
            for day in (73, 145, 218, 291, 364):
                ds.dayofyear[ds.dayofyear>=day] = ds.dayofyear[ds.dayofyear>=day]+1

        # Check directory exists and save
        nc_fname = Path(self.output_directory) / name+'.nc'
        Path(nc_fname).parent.mkdir(exist_ok=True, parents=True)
        print("nc_fname", nc_fname)
        ds.to_netcdf(nc_fname, encoding={name: {'dtype': 'float32'}}, engine='h5netcdf')
        return nc_fname

    def run_computation(self):
        # check that the output directory exists
        Path(self.output_directory).mkdir(exist_ok=True, parents=True)

        #######################################################
        self.load_data()

        #######################################################
        self.convert_to_celsius()

        #######################################################
        self.check_alignment()

        #######################################################
        self.get_humratio()

        #######################################################
        self.calculate_WBGT()
