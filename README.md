# Climate risk to rice labour

## Getting started
The notebook has already been evaluated, so have a look [reduced_example](notebooks/reduced_example.ipynb).

You can also run it on [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/C-H-Simpson/HarvestOccupationalHeat/HEAD?filepath=notebooks%2Freduced_example.ipynb) or on your own machine.

CMIP6 data is retreived from the Pangeo GCS.

A script is provided to setup the conda environment from inside the notebook if required.


## Project Organization
```
├── Makefile
├── README.md
├── environment.yml    <- Conda environment specification.
├── env.sh             <- Conda environment setup script.
│
├── notebooks          <- Jupyter notebooks.
│   └──reduced_example.ipynb <- Example of heat/labour analysis using climate data and crop calendars.
│   └──reduced_example.py    <- Script version of above notebook, used for clean version control.
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├──dayofyear.py    <- Logic for subsetting data based on the day of the year.
│   ├──Labour.py       <- Formulae for assumptions about the effect of WBGT on labour.
│   └──RiceAtlas.py    <- Routine for loading RiceAtlas data.
```
