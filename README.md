# Climate risk to rice labour

## Getting started
The notebook has already been evaluated, so have a look [reduced_example](notebooks/reduced_example.ipynb).

To run the notebook yourself:
1. Create the environment:
* There is a known issue with the package manager conda, that causes it to use very large memory when trying to solve an environment. For this reason, the default `environment.yml` has all versions specified, and contains everything that was in the environment during development. Alternatively, if this isn't a problem, `environment_minimal.yml` has just those packages that are required.
* `conda env create -f environment.yml --prefix $PWD/env`
2. Open [reduced_example](notebooks/reduced_example.ipynb) and run it.


## Project Organization
```
├── Makefile
├── README.md
├── environment.yml    <- Conda environment specification.
├── env.sh             <- Conda environment setup script.
├── openID.sh          <- Script to set environment variables for ESGF openDAP access, not necessary when running interactively.
│
├── notebooks          <- Jupyter notebooks.
│   └──reduced_example.ipynb <- Example of heat/labour analysis using climate data and crop calendars.
│   └──reduced_example.py    <- Script version of above notebook, used for clean version control.
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├──dayofyear.py    <- Logic for subsetting data based on the day of the year.
│   ├──esgf_opendap.py <- Routine for loading climate data from ESGF.
│   ├──Labour.py       <- Formulae for assumptions about the effect of WBGT on labour.
│   └──RiceAtlas.py    <- Routine for loading RiceAtlas data.
```
