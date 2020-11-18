# Climate risk to rice labour

## Getting started
The notebook has already been evaluated, so have a look [reduced_example](notebooks/reduced_example.ipynb).

To run the notebook yourself:
1. Create the environment:
* There is a known issue with the package manager conda, that causes it to use very large memory when trying to solve an environment. If this turns out to be a problem for you, get around this by usings "full_env.yml" instead of "environment.yml".
* `conda env create -f environment.yml --prefix $PWD/env`
2. Open [reduced_example](notebooks/reduced_example.ipynb) and run it.


## Project Organization
```
├── Makefile
├── README.md
├── environment.yml    <- Minimal conda environment to run examples.
├── dev_environment.yml    <- Conda environment used in development.
├── openID.sh          <- Script to set environment variables for ESGF openDAP access.
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
