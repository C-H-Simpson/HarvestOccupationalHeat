# Climate risk to rice labour

## Getting started
The notebook has already been evaluated, so have a look [reduced_example](notebook/reduced_example.ipynb).

To run the notebook yourself:
1. Run `make setup`
2. Run `source openID.sh` and set your ESGF login info.
3. Open the notebook [reduced_example](notebook/reduced_example.ipynb).


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

