from setuptools import find_packages, setup

setup(
        name="src",
        version="0.0.1",
        author="Charles Simpson",
        author_email="champs@bas.ac.uk",
        description="Estimating the effect of occupational heat exposure on rice harvesting",
        url="https://github.com/C-H-Simpson/HarvestOccupationalHeat",
        platforms=['Unix'],
        packages=find_packages('.'),
)
