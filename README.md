# Unsupervised Machine Learning In Azimuthal Electron Data
## P4 Applied Computer Lab 2762562B

This package was developed within a project to utilise unsupervised machine learning (UML) 
to denoise azimuthal electron datasets.

### Dependencies
The following packages are needed for this package to run (These are already declared in the .toml file):

**numpy >= 2.0.2**

**matplotlib>=3.9.2**

**scipy>=1.13.1**

**hyperspy>=2.3.0**

**tqdm>=4.67.2**

### Navigation
The "PhysComp" file contains all of the devloped functions seperated into .py files grouped by area of the project.
The files are the following:

**uml.py** - Functions relating to decomposing a dataset and reconstructing using UML

**fitting.py** - Functions relating to fitting a two-fold periodic function to a dataset

**assess.py** - Functions relating to assessment of the denoising and assosiated fits

**sim.py** - Funtion defined to simulate a dataset for testing purposes

**visual.py** - Functions relating to visualising the data, both 3D and 2D

Within this package there is a "Demo" folder on the top level containing notebooks demonstrating the useage of each function 

### Installation
This package should install on a python terminal with the following command:
- "pip install git+https://github.com/aedanbuchan/2762562B-P4-Computer-Lab.git"
