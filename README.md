# Drone charging

## Installation
Create a new conda environment based on `environment.yml`:
```
conda env create -f environment.yml
```
Install `networkx-metis` manually:
```
cd <workspace directory>
git clone https://github.com/networkx/networkx-metis.git
cd networkx-metis
pip install Cython # optional
python setup.py install
```
The conda installation should ship with the correct Gurobi command line tools, including those needed for handling the licenses.
Obtain a Gurobi key and add the key to the machine that you are running the code on:
```
grbgetkey <license key> 
```
The license key should come from a page such as [this](https://www.gurobi.com/downloads/free-academic-license/).