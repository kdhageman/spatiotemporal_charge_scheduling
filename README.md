# Supplemental material 
This repository contains the source code, experiments and data processing code for the paper
```
K. Hageman and R.H Jacobsen, "On the Spatio-Temporal Allocation of Charging Stations to Autonomous Drone Fleets", 2023
```
The relevant files can be found in the following directory:
- The source code of the simulator and Pyomo optimization model can be found in `pyomo_models`, `simulate` and `util`.
- The experiments can be found in `experiments`
- The Jupyter notebooks for the data processing can be foudn in `notebooks_experiments`
- The `config` and `pointclouds` directories are (nearly) empty and are intended to be populated by relevant configuration and data files.

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

## How to run
> Note that these command are only tested on MacOS and Windows and Linux remain untested.

### 1. Preparation
It is necessary to have access to a 3d point cloud (in `.ply` format) of a structure.
Prior to running a simulation, a set of flight paths must be generated first for this structure.
To do so, prepare a configuration file (use `config/flight_sequence/template.yml` as a template).
Then run the following two commands:
```
$ python experiment/plan_flight_sequence_save_viewpoint.py <path to config file>
$ python experiment/plan_flight_sequence_for_pcl.py <path to config file>
```
The first command is used to generate a viewpoint file for Open3D, and the second produces a set of waypoints for each drone (saved in `flight_sequences.pkl`).
It also generates a series of figures that illustrates the processing the input point cloud into these flight sequences.
Note that both commands can use the same configuration file.

### 2. Simulation
To run a simulation and collect the results, first prepare a configuration file (use `config/charge_scheduling/template.yml` as a template).
Then run the following command:
```
$ python experiment/1_optimal_per_degredation.py
$ python experiment/2_charging_station_utility.py
$ python experiment/3_gridsearch.py
```

> Note that the configuration file used in these experiments serves as a "base" configuration only.

To run three individual sets of experiments
For each experiment, three files will be generated:
- `battery.pdf` shows the battery profile over time for all drones
- `occupancy.pdf` shows the utilization of the charging stations over time 
- `result.json` outputs detailed information collected during the simulation
