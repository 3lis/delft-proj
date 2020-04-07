# Visiting project at TU Delft

This repository contains the code developed as visiting project at the Intelligent Vehicles Group of TU Delft.

## Quick Start

### Prerequisites
- keras (backend tensorflow)
- nuscenes-devkit
- pillow
- pyquaternion
- matplotlib
- opencv
- h5py

Tested with Python 3.6.5, on MacOS 10.14.3 and Ubuntu 18.04.4 LTS.


### Data Preparation
Download the [nuScenes](https://www.nuscenes.org/download) dataset (v1.0) and place it in `dataset/nuScenes/orig`. Then, execute the script `src/extract_nu.py` to:

- extract and process the relevant ground truth data, which is placed in `dataset/nuScenes/data`
- generate the file `dataset/nuScenes/black_list.npy` indicating the nuScenes scenes where: there are no visible vehicles, the ego-vehicle is stationary, the environment conditions are bad (night/rain).


### Execution
The program can be run by executing the main script `src/exec_main.py `. The script supports the following command line arguments:

```
main_exec.py [-h] -c <file> -g <num> [-f <frac>] [-l <model>] [-Ttrs]
```

- `-c <file>`, `--config <file>` pass the configuration file (without path nor extension) describing the model architecture and training parameters.
- `-f <frac>`, `--fgpu <frac>` set the fraction of GPU memory to allocate *[default: 0.90]*.
- `-g <num>`, `--gpu <num>` set the number of GPUs to use (0 if CPU) or list of GPU indices.
- `-h`, `--help` show the help message with description of the arguments.
- `-l <model>`, `--load <model>` pass a HDF5 file to load as weights of the model.
- `-r`, `--redir` redirect _stdout_ and _stderr_ to log files.
- `-s`, `--save` archive the used python scripts and configuration file.
- `-t`, `--test` test the model.
- `-T`, `--train` train the model.

As example, the following command creates a new model using the parameters defined in the configuration file `src/cnfg/net1.py`. It trains the model using the first two GPUs on the machine, then it tests the results, saves all the files required to reproduce the experiment, and redirects all console messages to log files:

```
python src/main_exec.py -c net1 -g 0,1 -Ttssr
```

Another example, this command loads an already trained model and executes all the test routines on CPU:

```
python src/main_exec.py -l log/nicemodel.h5 -c net1 -g 0 -t
```
