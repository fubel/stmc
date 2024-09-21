# Spatial-Temporal Multi-Cuts for Online Multiple-Camera Vehicle Tracking
![arXiv Badge](https://img.shields.io/badge/Paper-arXiv.0000.0000-b31b1b.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Authors

[Fabian Herzog](https://github.com/fubel), [Johannes Gilg](https://github.com/Blueblue4), [Philipp Wolters](https://github.com/phi-wol), [Torben Teepe](https://github.com/tteepe/), and Gerhard Rigoll

## Installation

Only tested with Python 3.8, CUDA 11.8, GCC >= 9.4.0 on NVIDIA RTX 3090, PyTorch 2.0.1 on Ubuntu 22.04.

```bash
# Setup with miniconda
conda create -n stmc python=3.8
conda activate stmc

# Setup torch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Setup RAMA
# (cf. https://github.com/pawelswoboda/RAMA/)
git clone git@github.com:pawelswoboda/RAMA.git
mkdir -p RAMA/build && cd RAMA/build
cmake ..
make -j 4

# Setup Python bindings
python -m pip install git+https://github.com/pawelswoboda/RAMA.git

# Install remaining dependencies
python -m pip install -r requirements.txt
```

## Data Setup

The config files assume the datasets are stored in `./data/`. You can setup a symlink to a different location or adjust the paths in the config. The datasets are available at:

* [CityFlow](https://www.aicitychallenge.org)
* [Synthehicle](https://github.com/fubel/synthehicle)

You need to provide the camera calibrations in `calibration.json` files. They are available in the releases.

## Running the Code

For a multi-camera scene, adjust the `config.yaml`. To track the Synthehicle scene `Town06-O-dawn`, run

```bash
# for Synthehicle, Town06-O-dawn
python main.py +experiment=Synthehicle dataset.scene_path=./test/Town06-O-dawn/
```

To track the CityFlow scene S02, run

```bash
# for Synthehicle, Town06-O-dawn
python main.py +experiment=CityFlow
```

❗️ We'll provide all pre-extracted detections and features soon!

## Features and Detections

Our resources are formatted in the MOT-Challenge format, with the addition that the last N columns of a resource file store the appearance feature vector of that object. Detections and features are available in the releases.

❗️ We'll provide all pre-extracted detections and features soon!

## Evaluation

The results are saved in the output directory specified in the config. 

**🚨 Please use the evaluation scripts provided by the respective datasets to evaluate the final results!**

Our in-built evaluation follows the evaluation protocol of Synthehicle which differs from the CityFlow official evaluation script (our eval does not filter single-cam trajectories, for instance). 

## Acknowledgements

We'd like to thank the authors of the following repositories for providing code used in our work:

* We use the [RAMA](https://github.com/pawelswoboda/RAMA.git) solver which enables fast multi-cuts on the GPU.
* The features for CityFlow are from [LCFractal](https://github.com/LCFractal/AIC21-MTMC).

## Citation
