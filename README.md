# Computron

Serving multiple parallelized deep learning models on the same cluster, with dynamic offloading.

Final project for [CS 267: Parallel Computing](https://sites.google.com/lbl.gov/cs267-spr2023).

## Install

### For Development

Clone this repo:

```shell
git clone --recurse-submodules git@github.com:dlzou/cs267-project.git
```

Create an environment, install torch and Colossal-AI from PIP, then install Energon-AI and AlpaServe from the included submodules. Finally, install Computron from source.

```shell
conda create -n computron python=3.10
conda activate computron
pip install torch==1.13 torchvision
pip install colossalai transformers
pip install -e energonai/
pip install -e alpa_serve/
pip install -e .
```
