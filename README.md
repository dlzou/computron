# Computron

## Abstract

Many of the most performant deep learning models today in fields like language and image understanding are fine-tuned models that contain billions of parameters. In anticipation of workloads that involve serving many of such large models to handle different tasks, we develop Computron, a system that uses memory swapping to serve multiple distributed models on a shared GPU cluster. Computron implements a model parallel swapping design that takes advantage of the aggregate CPU-GPU link bandwidth of a cluster to speed up model parameter transfers. This design makes swapping large models feasible and can improve resource utilization. We demonstrate that Computron successfully parallelizes model swapping on multiple GPUs, and we test it on randomized workloads to show how it can tolerate real world variability factors like burstiness and skewed request rates.

## Installation for Development

Clone this repository and its submodules:

```shell
git clone --recurse-submodules git@github.com:dlzou/computron.git
```

Create an environment, install torch and Colossal-AI from PIP, then install Energon-AI and AlpaServe from the included submodules. Finally, install Computron from source.

```shell
conda create -n computron python=3.10
conda activate computron
pip install torch==1.13 torchvision colossalai transformers
pip install -e energonai/
pip install -e alpa_serve/
pip install -e .
```
