## Install

```
# 1. Create a conda virtual environment.
conda create -n fl python=3.9 -y
conda activate fl


# 2. Install PyTorch (I use PyTorch 2.0 built under cuda 11.8)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 3.
pip install dill

```