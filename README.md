# Federated Learning

<p style="text-align:right;">Wei, 2023/5</p>

This is my PyTorch approach to simulate and implement the interactions between clients and the cloud server in horizontal Federated Learning mode to realize a simple MNIST classification. The details are listed as follows, just the same as FedAvg algorithm.

- **server**: create N threads, one thread per client
  - randomly choose M out of N clients
  - send global weight to M clients
  - receive local weight from them
  - average the weight from N clients (N-M clients will use old weight)
- **client**: create N processes, one process per client
  - receive global weight
  - train the local model on its local data
  - send local weight to the server

Transferring large pickle files via sockets and using condition variables for synchronization are the two most difficult parts in the implementation. For more details, please refer to the code and `report.pdf`.

## 1. Install

```
# 1. Create a conda virtual environment.
conda create -n fl python=3.9 -y
conda activate fl

# 2. Install PyTorch (I use PyTorch 2.0 built under cuda 11.8)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Install dill
pip install dill
```

## 2. Dataset

- Put MNIST dataset in `data/MNIST/raw` folder
- Client dataset is pre-divided MNIST dataset, and should be put in `data/client` folder. You can download from https://pan.baidu.com/s/1zzVC420DMYYo_y54OTujeg?pwd=ypzz 

The final structure should be like:

```
.
├── client.py
├── data
│   ├── client
│   │   ├── Client10.pkl
│   │   ├── Client11.pkl
│   │   ├── Client12.pkl
│   │   ├── Client13.pkl
│   │   ├── Client14.pkl
│   │   ├── Client15.pkl
│   │   ├── Client16.pkl
│   │   ├── Client17.pkl
│   │   ├── Client18.pkl
│   │   ├── Client19.pkl
│   │   ├── Client1.pkl
│   │   ├── Client20.pkl
│   │   ├── Client2.pkl
│   │   ├── Client3.pkl
│   │   ├── Client4.pkl
│   │   ├── Client5.pkl
│   │   ├── Client6.pkl
│   │   ├── Client7.pkl
│   │   ├── Client8.pkl
│   │   └── Client9.pkl
│   └── MNIST
│       └── raw
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-images-idx3-ubyte.gz
│           ├── t10k-labels-idx1-ubyte
│           ├── t10k-labels-idx1-ubyte.gz
│           ├── train-images-idx3-ubyte
│           ├── train-images-idx3-ubyte.gz
│           ├── train-labels-idx1-ubyte
│           └── train-labels-idx1-ubyte.gz
├── network.py
├── server.py
└── utils.py
```

## 3. Run

Create N clients; M out of N clients will participate in the update

```
python server.py --M 10
python client.py --N 20
```

