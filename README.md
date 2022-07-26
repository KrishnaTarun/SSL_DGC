# Dynamic Channel Selection in Self-Supervised Learning

This is a official Pytorch based implementation of [Dynamic Channel Selection in Self-Supervised Learning](https://arxiv.org/abs/2207.12065) accepted in [IMVIP 2022](https://imvipconference.github.io/). Code from channel gating is derived from https://imvipconference.github.io/ while we train the self-supervised approach based on [Solo-learn library](https://github.com/vturrisi/solo-learn).

## Getting Started 

### Requirements

The main requirements of this work are:

- Python 3.8  
- PyTorch == 1.10.0  
- Torchvision == 	0.11.1
- CUDA 10.2

We recommand using conda env to setup the experimental environments.

# Install other requirements
```shell script
pip install -r requirements.txt

# Clone repo
git clone https://github.com/KrishnaTarun/SSL_DGC.git
cd ./SSL_DGC
```

### Pre-Training

```shell script

# CIFAR-100
bash bash_files/pretrain/imagent100/simsiam.sh

```

