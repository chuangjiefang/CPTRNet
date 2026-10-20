# CPTRNet
Code of GKConv-Guided CP Decomposition for Hyperspectral and Multispectral Image Fusion

# Getting started

## Setup environment

1. clone the repository

```bash
git clone https://github.com/chuangjiefang/CPTRNet.git
cd CPTRNet
```

2. Install Python environment

```bash
conda create -n your_env_name python=3.8.20
```


## Prepare dataset

Datasets can be downloaded from: [data](https://pan.baidu.com/s/1-dpIwSvNUnsuC1N6uobfxQ?pwd=cy97)


## Dataset Directory Structure

Your dataset should be organized as follows under the `data/` directory:

```

data/
├── CAVE/
├── Harvard/
├── Chikusei/
└── Pavia/
```

## Training
1. **Edit `option.py`**
2. **Run the main script**:

```bash
python train.py
```
