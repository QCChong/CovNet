## Point Cloud Completion with Geometric Inference and Refinement using Covariance

Xiangyang Wu, Chongchong Qu, Haixin Zhou, Yongwei Miao

-----------

This repository contains the source code for the paper “Point Cloud Completion with Geometric Inference and Refinement using Covariance”.

-----


### Usage

#### 1) Envrionment & prerequisites

- CUDA 10.0
- Python 3.7
- Pytorch 1.4.0
- Pytorch-lightning 1.3.8

#### 2) Compile

Install this library by running the following command:
Compile our extension modules:  

```bash
    cd loss/emd
    python setup.py install

    cd utils/pointnet2_ops_lib
    python setup.py install
```

#### 3) Download data and trained models
```bash
    cd data
    bash get_dataset.sh
```

#### 4) Train or validation. 
```bash
    # 1. train the model
    python train.py 
    
    # 2. validate the model, The pretrained model is [here](https://drive.google.com/file/d/1fgLk4ueRjanDFRjZDo3BAAoFDtKdJ9E9/view?usp=sharing).Download them here, and unzip it under logs/.
    python evalution.py
```

## Acknowledgement

This code is based on  [MSN](https://github.com/Colin97/MSN-Point-Cloud-Completion) and [Pointnet2.Pytorch](https://github.com/erikwijmans/Pointnet2_PyTorch).
