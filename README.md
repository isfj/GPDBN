# GPDBN: Deep bilinear network integrating both genomic data and pathological images for breast cancer prognosis prediction

This is an implementation of GPDBN in Python 2.7.12 under Linux with CPU Intel Xeon 4110 @ 2.10GHz, GPU NVIDIA GeForce RTX 2080 Ti, and 192GB of RAM. It needs Keras libraries to be installed.

It uses Keras library with the Tensorflow backend, and does not work on the Theano backend. Because the loss function of the network is written with Tensorflow.

## Installation
```
git clone https://github.com/isfj/GPDBN.git
cd GPDBN
```
## Running Experiments
Our proposed GPDBN framework is in the `GPDBN/model/inter_intra.py ` directorty.
Before running the experiments, make sure the required environment is configured. After configuring the required environment, you can run GPDBN by the following codes
```
cd GPDBN/model
python inter_intra.py
```

## Dataset
Breast cancer patient samples adopted in this study include matched digital whole-slide images and gene expression profiles, which are acquired from The Cancer Genome Atlas (TCGA) data portal 

## Contact
Please feel free to contact us if you need any help: ustc023w@mail.ustc.edu.cn

All rights reserved
