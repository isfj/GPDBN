# GPDBN: Deep bilinear network integrating both genomic data and pathological images for breast cancer prognosis prediction

This is an implementation of GPDBN for survival analysis in Python 3.7.7 under Linux with CPU Intel Xeon 4110 @ 2.10GHz, GPU NVIDIA GeForce RTX 2080 Ti, and 192GB of RAM. 



## Installation
```
git clone https://github.com/isfj/GPDBN.git
```
## Running Experiments
Before running the experiments, make sure the required environment in 'requirment.txt' is configured. After configuring the required environment, you can run GPDBN supervised with cox objective by the following codes
```
cd GPDBN/GPDBN_Cox
python train_cv.py
```

## Dataset
Breast cancer patient samples adopted in this study include matched digital whole-slide images and gene expression profiles, which are acquired from The Cancer Genome Atlas (TCGA) data portal.


## Contact
Please feel free to contact us if you need any help: ustc023w@mail.ustc.edu.cn

All rights reserved
