# Hierarchical Few-shot Learning via Multi-granularity Global-local Feature Cross-measure Mechanism

PyTorch implementation of 

**HGLCM achieves new state-of-the-art performance on five few-shot learning benchmarks with significant advantages. The result is obtained without using any extra data for training or testing (tranductive setting).**


## Prerequisites

The following packages are required to run the scripts:

- [PyTorch >= version 1.1](https://pytorch.org)

- [CVXPY](https://www.cvxpy.org/)

- [OpenCV-python](https://pypi.org/project/opencv-python/)

- [tensorboard](https://www.tensorflow.org/tensorboard)
## Dataset
Please click the Google Drive [link](https://drive.google.com/drive/folders/1sXJgi9pXo8i3Jj1nk08Sxo6x7dAQjf9u?usp=sharing) or [Baidu Drive (uk3o)](https://pan.baidu.com/s/17hbnrRhM1acpcjR41P3J0A) for downloading the 
following datasets, or running the downloading bash scripts in folder `datasets/` to download.


### TieredImageNet Dataset
TieredImageNet is also a subset of ImageNet, which includes 608 classes from 34 super-classes. Compared with  miniImageNet, the splits of meta-training(20), meta-validation(6) and meta-testing(8) are set according to the super-classes to enlarge the domain difference between  training and testing phase. The dataset also include more images for training and evaluation (779,165 images in total).


### FC100 Dataset
FC100 is a few-shot classification dataset built on CIFAR100. We follow the split division proposed in [TADAM](https://papers.nips.cc/paper/7352-tadam-task-dependent-adaptive-metric-for-improved-few-shot-learning.pdf), where 36 super-classes were divided into 12 (including 60 classes), 4 (including 20 classes), 4 (including 20 classes), for meta-training, meta-validation and meta-testing, respectively, and each class contains 600 images.

### CIFAR-FS dataset (not in paper)
CIFAR-FS was also built upon CIFAR100,proposed in [here](https://arxiv.org/pdf/1805.08136.pdf). It contains 64, 16, 20 classes for training, validation and testing.



## Acknowledgment
Our project references the codes in the following repos.
- [DeepEMD](https://git.io/DeepEMD)




