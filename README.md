# L-TTA: Lightweight Test-Time Adaptation Using a Versatile Stem Layer (NeurIPS 2024)
This repository provides the official PyTorch implementation of the following paper:

> **Abstract:** 
*Test-time adaptation (TTA) is the most realistic methodology for adapting deep learning models to the real world using only unlabeled data from the target domain. Numerous TTA studies in deep learning have aimed at minimizing entropy. However, this necessitates forward/backward processes across the entire model and is limited by the incapability to fully leverage data based solely on entropy. This study presents a groundbreaking TTA solution that involves a departure from the conventional focus on minimizing entropy. Our innovative approach uniquely remodels the stem
layer (i.e., the first layer) to emphasize minimizing a new learning criterion, namely, uncertainty. This method requires minimal involvement of the
model’s backbone, with only the stem layer participating in the TTA process. This approach significantly reduces the memory required for training and enables rapid adaptation to the target domain with minimal parameter updates. Moreover, to maximize data utility, the stem layer applies a discrete wavelet transform to the input features. It extracts multi-frequency domains and focuses on minimizing their individual uncertainties. The proposed method integrated into ResNet-26 and ResNet-50 models demonstrates its robustness by achieving state-of-the-art TTA performance while using the least amount of memory compared to existing studies on CIFAR-10-C, CIFAR-100-C, and Cityscapes-C benchmark datasets.*

<p align="center">
  <img src="assets/NeurIPS_presentation.jpg" />
</p>

## Pytorch Implementation
### Installation

Easy conda environment setting: you only need to import robust2.yml.

```
 $ conda env create --name robust2 --file=robust2.yml
 $ conda activate robust2
```

The easiest way to install pytorch_wavelets for DWT operations (i.e., 2D)

```
 $ git clone https://github.com/fbcotter/pytorch_wavelets.git
 $ cd pytorch_wavelets
 $ pip install .
```

## How to TTA

### TTA
You can easily perform the TTA process with the bash command below.
Only by changing the code in brackets

+ CIFAR-10-C Download Link: (https://zenodo.org/records/2535967)
+ ImageNet-C Download Link: (https://zenodo.org/records/2235448)

+ ResNet-26 Model (CIFAR-10-C) ResNet-50 Model (ImageNet-C) Download Link: (https://zenodo.org/records/13917882)

+ modify
    + dataset path 
        + ex) /home/users/~
    + epochs
        + ex) 1 # only 1
    + SE BLOCK OPT
        + ex) 1 321 0
    + DWT_KERNEL_SIZE
        + ex) 3 3 3
    + DWT_LEVEL
        + ex) 2 2 2
    + ITERATION FOR TTA
        + ex) 0 ~ 
```
CUDA_VISIBLE_DEVICES=[GPU] python train_dwt_tta_se.py [CORRUPTION_DATASET] --dwt-kernel-size  [DWT_KERNEL_SIZE] --dwt_level [DWT_LEVEL] --dwt_bn [SE BLOCK OPT] --model resnet26_dwt_se --lr 0.05 --epochs 1 --sched cosine -b 128 -j 5 --val-split val --input-size 3 224 224 --output TEST --num-classes 10 --experiment TEST --no-prefetcher --resume [CHECK_POINT] --ada 1 --lbatch [ITERATION FOR TTA]
```

+ Example of TTA 
```
CUDA_VISIBLE_DEVICES=0 python train_dwt_tta_se.py /home/datasets/test-C-10/gaussian_noise_5 --dwt-kernel-size  3 3 3 --dwt_level 2 2 2 --dwt_bn 1 321 0 --model resnet26_dwt_se --lr 0.05 --epochs 1 --sched cosine -b 128 -j 5 --val-split val --input-size 3 224 224 --output TEST --num-classes 10 --experiment TEST --no-prefetcher --resume weight/model_best.pth.tar --ada 1 --lbatch 75
```

## Acknowledgments
Our pytorch implementation is heavily derived from [TIMM](https://github.com/huggingface/pytorch-image-models), [Pytorch Wavelets](https://github.com/fbcotter/pytorch_wavelets).

Thanks to the TIMM, and Pytorch_Wavelets implementations.