## Introduction

This repo contains the official PyTorch implementation of D\&R

## Quick Start

**1. Check Requirements**
* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.6 & [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch version.
* CUDA 10.1, 10.2
* GCC >= 4.9

**2. Build**
  
* Create a virtual environment (optional)
  ```
  conda create -n dandr python=3.7
  conda activate dandrzq
  ```
* Install PyTorch according to your CUDA version 
  
* Install Detectron2 (the version of Detectron2 must  be 0.3)
  ```angular2html
  python3 -m pip install detectron2==0.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
  ```
* Install other requirements. 
  ```angular2html
  python3 -m pip install -r requirements.txt
  ```

**3. Prepare Data and Weights**
* Data Preparation (from DeFRCN)

    | Dataset | Size | GoogleDrive | BaiduYun | Note |
    |:---:|:---:|:---:|:---:|:---:|
    |VOC2007| 0.8G |[download](https://drive.google.com/file/d/1BcuJ9j9Mtymp56qGSOfYxlXN4uEVyxFm/view?usp=sharing)|[download](https://pan.baidu.com/s/1kjAmHY5JKDoG0L65T3dK9g)| - |
    |VOC2012| 3.5G |[download](https://drive.google.com/file/d/1NjztPltqm-Z-pG94a6PiPVP4BgD8Sz1H/view?usp=sharing)|[download](https://pan.baidu.com/s/1DUJT85AG_fqP9NRPhnwU2Q)| - |
    |vocsplit| <1M |[download](https://drive.google.com/file/d/1BpDDqJ0p-fQAFN_pthn2gqiK5nWGJ-1a/view?usp=sharing)|[download](https://pan.baidu.com/s/1518_egXZoJNhqH4KRDQvfw)| refer from [TFA](https://github.com/ucbdrive/few-shot-object-detection#models) |
    |COCO| ~19G | - | - | download from [offical](https://cocodataset.org/#download)|
    |cocosplit| 174M |[download](https://drive.google.com/file/d/1T_cYLxNqYlbnFNJt8IVvT7ZkWb5c0esj/view?usp=sharing)|[download](https://pan.baidu.com/s/1NELvshrbkpRS8BiuBIr5gA)| refer from [TFA](https://github.com/ucbdrive/few-shot-object-detection#models) |
  - Unzip the downloaded data-source to `datasets` and put it into your project directory:
    ```angular2html
      ...
      datasets
        | -- coco (trainval2014/*.jpg, val2014/*.jpg, annotations/*.json)
        | -- cocosplit
        | -- VOC2007
        | -- VOC2012
        | -- vocsplit
      defrcn
      tools
      ...
    ```
* Weights Preparation
  - DeFRCN use the imagenet pretrain weights to initialize the model. 
  Download the same models from (given by DeFRCN): [GoogleDrive](https://drive.google.com/file/d/1rsE20_fSkYeIhFaNU04rBfEDkMENLibj/view?usp=sharing) [BaiduYun](https://pan.baidu.com/s/1IfxFq15LVUI3iIMGFT8slw)
  - Put the chekpoints into ImageNetPretrained/MSRA/R-101.pkl, ImageNetPretrained/torchvision, respectively
  - We provide the BASE_WEIGHT (refer to run_*.sh) we used.
  | Dataset | Split | Size | GoogleDrive |
    |:---:|:---:|:---:|:---:|
    |VOC2007| 1 | 203.8M | [download](https://drive.google.com/file/d/19LxiN9cj92YePs02k9E4-KyGY5ohTU9w/view?usp=share_link)|
    |VOC2007| 2 | 203.8M | [download](https://drive.google.com/file/d/1t1bbJ-YsXohIDUsQvUiF8pC6f7vxh0Z3/view?usp=share_link)|
    |VOC2007| 3 | 203.8M | [download](https://drive.google.com/file/d/1bWiS0fBrQDljTnpBFFldbZ8ZmZUmhB8w/view?usp=share_link)|
    | COCO | - | 206.2MB | [download](https://drive.google.com/file/d/1pH-7b_1B3qm_rJo-_nEfcrHmy-PCHHZ7/view?usp=share_link) |

* Text Embeddings Preparation
  - Refer to the official implementation of [CLIP](https://github.com/openai/CLIP) for text embedding generation.
  - Put the generated text embeddings into 'dataset/clip'

**4. Training and Evaluation**

* To reproduce the results on VOC, 
  ```angular2html
  sh run_voc.sh SPLIT_ID (1, 2 or 3)
  ```
* To reproduce the results on COCO
  ```angular2html
  sh run_coco.sh
  ```
* Please read the details of few-shot object detection pipeline in `run_*.sh`.

## Acknowledgement
This repo is developed based on DeFRCN and [Detectron2](https://github.com/facebookresearch/detectron2). Please check them for more details and features.
```
