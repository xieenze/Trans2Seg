# Transparent Transformer Segmentation
## Introduction
This repository contains the data and code for "Trans2Seg: Transparent Object Segmentation with Transformer 	".


## Environments

- python 3
- torch = 1.4.0
- torchvision
- pyyaml
- Pillow
- numpy

## INSTALL

```
python setup.py develop --user
```

## Data Preparation
1. create dirs './datasets/transparent/Trans10K_v2' 
2. put the train/validation/test data under './datasets/transparent/Trans10K_v2'. 
Data Structure is shown below.
```
Trans10K_v2
├── test
│   ├── images
│   └── masks_12
├── train
│   ├── images
│   └── masks_12
└── validation
    ├── images
    └── masks_12
```
Dataset will be released soon.

## Network Define
The code of Network pipeline is in `segmentron/models/trans2seg.py`.

The code of Transformer Encoder-Decoder is in `segmentron/modules/transformer.py`.

## Train
Our experiments are based on one machine with 8 V100 GPUs with 32g memory, about 1 hour training time.

```
bash tools/dist_train.sh $CONFIG-FILE $GPUS
```

For example:
```
bash tools/dist_train.sh configs/trans10kv2/trans2seg/trans2seg_medium.yaml 8
```

## Test
```
bash tools/dist_train.sh $CONFIG-FILE $GPUS --test TEST.TEST_MODEL_PATH $MODEL_PATH
```


## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.

```
@article{xie2021segmenting,
  title={Segmenting transparent object in the wild with transformer},
  author={Xie, Enze and Wang, Wenjia and Wang, Wenhai and Sun, Peize and Xu, Hang and Liang, Ding and Luo, Ping},
  journal={arXiv preprint arXiv:2101.08461},
  year={2021}
}
```
