# Counterfactual Debiasing Inference for Compositional Action Recognition

<!-- This codebase is created by Pengzhan Sun, Xunsong Li at UESTC (University of Electronic Science and Technology of China). -->

This code base is the pytorch implementation of Counterfactual Debiasing Inference for Compositional Action Recognition **(ACM MM'21)**. 


- [Introduction](#introduction)
- [Task Setting](#task-setting)
- [Method](#method)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Checkpoints](#checkpoints)
- [Getting Started](#getting-started)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)


## Introduction

<div align=center><img width = '400' src ="https://github.com/pengzhansun/CF-CAR/blob/main/demo_images/setting_car.png"/></div>
The Counterfactual Debiasing Network (CDN) is designed to remove the bad object appearance bias while keep the good object appearance cue for action recognition. Recent action recognition models may tend to rely on object appearance as a shortcut and thus fail to sufficiently learn the real action knowledge. On the one hand, object appearance is the bias which cheats the model to make the wrong prediction because of different action classes it co-appears between the training stage and test stage. On the other hand, the object appearance is a meaningful cue which can help the model to learn the knowledge of action.

## Task Setting
There are two disjoint action sets _\{1, 2\}_ and two disjoint object sets _\{A, B\}_. For the compositional action recognition task, the training set of the model is _\{1A + 2B\}_, and the verification set is _\{1B + 2A\}_. Under this challenging setting, the model needs to be able to recognize new combinations of actions and objects. In this problem setting, there are 174 action categories with 54,919 training and 57,876 validation instances. More details can be found in Something-Else.

## Method
We empower models the ability of counterfactual analysis so that a more accurate classification result can be gained by comparing factual inference outcome and counterfactual inference outcome.

<div align=center><img width = '600' src ="https://github.com/pengzhansun/CF-CAR/blob/main/demo_images/idea.png"/></div>

- We observe that prior knowledge learned from appearance information is mixed with the spurious correlation between action and instance appearance, which badly inhibits the model’s ability of action learning.
<div align=center><img width = '400' src ="https://github.com/pengzhansun/CF-CAR/blob/main/demo_images/contribution1.png"/></div>

- We remove the pure appearance effect from total effect by counterfactual debiasing inference on our novel framework CDN proposed for compositional action recognition.

- We achieve state-of-the-art performance for compositional action recognition on the Something-Else dataset.

<!-- | Method | Acc-1 | Acc-5 |
|:--------:|:--------:|:--------:|
| I3D | 50.5 | 76.9 |
| STIN[1] | 51.4 | 79.3 |
| STIN + I3D[1] | 54.6 | 79.4 |
| Interactive Fusion[2] | 59.6 | 85.8 |
| SAFCAR[3] | 60.5 | 84.3 |
| Our CDN w/o CF | **62.8** | **87.3** |
| Our CDN | **64.5** | **88.2** |

[1]: Something-Else: Compositional Action Recognition with Spatial-Temporal Interaction Networks<br>
[2]: Interactive Fusion of Multi-level Features for Compositional Activity Recognition<br>
[3]: SAFCAR: Structured Attention Fusion for Compositional Action Recognition<br> -->

## Requirements
```
pip install -r requirements.txt
```

## Dataset
Download Something-Something Dataset and Something-Else Annotation from [Something-Else repo](https://github.com/joaanna/something_else) (Joaana et al., 2020). Note that we also provide the [annotation per video](https://drive.google.com/file/d/1tNYKIT3bSXyZq-q5-sKHzEibSrzkQF_6/view?usp=sharing) for users with limited computing resources by spliting Something-Else Annotation mentioned above.

## Checkpoints
Download [our models](https://drive.google.com/drive/folders/1nXqJYcXqMQBxgi5y0gvQ2A5DsUou_G2g?usp=sharing) reported on the paper. 

## Getting Started
To train, test or conduct counterfactual debiasing inference, please run these [scripts](https://github.com/pengzhansun/CF-CAR/tree/main/scripts).

## Citation
If you use this code repository in your research, please cite this project.

```
@inproceedings{counterfactual2021,
  title={Counterfactual Debiasing Inference for Compositional Action Recognition},
  author={Sun, Pengzhan and Wu, Bo and Li, Xunsong and Li, Wen and Duan, Lixin and Gan, Chuang},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  year={2021}
}
```


## Acknowledgments
This repository is implemented as a fork of [Something-Else](https://github.com/joaanna/something_else). We used parts of code from following repositories:

https://github.com/joaanna/something_else

https://github.com/ruiyan1995/Interactive_Fusion_for_CAR

Contact: pengzhansun6@gmail.com
