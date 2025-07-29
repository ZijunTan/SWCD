# SWCD: Towards Accurate Change Detection via Similarity-Awareness Weakly Supervised Learning
## Abtract
Change detection (CD) is one of the prominent research topics in the fields of Earth science and remote sensing. Recently, an increasing number of deep learning-based CD methods have been developed. Most of the current CD methods require lots of pixel-level labels for supervised learning. However, annotating all the changed pixels in bitemporal images is both challenging and time-consuming. In this work, as a first attempt in the field of CD, we propose a novel CD framework, similarity-awareness weakly supervised change detection (SWCD) to achieve accurate CD, which uses weakly supervised learning as an auxiliary task to guide the model in both semi-supervised and supervised learning. In the weakly supervised branch, we incorporate the concept of similarity and introduce similarity information into the supervised branch to guide pixel-level CD learning, thus enhancing feature continuity. Moreover, large kernel convolution attention is introduced to enhance multi-scale feature learning. In the supervised branch, we reevaluate the approach to multi-scale feature aggregation and introduce an adaptive feature module to integrate features from both global and local perspectives. Furthermore, our method can serve as a general framework that is compatible with existing CD approaches. Experimental results on four CD datasets demonstrate the superior effectiveness and generalization of our proposed method.
## Get Strat
### Data Preparation
Download three dataset LEVIR-CD, BCDD, SYSU-CD.
Use generate_weak_label.py to convert the labels to weaks and then you need to divide the data from the dataset into which data has full supervision labels and generate a txt file.
Finally, you can prepare datasets into following structure.
```
├─Train
    ├─A
    ├─B
    ├─label
    ├─label_weak
    └─list
      ├─train.txt
      ├─train_semi_10.txt
      ├─train_semi_30.txt
      ├─train_semi_50.txt
├─Val
    ├─A
    ├─B
    ├─label
    ├─label_weak
    └─list
    ├─val.txt
├─Test
    ├─A
    ├─B
    ├─label
    ├─label_weak
    └─list
    ├─test.txt
```
## Acknowledgement
This repository is built under the help of the projects [CLAFA](https://github.com/xingronaldo/CLAFA) for academic use only.
