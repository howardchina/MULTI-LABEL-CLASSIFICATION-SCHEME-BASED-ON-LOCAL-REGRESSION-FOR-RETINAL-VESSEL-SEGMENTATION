# MULTI-LABEL CLASSIFICATION SCHEME BASED ON LOCAL REGRESSION FOR RETINAL VESSEL SEGMENTATION
ICIP2018 paper [link](https://ieeexplore.ieee.org/document/8451415 "ICIP2018: Multi-Label Classification Scheme Based on Local Regression for Retinal Vessel Segmentation")

### config
1. Set keras to channel_first mode. location: ./keras
2. using Tensorflow as backend
3. install keras-gpu, python2.7 by conda
4. more information about installation of keras-gpu, go to official tutorial.


### Ablation experiment

| Methods            | Accuracy |Sensitivity | Specificity | F1 score | ROC |
|------------------|-----------|---------|--------------------|----------------|---------------------|
| Binary classification (raw CNN) | 0.9552 | 0.8224 | 0.9745 | 0.8239　| 0.9782 |
| + Local deregression (step 1)  | 0.9546 | 0.7407 | 0.9758 | 0.8059 | - |
| + Non-linear enhancement (step 2)| 0.9504 | 0.8221 | 0.9691 | 0.8085 | - |
| + Local regression (step 3) | 0.9536 | 0.7698 | 0.9804 | 0.8086 | - |

### Performance of different methods on DRIVE.

| Methods | Year | Accuracy |Sensitivity | Specificity | F1 score | ROC |
|------------------|-----------|---------|--------------------|----------------|---------------------|
| DRIU[18] | 2016 | 0.9528 | 0.8330 | 0.9714 | 0.8262　| 0.9796 |
| CNN[20] | 2016 | 0.9517 | 0.8295 | 0.9707 | 0.8221 | 0.9763 |
| HED[16] | 2015 | 0.9462 | 0.8009 | 0.9688 | 0.8002 | 0.9699 |
| Proposed | 2018 | 0.9519 | 0.7761 | 0.9792| 0.8129 | - |

### Performance of different methods on STARE.

| Methods | Year | Accuracy |Sensitivity | Specificity | F1 score | ROC |
|------------------|-----------|---------|--------------------|----------------|---------------------|
| DRIU[18] | 2016 | 0.9669 | 0.8641 | 0.9792 | 0.8489　| 0.9902 |
| CNN[20] | 2016 | 0.9738 | 0.8833 | 0.9747 | 0.8791 | 0.9927 |
| HED[16] | 2015 | 0.9603 | 0.8398 | 0.9749 | 0.8201 | 0.9860 |
| Proposed | 2018 | 0.9704 | 0.8120 | 0.9895| 0.8553 | - |

## About

If you used the code for your research, please, cite the paper:

    @inproceedings{he2018multi, 
      title={Multi-Label Classification Scheme Based on Local Regression for Retinal Vessel Segmentation},
      author={He, Qi and Zou, Beiji and Zhu, Chengzhang and Liu, Xiyao and Fu, Hongpu and Wang, Lei},
      booktitle={2018 25th IEEE International Conference on Image Processing (ICIP)},
      pages={2765--2769},
      year={2018},
      organization={IEEE}
    }