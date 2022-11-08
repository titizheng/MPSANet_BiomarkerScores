# Automatic CD30 scoring method for whole slide images of primary cutaneous CD30+lymphoproliferative diseases(2022.07)
## Paper 
This repository contains the code and data to reproduce the main results from the paper:
[Automatic CD30 scoring method for whole slide images of primary cutaneous CD30+lymphoproliferative diseases](https://jcp.bmj.com/content/early/2022/07/21/jcp-2022-208344.abstract)

If you have any quesions, please post it on github issues

 ## Prerequisites
* Python (3.6)
* Numpy (1.14.3)
* Scipy (1.0.1)
* PyTorch (1.8.1+cu111)/CUDA Version: 11.2
* torchvision 0.9.1+cu111)
* PIL (5.1.0)
* scikit-image (0.13.1)
* OpenSlide 1.1.2
* matplotlib (2.2.2)


## Date
The main data are the whole slide images (WSI) in *.ndpi.Data are available on reasonable request. The Yale 
HER2 cohort dataset was obtained from 
[SamanFarahmand’s project](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=119702524)

As the use of this dataset is subject to critical review by the hospital, if other researchers require 
this dataset, you can contact the corresponding author of this paper.

## Model
![MPAANet](/images/MPSANet.jpg)



## Citation 
If our paper helps your research, please cite it in your publications:
```
@article{zheng2022automatic,
  title={Automatic CD30 scoring method for whole slide images of primary cutaneous CD30+ lymphoproliferative diseases},
  author={Zheng, Tingting and Zheng, Song and Wang, Ke and Quan, Hao and Bai, Qun and Li, Shuqin and Qi, Ruiqun and Zhao, Yue and Cui, Xiaoyu and Gao, Xinghua},
  journal={Journal of Clinical Pathology},
  year={2022},
  publisher={BMJ Publishing Group}
}
```
