# High-resolution networks (HRNets) for facial landmark detection
## Introduction
This is the official code of [High-Resolution Representations for Facial Landmark Detection](https://arxiv.org/pdf/1904.04514.pdf). We extend the high-resolution representation (HRNet) [1] by augmenting the high-resolution representation by aggregating the (upsampled) representations from all the parallel
convolutions, leading to stronger representations.

<!-- <div align=center>

![](images/hrnetv2p.png)

</div> -->

## Performance
### ImageNet pretrained models
HRNetV2 ImageNet pretrained models are now available! Codes and pretrained models are in [HRNets for Image Classification](https://github.com/HRNet/HRNet-Image-Classification)


### COFW

The model is trained on COFW *train* and evaluated on COFW *test*.

| Model | #Params | GFLOPs | NME | FR<sub>0.1</sub>|pretrained model|model|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|HRNetV2 | 9.3M | 4.3 | 3.45 | 0.20 | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [HR18-COFW.pth](https://1drv.ms/u/s!AiWjZ1LamlxzdFIsEUQl8jgUaMk)|


### AFLW
The model is trained on AFLW *train* and evaluated on AFLW *full* and *frontal*.

| Model | #Params | GFLOPs | NME(*full*) | NME(*frontal*) | pretrained model|model|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|HRNetV2 | 9.3M | 4.3 | 1.57 | 1.46 | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [HR18-AFLW.pth](https://1drv.ms/u/s!AiWjZ1Lamlxzc7xumEw810iBLTc)|

### WFLW

| Model | #Params | GFLOPs | NME(*test*) | NME(*pose*) | NME(*illumination*) | NME(*occlution*) | NME(*blur*) | NME(*makeup*) | NME(*expression*) | pretrained model|model|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|HRNetV2 | 9.3M | 4.3 | 4.60 | 7.86 | 4.57 | 5.42 | 5.36 | 4.26 | 4.78 | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [HR18-WFLW.pth](https://1drv.ms/u/s!AiWjZ1LamlxzdTsr_9QZCwJsn5U)|

### 300W

| Model | #Params | GFLOPs | NME(*common*) | NME(*challenge*) | NME(*full*) | NME(*test*) |  pretrained model|model|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|HRNetV2 | 9.3M | 4.3 | 1.57 | 1.46 | | | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [HR18-300W.pth]()|

## Quick start
#### Environment
This code is developed using on Python 3.6 and PyTorch 1.0.0 on Ubuntu 16.04 with NVIDIA GPUs. Training and testing are 
performed using 1 NVIDIA P40 GPU with CUDA 9.0 and cuDNN 7.0. Other platforms or GPUs are not fully tested.

#### Install
1. Install PyTorch 1.0 following the [official instructions](https://pytorch.org/)
2. Install dependencies
````bash

pip install -r requirements.txt
````
3. Clone the project
````bash 
git clone https://github.com/HRNet/HRNet-Facial-Landmark-Detection.git
````

#### HRNetV2 pretrained models
```bash
cd HRNet-Facial-Landmark-Detection
# Download pretrained models into this folder
mkdir hrnetv2_pretrained
```
#### Data



#### Train (multi-gpu training)
Please specify the configuration file in `experiments` (learning rate should be adjusted when the number of GPUs is changed).
````bash
python tools/train.py --cfg <CONFIG-FILE>
# example:
python tools/train.py --cfg experiments/wflw/face_alignment_wflw_hrnet_w18.yaml
````

#### Test
````bash
python tools/test.py --cfg <CONFIG-FILE> --model-file <MODEL WEIGHT> 
# example:
python tools/test.py --cfg experiments/wflw/face_alignment_wflw_hrnet_w18.yaml --model-file HR18-WFLW.pth
````

 
## Other applications of HRNets (codes and models):
* [Human pose estimation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
* [Semantic segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) (coming soon)
* [Object detection](https://github.com/HRNet/HRNet-Object-Detection)
* [Image classification](https://github.com/HRNet/HRNet-Image-Classification)
 
## Citation
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{SunZJCXLMWLW19,
  title={High-Resolution Representations for Labeling Pixels and Regions},
  author={Ke Sun and Yang Zhao and Borui Jiang and Tianheng Cheng and Bin Xiao 
  and Dong Liu and Yadong Mu and Xinggang Wang and Wenyu Liu and Jingdong Wang},
  journal   = {CoRR},
  volume    = {abs/1904.04514},
  year={2019}
}
````

## Reference
[1] Deep High-Resolution Representation Learning for Human Pose Estimation. Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. CVPR 2019. [download](https://arxiv.org/pdf/1902.09212.pdf)

