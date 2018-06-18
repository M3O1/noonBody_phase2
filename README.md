## Human Body Segmentation

### Objective

> 영상 내에서 사람 신체의 Outline을 따는 것을 목표로 함

### Requirements

* Photo-Realistic하도록, 사람 형상에 가깝게 Segmentation
* 사람이 여러 명이 있었을 경우 구분지어서 Segmentation할 수 있도록 함 ( Instance Segmentation)
* 신체 일부분만 있더라도 Segmentation
* 신체 일부분이 가려져 있더라도, 예측하여 Segmentation
* (Optional) 신체 부위(Landmark) Detection
* 위 모든 기능이 Mobile 환경에서도 충분히 돌아가도록 함 (But Not Real-time 60FPS에 맞출 필요는 없음)

### Data-Set

* The Dataset of Baidu People Segmentation Competition

  Labeling된 데이터 수 : 5387장 ( 이 중 영상 내 사람 한명만 있는 경우: 3909장)

### Reference

**논문**

* Image-to-Image Translation with Conditional Adversarial Networks, Phillip Isola, 2016

* Precomputed real-time texture synthesis with markovian generative adversarial networks, C. Li and M. Wand. , 2016

  

**블로그**

* [(pix2pix) image-to-image Translation with Conditional Adversarial Networks](https://kakalabblog.wordpress.com/2017/08/10/pix2pix-image-to-image-translation-with-conditional-adversarial-networks/)

* [GAN을 이용한 image to image translation: pix2pix, CycleGAN, discoGAN](https://taeoh-kim.github.io/blog/gan%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-image-to-image-translation-pix2pix-cyclegan-discogan/)

  

**GITHUB**

* [keras implementation of px2pix](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix)

### To-Do

- [x] '18.06.18 : Data 정제 및 Data Generator 구현
- [ ] 



-----

### Explanation

1. **Photo-Realistic하도록, 사람 형상에 가깝게 Segmentation**

   이 문제를 해결하기 위해서는, Photo-Realistic의 Loss를 Generator에 주입해야 하므로, GAN을 이용하기로 함. 유사한 형태로서는 Pix2Pix가 있어서, 이를 이용함 PIX2PIX를 구현하기 위해서는 

   * U-NET
   * DCGAN
   * PATCHGAN

   개념을 바탕으로 한다. 