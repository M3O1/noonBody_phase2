## Human Body Segmentation

### Objective

> 영상 내에서 사람 신체의 Outline을 따는 것을 목표로 함

### Requirements

* Photo-Realistic하도록, 사람 형상에 가깝게 Segmentation
* 신체 일부분만 있더라도 Segmentation
* 사람이 여러 명이 있었을 경우 구분지어서 Segmentation할 수 있도록 함 ( Instance Segmentation)
* 신체 일부분이 가려져 있더라도, 예측하여 Segmentation
* (Optional) 신체 부위(Landmark) Detection
* 위 모든 기능이 Mobile 환경에서도 충분히 돌아가도록 함 (But Not Real-time 60FPS에 맞출 필요는 없음)

### Data-Set

* The Dataset of Baidu People Segmentation Competition

  Labeling된 데이터 수 : 5387장 ( 이 중 영상 내 사람 한명만 있는 경우: 3909장)

* 크롤링한 데이터셋 2946장

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



-----

### Explanation

1. **Photo-Realistic하도록, 사람 형상에 가깝게 Segmentation**

   이 문제를 해결하기 위해서는, Photo-Realistic의 Loss를 Generator에 주입해야 하므로, GAN을 이용하기로 함. 유사한 형태로서는 Pix2Pix가 있어서, 이를 이용함 PIX2PIX를 구현하기 위해서는 

   * U-NET
   * DCGAN
   * PATCHGAN

   개념을 바탕으로 한다. 

2. **신체 일부분만 있더라도 Segmentation**

   데이터 샘플에 신체 일부분만 찍은 경우들을 섞어서 넣었다. 그리고 Data-Augumentation 작업 중에도, 이미지에서 일부 부분만 뽑아져 나오도록, Random Crop 및 Resize를 하였기 때문에, 일부 부분만 있더라도 유의미하게 동작하도록 함.

---------

### 깨달은 점

1. 이미지 Segmentation 할 때에는 Batch Size가 작을 수록 학습을 잘한다.

   배치 사이즈가 4일때(위) , 배치 사이즈가 64일 때 (아래)

   ![batch_size_4](/home/ksj/projects/segment_body/src/batchsize_4_training.png)

   ![batch_size_64](/home/ksj/projects/segment_body/src/batchsize_64_training.png)

   Classification을 할 때에는, 영상 내에서 보편적인 특성을 위주로 학습을 해야 하지만, Segmentation을 할 때에는 각 영상마다의 고유한 특성들을 학습해야 함. Batch가 커지면 오히려 세부 정보들을 죽이는 효과가 나오기 때문에, 오히려 학습이 떨어진다고 파악하고 있음.

   배치 노말의 경우, 배치 사이즈가 1인 경우에는 배치 노말이 있으면 아래와 같이 끔직한 형태로 학습이 진행된다. 그렇기 때문에, 배치 사이즈가 1인 경우에는 배치 노말을 제거해주어야(위 케이스) 정상적으로 학습된다.

   ![batch_size_1](/home/ksj/projects/segment_body/src/batchsize_1_no_bn.png)         

   ![batch_size_1](/home/ksj/projects/segment_body/src/batchsize_1_training.png)

   2. 모바일 환경에서 돌아가기 위해서는, 모델은 단순해야 한다. 

      모델의 연산량을 좌지우지하는 것은 얼마나 중간 결과물들을 많이 만들어 내냐에 달려 있다. U-NET에 Dense-Block을 합쳐 놓은 Tiramisu Network는 U-Net에 비해 정확도가 높을 지라도, 매우 큰 RAM을 필요로 한다. 

      U-NET은 (nb_filter = 16, depth = 4)를 기준으로 114MB밖에 차지 안하지만, Tiramisu는 논문에 따라 구현하면, 대략 2.07GB정도 차지하게 된다. Feed-Forwarding을 할때에는 당연히 이만큼의 Memory가 필요없겟지만, 연산량도 램에 비례에서 늘어나기 때문에, 대략 20배가량 증가한다고 보면 된다. 조금의 성능향상을 위해, 너무 무거워질 우려가 있기 때문에 Tiramisu Network는 배제하는 것이 옳다고 본다.

      ````python
      # 케라스 모델의 메모리 사용량을 측정하기 위한 코드
      def get_model_memory_usage(batch_size, model):
          import numpy as np
          from keras import backend as K
      
          shapes_mem_count = 0
          for l in model.layers:
              single_layer_mem = 1
              for s in l.output_shape:
                  if s is None:
                      continue
                  single_layer_mem *= s
              shapes_mem_count += single_layer_mem
      
          trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
          non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
      
          total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
          gbytes = np.round(total_memory / (1024.0 ** 3), 3)
          return gbytes
      ````

   3. 