import h5py
import random
import numpy as np
import cv2

from skimage.transform import rescale, rotate
from skimage.util import random_noise

__all__ = ['HumanSegGenerator']

def HumanSegGenerator(dataset, input_size, batch_size=64,is_train=True):
    '''
    dataset : total dataset of human segmentation
        data는 (384,384,4)로 구성되어 잇는데,
        image, profile = data[384,384,:3], data[384,384,-1]

        with h5py.File("../data/baidu_segmentation.h5") as file:
            dataset = file['384x384'][:]
    input_size : 모델에 feeding하기 위한 input size

    is_train : train일 경우, data Augumentation, 아닌 경우
    '''
    if batch_size is None:
        # batch_size = None -> Full batch
        batch_size = dataset.shape[0]

    counter = 0; batch_image = []; batch_profile = []
    while True:
        np.random.shuffle(dataset)
        for data in dataset:
            counter += 1
            if is_train:
                #train일 경우, data Augumentation PipeLine 거침
                data = apply_rotation(data)
                data = apply_rescaling(data)
                data = apply_flip(data)
                data = apply_random_crop(data, input_size)
                # give instance Noise for training
                data[:,:,:3] = random_noise(data[:,:,:3],
                                            mode='gaussian',
                                            mean=0,var=0.001)
            else:
                #test일 경우, data augmentation 하지 않음
                data = data / 255. # normalize data
                data = cv2.resize(data, input_size)

            # dataset을 image와 profile로 나눔
            image, profile = data[:,:,:3], data[:,:,-1]
            # adjust the range of value
            image = np.clip(image,0.,1.)
            profile = (profile>0.7).astype(int)

            batch_image.append(image); batch_profile.append(profile)
            if counter == batch_size:
                yield np.stack(batch_image, axis=0), np.stack(batch_profile, axis=0)
                counter = 0; batch_image = []; batch_profile = []

# data augumentation pipeline
def apply_rotation(image):
    rotation_angle = random.randint(-8,8)
    return rotate(image, rotation_angle)

def apply_rescaling(image):
    rescale_ratio = 0.8 + random.random() * 0.4 # from 0.8 to 1.2
    return rescale(image, rescale_ratio,mode='reflect')

def apply_flip(image):
    if random.random()>0.5:
        return image[:,::-1,:]
    else:
        return image

def apply_random_crop(image, input_size):
    iv = random.randint(0,image.shape[0]-input_size[0])
    ih = random.randint(0,image.shape[1]-input_size[1])
    return image[iv:iv+input_size[0],ih:ih+input_size[1],:]
