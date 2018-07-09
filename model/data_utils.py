import h5py
import random
import numpy as np
import cv2

from skimage.transform import rescale, rotate
from skimage.util import random_noise
from skimage.exposure import adjust_gamma

__all__ = ['HumanSegGenerator','load_dataset']

def load_dataset(dataset_name='train',
                 h5_path="../data/baidu-segmentation.h5"):
    with h5py.File(h5_path) as file:
        return file[dataset_name][:]

def HumanSegGenerator(dataset,
                      img_dim,
                      batch_size=64,
                      sigmoid=True,
                      bg_removal=False,
                      crop=False,
                      crop_pad=3,
                      crop_keep_shape=False,
                      aug_funcs=[],
                      prep_funcs=[]):
    '''
    dataset : total dataset of human segmentation
        data는 (384,384,4)로 구성되어 잇는데,
        image, profile = data[384,384,:3], data[384,384,-1]

        with h5py.File("../data/baidu-segmentation.h5") as file:
            dataset = file['384x384'][:]

    img_dim : 모델에 feeding하기 위한 input size
    batch_size : batch size
    sigmoid : 값 범위를 (0,1) or (-1,1)
    aug_funcs : list of data augmentation func
    prep_func : list of preprocessing func
    '''
    if batch_size is None:
        # batch_size = None -> Full batch
        batch_size = dataset.shape[0]

    counter = 0; batch_image = []; batch_profile = []
    while True:
        np.random.shuffle(dataset)
        for data in dataset:
            counter += 1
            # data Normalization
            data = cv2.normalize(data,np.zeros_like(data),
                                 alpha=0.,beta=1.,
                                 norm_type=cv2.NORM_MINMAX,
                                 dtype=cv2.CV_32F)

            # apply image augumentation
            for aug_func in aug_funcs:
                data = aug_func(data)

            if crop:
                data = crop_data(data,pad=crop_pad,keep_shape=crop_keep_shape)

            data = data.astype(np.float32) # adjust data type for opencv resize method
            data = cv2.resize(data,img_dim[:2])

            # apply image preprocessing
            for prep_func in prep_funcs:
                data = prep_func(data)

            # dataset을 image와 profile로 나눔
            image, profile = data[:,:,:-1], data[:,:,-1]
            # adjust the range of value
            image = np.clip(image,0.,1.)

            profile = (profile>0.7).astype(np.uint8)
            if bg_removal:
                profile = cv2.bitwise_and(image, image, mask=profile)
            else:
                profile = np.expand_dims(profile,axis=-1)

            if not sigmoid:
                # normalize from (0,1) to (-1,1)
                image, profile = to_tanh(image), to_tanh(profile)

            batch_image.append(image); batch_profile.append(profile)
            if counter == batch_size:
                yield np.stack(batch_image, axis=0), np.stack(batch_profile, axis=0)
                counter = 0; batch_image = []; batch_profile = []

def HumanDetectGenerator(dataset,
                      img_dim,
                      batch_size=64,
                      sigmoid=True,
                      aug_funcs=[],
                      prep_funcs=[]):
    '''
    dataset : total dataset of human segmentation
        data는 (384,384,4)로 구성되어 잇는데,
        image, profile = data[384,384,:3], data[384,384,-1]

        with h5py.File("../data/baidu-segmentation.h5") as file:
            dataset = file['384x384'][:]

    img_dim : 모델에 feeding하기 위한 input size
    batch_size : batch size
    sigmoid : 값 범위를 (0,1) or (-1,1)
    aug_funcs : list of data augmentation func
    prep_func : list of preprocessing func
    '''
    if batch_size is None:
        # batch_size = None -> Full batch
        batch_size = dataset.shape[0]

    height, width = img_dim[:2]

    counter = 0; batch_image = []; batch_pos = []
    while True:
        np.random.shuffle(dataset)
        for data in dataset:
            counter += 1
            # data Normalization
            data = cv2.normalize(data,np.zeros_like(data),
                                 alpha=0.,beta=1.,
                                 norm_type=cv2.NORM_MINMAX,
                                 dtype=cv2.CV_32F)

            # apply image augumentation
            for aug_func in aug_funcs:
                data = aug_func(data)

            data = data.astype(np.float32) # adjust data type for opencv resize method
            data = cv2.resize(data,img_dim[:2])

            # apply image preprocessing
            for prep_func in prep_funcs:
                data = prep_func(data)

            # dataset을 image와 profile로 나눔
            image, profile = data[:,:,:-1], data[:,:,-1]
            # adjust the range of value
            image = np.clip(image,0.,1.)

            profile = (profile>0.7).astype(np.uint8)

            y , x  = np.argwhere(profile==1).min(axis=0)
            y2, x2 = np.argwhere(profile==1).max(axis=0)
            h, w = y2-y, x2-x
            pos = np.array([x/width, y/height, w/width, h/height])
            if not sigmoid:
                # normalize from (0,1) to (-1,1)
                image = to_tanh(image)

            batch_image.append(image); batch_pos.append(pos)
            if counter == batch_size:
                yield np.stack(batch_image, axis=0), np.stack(batch_pos, axis=0)
                counter = 0; batch_image = []; batch_pos = []

# image preprocessing pipeline
def clahe_func(clipLimit=2.0,tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    def func(data):
        # clahe는 uint8에서만 연산 가능하므로
        # uint8로 바꾸어줘야 함
        image = cv2.normalize(data[:,:,:3],np.zeros_like(data[:,:,:3]),
                              alpha=0,beta=255,
                              norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_8U)
        # apply clahe
        image = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
        image[:,:,0] = clahe.apply(image[:,:,0])
        image = cv2.cvtColor(image,cv2.COLOR_LAB2RGB)

        # normalize (0,1)
        data[:,:,:3] = cv2.normalize(image,np.zeros_like(image),
                                     alpha=0.,beta=1.,
                                     norm_type=cv2.NORM_MINMAX,
                                     dtype=cv2.CV_32F)
        return data
    return func

def gray_func():
    def func(data):
        image = data[:,:,:3]
        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        return np.stack((gray,data[:,:,3]),axis=-1)
    return func

def homomorphic_func(sigma=10,gamma1=0.5,gamma2=1.5):
    def func(data):
        img = data[:,:,:3]


        img_YUV = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        y = img_YUV[:,:,0]

        rows, cols = y.shape[:2]
        M, N = 2*rows + 1, 2*cols + 1
        ### illumination elements와 reflectance elements를 분리하기 위해 log를 취함
        imgLog = np.log1p(y) # y값을 0~1사이로 조정한 뒤 log(x+1)

        X, Y = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M)) # 0~N-1(and M-1) 까지 1단위로 space를 만듬
        Xc, Yc = np.ceil(N/2), np.ceil(M/2) # 올림 연산
        gaussianNumerator = (X - Xc)**2 + (Y - Yc)**2 # 가우시안 분자 생성

        ### low pass filter와 high pass filter 생성
        LPF = np.exp(-gaussianNumerator / (2*sigma*sigma))
        HPF = 1 - LPF

        ### LPF랑 HPF를 0이 가운데로 오도록iFFT함.
        ### 사실 이 부분이 잘 이해가 안 가는데 plt로 이미지를 띄워보니 shuffling을 수행한 효과가 났음
        ### 에너지를 각 귀퉁이로 모아 줌
        LPF_shift = np.fft.ifftshift(LPF.copy())
        HPF_shift = np.fft.ifftshift(HPF.copy())

        ### Log를 씌운 이미지를 FFT해서 LPF와 HPF를 곱해 LF성분과 HF성분을 나눔
        img_FFT = np.fft.fft2(imgLog.copy(), (M, N))
        img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N))) # low frequency 성분
        img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N))) # high frequency 성분

        ### 각 LF, HF 성분에 scaling factor를 곱해주어 조명값과 반사값을 조절함
        img_adjusting = gamma1*img_LF[0:rows, 0:cols] + gamma2*img_HF[0:rows, 0:cols]

        ### 조정된 데이터를 이제 exp 연산을 통해 이미지로 만들어줌
        img_exp = np.expm1(img_adjusting) # exp(x) + 1
        img_out = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp)) # 0~1사이로 정규화

        ### 마지막으로 YUV에서 Y space를 filtering된 이미지로 교체해주고 RGB space로 converting
        img_YUV[:,:,0] = img_out
        data[:,:,:3] = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2RGB)
        return data
    return func

# data augumentation pipeline
def rotation_func(min_angle=-8,max_angle=8):
    def func(data):
        rotation_angle = random.randint(min_angle,max_angle)
        return rotate(data, rotation_angle, mode='constant', cval=0.)
    return func

def rescaling_func(scale=0.1):
    def func(data):
        rescale_ratio = (1-scale) + random.random() * scale * 2
        return rescale(data, rescale_ratio,mode='reflect')
    return func

def flip_func():
    def func(data):
        if random.random()>0.5:
            return data[:,::-1,:]
        else:
            return data
    return func

def random_crop_func(crop_dim=(256,256)):
    def func(data):
        iv = random.randint(0,data.shape[0]-crop_dim[0])
        ih = random.randint(0,data.shape[1]-crop_dim[1])
        return data[iv:iv+crop_dim[0],ih:ih+crop_dim[1],:]
    return func

def random_noise_func(var=0.001):
    def func(data):
        data[:,:,:3] = random_noise(data[:,:,:3],
                                    mode='gaussian',
                                    mean=0,var=var)
        return data
    return func

def gamma_func(min_gamma=0.8,max_gamma=1.4):
    def func(data):
        gamma = np.random.uniform(min_gamma,max_gamma)
        data[:,:,:3] = adjust_gamma(data[:,:,:3],gamma)
        return data
    return func

def color_gamma_func(min_gamma=0.6,max_gamma=1.4):
    def func(data):
        for i in range(3):
            gamma = np.random.uniform(min_gamma,max_gamma)
            data[:,:,i] = adjust_gamma(data[:,:,i], gamma)
        return data
    return func

def margin_to_square(image):
    h, w = image.shape[:2]
    if h - w > 0:
        diff = h - w
        margin1, margin2 = diff // 2 , diff - diff//2
        margined = np.pad(image,((0,0),(margin1,margin2),(0,0)),'constant',constant_values=0)
    elif h - w < 0:
        diff = w - h
        margin1, margin2 = diff // 2 , diff - diff//2
        margined = np.pad(image,((margin1,margin2),(0,0),(0,0)),'constant',constant_values=0)
    else:
        margined = image
    return margined

def crop_data(data,pad=3,keep_shape=False):
    mask = data[:,:,3] > 0.2
    h,w = mask.shape

    top, left  = np.argwhere(mask==True).min(axis=0)
    bot, right = np.argwhere(mask==True).max(axis=0)

    top_pad = top - pad if top - pad > 0 else 0
    bot_pad = bot + pad if bot + pad < h else h
    left_pad = left - pad if left - pad > 0 else 0
    right_pad = right + pad if right + pad < w else w

    cropped = data[top_pad:bot_pad,left_pad:right_pad]
    if keep_shape:
        return margin_to_square(cropped)
    else:
        return cropped

# normalization
def to_tanh(X):
    # value range => (-1,1)
    X = X.astype(np.float32)
    return cv2.normalize(X,np.zeros_like(X),
            -1.,1.,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)

def to_sigmoid(X):
    # value range => (0,1)
    X = X.astype(np.float32)
    return cv2.normalize(X,np.zeros_like(X),
        0.,1.,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)

def to_uint8(X):
    X = X.astype(np.float32)
    return cv2.normalize(X,np.zeros_like(X),0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
