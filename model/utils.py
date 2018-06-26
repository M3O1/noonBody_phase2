import h5py
import random
import numpy as np
import cv2

from skimage.transform import rescale, rotate
from skimage.util import random_noise

__all__ = ['HumanSegGenerator','load_dataset']

def load_dataset(dataset_name='train',h5_path="../data/baidu-segmentation.h5"):
    with h5py.File(h5_path) as file:
        return file[dataset_name][:]

def HumanSegGenerator(dataset,
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

            data = data.astype(np.float32) # adjust data type for opencv resize method
            data = cv2.resize(data,img_dim[:2])

            # apply image preprocessing
            for prep_func in prep_funcs:
                data = prep_func(data)

            # dataset을 image와 profile로 나눔
            image, profile = data[:,:,:-1], data[:,:,-1]
            # adjust the range of value
            image = np.clip(image,0.,1.)

            profile = (profile>0.7).astype(int)
            profile = np.expand_dims(profile,axis=-1)

            if not sigmoid:
                # normalize from (0,1) to (-1,1)
                image, profile = to_tanh(image), to_tanh(profile)

            batch_image.append(image); batch_profile.append(profile)
            if counter == batch_size:
                yield np.stack(batch_image, axis=0), np.stack(batch_profile, axis=0)
                counter = 0; batch_image = []; batch_profile = []

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
    return cv2.normalize(X,np.zeros_like(X),
              0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)

# Process batch Image
def extract_patches(X, patch_size):
    row, col = patch_size
    list_row_idx = [(i * row, (i + 1) * row) for i in range(X.shape[1] // row)]
    list_col_idx = [(i * col, (i + 1) * col) for i in range(X.shape[2] // col)]

    list_X = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    return list_X

def get_disc_batch(image_batch, profile_batch, generator, batch_counter, patch_size,
                   label_smoothing=False, label_flipping=0):
    batch_size = image_batch.shape[0]
    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = generator.predict(image_batch)
        y_disc = np.zeros((batch_size, 2), dtype=np.uint8)
        y_disc[:, 0] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]
    else:
        X_disc = profile_batch
        y_disc = np.zeros((batch_size, 2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form X_disc
    X_disc = extract_patches(X_disc, patch_size)
    return X_disc, y_disc

def gen_sample(generator, nb_sample):
    image_sample, profile_sample= [], []
    image_batch, profile_batch = next(generator)
    nb_batch = image_batch.shape[0]
    tot_batch = 0
    while nb_sample >= tot_batch:
        image_sample.append(image_batch); profile_sample.append(profile_batch)
        tot_batch += nb_batch
        image_batch, profile_batch = next(generator)

    image_sample = np.concatenate(image_sample)[:nb_sample]
    profile_sample = np.concatenate(profile_sample)[:nb_sample]
    return image_sample, profile_sample

def plot_sample_image(images, profiles, plot_path):
    images = np.stack([to_uint8(image) for image in np.squeeze(images)])
    profiles = np.stack([to_uint8(image) for image in np.squeeze(profiles)])

    samples = []
    for image, profile in zip(images, profiles):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        result = fill_mask(image, profile)
        samples.append(np.concatenate((image,result),axis=1))
    samples = np.concatenate(samples,axis=0)
    samples = cv2.cvtColor(samples, cv2.COLOR_BGR2RGB)
    cv2.imwrite(plot_path, samples)

def extract_contour(image):
    kernel = np.ones((5,5), np.uint8)
    return image - cv2.erode(image,kernel,iterations=1)

def fill_mask(image, profile):
    mask = np.expand_dims(profile,axis=-1) * np.array([1,0,0],dtype=np.uint8)
    contour = extract_contour(profile)
    contour = np.expand_dims(contour,axis=-1) * np.array([1,1,1],dtype=np.uint8)
    result = cv2.addWeighted(image,0.5,mask,0.5,0.8)
    return cv2.add(result,contour)
