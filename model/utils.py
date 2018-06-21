import h5py
import random
import numpy as np
import cv2

from skimage.transform import rescale, rotate
from skimage.util import random_noise
from PIL import Image

__all__ = ['HumanSegGenerator','load_dataset']

def HumanSegGenerator(dataset, img_dim, batch_size=64,is_train=True):
    '''
    dataset : total dataset of human segmentation
        data는 (384,384,4)로 구성되어 잇는데,
        image, profile = data[384,384,:3], data[384,384,-1]

        with h5py.File("../data/baidu_segmentation.h5") as file:
            dataset = file['384x384'][:]
    img_dim : 모델에 feeding하기 위한 input size

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
                data = apply_random_crop(data, img_dim)
                # give instance Noise for training
                data[:,:,:3] = random_noise(data[:,:,:3],
                                            mode='gaussian',
                                            mean=0,var=0.001)
            else:
                #test일 경우, data augmentation 하지 않음
                data = data / 255. # normalize data
                data = cv2.resize(data, img_dim[:2])

            # dataset을 image와 profile로 나눔
            image, profile = data[:,:,:3], data[:,:,-1]
            # adjust the range of value
            image = np.clip(image,0.,1.)

            profile = (profile>0.7).astype(int)
            profile = np.expand_dims(profile,axis=-1)

            # normalize from (0,1) to (-1,1)
            image, profile = normalization(image), normalization(profile)
            batch_image.append(image); batch_profile.append(profile)
            if counter == batch_size:
                yield np.stack(batch_image, axis=0), np.stack(batch_profile, axis=0)
                counter = 0; batch_image = []; batch_profile = []

def normalization(X):
    # value range : (0,1) => (-1,1)
    return (X - 0.5) * 2.

def inverse_normalization(X):
    # value range : (-1,1) => (0,1)
    res = (X + 1.) / 2.
    return np.clip(res, 0.,1.)

# data augumentation pipeline
def apply_rotation(image):
    rotation_angle = random.randint(-8,8)
    return rotate(image, rotation_angle, mode='constant', cval=1.)

def apply_rescaling(image):
    rescale_ratio = 0.9 + random.random() * 0.1 # from 0.8 to 1.2
    return rescale(image, rescale_ratio,mode='reflect')

def apply_flip(image):
    if random.random()>0.5:
        return image[:,::-1,:]
    else:
        return image

def apply_random_crop(image, img_dim):
    iv = random.randint(0,image.shape[0]-img_dim[0])
    ih = random.randint(0,image.shape[1]-img_dim[1])
    return image[iv:iv+img_dim[0],ih:ih+img_dim[1],:]

def load_dataset(dataset_name='384x384',h5_path="../data/baidu_segmentation.h5"):
    with h5py.File(h5_path) as file:
        return file[dataset_name][:]

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

def plot_generated_batch(image_sample, profile_sample, generator, plot_path):
    nb_sample = image_sample.shape[0]
    rows = []
    for i in range(nb_sample):
        image = inverse_normalization(np.squeeze(image_sample[i]))
        real_profile = np.squeeze(inverse_normalization(profile_sample[i]))
        y = generator.predict(image_sample[i:i+1])
        fake_profile = np.squeeze(inverse_normalization(y))
        real_profile = np.stack([real_profile]*3,axis=-1)
        fake_profile = np.stack([fake_profile]*3,axis=-1)

        rows.append(np.concatenate([image, real_profile, fake_profile],axis=1))
    sample = np.concatenate(rows,axis=0)
    Image.fromarray((sample * 255).astype(np.uint8)).save(plot_path)
