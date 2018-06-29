import os
import numpy as np
import cv2
import keras
from slacker import Slacker
import keras.backend as K
import tensorflow as tf
from datetime import datetime
import json

class PlotCheckpoint(keras.callbacks.Callback):
    def __init__(self, unet, images, plot_dir,threshold=0.5, bg_removal=False):
        self.unet = unet
        self.images = images
        self.plot_dir = plot_dir
        self.thr = threshold
        self.bg_removal = bg_removal

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        plot_path = os.path.join(self.plot_dir,"{:02d}_epoch_sample.png".format(epoch))

        pred = self.unet.predict_on_batch(self.images)
        if self.bg_removal:
            plot_bg_removal_sample_image(self.images, pred, plot_path)
        else:
            pred = (pred >= self.thr).astype(np.uint8)
            plot_sample_image(self.images, pred, plot_path)

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

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

def plot_bg_removal_sample_image(images, profiles, plot_path):
    images = np.stack([to_uint8(image) for image in np.squeeze(images)])
    profiles = np.stack([to_uint8(image) for image in np.squeeze(profiles)])

    samples = []
    for image, profile in zip(images, profiles):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if len(profile.shape) == 2:
            image = cv2.cvtColor(profile, cv2.COLOR_GRAY2RGB)
        samples.append(np.concatenate((image,profile),axis=1))
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

def to_uint8(X):
    X = X.astype(np.float32)
    return cv2.normalize(X,np.zeros_like(X),
              0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)

def send_report_to_slack(title,description,image_paths):
    try:
        with open("../.slack_token",'r') as file:
            token = json.load(file)['token']
    except FileNotFoundError as error:
        print("File(.slack_token) Not Found")

    slack = Slacker(token)
    attachment = dict()
    attachment = {
        "title": title ,
        "text" : description
    }

    attachments = [attachment]
    slack.chat.post_message('#noonbody',attachments=attachments)
    for title, image_path in image_paths.items():
        try:
            slack.files.upload(image_path, title=title, channels='#noonbody')
        except:
            pass

def set_directory(save_dir,model_name):
    result_dir = os.path.join(save_dir,"{}/{}/".format(model_name,
                                                      datetime.now().strftime('%m%d_%H')))
    # set the path of weights
    weights_dir = os.path.join(result_dir,'weights')
    os.makedirs(weights_dir,exist_ok=True)
    # set the path of model
    model_dir = os.path.join(result_dir, "model-arch")
    os.makedirs(model_dir, exist_ok=True)
    # set the path of sample iamge
    sample_dir = os.path.join(result_dir,"sample")
    os.makedirs(sample_dir, exist_ok=True)
    return weights_dir, model_dir, sample_dir

def mean_iou(y_pred, y_true):
    y_pred = K.cast(y_pred >= 0.5, dtype=tf.float32)
    y_true = K.cast(y_true >= 0.5, dtype=tf.float32)
    intersect = y_pred * y_true
    union = K.ones_like(y_pred) - ((1-y_pred)*(1-y_true))
    return K.sum(intersect) / K.sum(union)

def np_mean_iou(y_pred, y_true, thr=0.5):
    y_pred = (y_pred>thr).astype(np.int)

    intersect = y_pred * y_true
    union = np.ones_like(y_pred) - ((1-y_pred) * (1-y_true))
    return np.sum(intersect) / np.sum(union)

def get_disc_batch(image_batch, profile_batch, generator, batch_counter,
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

    X_disc = np.concatenate([image_batch, X_disc],axis=-1)

    return X_disc, y_disc
