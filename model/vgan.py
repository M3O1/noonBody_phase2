from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Flatten, Dense, Reshape, concatenate
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

import keras.backend as K


def VGAN(generator, discriminator):
    image = generator.inputs[0]
    machine_seg = generator.outputs[0]

    # machine
    gen_concat = concatenate([image,machine_seg],axis=3)
    dis_output = discriminator(gen_concat)

    gan = Model(image, [machine_seg, dis_output], name='gan')
    return gan

###################################
# Generator
###################################
def UNET(img_dim=(256,256,3), depth=4,
         nb_filter=32, bn=True, activation='relu',
         out_activation='sigmoid', output_dim=(256,256,1)):

    input_size = img_dim[:2]
    nb_out_channel = output_dim[2]
    x = Input(img_dim,name="input")

    # prepare encode filters
    list_filters = [min(512, nb_filter * (2 ** i)) for i in range(depth)]

    # Down part
    p = None
    convs = []
    for i in range(depth):
        if p is None:
            c, p = unet_convBlock(x, list_filters[i], "conv{}-".format(i), bn, activation)
        else:
            c, p = unet_convBlock(p, list_filters[i], "conv{}-".format(i), bn, activation)
        convs.append(c)

    # Bottom part
    p = conv_bn_activation(p, list_filters[-1]*2, 'mid-1', bn, activation)
    p = conv_bn_activation(p, list_filters[-1]*2, 'mid-2', bn, activation)

    # Up part
    for i in range(depth-1,-1,-1):
        if i == depth-1:
            c = unet_upconvBlock(p, convs[i], list_filters[i], "upconv{}-".format(i), bn, activation)
        else:
            c = unet_upconvBlock(c, convs[i], list_filters[i], "upconv{}-".format(i), bn, activation)

    y = Conv2D(nb_out_channel, (1, 1), activation=out_activation) (c)

    model = Model(inputs=x, outputs=y, name='unet')
    return model

def unet_convBlock(x, filters, block_name, bn=True, activation='relu'):
    conv = conv_bn_activation(x, filters, block_name+'conv1', bn, activation)
    conv = conv_bn_activation(conv, filters, block_name+'conv2', bn, activation)
    out = MaxPooling2D((2, 2), name=block_name+"pool") (conv)
    return conv, out

def unet_upconvBlock(x, connect_layer, filters, block_name, bn=True, activation='relu'):
    upconv = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=block_name+"upconv1") (x)
    concat = concatenate([upconv, connect_layer], axis=3, name=block_name+"concat")

    conv = conv_bn_activation(concat, filters, block_name+'conv1', bn, activation)
    conv = conv_bn_activation(conv, filters, block_name+'conv2', bn, activation)

    return conv

###################################
# Discriminator
###################################
def DISCRIMINATOR(img_dim=(256,256,3), nb_filter=32, output_dim=(256,256,1)):
    h, w, ch = img_dim
    _, _, out_ch = output_dim
    inputs = Input((h, w, ch + out_ch))

    conv = conv_bn_activation(inputs, nb_filter, 'conv1-1', bn=True, activation='relu')
    conv = MaxPooling2D(pool_size=(2, 2))(conv)
    conv = conv_bn_activation(conv, nb_filter, 'conv1-2', bn=True, activation='relu')
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    conv = conv_bn_activation(pool, 2*nb_filter, 'conv2-1', bn=True, activation='relu')
    conv = MaxPooling2D(pool_size=(2, 2))(conv)

    conv = conv_bn_activation(conv, 2*nb_filter, 'conv2-2', bn=True, activation='relu')

    for idx in range(2,5):
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        conv = conv_bn_activation(pool, (2**(idx)*nb_filter),
            'conv{}-1'.format(idx+1), bn=True, activation='relu')
        conv = conv_bn_activation(conv, (2**(idx)*nb_filter),
            'conv{}-2'.format(idx+1), bn=True, activation='relu')

    gap = GlobalAveragePooling2D()(conv)
    outputs = Dense(1, activation='sigmoid')(gap)

    model = Model(inputs, outputs, name='discriminator')
    return model

###################################
# generic layer
###################################
def conv_bn_activation(x, filters, block_name, bn=True, activation='relu'):
    conv = Conv2D(filters, (3, 3), activation='linear', padding='same', name=block_name) (x)
    if bn:
        conv = BatchNormalization(scale=False, axis=3)(conv)
    if activation=='LeakyReLU':
        conv = LeakyReLU(0.2)(conv)
    else:
        conv = Activation('relu')(conv)
    return conv
