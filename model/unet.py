from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import BatchNormalization, concatenate
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from keras import backend as K

def UNET(img_dim=(256,256,3), depth=6, nb_filter=16, output_dim=(256,256,2), output_activation='tanh'):
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
            c, p = unet_convBlock(x, list_filters[i], block_name="conv{}-".format(i))
        else:
            c, p = unet_convBlock(p, list_filters[i], block_name="conv{}-".format(i))
        convs.append(c)

    # Bottom part
    p = Conv2D(list_filters[-1], (3, 3), activation='linear', padding='same', name='mid-1') (p)
    p = LeakyReLU(0.2)(p)
    p = Conv2D(list_filters[-1], (3, 3), activation='linear', padding='same', name='mid-2') (p)
    p = LeakyReLU(0.2)(p)

    # Up part
    for i in range(depth-1,-1,-1):
        if i == depth-1:
            c = unet_upconvBlock(p, convs[i], list_filters[i], block_name="upconv{}-".format(i))
        else:
            c = unet_upconvBlock(c, convs[i], list_filters[i], block_name="upconv{}-".format(i))

    if output_activation == 'tanh':
        y = Conv2D(nb_out_channel, (1, 1), activation='tanh') (c)
    elif output_activation == 'sigmoid':
        y = Conv2D(nb_out_channel, (1, 1), activation='sigmoid') (c)
    else:
        NotImplementedError()

    model = Model(inputs=x, outputs=y)
    return model

def unet_convBlock(x, filters, block_name):
    conv = Conv2D(filters, (3, 3), activation='linear', padding='same', name=block_name+"conv1") (x)
    conv = LeakyReLU(0.2)(conv)
    conv = Conv2D(filters, (3, 3), activation='linear', padding='same', name=block_name+"conv2") (conv)
    conv_lr = LeakyReLU(0.2)(conv)
    out = MaxPooling2D((2, 2), name=block_name+"pool") (conv_lr)
    return conv, out

def unet_upconvBlock(x, connect_layer, filters, block_name):
    upconv = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=block_name+"upconv1") (x)
    concat = concatenate([upconv, connect_layer], axis=3, name=block_name+"concat")
    conv = Conv2D(filters, (3, 3), activation='linear', padding='same', name=block_name+"conv1") (concat)
    conv = LeakyReLU(0.2)(conv)
    out = Conv2D(filters, (3, 3), activation='linear', padding='same', name=block_name+"conv2") (conv)
    conv = LeakyReLU(0.2)(conv)
    return out
