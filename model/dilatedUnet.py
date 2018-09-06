from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D
from keras.layers import BatchNormalization, concatenate
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from ops import InstanceNormalization, conv_bn_activation, PermaDropout

def DILATEDUNET(img_dim=(256,256,3), depth=4,
         nb_filter=32, bn=True, instance=False, drop_rate=0, activation='relu',
         sigmoid=True, output_dim=(256,256,1),dilation_rate=1):
    '''
    From pix2pix
    1. BatchNorm => Instance Norm
    2. Instance Norm is not applied to the first layer
    3. Activation
        encoder => leaky relu
        decoder => relu
    4. Dropout
        permument Dropout

    From Unet
    1. Width : 2 ( double conv + )
    '''
    input_size = img_dim[:2]
    nb_out_channel = output_dim[2]
    x = Input(img_dim,name="input")

    # prepare encode filters
    list_filters = [min(512, nb_filter * (2 ** i)) for i in range(depth)]

    # Down part - Encoder
    p = None
    convs = []
    for i in range(depth):
        if p is None:
            c, p = unet_convBlock(x, list_filters[i], "conv{}-".format(i), False, instance, drop_rate, activation, dilation_rate)
        else:
            c, p = unet_convBlock(p, list_filters[i], "conv{}-".format(i), bn, instance, drop_rate, activation, dilation_rate)
        convs.append(c)

    # Bottom part
    p = conv_bn_activation(p, list_filters[-1]*2, 'mid-1', bn, instance, activation)
    p = conv_bn_activation(p, list_filters[-1]*2, 'mid-2', bn, instance, activation)

    # Up part - Decoder
    for i in range(depth-1,-1,-1):
        if i == depth-1:
            c = unet_upconvBlock(p, convs[i], list_filters[i],
                "upconv{}-".format(i), bn, instance, drop_rate, 'relu')
        else:
            c = unet_upconvBlock(c, convs[i], list_filters[i],
                "upconv{}-".format(i), bn, instance, drop_rate, 'relu')

    if sigmoid:
        y = Conv2D(nb_out_channel, (1, 1), activation='sigmoid') (c)
    else:
        y = Conv2D(nb_out_channel, (1, 1), activation='tanh') (c)

    model = Model(inputs=x, outputs=y, name='unet')
    return model

def unet_convBlock(x, filters, block_name, bn=True, instance=False, drop_rate=0,activation='relu', dilation_rate=2):
    conv = conv_bn_activation(x, filters, block_name+'conv1', 1, bn, instance, activation)
    conv = dilated_bn_activation(conv, filters, block_name+'conv2', 1, bn, instance, activation, dilation_rate=dilation_rate)
    concat = concatenate([x, conv], axis=3, name=block_name+"concat")
    squeeze = Conv2D(filters, (1,1), activation='linear', name=block_name+"pointwise", use_bias=False)(concat)
    out = AveragePooling2D((2, 2), name=block_name+"pool")(squeeze)
    return conv, out

def unet_upconvBlock(x, connect_layer, filters, block_name, bn=True, instance=False, drop_rate=0, activation='relu', dilation_rate=2):
    upconv = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=block_name+"upconv1") (x)
    concat = concatenate([upconv, connect_layer], axis=3, name=block_name+"concat")

    conv = conv_bn_activation(concat, filters, block_name+'conv1', 1, bn, instance, activation)
    conv = conv_bn_activation(conv, filters, block_name+'conv2', 1, bn, instance, activation)
    concat = concatenate([concat, conv], axis=3, name=block_name+"concat2")
    squeeze = Conv2D(filters, (1,1), activation='linear', name=block_name+"pointwise", use_bias=False)(concat)
    return squeeze

def dilated_bn_activation(x, filters, block_name, strides=1, bn=True, instance=False, activation='relu', dilation_rate=2):
    if bn:
        conv = Conv2D(filters, (3, 3), activation='linear', padding='same', strides=strides, 
                    name=block_name, use_bias=False, dilation_rate=dilation_rate) (x)
        if instance:
            conv = InstanceNormalization(axis=3)(conv)
        else:
            conv = BatchNormalization(axis=3)(conv)
    else:
        conv = Conv2D(filters, (3, 3), activation='linear', padding='same', strides=strides, 
                    name=block_name, dilation_rate=dilation_rate) (x)

    if activation=='LeakyReLU':
        conv = LeakyReLU(0.2)(conv)
    else:
        conv = Activation('relu')(conv)
    return conv
