from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import BatchNormalization, concatenate
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from ops import InstanceNormalization, conv_bn_activation, mobile_conv_block

def MOBILEUNET(img_dim=(256,256,3), depth=4,
         nb_filter=32, depth_multiplier=1, bn=True, instance=False, drop_rate=0, activation='relu',
         sigmoid=True, output_dim=(256,256,1)):
    '''
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
            c, p = unet_convBlock(x, list_filters[i], "conv{}-".format(i), False, instance, drop_rate, activation,depth_multiplier)
        else:
            c, p = unet_convBlock(p, list_filters[i], "conv{}-".format(i), bn, instance, drop_rate, activation,depth_multiplier)
        convs.append(c)

    # Bottom part
    p = mobile_conv_block(p, list_filters[-1]*2, 'mid-1', bn, instance, activation, depth_multiplier)
    p = mobile_conv_block(p, list_filters[-1]*2, 'mid-2', bn, instance, activation, depth_multiplier)

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

def unet_convBlock(x, filters, block_name, bn=True, instance=False, drop_rate=0, activation='relu', depth_multiplier=1):
    conv = mobile_conv_block(x, filters, block_name+'conv1', 1, bn, instance, activation, depth_multiplier)
    conv = mobile_conv_block(conv, filters, block_name+'conv2', 1, bn, instance, activation, depth_multiplier)
    out = MaxPooling2D((2, 2), name=block_name+"pool") (conv)
    return conv, out

def unet_upconvBlock(x, connect_layer, filters, block_name, bn=True, instance=False, drop_rate=0, activation='relu', depth_multiplier=1):
    upconv = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=block_name+"upconv1") (x)
    concat = concatenate([upconv, connect_layer], axis=3, name=block_name+"concat")

    conv = mobile_conv_block(concat, filters, block_name+'conv1', 1, bn, instance, activation, depth_multiplier)
    conv = mobile_conv_block(conv, filters, block_name+'conv2', 1, bn, instance, activation, depth_multiplier)
    return conv
