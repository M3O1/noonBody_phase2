from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import BatchNormalization, concatenate
from keras.layers.core import Activation

def TIRAMISU(img_dim=(256,256,3), nb_layers=[4,5,7,10],
             nb_filter=16, growth_rate = 16, bn=True, sigmoid=True,
             output_dim=(256,256,1)):
    nb_out_channel = output_dim[2]
    x = Input(img_dim,name='intput')


    p = Conv2D(nb_filter,(3,3),padding='same',kernel_initializer='he_uniform')(x)
    skips, added = down_path(p, nb_layers, growth_rate, bn=bn)
    out = up_path(added, skips[:-1][::-1], nb_layers[:-1][::-1], growth_rate, bn=bn)

    if sigmoid:
        y = Conv2D(nb_out_channel, (1,1),activation='sigmoid',
                   kernel_initializer='he_uniform')(out)
    else:
        y = Conv2D(nb_out_channel, (1,1),activation='sigmoid',
                   kernel_initializer='he_uniform')(out)

    model = Model(inputs=x, outputs=y, name='tiramisu')
    return model

def transition_dn(x,bn=True):
    input_nb_filter = x.get_shape().as_list()[-1]
    return bn_relu_conv(x, input_nb_filter, strides=2, bn=bn)

def down_path(x, nb_layers, growth_rate, bn=True):
    skips = []
    for i,n in enumerate(nb_layers):
        x, added = dense_block(x,n,growth_rate,bn=bn)
        skips.append(x)
        x = transition_dn(x, bn=bn)
    return skips, added

def transition_up(added):
    x = concatenate(added)
    nb_filter = x.get_shape().as_list()[-1]
    return Conv2DTranspose(nb_filter,(3,3),strides=(2,2),
                           kernel_initializer='he_uniform',
                           padding='same')(x)

def up_path(added, skips, nb_layers, growth_rate,bn=True):
    for i, n in enumerate(nb_layers):
        x = transition_up(added)
        x = concatenate([x,skips[i]])
        x, added = dense_block(x,n,growth_rate,bn=bn)
    return x

def dense_block(x, nb_block, growth_rate, bn=True):
    added = []
    for _ in range(nb_block):
        b = bn_relu_conv(x, growth_rate,bn=True)
        x = concatenate([x,b])
        added.append(b)
    return x, added

def bn_relu_conv(x, nb_filter, strides=1, bn=True):
    if bn:
        conv = BatchNormalization(axis=3)(x)
        conv = Activation('relu')(conv)
        conv = Conv2D(nb_filter, (3,3), strides=strides,
                      activation='linear', padding='same',
                      kernel_initializer='he_uniform',
                      use_bias=False)(conv)
    else:
        conv = Activation('relu')(x)
        conv = Conv2D(nb_filter, (3,3), strides=strides,
                      activation='linear', padding='same',
                      kernel_initializer='he_uniform')(conv)
    return conv
