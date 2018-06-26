from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Dense, concatenate
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU

def VGAN(generator, discriminator):
    image = generator.inputs[0]
    machine_seg = generator.outputs[0]

    # machine
    gen_concat = concatenate([image,machine_seg],axis=3)
    dis_output = discriminator(gen_concat)

    gan = Model(image, [machine_seg, dis_output], name='gan')
    return gan

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
