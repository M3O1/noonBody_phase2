from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Dense, concatenate
from keras.layers.core import Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from ops import InstanceNormalization, conv_bn_activation

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
def DISCRIMINATOR(img_dim=(256,256,3), nb_filter=32,activation='LeakyReLu',
                bn=True, instance=False, sigmoid=True, output_dim=(256,256,1)):
    h, w, ch = img_dim
    _, _, out_ch = output_dim
    inputs = Input((h, w, ch + out_ch))

    conv = conv_bn_activation(inputs, nb_filter, 'conv1-1', bn=bn, instance=instance,  activation=activation)
    conv = MaxPooling2D(pool_size=(2, 2))(conv)
    conv = conv_bn_activation(conv, nb_filter, 'conv1-2', bn=bn, instance=instance, activation=activation)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    conv = conv_bn_activation(pool, 2*nb_filter, 'conv2-1', bn=bn, instance=instance,  activation=activation)
    conv = MaxPooling2D(pool_size=(2, 2))(conv)

    conv = conv_bn_activation(conv, 2*nb_filter, 'conv2-2', bn=bn, instance=instance,  activation=activation)

    for idx in range(2,5):
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        conv = conv_bn_activation(pool, (2**(idx)*nb_filter),
            'conv{}-1'.format(idx+1), 1, bn=bn, instance=instance, activation=activation)
        conv = conv_bn_activation(conv, (2**(idx)*nb_filter),
            'conv{}-2'.format(idx+1), 1, bn=bn, instance=instance, activation=activation)

    gap = GlobalAveragePooling2D()(conv)
    if sigmoid:
        outputs = Dense(2, activation='sigmoid')(gap)
    else:
        outputs = Dense(2, activation='tanh')(gap)

    model = Model(inputs, outputs, name='discriminator')
    return model

def PIC_DISCRIMINATOR(img_dim=(256,256,3), nb_filter=64, depth=6,
                    activation='LeakyReLu',bn=True, instance=False,
                    sigmoid=True, output_dim=(256,256,1)):
    h, w, ch = img_dim
    _, _, out_ch = output_dim
    inputs = Input((h, w, ch + out_ch))

    list_filters = [nb_filter * min(8, (2 ** i)) for i in range(depth)]

    conv = None
    for i, nb_filter in enumerate(list_filters):
        if i == 0:
            conv = conv_bn_activation(inputs, nb_filter, 'conv-{}'.format(i), 2,
                                        bn=bn, instance=instance, activation=activation)
        else:
            conv = conv_bn_activation(conv, nb_filter, 'conv-{}'.format(i), 2,
                                        bn=bn, instance=instance, activation=activation)
    flat = Flatten()(conv)
    if sigmoid:
        outputs = Dense(2, activation='softmax')(flat)
    else:
        outputs = Dense(2, activation='tanh')(flat)

    model = Model(inputs, outputs, name='discriminator')
    return model
