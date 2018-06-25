from keras.models import Model, Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
import numpy as np

__all__ = ["PIC2PIC", "GENERATOR", "DISCRIMINATOR", 'PATCHBLOCK']

'''
GAN - merge Generator with discriminator
'''
def PIC2PIC(generator, discriminator):
    gen_input = generator.input
    gen_output = generator.output

    # get the patch_size, patch_size is the input node size of discriminator
    patch_node = discriminator.get_input_at(0)[0]
    _, ph, pw, _ = patch_node.shape.as_list()
    patch_size = (ph,pw)

    list_gen_patch = gen_patch(gen_input, gen_output, patch_size)
    patch_output = discriminator(list_gen_patch)

    gan =  Model(inputs=[gen_input],
                    outputs=[gen_output, patch_output],
                    name='PIC2PIC_GAN')
    return gan

def gen_patch(gen_input, gen_output, patch_size):
    _, h, w, _ = gen_input.shape.as_list()
    ph, pw = patch_size

    list_row_idx = [(i * ph, (i + 1) * ph) for i in range(h // ph)]
    list_col_idx = [(i * pw, (i + 1) * pw) for i in range(w // pw)]

    list_gen_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1],
                                         col_idx[0]:col_idx[1], :])(gen_output)
            list_gen_patch.append(x_patch)

    return list_gen_patch
'''
generator - Unet

    The encoder-decoder architecture consists of:

encoder:

    C64-C128-C256-C512-C512-C512-C512-C512
    => C64-C128-C256-C512-C512-C512
decoder:

    CD512-CD512-CD512-C512-C512-C256-C128-C64
    => CD512-C512-C512-C256-C128-C64

    After the last layer in the decoder, a convolution is applied
    to map to the number of output channels (3 in general,
    except in colorization, where it is 2), followed by a Tanh
    function. As an exception to the above notation, BatchNorm
    is not applied to the first C64 layer in the encoder.
    All ReLUs in the encoder are leaky, with slope 0.2, while
    ReLUs in the decoder are not leaky
'''
def GENERATOR(img_dim=(256,256,3), nb_filter=64, output_dim=(256,256,2)):
    input_layer = Input(shape=img_dim, name='unet_input')

    # prepare encode filters
    nb_conv = int(np.floor(np.log2(img_dim[1])))
    list_filters = [nb_filter * min(8, (2 ** i)) for i in range(nb_conv)]

    # encoder
    list_encoder = []
    list_encoder.append(Conv2D(list_filters[0], (3,3),
                               strides=(2,2), name='unet_conv2D_1',
                               padding='same')(input_layer))
    for i, filters in enumerate(list_filters[1:]):
        name = "unet_conv2D_{}".format(i + 2)
        conv = conv_block_unet(list_encoder[-1], filters, True,name)
        list_encoder.append(conv)

    # Prepare decoder filters
    list_filters = list_filters[:-1][::-1]
    if len(list_filters) < nb_conv - 1:
        list_filters.append(nb_filter)

    # decoder
    list_decoder = []
    list_decoder.append(deconv_block_unet(list_encoder[-1], list_encoder[-2],
                                        list_filters[0], bn=True, dropout=True,
                                        name='unet_deconv2D_1'))
    for i, filters in enumerate(list_filters[1:]):
        name = 'unet_deconv2D_{}'.format(i+2)

        if i < 2: d = True
        else: d = False

        conv = deconv_block_unet(list_decoder[-1], list_encoder[-(i+3)], filters,
                                bn=True, dropout=d,name=name)
        list_decoder.append(conv)

    x = Activation('relu')(list_decoder[-1])
    out_channels = output_dim[-1]
    x = Conv2DTranspose(out_channels, (3,3), strides=(2,2), padding='same')(x)
    x = Activation('tanh')(x)
    return Model(inputs=input_layer, outputs=x)

def conv_block_unet(x, filters, bn, name):
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters, (3, 3), strides=(2,2), name=name, padding="same")(x)
    if bn:
        x = BatchNormalization()(x)
    return x

def deconv_block_unet(x, encoded, filters, bn, dropout, name):
    x = Activation("relu")(x)
    x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same")(x)
    if bn:
        x = BatchNormalization()(x)
    # keras에서는 inference 과정에서 dropout이 가능하도록 하지 못하게 설계되어 있음
    # if dropout:
    #     x = Dropout(0.5)(x)
    x = Concatenate()([x, encoded])
    return x
'''
discriminator - PatchGAN

    [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]
    PatchGAN only penalizes structure at the scale of patches. This
    discriminator tries to classify if each N x N patch in an
    image is real or fake. We run this discriminator convolutationally
    across the image, averaging all responses to provide
    the ultimate output of D.

    The 70 × 70 discriminator architecture is:
    C64-C128-C256-C512
    After the last layer, a convolution is applied to map to a 1
    dimensional output, followed by a Sigmoid function. As an
    exception to the above notation, BatchNorm is not applied
    to the first C64 layer. All ReLUs are leaky, with slope 0.2.
    All other discriminators follow the same basic architec-
    ture, with depth varied to modify the receptive field size:
'''

def DISCRIMINATOR(img_dim, patch_size, nb_filter=64):
    """
    The discriminator has two parts. First part is the actual discriminator
    seconds part we make it a PatchGAN by running each image patch through the model
    and then we average the responses
    Discriminator does the following:
    1. Runs many pieces of the image through the network
    2. Calculates the cost for each patch
    3. Returns the avg of the costs as the output of the network
    """
    patch_dim = *patch_size, img_dim[2]
    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch = get_nb_patch(img_dim, patch_size)

    # generate a list of inputs for the difference patches to the network
    list_input = [Input(shape=patch_dim, name="patch_gan_input_{}".format(i)) for i in range(nb_patch)]

    # generate individual losses for each patch
    patchblock = PATCHBLOCK(patch_dim,img_dim, nb_filter)
    patches = [patchblock(patch) for patch in list_input]

    # generate minibatch discriminator among patches
    x, x_mbd = list(zip(*patches))
    x_out = MINIBATCH_DISCRIMINATOR(list(x), list(x_mbd))

    return Model(inputs=list_input, outputs=[x_out], name='discriminator')

def PATCHBLOCK(patch_dim, img_dim, nb_filter):
    # We have to build the discriminator dinamically because
    # the size of the disc patches is dynamic

    # 의심스러운 구석 중 하나. patch_dim으로 바뀌어야 할 거 같은데... 그래서 바꾸었음
    # 70 x 70 Discriminator architecture is:
    # C64 - C128 - C256 - C512
    #nb_conv = int(np.floor(np.log2(img_dim[1])))
    nb_conv = int(np.floor(np.log2(patch_dim[1])-np.log2(4)))
    list_filters = [nb_filter * min(8, (2 ** i)) for i in range(nb_conv)]

    # INPUT
    input_layer = Input(shape=patch_dim)
    # CONV 1
    # Do first conv bc it is different from the rest
    # paper skips batch norm for first layer
    x = Conv2D(list_filters[0], (3, 3),
                      strides=(2, 2), name="disc_conv2d_1", padding="same")(input_layer)
    x = LeakyReLU(alpha=0.2)(x)

    # CONV 2 - CONV N
    # do the rest of the convs based on the sizes from the filters
    for i, filter_size in enumerate(list_filters[1:]):
        name = 'disc_conv2d_{}'.format(i+2)
        x = Conv2D(filter_size, (3, 3), strides=(2, 2), name=name, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU(0.2)(x)

    out_flat = Flatten()(x)
    out = Dense(2, activation='softmax', name='disc_dense')(out_flat)

    return Model(inputs=input_layer, outputs=[out,out_flat], name='patch_gan')

def MINIBATCH_DISCRIMINATOR(x, x_mbd):
    # merge layers if have multiple patches (aka perceptual loss)
    if len(x) > 1:
        x = Concatenate()(x)
    else:
        x = x[0]
    # merge mbd if needed
    # mbd = mini batch discrimination
    # https://arxiv.org/pdf/1606.03498.pdf
    if len(x_mbd) > 1:
        x_mbd = Concatenate()(x_mbd)
    else:
        x_mbd = x_mbd[0]

    num_kernels = 100
    dim_per_kernel = 5

    x_mbd = Dense(num_kernels * dim_per_kernel, use_bias=False, activation=None)(x_mbd)
    x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
    x_mbd = Lambda(minibatch_disc, output_shape=lambda shape : shape[:2], name='disc_mbd')(x_mbd)

    x = Concatenate()([x,x_mbd])
    return Dense(2, activation='softmax', name='disc_output')(x)

def minibatch_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)

    return x

def get_nb_patch(img_dim, patch_size):
    # 이미지를 patch 사이즈만큼 잘라, 몇 개가 나오는지 연산하는 함수
    assert img_dim[0] % patch_size[0] == 0, "patch_size does not divide height"
    assert img_dim[1] % patch_size[1] == 0, "patch_size does not divide width"

    nb_patch = (img_dim[0] // patch_size[0]) * (img_dim[1] // patch_size[1])
    return nb_patch
