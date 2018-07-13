from keras.models import Model
from keras.layers import Input, MaxPool2D,  Conv2D, Dropout, concatenate, Reshape, Lambda, AveragePooling2D
from keras import backend as K
from keras.initializers import TruncatedNormal
from keras.regularizers import l2
from keras.layers import Flatten, Dense

def SQUEEZEDET(img_dim=(256,256,3), nb_filter=16,drop_rate=0.3, nb_fc=256):
    input_layer = Input(img_dim,name='input')

    conv0 = Conv2D(64, (3,3),strides=(2,2),padding='SAME', activation='relu',
                  kernel_initializer=TruncatedNormal(stddev=0.001),
                  kernel_regularizer=l2(1e-3))(input_layer)
    pool0 = MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding='SAME', name='pool0')(conv0)

    nb_f1 = nb_filter
    fire1 = fire_layer(pool0, nb_f1, name='fire1')
    fire2 = fire_layer(fire1, nb_f1, name='fire2')
    fire3 = fire_layer(fire2, nb_f1, name='fire3')
    pool1 = MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding='SAME', name='pool1')(fire3)

    nb_f2 = nb_filter*2
    fire4 = fire_layer(pool1, nb_f2, name='fire4')
    fire5 = fire_layer(fire4, nb_f2, name='fire5')
    pool2 = MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool2")(fire5)

    nb_f3 = nb_filter*3
    fire6 = fire_layer(pool2, nb_f3, name='fire6')
    fire7 = fire_layer(fire6, nb_f3, name='fire7')

    nb_f4 = nb_filter*4
    fire8 = fire_layer(fire7, nb_f4, name='fire8')
    fire9 = fire_layer(fire8, nb_f4, name='fire9')

    nb_f5 = nb_filter*5
    fire10 = fire_layer(fire9, nb_f5, name='fire10')
    fire11 = fire_layer(fire10, nb_f5, name='fire11')

    dropout = Dropout(rate=drop_rate, name='drop1')(fire11)

    preds = Conv2D(nb_filter, (1,1), activation='relu', padding='SAME',
            kernel_initializer=TruncatedNormal(stddev=0.001),
            kernel_regularizer=l2(1e-3))(dropout)

    flat = Flatten()(preds)
    flat = Dense(nb_fc, activation='relu')(flat)
    out = Dense(4)(flat)

    model = Model(inputs=input_layer, outputs=out,name='squeezedet')
    return model

def fire_layer(x, nb_filter, stdd=0.001, w_decay=1e-3, name=''):
    '''
    wrapper for fire layer constructions

    :param name: name for layer
    :param input: previous layer
    :param nb_filter: number of filters for squeezing
    :param stdd: standard deviation used for initialization
    :return: a keras fire layer
    '''
    sq1x1 = Conv2D(nb_filter, (1,1), strides=(1,1), padding='SAME',
                   kernel_initializer=TruncatedNormal(stddev=stdd), activation='relu',
                   kernel_regularizer=l2(w_decay),name=name+'/squeeze1x1')(x)
    ex1x1 = Conv2D(4*nb_filter, (1,1), strides=(1,1), padding='SAME',
                   kernel_initializer=TruncatedNormal(stddev=stdd), activation='relu',
                   kernel_regularizer=l2(w_decay),name=name+'/expand1x1')(sq1x1)
    ex3x3 = Conv2D(4*nb_filter, (3,3), strides=(1,1), padding='SAME',
                   kernel_initializer=TruncatedNormal(stddev=stdd), activation='relu',
                   kernel_regularizer=l2(w_decay),name=name+'/expand3x3')(sq1x1)
    return concatenate([ex1x1,ex3x3], axis=3)
