from keras import backend as K
from keras.layers.core import Activation, Dense, Dropout, Lambda
import sys

def cosine_distance(vecs, normalize=False):
    x, y = vecs
    if normalize:
        x = K.l2_normalize(x, axis=0)
        y = K.l2_normalize(x, axis=0)
    return K.prod(K.stack([x, y], axis=1), axis=1)


def cosine_distance_output_shape(shapes):
    return shapes[0]


def sim_model(v1, v2):

    print('v1', v1)
    print('v2', v2)

    merged = Lambda(cosine_distance,
                    output_shape=cosine_distance_output_shape)([v1, v2])
    print('merge', merged)
    fc1 = Dense(512, kernel_initializer="glorot_uniform")(merged)
    fc1 = Dropout(0.2)(fc1)
    fc1 = Activation("relu")(fc1)

    fc2 = Dense(128, kernel_initializer="glorot_uniform")(fc1)
    fc2 = Dropout(0.2)(fc2)
    fc2 = Activation("relu")(fc2)

    pred = Dense(2, kernel_initializer="glorot_uniform")(fc2)
    pred = Activation("softmax")(pred)
    print('pred', pred)
    return pred


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))