# pos + pos -> 1
# pos + neg -> 0


from keras.applications import VGG16
import os
import DataHandler as dh
import BaseModel as bm
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import random


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    lhs_images = []
    rhs_images = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            # pairs += [[x[z1], x[z2]]]
            lhs_images += [x[z1]]
            rhs_images += [x[z2]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            lhs_images += [x[z1]]
            rhs_images += [x[z2]]
            # pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(lhs_images), np.array(rhs_images), np.array(labels)

IM_SIZE = 224
EPOCHS = 20
BATCH_SIZE = 32

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_dim = 784
epochs = 20

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(10)]
tr_lhs, tr_rhs, tr_y = create_pairs(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(10)]
te_lhs, te_rhs, te_y = create_pairs(x_test, digit_indices)

print("tr", tr_lhs.shape)
print('y', tr_y.shape)

vgg_1 = VGG16(weights='imagenet', include_top=True)
vgg_2 = VGG16(weights='imagenet', include_top=True)

for layer in vgg_1.layers:
    layer.trainable = False
    layer.name = layer.name + "_1"
for layer in vgg_2.layers:
    layer.trainable = False
    layer.name = layer.name + "_2"
print('_'*12, 'VGG16', '-'*12)
vgg_1.summary()


v1 = vgg_1.get_layer("flatten_1").output
v2 = vgg_2.get_layer("flatten_2").output

pred = bm.sim_model(v1, v2)

model = Model(inputs=[vgg_1.input, vgg_2.input], outputs=pred)

print('_'*12, 'SIAMESE', '-'*12)
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit([tr_lhs, tr_rhs], tr_y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
