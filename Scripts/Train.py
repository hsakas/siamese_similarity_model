import DownloadData

from keras.applications import VGG16
import os
import DataHandler as dh
import BaseModel as bm
from keras.models import Model

DATA_DIR = os.path.abspath("../data")
IMAGE_TEMP_DIR = os.path.join(DATA_DIR, "tmp")
IMAGE_DIR = os.path.join(DATA_DIR, "images", "jpg")
IM_SIZE = 224
EPOCHS = 20
BATCH_SIZE = 32

print('Creating triples...')
triples = dh.create_image_triples(DATA_DIR)

print('Loading images...')
lhs, rhs, y = dh.load_image_triplets(image_dir=DATA_DIR,
                                  image_triples=triples,
                                  image_size=IM_SIZE, shuffle=True)
print('y', y.shape)
print('lhs', lhs.shape)
print('rhs', rhs.shape)

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

model.fit([lhs, rhs], y, epochs=EPOCHS, batch_size=BATCH_SIZE)
# pos + pos -> 1
# pos + neg -> 0

