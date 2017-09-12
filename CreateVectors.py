
from __future__ import division, print_function
from scipy.misc import imresize
from keras.applications import vgg16, vgg19, inception_v3, resnet50, xception
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import os


DATA_DIR = os.path.abspath("data")
IMAGE_DIR = os.path.join(DATA_DIR, "images", "jpg")


def image_batch_generator(image_names, batch_size):
    num_batches = len(image_names) // batch_size
    for i in range(num_batches):
        batch = image_names[i * batch_size: (i + 1) * batch_size]
        yield batch
    batch = image_names[(i + 1) * batch_size:]
    yield batch


def vectorize_images(image_dir, image_size, preprocessor,
                     model, vector_file, batch_size=32):
    image_names = os.listdir(image_dir)
    num_vecs = 0
    fvec = open(vector_file, "w")
    for image_batch in image_batch_generator(image_names, batch_size):
        batched_images = []
        for image_name in image_batch:
            image = plt.imread(os.path.join(image_dir, image_name))
            image = imresize(image, (image_size, image_size))
            batched_images.append(image)
        X = preprocessor(np.array(batched_images, dtype="float32"))
        vectors = model.predict(X)
        for i in range(vectors.shape[0]):
            if num_vecs % 100 == 0:
                print("{:d} vectors generated".format(num_vecs))
            image_vector = ",".join(["{:.5e}".format(v) for v in vectors[i].tolist()])
            fvec.write("{:s}\t{:s}\n".format(image_batch[i], image_vector))
            num_vecs += 1
    print("{:d} vectors generated".format(num_vecs))
    fvec.close()


IMAGE_SIZE = 224
VECTOR_FILE = os.path.join(DATA_DIR, "vgg16-vectors.tsv")

vgg16_model = vgg16.VGG16(weights="imagenet", include_top=True)
vgg16_model.summary()

model = Model(input=vgg16_model.input,
             output=vgg16_model.get_layer("fc2").output)
preprocessor = vgg16.preprocess_input

vectorize_images(IMAGE_DIR, IMAGE_SIZE, preprocessor, model, VECTOR_FILE)
