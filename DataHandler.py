import os
import numpy as np
import itertools
from keras.utils import np_utils
from scipy.misc import imresize
import matplotlib.pyplot as plt


def create_image_triples(image_dir):
    image_groups = {}
    for image_name in os.listdir(image_dir):
        base_name = image_name[0:-4]
        group_name = base_name[0:4]
        if group_name in image_groups.keys():
            image_groups[group_name].append(image_name)
        else:
            image_groups[group_name] = [image_name]
    num_sims = 0
    image_triples = []
    group_list = sorted(list(image_groups.keys()))
    for i, g in enumerate(group_list):
        if num_sims % 100 == 0:
            print("Generated {:d} pos + {:d} neg = {:d} total image triples"
                  .format(num_sims, num_sims, 2*num_sims))
        images_in_group = image_groups[g]
        sim_pairs_it = itertools.combinations(images_in_group, 2)
        # for each similar pair, generate a corresponding different pair
        for ref_image, sim_image in sim_pairs_it:
            image_triples.append((ref_image, sim_image, 1))
            num_sims += 1
            while True:
                j = np.random.randint(low=0, high=len(group_list), size=1)[0]
                if j != i:
                    break
            dif_image_candidates = image_groups[group_list[j]]
            k = np.random.randint(low=0, high=len(dif_image_candidates), size=1)[0]
            dif_image = dif_image_candidates[k]
            image_triples.append((ref_image, dif_image, 0))
    print("Generated {:d} pos + {:d} neg = {:d} total image triples"
          .format(num_sims, num_sims, 2*num_sims))
    return image_triples


def load_image_triplets(image_dir, image_triples, image_size, shuffle=False):
    # loop once per epoch
    if shuffle:
        indices = np.random.permutation(np.arange(len(image_triples)))
    else:
        indices = np.arange(len(image_triples))
    shuffled_triples = [image_triples[ix] for ix in indices]
    # num_batches = len(shuffled_triples) // batch_size
    # for bid in range(num_batches):
        # loop once per batch
    images_left, images_right, labels = [], [], []
    lhs_images = []
    rhs_images = []
    # batch = shuffled_triples[bid * batch_size: (bid + 1) * batch_size]
    for i in range(len(shuffled_triples)):

        # print(i, 'eh', shuffled_triples, len(shuffled_triples))
        lhs, rhs, label = shuffled_triples[i]

        lhs = load_image(image_dir=image_dir, image_name=lhs, image_size=image_size)
        rhs = load_image(image_dir=image_dir, image_name=rhs, image_size=image_size)

        # lhs = np.array(lhs).reshape((1, image_size*image_size, 3))/255
        # rhs = np.array(rhs).reshape((1, image_size*image_size, 3))/225
        # pairs += [[lhs, rhs]]
        lhs_images += [lhs]
        rhs_images += [rhs]
        labels.append(label)
    Y = np_utils.to_categorical(np.array(labels), num_classes=2)
    return (np.array(lhs_images), np.array(rhs_images), Y)


image_cache = {}


def load_image(image_dir, image_name, image_size):
    if not (image_name in image_cache.keys()):
        image = plt.imread(os.path.join(image_dir, image_name)).astype(np.float32)
        image = imresize(image, (image_size, image_size))
        image = np.divide(image, 256)
        image_cache[image_name] = image
    return image_cache[image_name]

