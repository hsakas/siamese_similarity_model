from keras.models import load_model
import os
import itertools

DATA_DIR = os.path.abspath("data")
IMAGE_DIR = os.path.join(DATA_DIR, "images", "eval")

BATCH_SIZE = 12

model = load_model(os.path.join(DATA_DIR, "models", "vgg16-dot-best.h5"))
for layer in model.layers:
    print(layer.name, layer.input_shape, layer.output_shape)


def get_holiday_triples(image_dir):
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
                  .format(num_sims, num_sims, 2 * num_sims))
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
          .format(num_sims, num_sims, 2 * num_sims))
    return image_triples

image_triples = get_holiday_triples(IMAGE_DIR)


def batch_to_vectors(batch, vec_size, vec_dict):
    X1 = np.zeros((len(batch), vec_size))
    X2 = np.zeros((len(batch), vec_size))
    Y = np.zeros((len(batch), 2))
    for tid in range(len(batch)):
        X1[tid] = vec_dict[batch[tid][0]]
        X2[tid] = vec_dict[batch[tid][1]]
        Y[tid] = [1, 0] if batch[tid][2] == 0 else [0, 1]
    return (X1, X2, Y)

def data_generator(triples, vec_size, vec_dict, batch_size=32):
    while True:
        # shuffle once per batch
        indices = np.random.permutation(np.arange(len(triples)))
        # num_batches = len(triples) // batch_size
        for i in range(len(triples)):
            # batch_indices = indices[bid * batch_size: (bid + 1) * batch_size]
            batch = [triples[i] for i in indices]
            yield batch_to_vectors(batch, vec_size, vec_dict)

def load_vectors(vector_file):
    vec_dict = {}
    fvec = open(vector_file, "r")
    for line in fvec:
        image_name, image_vec = line.strip().split("\t")
        vec = np.array([float(v) for v in image_vec.split(",")])
        vec_dict[image_name] = vec
    fvec.close()
    return vec_dict

VECTOR_SIZE = 4096
VECTOR_FILE = os.path.join(DATA_DIR, "vgg16-eval-vectors.tsv")

vec_dict = load_vectors(VECTOR_FILE)

X1, X2, Y = batch_to_vectors(image_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)

Ytest_ = model.predict([X1, X2])

print('pred', Ytest_)
print('actual', Y)
