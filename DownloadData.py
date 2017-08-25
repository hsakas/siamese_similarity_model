from urllib.request import urlretrieve
import os
import tarfile


DATA_DIR = os.path.abspath("data")
print('data dir', DATA_DIR)

if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR)
    print('VGG directory created!')

# check if the model trained parameters file is present
if not os.path.isfile(os.path.join(DATA_DIR, "jpg1.tar.gz")):
    try:
        print("Downloading data...")
        urlretrieve(
            'ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg1.tar.gz',
            os.path.join(DATA_DIR, "jpg1.tar.gz"))
        print('\nDone!')

        print('Extracting data...')
        with tarfile.open(os.path.join(DATA_DIR, "jpg1.tar.gz"), 'r:gz') as tar:
            tar.extractall(path=os.path.join(DATA_DIR, "images"))
        print('Done!')
    except:
        if os.path.isfile(os.path.join(DATA_DIR, "jpg1.tar.gz")):
            os.remove(os.path.join(DATA_DIR, "jpg1.tar.gz"))
else:
    print("Parameter file already exists!")
