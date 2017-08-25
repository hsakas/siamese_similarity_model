from urllib.request import urlretrieve
import os
import tarfile
import time
import sys


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        time.sleep(3)
        return

    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
(percent, progress_size / (1024 * 1024), speed, duration))


DATA_DIR = os.path.abspath("../data")

if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR)
    print('VGG directory created!')

# check if the model trained parameters file is present
if not os.path.isfile(os.path.join(DATA_DIR, "jpg1.tar.gz")):
    try:
        print("Downloading data...")
        urlretrieve(
            'ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg1.tar.gz',
            os.path.join(DATA_DIR, "jpg1.tar.gz"), reporthook=reporthook)
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
