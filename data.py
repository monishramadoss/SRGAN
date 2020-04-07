import os
from tqdm.auto import tqdm
import urllib.request
import zipfile

DEVSET_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
TRAINSET_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DEVSET = "./data/DIV2K_valid_HR.zip"
TRAINSET = "./data/DIV2K_train_HR.zip"
DEVDATA_FOLDER = "./data/DIV2K_valid_HR"
TRAINDATA_FOLDER = "./data/DIV2K_train_HR"

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


if not os.path.exists('./data'):
    os.makedirs('./data')
if not os.path.exists('./data/train'):
    os.makedirs('./data/train')
if not os.path.exists('./data/dev'):
    os.makedirs('./data/dev')
if not os.path.exists('./checkpoint'):
    os.makedirs('./checkpoint')


if not os.path.exists(TRAINSET):
    with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc="Div2k Train Set") as t:
        urllib.request.urlretrieve(TRAINSET_URL, TRAINSET, reporthook=t.update_to)
    with zipfile.ZipFile(TRAINSET, 'r') as zip_ref:
        zip_ref.extractall('./data/train')
if not os.path.exists(DEVSET):
    with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc="Div2k Valid Set") as t:
        urllib.request.urlretrieve(DEVSET_URL, DEVSET, reporthook=t.update_to)
    with zipfile.ZipFile(DEVSET, 'r') as zip_ref:
        zip_ref.extractall('./data/dev')
