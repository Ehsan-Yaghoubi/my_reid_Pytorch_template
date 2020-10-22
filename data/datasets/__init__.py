# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .dataset_loader import ImageDataset
from .ltcc_noneID import LTCC_noneID
from .ltcc_orig import LTCC_Orig

__factory = {
    'ltcc_noneID': LTCC_noneID,
    'ltcc_orig': LTCC_Orig
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
