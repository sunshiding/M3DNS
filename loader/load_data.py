# -*- coding: utf-8 -*-
import os
import pickle
import torch
import numpy as np
from loader import data_wzry
from loader import data_coco
from loader import data_flickr
from loader import data_iapr
from loader import data_nus
from loader import common

def load_data(hp):
    datadir = '/data/yangy/data_prepare/'
    if hp['dataname'].lower() == "wzry":
        datadir += 'wzry/'
        train_data, test_data = data_wzry.load_data(datadir,hp)
    elif hp['dataname'].lower() == "coco":
        datadir += 'COCO/'
        train_data, test_data = data_coco.load_data(datadir,hp)
    elif hp['dataname'].lower() == "flickr":
        datadir += 'mirflickr/'
        train_data, test_data = data_flickr.load_data(datadir,hp)
    elif hp['dataname'].lower() == "iapr":
        datadir += 'iapr/'
        train_data, test_data = data_iapr.load_data(datadir,hp)
    elif hp['dataname'].lower() == "nus":
        datadir += 'NUS-WIDE/'
        train_data, test_data = data_nus.load_data(datadir,hp)
    return train_data, test_data

def get_batch(data,indices,hp):
    if hp['dataname'].lower() == "wzry":
        batch_data = data_wzry.get_batch(data,indices)
    else:
        batch_data = common.get_batch(data,indices)
    return batch_data
