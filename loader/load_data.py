# -*- coding: utf-8 -*-
import os
import pickle
import torch
import numpy as np
from loader import data_wzry
def load_data(hp):
    datadir = '/data/yangy/data_prepare/'
    if hp['dataname'].lower() == "wzry":
        datadir += 'wzry/'
        train_data, test_data = data_wzry.load_data(datadir,hp)
    return train_data, test_data

def get_batch(data,indices,hp):
    if hp['dataname'].lower() == "wzry":
        batch_data = data_wzry.get_batch(data,indices)
    return batch_data
