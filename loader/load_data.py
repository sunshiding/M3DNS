# -*- coding: utf-8 -*-
import os
import pickle
import torch
import numpy as np

def load_data(data_name,ratio):
    train_id = np.load("{}/train_id.npy".format(path))
    semi_id = np.load("{}/semi_id.npy".format(path))
    test_id = np.load("{}/test_id.npy".format(path))

    return [all_label_data, text_label_data, img_label_data, all_semi_data, text_semi_data, img_semi_data], test_data

def get_batch(data,indices):

    return batch_data