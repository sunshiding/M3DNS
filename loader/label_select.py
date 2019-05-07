import os
import numpy as np
import pickle
import csv

def select(path,hp):
    num = hp['label']
    if hp['dataname'].lower() == "coco":
        label = np.load(path+'coco_label.npy')
        count = np.sum(label,axis = 0)
    elif hp['dataname'].lower() == "flickr":
        label = np.load(path + 'label_data.npy')
        count = np.sum(label,axis = 0)
    elif hp['dataname'].lower() == "nus":
        label = np.load(path+'label.npy')
        count = np.sum(label,axis = 0)
    elif hp['dataname'].lower() == "iapr":
        error_id = ['4072']
        count = np.zeros(255)
        with open(path + "label.csv", "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                for i in range(1, len(row)):
                    count[int(row[i])] += 1
    elif hp['dataname'].lower() == "wzry":
        count = np.zeros(1744)
        data_id = list(range(121))
        for i in range(len(data_id)):
            pkl_id = data_id[i]
            filepath = path + 'wzry-new-' + str(pkl_id) + '.pkl'
            if os.path.exists(filepath) is True:
                data_temp = pickle.load(open(filepath, 'rb'))
                for key in data_temp.keys():
                    single_data = data_temp[key]
                    count += single_data[-1].reshape(-1)
    label_select = []
    for i in range(num):
        index = np.argmax(count)
        label_select.append(index)
        count[index] = 0
    label_select.sort()
    print(label_select)
    np.save(hp['rootdir']+'label_select.npy',label_select)

    return label_select
