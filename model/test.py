from measure.measure_average_precision import average_precision
from measure.measure_hamming_loss import hamming_loss
from measure.measure_ranking_loss import ranking_loss
from measure.measure_example_auc import example_auc
from measure.measure_example_f1 import example_f1
from measure.measure_one_error import one_error
from measure.measure_macro_auc import macro_auc
from measure.measure_micro_auc import micro_auc
from measure.measure_macro_f1 import macro_f1
from measure.measure_micro_f1 import micro_f1
from measure.measure_coverage import coverage
from loader.load_data import get_batch

from torch.autograd import Variable
import numpy as np


evaluation = ['average_precision', 'coverage', 'ranking_loss', 'macro_auc', 'micro_auc', 'example_auc',
              'hamming_loss', 'one_error', 'macro_f1', 'micro_f1', 'example_f1']  # 评价指标

view_name = ['Image', 'Text', 'Title']


def test_single_view(x, y, hp):
    result = {}
    if 0 in hp['eval']:
        result[evaluation[0]] = average_precision(x, y)
    if 1 in hp['eval']:
        result[evaluation[1]] = coverage(x, y)
    if 2 in hp['eval']:
        result[evaluation[2]] = ranking_loss(x, y)
    if 3 in hp['eval']:
        result[evaluation[3]] = macro_auc(x, y)
    if 4 in hp['eval']:
        result[evaluation[4]] = micro_auc(x, y)
    if 5 in hp['eval']:
        result[evaluation[5]] = example_auc(x, y)
    if 6 in hp['eval']:
        if 'thread' in hp.keys():
            result[evaluation[6]] = hamming_loss(x, y, thread=hp['thread'])
        else:
            print('No thread for prediction(default = 0.5)!!')
            result[evaluation[6]] = hamming_loss(x, y)
    if 7 in hp['eval']:
        result[evaluation[7]] = one_error(x, y)
    if 8 in hp['eval']:
        if 'thread' in hp.keys():
            result[evaluation[8]] = macro_f1(x, y, thread=hp['thread'])
        else:
            print('No thread for prediction(default = 0.5)!!')
            result[evaluation[8]] = macro_f1(x, y)
    if 9 in hp['eval']:
        if 'thread' in hp.keys():
            result[evaluation[9]] = micro_f1(x, y, thread=hp['thread'])
        else:
            print('No thread for prediction(default = 0.5)!!')
            result[evaluation[9]] = micro_f1(x, y)
    if 10 in hp['eval']:
        if 'thread' in hp.keys():
            result[evaluation[0]] = example_f1(x, y, thread=hp['thread'])
        else:
            print('No thread for prediction(default = 0.5)!!')
            result[evaluation[0]] = example_f1(x, y)
    return result


def test(test_data, hp, models, stage):
    print("----------start testing models----------")
    view_num = len(models)
    for i in range(view_num):
        models[i].cuda()
        models[i].eval()

    # calculate output
    bag_num = len(test_data)
    batch_size = hp['test_size'][0]
    max_step = int(bag_num / batch_size)
    while max_step * batch_size < bag_num:
        max_step += 1

    h = [None for i in range(view_num)]
    label = [None for i in range(view_num)]
    for step in range(max_step):
        # get data
        step_data = get_batch(test_data,list(range(step * batch_size,min((step + 1) * batch_size,bag_num))),hp)
        x1, x2, bag1, bag2, y = step_data
        print('img_bag',bag1)
        print('txt_bag',bag2)
        x_img = Variable(x1, volatile=True).cuda()
        x_text = Variable(x2, volatile=True).cuda()
        h1,_,_ = models[0](x_img,bag1)
        h2,_,_ = models[1](x_text,bag2)

        if step == 0:
            h[0] = h1.cpu().data.numpy()
            label[0] = y.numpy()
        else:
            h[0] = np.concatenate((h[0], h1.cpu().data.numpy()))
            label[0] = np.concatenate((label[0], y.numpy()))

        if step == 0:
            h[1] = h2.cpu().data.numpy()
            label[1] = y.numpy()
        else:
            h[1] = np.concatenate((h[1], h2.cpu().data.numpy()))
            label[1] = np.concatenate((label[1], y.numpy()))

    result = {}
    # test single view
    for i in range(view_num):
        print(i)
        # test
        result[view_name[i]] = test_single_view(h[i], label[i], hp)

        # show test result
        print("test result : ", view_name[i])
        for key in result[view_name[i]].keys():
            print(key, result[view_name[i]][key], '\n')
    np.save('{}predict-single-{}.npy'.format(hp['rootdir'],stage), result)
    # test all view
    h_average = []
    h_max = []
    for i in range(h[0].shape[0]):
        z = []
        for j in range(view_num):
            z.append(h[j][i].reshape(1, -1))
        z = np.concatenate(z)
        h_average.append(np.mean(z, axis=0).reshape(1, -1))
        h_max.append(np.max(z, axis=0).reshape(1, -1))
    h_average = np.concatenate(h_average, axis=0)
    h_max = np.concatenate(h_max, axis=0)

    result['avg'] = test_single_view(h_average, label[0], hp)
    print("test result : average all")
    for key in result['avg'].keys():
        print(key, result['avg'][key], '\n')
    result['max'] = test_single_view(h_max, label[0], hp)
    print("test result : max all")
    for key in result['max'].keys():
        print(key, result['max'][key], '\n')
    print("----------end testing models----------")
    np.save('{}predict-all-{}.npy'.format(hp['rootdir'],stage), result)
    save_result(hp['rootdir'],stage,result)
    return result


def save_result(filepath, stage, result):
    path = "{}test-result-{}.txt".format(filepath,stage)
    with open(path, 'w') as f:    
        for key in result.keys():
            f.write(key + '\n')
            for k in result[key].keys():
                f.write("\t" + k + ' ' + str(result[key][k]) + '\n')
