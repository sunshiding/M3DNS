import os
import time
from argparse import ArgumentParser

all_datas = ['wzry','flickr','iapr','nus','coco']
label_num = {'wzry':54, 'flickr': 20,'iapr': 20,'nus': 20,'coco': 20}

def parse_datas(datas):
    if datas == "all":
        return all_datas
    else:
        if "-" in datas:
            data_list = datas.split('-')
            return [all_datas[int(d)] for d in data_list]
        else:
            return [all_datas[int(datas)]]
def parse_gpu(gpus):
    if "-" in gpus:
        gpu_list = gpus.split('-')
        return gpu_list
    else:
        return [gpus]

if __name__ == '__main__':
    parser  = ArgumentParser('M3DN')
    parser.add_argument('--gpu',type=str)
    parser.add_argument('--data',type=str)
    parser.add_argument('--fixed',type=int,default=0)
    parser.add_argument('--ratio',type=float,default=0)
    args = parser.parse_args()
    
    datas = parse_datas(args.data)
    gpus = parse_gpu(args.gpu)
    time_str = time.strftime("%m%d-%H%M",time.localtime(time.time()))
    
    for i in range(len(datas)):
        data = datas[i]
        gpu = gpus[i]
        screen_name = data + "-" + time_str + "-" + gpu
        cmdstr = "python run.py --gpu={} --dataname={} --seed=0 --pre_epoch=20 --pre_size=32 --epoch=20 --epoch_1=2 --batch_size=16 --label={} --fixed={} --pretrain=1 --ratio={}".format(gpu,data,str(label_num[data]),args.fixed,str(args.ratio))
        os.system("screen -dmS {}".format(screen_name))
        os.system('screen -X -S {} -p 0 -X stuff "{}"'.format(screen_name,cmdstr))
        os.system('screen -X -S {} -p 0 -X stuff "\n"'.format(screen_name))
