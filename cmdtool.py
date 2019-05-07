import os
import time
from argparse import ArgumentParser

all_datas = ['wzry','flickr','iapr','nus','coco']
label_num = {'wzry':80, 'flickr': 30,'iapr': 27,'nus': 30,'coco': 23}

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
    parser.add_argument('--fixed',type=str,default='0')
    parser.add_argument('--ratio',type=str,default='0')
    parser.add_argument('--semi',type=str,default='1')
    parser.add_argument('--lr',type=str,default='6')
    parser.add_argument('--pretrain',type=int,default=1)
    parser.add_argument('--screen',type=int,default=1)
    args = parser.parse_args()
    
    datas = parse_datas(args.data)
    gpus = parse_gpu(args.gpu)
    semis = parse_gpu(args.semi)
    ratios = parse_gpu(args.ratio)
    fixeds = parse_gpu(args.fixed)
    lrs = ["1e-" + w for w in parse_gpu(args.lr)
    time_str = time.strftime("%m%d-%H%M",time.localtime(time.time()))
    
    for i in range(len(datas)):
        data = datas[i]
        if i > len(gpus) - 1:
            gpu = gpus[-1]
        else:
            gpu = gpus[i]
        for semi in semis:
            for fixed in fixeds:
                for ratio in ratios:
                    for lr in lrs:
                        screen_name = "{}-{}-{}-{}-{}-{}-{}".format(data,gpu,semi,fixed,ratio,lr,time_str)
                        test_size = 16 if data == "wzry" else 32
                        #if args.semi == 1 and args.fixed == 0 and args.ratio == 0:
                        #    epoch = 2
                        #else:
                        epoch = 2
                        cmdstr = "python run.py --gpu={} --dataname={} --seed=0 --pre_epoch=20 --pre_size=32 --epoch={} --epoch_1=2 --batch_size=32 --test_size={} --label={} --fixed={} --semi={} --pretrain={} --ratio={} --lr={}".format(gpu,data,str(epoch),str(test_size),str(label_num[data]), fixed, semi, args.pretrain, str(ratio),lr)
                        if args.screen == 1:
                            os.system("screen -dmS {}".format(screen_name))
                            os.system('screen -X -S {} -p 0 -X stuff "{}"'.format(screen_name,cmdstr))
                            os.system('screen -X -S {} -p 0 -X stuff "\n"'.format(screen_name))
                        else:
                            os.system(cmdstr)
