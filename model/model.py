import os
import torch
from torch import nn
from torchvision import models

class ImageNet(nn.Module):
    def __init__(self, label_num):
        super(ImageNet, self).__init__()
        self.feature = models.resnet18(pretrained=True)
        self.feature.fc = nn.Linear(512, 256)
        self.encoder = make_layers([256,128,64])
        self.decoder = make_layers([64,128,256])
        self.fc = nn.Linear(64, label_num)
        self.sig = nn.Sigmoid()
        self._initialize_weights()

    def forward(self, x, bag):
        feature = self.feature(x)
        encoder = self.encoder(feature)
        decoder = self.decoder(encoder)
        y = []
        s = 0
        #print(bag)
        x = self.fc(encoder)
        for i in range(len(bag)):
            z = nn.MaxPool2d(kernel_size=(bag[i], 1))(x[s:(s + bag[i])].view(1, 1, bag[i], x.size(1)))
            s += bag[i]
            y.append(z.view(1, -1))
        y = torch.cat(y)
        y = self.sig(y)
        return y, feature.detach(), decoder

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()


class TextNet(nn.Module):
    def __init__(self, neure_num):
        super(TextNet, self).__init__()
        self.encoder = make_layers(neure_num[:3])
        self.decoder = make_layers(neure_num[:3])[::-1]
        self.feature = make_layers(neure_num[2:-1])
        self.fc = nn.Linear(neure_num[-2], neure_num[-1])
        self.sig = nn.Sigmoid()
        self._initialize_weights()

    def forward(self, x, bag):
        encoder = self.encoder(x)
        decoder = self.decoder(x)
        x = self.feature(encoder)
        x = self.fc(x)
        y = []
        s = 0
        for i in range(len(bag)):
            z = nn.MaxPool2d(kernel_size=(bag[i], 1))(x[s:(s + bag[i])].view(1, 1, bag[i], x.size(1)))
            s += bag[i]
            y.append(z.view(1, -1))
        y = torch.cat(y)
        y = self.sig(y)
        return y, x, decoder

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()


def make_layers(cfg):
    layers = []
    n = len(cfg)
    input_dim = cfg[0]
    for i in range(1, n):
        output_dim = cfg[i]
        if i < n - 1:
            layers += [nn.Linear(input_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU(inplace=True)]
        else:
            layers += [nn.Linear(input_dim, output_dim), nn.ReLU(inplace=True)]
        input_dim = output_dim
    return nn.Sequential(*layers)


def load_model(label_num, neure_num, pre_train, rootpath):
    print("----------start loading models----------")
    my_models = [ImageNet(label_num), TextNet(neure_num+[label_num])]
    if pre_train is True:
        for i in range(len(my_models)):
            path = "{}model_{}.pkl".format(rootpath,str(i))
            if os.path.exists(path):
                my_models[i].load_state_dict(torch.load(path))
            else:
                print(path)
                print("No such model parameter !!")

    print("----------end loading models----------")
    print("model information: ")
    for i in range(len(my_models)):
        print(my_models[i])
    return my_models


def save_model(models, rootpath):
    for i in range(len(models)):
        path = "{}model_{}.pkl".format(rootpath,str(i))
        torch.save(models[i].state_dict(), open(path, 'wb'))
