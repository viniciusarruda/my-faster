import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math
import config

##
# TODO: Testar:
#           1) Tem que entrar e sair tensores de mesmo shape que antes com a versao nao completa !
#           2) Ver direito quem fica com require_grad False/True
#           3) Ver quando usar o eval() 
#           4) Ver como que faz para passar os parametros treinaveis para o otimizador..no momento esta indo tudo.. n sei se esta certo
#           5) Ver sempre as anchoras ..
#           6) Falta usar o children[-2] depois do roialign
##

# TODO: RCNN_top !

# Acho que na faster n eh o avarage pooling que faz a magica para funcionar com todos os tamanhos e sim o ROI Align/Pooling e variantes

class FeatureExtractorNetComplete(nn.Module):

    def  __init__(self):
        super(FeatureExtractorNetComplete, self).__init__()

        h, w = config.input_img_size
        assert h == w

        full_model = models.resnet18(pretrained=True)

        removed = list(full_model.children())[:-3]
        self.model = torch.nn.Sequential(*removed) # equivalent to RCNN_base

        for l in [0, 1, 4]:
            for param in self.model[l].parameters():
                param.requires_grad = False 

        def set_bn_fix(m):
            if type(m) == nn.BatchNorm2d:
                for p in m.parameters():
                    p.requires_grad=False

        self.model.apply(set_bn_fix)


        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        # self.conv3 = nn.Conv2d(12, 24, 3, padding=1)

        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0)
        # self.linear1 = nn.Linear(1*32*32, 128)
        # self.linear2 = nn.Linear(128, 12*32*32)

        self.out_dim = 256 # 12
        self.receptive_field_size = 16 #4 # 2 ^ number_of_maxpool_stride_2
        self.feature_extractor_size = math.ceil(h / self.receptive_field_size) #14 #32

    # overriding nn.Module.train()
    def train(self, mode=True):
        
        super().train(mode)

        if mode == True:

            self.model.eval()
            self.model[5].train()
            self.model[6].train()

            def set_bn_eval(m):
                if type(m) == nn.BatchNorm2d: 
                    m.eval()

            self.model.apply(set_bn_eval)


    def forward(self, x):

        # print(x.size())
        x = self.model(x)
        # print(x.size())
        # exit()
        return x
        # print(x.size())
        # x = F.relu(self.conv1(x))
        # print(x.size())
        # x = F.relu(self.conv2(x))
        # print(x.size())
        # x = F.relu(self.conv3(x))
        # print(x.size())
        # x = F.relu(self.conv4(x))
        # print(x.size())
        # x = F.max_pool2d(x, 2)
        
        # x = self.conv1(x)
        # x = F.max_pool2d(x, 4)

        # x = x.view(-1, 1*32*32)

        # # x = torch.ones((1, 128))
        # x = self.linear1(x)
        # x = torch.sigmoid(x)
        # # print(x.size())
        # x = self.linear2(x)
        # # print(x.size())
        # x = torch.sigmoid(x)
        # # print(x.size())
        # x = x.view(-1, 12, 32, 32)
        # # print(x.size())

        # return x


# for i,e in enumerate(list(full_model.children())):
#     print(i)
#     print(e)
#     print('\n')
# exit()

        # def pp(m):
        #     classname = m.__class__.__name__
        #     print(classname, classname.find('BatchNorm'), type(m), type(m) == nn.BatchNorm2d, [p.requires_grad for p in m.parameters()])
        #     # if classname.find('BatchNorm') != -1:
        #     #     for p in m.parameters(): p.requires_grad=False
        # self.model.apply(pp)

        # exit()

# ResNet 18
# -> [8, 3, 224, 224]
# (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# -> [8, 64, 112, 112]
# (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# -> [8, 64, 112, 112]
# (2): ReLU(inplace)
# -> [8, 64, 112, 112]
# (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
# -> [8, 64, 56, 56]
# (4): 2 X BasicBlock
# -> [8, 64, 56, 56]
# (5): 2 X BasicBlock
# -> [8, 128, 28, 28]
# (6): 2 X BasicBlock
# -> [8, 256, 14, 14]
# (7): 2 X BasicBlock
# -> [8, 512, 7, 7]
# (8): AvgPool2d(kernel_size=7, stride=1, padding=0)
# -> [8, 512, 1, 1]
# (9): Linear(in_features=512, out_features=1000, bias=True)
# -> [8, 1000]

# ResNet 101
# -> [8, 3, 224, 224]
# (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# -> [8, 64, 112, 112]
# (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# -> [8, 64, 112, 112]
# (2): ReLU(inplace)
# -> [8, 64, 112, 112]
# (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
# -> [8, 64, 56, 56]
# (4): 3 X Bottleneck
# -> [8, 256, 56, 56]
# (5): 4 X Bottleneck
# -> [8, 512, 28, 28]
# (6): 23 X Bottleneck
# -> [8, 1024, 14, 14]
# (7): 3 X Bottleneck
# -> [8, 2048, 7, 7]
# (8): AvgPool2d(kernel_size=7, stride=1, padding=0)
# -> [8, 2048, 1, 1]
# (9): Linear(in_features=2048, out_features=1000, bias=True)
# -> [8, 1000]

# remove o último modulo de um modelo
# removed = list(model_ft.children())[:-2]
# model_ft = torch.nn.Sequential(*removed)

# if __name__ == "__main__":
#     fe = FeatureExtractorNetComplete()
#     print(dir(fe))