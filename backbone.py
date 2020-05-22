import torch
import torch.nn as nn
from torchvision import models
import math
import config


#
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


class ResNetBackbone(nn.Module):

    def __init__(self):
        super(ResNetBackbone, self).__init__()

        w, h = config.input_img_size
        # assert h == w

        # resnet 18 -> 512
        # resnet 101 -> 2048
        # https://miro.medium.com/max/1400/1*aq0q7gCvuNUqnMHh4cpnIw.png
        n_layers = int(config.backbone.replace('ResNet', ''))
        if n_layers == 18:
            full_model = models.resnet18(pretrained=True)
            self.out_dim = 256
            top_out_channels = 512
        elif n_layers == 50:
            full_model = models.resnet50(pretrained=True)
            self.out_dim = 1024
            top_out_channels = 2048
        else:
            raise NotImplementedError('{} is not implemented!'.format(config.backbone))

        # l = list(full_model.children())
        # for e in l:
        #     print(e)
        #     print()
        # print(len(l))
        # exit()

        child_list = list(full_model.children())
        self.base = torch.nn.Sequential(*child_list[:7])  # equivalent to RCNN_base
        self.top = torch.nn.Sequential(child_list[7])     # equivalent to RCNN_top
        self.cls = nn.Linear(top_out_channels, config.n_classes)
        self.reg = nn.Linear(top_out_channels, 4 * (config.n_classes - 1))

        for l in [0, 1, 4]:
            for param in self.base[l].parameters():
                param.requires_grad = False

        def set_bn_fix(m):
            if type(m) == nn.BatchNorm2d:
                for p in m.parameters():
                    p.requires_grad = False

        self.base.apply(set_bn_fix)
        self.top.apply(set_bn_fix)

        self.receptive_field_size = 16  # 4 # 2 ^ number_of_maxpool_stride_2
        self.feature_extractor_size = (math.ceil(w / self.receptive_field_size), math.ceil(h / self.receptive_field_size))  # 14 #32

    # overriding
    def train(self, mode=True):

        super().train(mode)  # essential

        if mode is True:

            self.base.eval()
            self.base[5].train()
            self.base[6].train()

            def set_bn_eval(m):
                if type(m) == nn.BatchNorm2d:
                    m.eval()

            self.base.apply(set_bn_eval)
            self.top.apply(set_bn_eval)

    def top_cls_reg(self, rois):

        x = self.top(rois).mean((2, 3))
        raw_cls = self.cls(x)
        raw_reg = self.reg(x)

        return raw_reg, raw_cls


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out


class ToyBackbone(nn.Module):
    def __init__(self):
        super(ToyBackbone, self).__init__()

        w, h = config.input_img_size
        self.max_h = h // 4
        self.max_w = w // 4
        # assert h == w

        self.out_dim = 12
        self.receptive_field_size = 16  # 4 # 2 ^ number_of_maxpool_stride_2
        self.feature_extractor_size = (w // self.receptive_field_size, h // self.receptive_field_size)  # 14 #32

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0)
        self.linear1 = nn.Linear(1 * self.max_h * self.max_w, 128)
        self.linear2 = nn.Linear(128, self.out_dim * self.feature_extractor_size[0] * self.feature_extractor_size[1])

        self.base = nn.Sequential(self.conv1,
                                  nn.MaxPool2d(4),
                                  View((1 * self.max_h * self.max_w,)),
                                  self.linear1,
                                  nn.Sigmoid(),
                                  self.linear2,
                                  nn.Sigmoid(),
                                  View((self.out_dim, self.feature_extractor_size[1], self.feature_extractor_size[0])))

        self.top = nn.Linear(7 * 7 * self.out_dim, 4096)
        self.cls = nn.Linear(4096, config.n_classes)  # background is already included in config.n_classes
        self.reg = nn.Linear(4096, 4 * (config.n_classes - 1))

    def top_cls_reg(self, rois):

        x = self.top(rois.view(rois.size(0), -1))
        raw_cls = self.cls(x)
        raw_reg = self.reg(x)

        return raw_reg, raw_cls


if __name__ == "__main__":
    fe = ResNetBackbone()
    print(dir(fe))
