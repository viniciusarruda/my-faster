import torch
from torch import nn
import torch.nn.functional as F
import config

class FeatureExtractorNet(nn.Module):
    def  __init__(self):
        super(FeatureExtractorNet, self).__init__()

        h, w = config.input_img_size
        self.max_h = h // 4
        self.max_w = w // 4
        assert h == w

        self.out_dim = 12
        self.receptive_field_size = 16 #4 # 2 ^ number_of_maxpool_stride_2
        self.feature_extractor_size = h // self.receptive_field_size #14 #32

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0)
        self.linear1 = nn.Linear(1*self.max_h*self.max_w, 128)
        self.linear2 = nn.Linear(128, self.out_dim*self.feature_extractor_size*self.feature_extractor_size)

    def forward(self, x):

        # print(x.size())
        x = self.conv1(x)
        # print(x.size())
        x = F.max_pool2d(x, 4)
        # print(x.size())
        x = x.view(-1, 1*self.max_h*self.max_w)
        # print(x.size())
        x = self.linear1(x)
        x = torch.sigmoid(x)
        # print(x.size())
        x = self.linear2(x)
        # print(x.size())
        x = torch.sigmoid(x)
        # print(x.size())
        x = x.view(-1, self.out_dim, self.feature_extractor_size, self.feature_extractor_size)
        # print(x.size())
        # exit()

        return x


# class FeatureExtractorNet(nn.Module):
#     def  __init__(self):
#         super(FeatureExtractorNet, self).__init__()

#         # self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
#         # self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
#         # self.conv3 = nn.Conv2d(12, 24, 3, padding=1)

#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0)
#         self.linear1 = nn.Linear(1*32*32, 128)
#         self.linear2 = nn.Linear(128, 12*32*32)
        
#         self.out_dim = 12
#         # self.conv4 = nn.Conv2d(24, self.out_dim, 3, padding=1)

#         self.receptive_field_size = 4 # 2 ^ number_of_maxpool_stride_2

#     def forward(self, x):

#         # print(x.size())
#         # x = F.relu(self.conv1(x))
#         # print(x.size())
#         # x = F.relu(self.conv2(x))
#         # print(x.size())
#         # x = F.relu(self.conv3(x))
#         # print(x.size())
#         # x = F.relu(self.conv4(x))
#         # print(x.size())
#         # x = F.max_pool2d(x, 2)
        
#         x = self.conv1(x)
#         x = F.max_pool2d(x, 4)

#         x = x.view(-1, 1*32*32)

#         # x = torch.ones((1, 128))
#         x = self.linear1(x)
#         x = torch.sigmoid(x)
#         # print(x.size())
#         x = self.linear2(x)
#         # print(x.size())
#         x = torch.sigmoid(x)
#         # print(x.size())
#         x = x.view(-1, 12, 32, 32)
#         # print(x.size())

#         return x
