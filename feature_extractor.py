from torch import nn
import torch.nn.functional as F

class FeatureExtractorNet(nn.Module):
    def  __init__(self):
        super(FeatureExtractorNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, 24, 3, padding=1)
        
        self.out_dim = 12
        self.conv4 = nn.Conv2d(24, self.out_dim, 3, padding=1)

        self.receptive_field_size = 2 # 2 ^ number_of_maxpool_stride_2

    def forward(self, x):

        # print(x.size())
        x = F.relu(self.conv1(x))
        # print(x.size())
        x = F.relu(self.conv2(x))
        # print(x.size())
        x = F.relu(self.conv3(x))
        # print(x.size())
        x = F.relu(self.conv4(x))
        # print(x.size())
        x = F.max_pool2d(x, 2)
        # print(x.size())
        
        return x