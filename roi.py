import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ROI(nn.Module):

    def  __init__(self, input_img_size):
    
        super(ROI, self).__init__()

        self.out_dim = 7
        self.input_img_size = input_img_size


    def forward(self, proposals, features):

        fx = features.size()[2] / self.input_img_size[0]
        fy = features.size()[3] / self.input_img_size[1]

        x = proposals[:, :, 0] * fx
        y = proposals[:, :, 1] * fy
        w = proposals[:, :, 2] * fx
        h = proposals[:, :, 3] * fy

        roi = torch.stack((x, y, w, h), dim=2)

        batch_rois = []

        for i in range(roi.size()[0]):
            rois = []
            for k in range(roi.size()[1]):

                x = roi[i, k, 0].long() # long -> torch.int64
                y = roi[i, k, 1].long()
                w = roi[i, k, 2].long()
                h = roi[i, k, 3].long()

                roi_feature = features[i, :, x:x+w, y:y+h][None, :]  # NAO DESISTE !!!

                roi_feature_interpolated = F.interpolate(roi_feature, size=(14, 14),  mode='bilinear', align_corners=True)

                roi_pooled = F.max_pool2d(roi_feature_interpolated, kernel_size=2)

                rois.append(roi_pooled)

            rois = torch.cat(rois, dim=0)
            batch_rois.append(rois)

        rois = torch.stack(batch_rois, dim=0)
                
        return rois
