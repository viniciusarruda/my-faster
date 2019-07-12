import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class RPN(nn.Module):

    def  __init__(self, input_img_size, feature_extractor_out_dim, receptive_field_size):
    
        super(RPN, self).__init__()

        ### Information about the feature extractor ###
        self.receptive_field_size = receptive_field_size
        self.input_img_size = input_img_size # w,h
        ###############################################

        ### Anchor related attributes ###
        self.anchor_ratios = [0.5, 1, 2] 
        self.anchor_scales = [8, 16, 32]
        self.k = len(self.anchor_scales) * len(self.anchor_ratios)
        self.anchors = self._get_anchors()
        self.anchors_parameters = self._get_anchors_parameters()
        #################################

        self.out_dim = 24
        self.conv_rpn = nn.Conv2d(in_channels=feature_extractor_out_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1, bias=True)

        self.cls_layer = nn.Conv2d(self.out_dim, self.k * 2, kernel_size=1, stride=1, padding=0)
        self.reg_layer = nn.Conv2d(self.out_dim, self.k * 4, kernel_size=1, stride=1, padding=0)


    def forward(self, x):

        x = F.relu(self.conv_rpn(x))

        cls_out = self.cls_layer(x)
        batch_size, _, w, h = cls_out.size()
        cls_out = cls_out.reshape((batch_size, w, h, self.k, 2))
        prob_object = F.softmax(cls_out, dim=4)[:, :, :, :, 0] # select just the probability of be a object
        prob_object = prob_object.reshape((batch_size, w, h, self.k))
        
        reg_out = self.reg_layer(x)
        proposals = self._anchors2proposals(reg_out)
        # proposals to bbox..

        ### NMS ###
        # TODO: Remove the for loop somehow !
        for i in range(batch_size):

            i_prob_object = prob_object[i].view(-1)
            i_proposals = proposals[i].view(4, -1)

            idxs = torch.argsort(i_prob_object, descending=True)

            i_prob_object = i_prob_object[idxs]
            i_proposals = i_proposals[:, idxs]

            n_proposals = idxs.size()[0]

            ### Remove iou > 0.7 ###
            # [tx, ty, tw, th]
            area_0 = i_proposals[2, 0] * i_proposals[3, 0]
            x0_0, y0_0, x1_0, y1_0 = i_proposals[0, 0], i_proposals[1, 0], i_proposals[0, 0] + i_proposals[2, 0], i_proposals[1, 0] + i_proposals[3, 0] 
            # print(area_0)
            for j in range(1, n_proposals):

                j = 1110
                x0_j, y0_j, x1_j, y1_j = i_proposals[0, j], i_proposals[1, j], i_proposals[0, j] + i_proposals[2, j], i_proposals[1, j] + i_proposals[3, j] 
                
                print(x0_0, y0_0, x1_0, y1_0)
                print(x0_j, y0_j, x1_j, y1_j)
                
                x0 = torch.max(x0_0, x0_j)
                y0 = torch.max(y0_0, y0_j)
                x1 = torch.min(x1_0, x1_j)
                y1 = torch.min(y1_0, y1_j)
                intersection = (x1 - x0) * (y1 - y0)
                w = max(x1 - x0, 0)
                h = max(y1 - y0, 0)
                intersection = 
                area_j = i_proposals[2, j] * i_proposals[3, j]
                
                union = area_0 + area_j - intersection
                iou = intersection / union
                
                print(iou, j)
                exit()

            # print(i_prob_object.size())
            # print(i_proposals.size())

            # print(i_prob_object[idx])


            exit()
        ### NMS ###

        
        # NMS
        # ROI pooling

        
        return cls_out, reg_out, proposals


    def _anchors2proposals(self, reg_out):
        """
        anchors: (N, 4) ndarray of float : anchors
        reg_out: (self.k, 4) ndarray of float
        reg_out -> [tx, ty, tw, th]

        pxi = axi + awi * txi
        pyi = ayi + ahi * tyi
        pwi = awi * exp(twi)
        phi = ahi * exp(thi)

        """
        ax, ay, aw, ah = self.anchors_parameters
    
        tx = reg_out[:, 0::4, :, :]
        ty = reg_out[:, 1::4, :, :]
        tw = reg_out[:, 2::4, :, :]
        th = reg_out[:, 3::4, :, :]

        px = ax + aw * tx
        py = ay + ah * ty
        pw = aw * torch.exp(tw)
        ph = ah * torch.exp(th)

        proposals = torch.cat((px, py, pw, ph), dim=1)

        return proposals


    def _get_anchors_parameters(self):

        print('pay attention with the order w,h or h,w')
        w, h = self.input_img_size

        # print('working with pixels !')
        # print('TODO: jogar tudo pra pytorch')
        anchors_center_width_offset = np.arange(0, w, self.receptive_field_size) + self.receptive_field_size // 2
        anchors_center_height_offset = np.arange(0, h, self.receptive_field_size) + self.receptive_field_size // 2

        ws = w // self.receptive_field_size
        hs = h // self.receptive_field_size

        anchors_center_width_offset = np.tile(anchors_center_width_offset.astype(np.float32, copy=False).reshape(1, 1, 1, -1), (hs, 1))
        anchors_center_height_offset = np.tile(anchors_center_height_offset.astype(np.float32, copy=False).reshape(1, 1, -1, 1), (1, ws))


        # wtmp = np.array([s / r for r in self.anchor_ratios for s in self.anchor_scales], dtype=np.float32).reshape((1, 1, -1))
        # htmp = np.array([s * r for r in self.anchor_ratios for s in self.anchor_scales], dtype=np.float32).reshape((1, 1, -1))

        # w = np.tile(wtmp * self.receptive_field_size, (ws, hs, 1))
        # h = np.tile(htmp * self.receptive_field_size, (ws, hs, 1))

        # print(anchors_center_width_offset.shape)
        # print(anchors_center_height_offset.shape)
        # print(w.shape)
        # print(h.shape)
        # exit()

        aw = np.array([s / r for r in self.anchor_ratios for s in self.anchor_scales], dtype=np.float32).reshape((1, -1, 1, 1))
        ah = np.array([s * r for r in self.anchor_ratios for s in self.anchor_scales], dtype=np.float32).reshape((1, -1, 1, 1))

        anchors_center_width_offset = torch.from_numpy(anchors_center_width_offset)
        anchors_center_height_offset = torch.from_numpy(anchors_center_height_offset)
        aw = torch.from_numpy(aw)
        ah = torch.from_numpy(ah)

        return anchors_center_width_offset, anchors_center_height_offset, aw, ah



    def _get_anchors(self):

        print('WARNING: implementation differ from source_base 1 and source_base 2 !')
        print('TODO: jogar tudo pra pytorch')
        print('Ta com -1 no h e w')

        anchors = np.zeros((self.k, 4))
        for i in range(len(self.anchor_ratios)):
            for j in range(len(self.anchor_scales)):
                h = self.receptive_field_size * self.anchor_scales[j] * self.anchor_ratios[i]
                w = self.receptive_field_size * self.anchor_scales[j] / self.anchor_ratios[i]

                ph = h / 2.
                pw = w / 2.

                index = i * len(self.anchor_scales) + j
                anchors[index, 0] = -(ph - 1)
                anchors[index, 1] = -(pw - 1)
                anchors[index, 2] = ph
                anchors[index, 3] = pw
        return anchors


