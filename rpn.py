import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from nms import nms
from bbox_utils import bbox2offset, offset2bbox, clip_boxes, filter_boxes
import time

class RPN(nn.Module):

    def  __init__(self, input_img_size, feature_extractor_out_dim, feature_extractor_size, receptive_field_size, device):
    
        super(RPN, self).__init__()

        ### Information about the feature extractor ###
        self.feature_extractor_size = feature_extractor_size
        self.receptive_field_size = receptive_field_size
        self.input_img_size = input_img_size # (n_rows, n_cols)
        ###############################################

        ### Anchor related attributes ###
        # acredito que ao implementar a resnet como feature extractor pode melhorar colocando mais ratios e scales, ficou ruim do jeito que esta
        self.anchor_ratios = [0.8, 1, 1.2]  #[1] #[0.5, 1, 2] 
        self.anchor_scales = [3.5, 4, 4.5] #[4] #[8, 16, 32]
        self.k = len(self.anchor_scales) * len(self.anchor_ratios)
        self.anchors_parameters, self.valid_anchors = self._get_anchors_parameters()
        self.anchors_parameters, self.valid_anchors = self.anchors_parameters.to(device), self.valid_anchors.to(device)
        #################################

        self.out_dim = 24 # why ? ahco que eu escolhi aleatoriamente
        
        self.conv_rpn = nn.Conv2d(in_channels=feature_extractor_out_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1, bias=True)

        self.cls_layer = nn.Conv2d(self.out_dim, self.k * 2, kernel_size=1, stride=1, padding=0)
        self.reg_layer = nn.Conv2d(self.out_dim, self.k * 4, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # x -> (batch_size, feature_extractor_out_dim, 64, 64)
        
        x = F.relu(self.conv_rpn(x))
        # x -> (batch_size, feature_extractor_out_dim, 64, 64)
    
        ### Compute the probability to be an object ###

        cls_out = self.cls_layer(x)
        # cls_out -> (batch_size, k * 2, 64, 64)
        # para cada ancora, prob de obj e ~obj

        batch_size, _, _, _ = cls_out.size()
        cls_out = cls_out.permute(0, 2, 3, 1).reshape((batch_size, -1, 2))
        # cls_out -> (batch_size, 64 * 64 * k, 2)
        ###############################################

        ### Compute the object proposals ###

        reg_out = self.reg_layer(x)
        # reg_out -> (batch_size, k * 4, 64, 64)

        reg_out = reg_out.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) # acredito que n precise do permute, talvez tirando fique mais rapido pois esse reshape esta mudando o tensor pois n eh contiguo - idem para o clss_out
        # reg_out -> (batch_size, 64 * 64 * k, 4)

        proposals, cls_out = self._anchors2proposals(reg_out, cls_out)
        # proposals -> (batch_size, k * 4, 64, 64)

        ####################################

        ####################################

        bboxes = offset2bbox(proposals)
        bboxes = clip_boxes(bboxes, self.input_img_size)

        probs_object = F.softmax(cls_out, dim=2)[:, :, 1] # it is 1 and not zero ! 
        # probs_object -> (batch_size, 64 * 64 * k)
        
        bboxes, probs_object = filter_boxes(bboxes, probs_object)
        # bboxes -> (batch_size, -1, 4)
        # probs_object -> (batch_size, -1)
        
        bboxes, probs_object = nms(bboxes, probs_object)
        # bboxes -> (batch_size, -1, 4)
        # probs_object -> (batch_size, -1)

        filtered_proposals = bbox2offset(bboxes)
        # filtered_proposals -> (batch_size, -1, 4)

        ####################################

        return proposals, cls_out, filtered_proposals, probs_object


    def _anchors2proposals(self, reg_out, cls_out):
        """
        anchors: (N, 4) ndarray of float : anchors
        reg_out: (self.k, 4) ndarray of float
        reg_out -> [tx, ty, tw, th]

        pxi = axi + awi * txi
        pyi = ayi + ahi * tyi
        pwi = awi * exp(twi)
        phi = ahi * exp(thi)

        """

        cls_out = cls_out[:, self.valid_anchors, :]

        ax = self.anchors_parameters[self.valid_anchors, 0]
        ay = self.anchors_parameters[self.valid_anchors, 1]
        aw = self.anchors_parameters[self.valid_anchors, 2]
        ah = self.anchors_parameters[self.valid_anchors, 3]

        tx = reg_out[:, self.valid_anchors, 0]
        ty = reg_out[:, self.valid_anchors, 1]
        tw = reg_out[:, self.valid_anchors, 2]
        th = reg_out[:, self.valid_anchors, 3]

        px = ax + aw * tx
        py = ay + ah * ty
        pw = aw * torch.exp(tw)
        ph = ah * torch.exp(th)

        proposals = torch.stack((px, py, pw, ph), dim=2).to(reg_out.device) # esse negocio de toda hora jogar pra device ta ruim.. 
        
        return proposals, cls_out


    def _get_anchors_parameters(self):

        # print('pay attention with the order w,h or h,w')
        # print('WARNING: implementation differ from source_base 1 and source_base 2 !')
        # print('TODO: jogar tudo pra pytorch')
        # print('Ta com -1 no h e w')
        # print('check all types and dims !')
        # print('TODO: tentar fazer com broadcast sem ter que gerar a matrix toda')
        
        # 16 = anchor base
        # ah * aw = 16*16 = 256
        # ah / aw = r -> ah * 1/aw = r
        # 
        # ah = r * aw       (2)
        # r * aw * aw = 256
        # aw^2 = 256/r
        # aw = sqrt(256/r)  (1)
        # ou seja, encontrar ah e aw para que mude o aspect ratio porem nao mude a area coberta pela ancora !

        base_anchor_area = self.receptive_field_size * self.receptive_field_size
        base_anchor_center_cols = 0.5 * self.receptive_field_size
        base_anchor_center_rows = 0.5 * self.receptive_field_size

        anchors = []

        for r in self.anchor_ratios:
            
            anchor_n_cols = np.round(np.sqrt(base_anchor_area / r))
            anchor_n_rows = np.round(anchor_n_cols * r)

            anchor_n_cols_mid = 0.5 * (anchor_n_cols - 1)
            anchor_n_rows_mid = 0.5 * (anchor_n_rows - 1)

            anchor_col_0 = base_anchor_center_cols - anchor_n_cols_mid
            anchor_col_1 = base_anchor_center_cols + anchor_n_cols_mid

            anchor_row_0 = base_anchor_center_rows - anchor_n_rows_mid
            anchor_row_1 = base_anchor_center_rows + anchor_n_rows_mid

            anchors.append([anchor_col_0, anchor_row_0, anchor_col_1, anchor_row_1])

        final_anchors = []

        for a in anchors:

            aw = a[2] - a[0] + 1
            ah = a[3] - a[1] + 1
            acw = a[0] + 0.5 * (aw - 1)
            ach = a[1] + 0.5 * (ah - 1)

            for s in self.anchor_scales:
                
                anchor_col_0 = acw - 0.5 * (aw * s - 1.0)
                anchor_col_1 = acw + 0.5 * (aw * s - 1.0)

                anchor_row_0 = ach - 0.5 * (ah * s - 1.0)
                anchor_row_1 = ach + 0.5 * (ah * s - 1.0)

                final_anchors.append([anchor_col_0, anchor_row_0, anchor_col_1, anchor_row_1])

        anchors = np.array(final_anchors)

        final_anchors = []

        for a in anchors:

            aw = a[2] - a[0] + 1
            ah = a[3] - a[1] + 1
            acw = a[0] + 0.5 * (aw - 1)
            ach = a[1] + 0.5 * (ah - 1)

            final_anchors.append([acw, ach, aw, ah])

        anchors = np.array(final_anchors, dtype=np.float32)
        anchors = anchors.reshape(-1)

        n_anchors = 4 * len(self.anchor_ratios) * len(self.anchor_scales)
        all_anchors = np.zeros((n_anchors, self.feature_extractor_size, self.feature_extractor_size), dtype=np.float32)
        for k in range(0, n_anchors, 4):
            for i in range(0, self.feature_extractor_size):
                for j in range(0, self.feature_extractor_size):
                    all_anchors[k + 0, i, j] = anchors[k + 0] + i * self.receptive_field_size 
                    all_anchors[k + 1, i, j] = anchors[k + 1] + j * self.receptive_field_size
                    all_anchors[k + 2, i, j] = anchors[k + 2]
                    all_anchors[k + 3, i, j] = anchors[k + 3]

        anchors = torch.from_numpy(all_anchors)
        anchors = anchors.permute(1, 2, 0).reshape(-1, 4)

        valid_mask = np.zeros(anchors.size(0), dtype=np.uint8)
        for i in range(anchors.size(0)):

            acw = anchors[i, 0]
            ach = anchors[i, 1]
            aw = anchors[i, 2]
            ah = anchors[i, 3]

            a0 = acw - 0.5 * (aw - 1)
            a1 = ach - 0.5 * (ah - 1)
            a2 = aw + a0 - 1
            a3 = ah + a1 - 1

            if a0 >= 0 and a1 >= 0 and a2 <= self.input_img_size[1] - 1 and a3 <= self.input_img_size[0] - 1:
                valid_mask[i] = 1

        valid_mask = torch.from_numpy(valid_mask)

        return anchors, valid_mask #anchors_center_cols_offset, anchors_center_rows_offset, aw, ah

