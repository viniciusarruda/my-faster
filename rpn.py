import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class RPN(nn.Module):

    def  __init__(self, input_img_size, feature_extractor_out_dim, receptive_field_size):
    
        super(RPN, self).__init__()

        ### Information about the feature extractor ###
        self.receptive_field_size = receptive_field_size
        self.input_img_size = input_img_size # (n_rows, n_cols)
        ###############################################

        ### Anchor related attributes ###
        self.anchor_ratios = [0.5, 1, 2] 
        self.anchor_scales = [8, 16, 32]
        self.k = len(self.anchor_scales) * len(self.anchor_ratios)
        self.anchors_parameters = self._get_anchors_parameters()
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

        batch_size, _, n_rows, n_cols = cls_out.size()
        cls_out = cls_out.reshape((batch_size, n_rows, n_cols, self.k, 2))
        # cls_out -> (batch_size, 64, 64, k, 2)

        prob_object = F.softmax(cls_out, dim=4)[:, :, :, :, 0] # select just the probability to be an object
        # prob_object -> (batch_size, 64, 64, k)

        ###############################################

        ### Compute the object proposals ###

        reg_out = self.reg_layer(x)
        # reg_out -> (batch_size, k * 4, 64, 64)
        
        proposals = self._anchors2proposals(reg_out)
        # proposals -> (batch_size, k * 4, 64, 64)

        ####################################

        proposals = proposals.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        # proposals -> (batch_size, k * 64 * 64, 4)
        
        prob_object = prob_object.view(batch_size, -1)
        # prob_object -> (batch_size, 64 * 64 * k)

        # torch.int64 -> index
        # torch.uint8 -> true or false (mask)
        cond = proposals[0, :, 2] >= 16.0  # filtrar bbox pequeno deveria ser depois do clip boxes n !!!! 
        proposals = proposals[:, cond, :]  # é depois de clip mesmo.. mas ai vai ter que fazer com formato em bbox e n offset 
        prob_object = prob_object[:, cond]

        cond = proposals[0, :, 3] >= 16.0
        proposals = proposals[:, cond, :]
        prob_object = prob_object[:, cond]

        proposals = self._offset2bbox(proposals)
        proposals = self._clip_boxes(proposals)

        # proposals -> (batch_size, -1, 4)
        # prob_object -> (batch_size, -1)
        
        # ATE AQUI TUDO CERTO !

        ### NMS ###
        # TODO: Remove the for loop somehow !
        batch_proposals = []
        batch_prob_object = []
        for i in range(batch_size):

            i_proposals_o = proposals[i, :, :]
            i_prob_object_o = prob_object[i, :]

            idxs = torch.argsort(i_prob_object_o, descending=True)
            n_proposals = 600
            idxs = idxs[:n_proposals]

            i_proposals = i_proposals_o[idxs, :]
            i_prob_object = i_prob_object_o[idxs]

            k = 0
            while k < i_proposals.size()[0]:

                ### Remove iou > 0.7 ###
                x0_0, y0_0, x1_0, y1_0 = i_proposals[k, 0], i_proposals[k, 1], i_proposals[k, 2], i_proposals[k, 3]
                area_0 = (x1_0 - x0_0 + 1) * (y1_0 - y0_0 + 1)
                assert x1_0 > x0_0 and y1_0 > y0_0 # just to ensure.. but this is dealt before I think

                marked_to_keep = []

                for j in range(k+1, i_proposals.size()[0]):

                    x0_j, y0_j, x1_j, y1_j = i_proposals[j, 0], i_proposals[j, 1], i_proposals[j, 2], i_proposals[j, 3]
                    
                    x0 = torch.max(x0_0, x0_j)
                    y0 = torch.max(y0_0, y0_j)
                    x1 = torch.min(x1_0, x1_j)
                    y1 = torch.min(y1_0, y1_j)
                    
                    intersection = torch.clamp(x1 - x0 + 1, min=0) * torch.clamp(y1 - y0 + 1, min=0)
                    area_j = (x1_j - x0_j + 1) * (y1_j - y0_j + 1)

                    union = area_0 + area_j - intersection
                    iou = intersection / union

                    if iou <= 0.7:
                        marked_to_keep.append(j)

                # keep
                i_proposals = torch.cat((i_proposals[:k+1, :], i_proposals[marked_to_keep, :]), dim=0)
                i_prob_object = torch.cat((i_prob_object[:k+1], i_prob_object[marked_to_keep]), dim=0)
                k += 1
            
            batch_proposals.append(i_proposals)
            batch_prob_object.append(i_prob_object)

        ### end of NMS ###

        proposals = torch.stack(batch_proposals, dim=0)
        prob_object = torch.stack(batch_prob_object, dim=0)

        proposals = self._bbox2offset(proposals)

        return proposals, prob_object


    def _bbox2offset(self, bboxes):
        """
        bboxes: batch_size, -1, 4
        proposals: batch_size, -1, 4

        """

        bx0 = bboxes[:, :, 0]
        by0 = bboxes[:, :, 1]
        bx1 = bboxes[:, :, 2]
        by1 = bboxes[:, :, 3]

        ox = bx0
        oy = by0
        ow = bx1 - bx0 + 1
        oh = by1 - by0 + 1

        offsets = torch.stack((ox, oy, ow, oh), dim=2)

        return offsets


    def _offset2bbox(self, proposals):
        """
        proposals: batch_size, -1, 4
        bboxes: batch_size, -1, 4

        """

        bx0 = proposals[:, :, 0]
        by0 = proposals[:, :, 1]
        bx1 = bx0 + proposals[:, :, 2] - 1
        by1 = by0 + proposals[:, :, 3] - 1

        bboxes = torch.stack((bx0, by0, bx1, by1), dim=2)

        return bboxes


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
        # ax, ay, aw, ah = self.anchors_parameters
        ax = self.anchors_parameters[:, 0].view(1, -1, 1, 1)
        ay = self.anchors_parameters[:, 1].view(1, -1, 1, 1)
        aw = self.anchors_parameters[:, 2].view(1, -1, 1, 1)
        ah = self.anchors_parameters[:, 3].view(1, -1, 1, 1)
    
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

    
    def _clip_boxes(self, bboxes):
        """
        bboxes: batch_size, -1, 4
        bboxes: batch_size, -1, 4

        """

        bx0 = bboxes[:, 0::4, :].clamp(0, self.input_img_size[0]-1)
        by0 = bboxes[:, 1::4, :].clamp(0, self.input_img_size[1]-1)
        bx1 = bboxes[:, 2::4, :].clamp(0, self.input_img_size[0]-1)
        by1 = bboxes[:, 3::4, :].clamp(0, self.input_img_size[1]-1)

        bboxes = torch.cat((bx0, by0, bx1, by1), dim=1)

        return bboxes


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

        n_rows, n_cols = self.receptive_field_size, self.receptive_field_size

        base_anchor_area = n_rows * n_cols
        base_anchor_center_cols = 0.5 * (n_cols - 1)
        base_anchor_center_rows = 0.5 * (n_rows - 1)

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
        anchors = torch.from_numpy(anchors)

        return anchors #anchors_center_cols_offset, anchors_center_rows_offset, aw, ah


if __name__ == "__main__":

    input_img_size = (128, 128)
    feature_extractor_out_dim = 12
    receptive_field_size = 16

    rpn = RPN(input_img_size, feature_extractor_out_dim, receptive_field_size)

