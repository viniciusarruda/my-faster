import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from nms import nms
from bbox_utils import clip_boxes, anchors_offset2bbox, anchors_bbox2offset
import config


class RPN(nn.Module):

    def __init__(self, input_img_size, feature_extractor_out_dim, feature_extractor_size, receptive_field_size):

        super(RPN, self).__init__()

        # ## Information about the feature extractor ###
        self.feature_extractor_size = feature_extractor_size  # (w, h)
        self.receptive_field_size = receptive_field_size
        self.input_img_size = input_img_size  # (w, h)
        ###############################################

        # ## Anchor related attributes ###
        # acredito que ao implementar a resnet como feature extractor pode melhorar colocando mais ratios e scales, ficou ruim do jeito que esta
        self.anchor_ratios = config.rpn_anchor_ratios
        self.anchor_scales = config.rpn_anchor_scales
        self.k = len(self.anchor_scales) * len(self.anchor_ratios)
        all_anchors, valid_anchors_mask = self._get_anchors()   # mudar para all_anchors, valid_anchors_mask e valid_anchors
        self.register_buffer('all_anchors', all_anchors)
        self.register_buffer('valid_anchors_mask', valid_anchors_mask)
        self.register_buffer('valid_anchors', all_anchors[valid_anchors_mask, :])
        # self.anchors, self.valid_anchors_mask = self.anchors.to(device), self.valid_anchors_mask.to(device)
        #################################

        self.conv_rpn = nn.Conv2d(in_channels=feature_extractor_out_dim, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)

        self.cls_layer = nn.Conv2d(in_channels=512, out_channels=self.k * 2, kernel_size=1, stride=1, padding=0)
        self.reg_layer = nn.Conv2d(in_channels=512, out_channels=self.k * 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x, pre_nms_top_n, pos_nms_top_n):
        assert x.size(0) == 1  # remove this assertion later..
        # x -> (batch_size, feature_extractor_out_dim, 64, 64)

        x = F.relu(self.conv_rpn(x))
        # x -> (batch_size, feature_extractor_out_dim, 64, 64)

        # ## Compute the probability to be an object ###

        cls_out = self.cls_layer(x)
        # cls_out -> (batch_size, k * 2, 64, 64)
        # para cada ancora, prob de obj e ~obj

        # print(cls_out.size())
        # print(cls_out.stride())
        # print(cls_out.is_contiguous())
        # print(cls_out.numel())
        # print()

        # Changed the above lines to the next ones
        # cls_out = cls_out.permute(0, 2, 3, 1).reshape((batch_size, -1, 2))
        # cls_out -> (batch_size, 64 * 64 * k, 2)

        cls_out = cls_out[0, :, :, :]
        # cls_out = cls_out.permute(1, 2, 0).reshape((-1, 2))
        # The below is the same as above, but explicit
        cls_out = cls_out.permute(1, 2, 0).contiguous().view(-1, 2)
        # cls_out -> (64 * 64 * k, 2)
        ###############################################

        # ## Compute the object proposals ###

        reg_out = self.reg_layer(x)
        # reg_out -> (batch_size, k * 4, 64, 64)

        # Changed the above lines to the next ones
        # reg_out = reg_out.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        # reg_out -> (batch_size, 64 * 64 * k, 4)

        reg_out = reg_out[0, :, :, :]
        reg_out = reg_out.permute(1, 2, 0).contiguous().view(-1, 4)
        # reg_out -> (64 * 64 * k, 4)

        proposals = self._anchors2proposals(reg_out)  # isso pode ser depois, pois na rpn_loss eu preciso do reg_out e n do proposals
        # proposals -> (#anchors, 4)
        # cls_out   -> (#anchors, 2)
        ####################################

        # TODO This second part should not be here.. instead.. is more a faster rcnn stuff than rpn one
        ####################################

        bboxes = clip_boxes(proposals, self.input_img_size)

        probs_object = F.softmax(cls_out, dim=1)[:, 1]  # it is 1 and not zero ! (TODO check this)
        # probs_object -> (batch_size, 64 * 64 * k)

        # Filter small bboxes (not implemented in that famous pytorch version)
        # cond = bboxes_filter_condition(bboxes)
        # bboxes, probs_object, labels_class = bboxes[cond, :], probs_object[cond], labels_class[cond]
        # bboxes -> (batch_size, -1, 4)
        # probs_object -> (batch_size, -1)

        # Filter the top pre_nms_top_n bboxes
        idxs = torch.argsort(probs_object, descending=True)
        n_bboxes = pre_nms_top_n

        if n_bboxes > 0:
            idxs = idxs[:n_bboxes]

        idxs_kept = nms(bboxes[idxs, :], probs_object[idxs], nms_threshold=0.7)
        idxs = idxs[idxs_kept]

        # Filter the top pos_nms_top_n bboxes
        n_bboxes = pos_nms_top_n
        if n_bboxes > 0:
            # already sorted by score due to `keep` indexing
            idxs = idxs[:n_bboxes]

        filtered_proposals = bboxes[idxs, :]
        probs_object = probs_object[idxs]
        # bboxes -> (batch_size, -1, 4)
        # probs_object -> (batch_size, -1)

        # In other codes after the pos_nms_top existis
        # a padding with zeros. I think it exists to avoid
        # empty tensor which may cause an error in the future.
        # Or, to serve as background cases? I do not know.
        # If this assertion fails, I need to inspect.
        assert probs_object.size(0) > 0, 'Inspect the following code behavior when this happens.'

        ####################################

        proposals = proposals[self.valid_anchors_mask, :]
        cls_out = cls_out[self.valid_anchors_mask, :]

        return proposals, cls_out, filtered_proposals, probs_object

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

        anchors = anchors_bbox2offset(self.all_anchors)

        ax = anchors[:, 0]
        ay = anchors[:, 1]
        aw = anchors[:, 2]
        ah = anchors[:, 3]

        tx = reg_out[:, 0]
        ty = reg_out[:, 1]
        tw = reg_out[:, 2]
        th = reg_out[:, 3]

        px = ax + aw * tx         # cw
        py = ay + ah * ty         # ch
        pw = aw * torch.exp(tw)   # w
        ph = ah * torch.exp(th)   # h

        proposals = torch.stack((px, py, pw, ph), dim=1)

        proposals = anchors_offset2bbox(proposals)

        return proposals

    def _get_anchors(self):

        # print('pay attention with the order w,h or h,w')
        # print('WARNING: implementation differ from source_base 1 and source_base 2 !')
        # print('TODO: jogar tudo pra pytorch')
        # print('Ta com -1 no h e w')
        # print('check all types and dims !')
        # print('TODO: tentar fazer com broadcast sem ter que gerar a matrix toda')

        # 16 = anchor base
        # ah * aw = 16*16 = 256
        # ah / aw = r -> ah * 1/aw = r
        # #
        # ah = r * aw       (2)
        # r * aw * aw = 256
        # aw^2 = 256/r
        # aw = sqrt(256/r)  (1)
        # ou seja, encontrar ah e aw para que mude o aspect ratio porem nao mude a area coberta pela ancora !

        # The base anchor is: [0, 0, self.receptive_field_size - 1, self.receptive_field_size - 1]
        base_anchor_area = self.receptive_field_size * self.receptive_field_size
        base_anchor_center_cols = (self.receptive_field_size - 1.0) * 0.5
        base_anchor_center_rows = (self.receptive_field_size - 1.0) * 0.5

        anchors = []

        for r in self.anchor_ratios:

            # An issue about this round: https://github.com/facebookresearch/Detectron/issues/227  TODO
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
        all_anchors = np.zeros((n_anchors, self.feature_extractor_size[1], self.feature_extractor_size[0]), dtype=np.float32)

        for k in range(0, n_anchors, 4):
            for i in range(0, self.feature_extractor_size[1]):
                for j in range(0, self.feature_extractor_size[0]):  # Jesus! There was a silent but dangerous bug here! Fixed!
                    all_anchors[k + 0, i, j] = anchors[k + 0] + j * self.receptive_field_size
                    all_anchors[k + 1, i, j] = anchors[k + 1] + i * self.receptive_field_size
                    all_anchors[k + 2, i, j] = anchors[k + 2]
                    all_anchors[k + 3, i, j] = anchors[k + 3]

        anchors = torch.from_numpy(all_anchors)
        anchors = anchors.permute(1, 2, 0).reshape(-1, 4)

        anchors = anchors_offset2bbox(anchors)

        valid_mask = (anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & (anchors[:, 2] < self.input_img_size[0]) & (anchors[:, 3] < self.input_img_size[1])

        print('A total of {} anchors.'.format(anchors.size(0)))
        # anchors = anchors[valid_mask, :]
        print('A total of {} valid anchors.'.format(valid_mask.nonzero().size(0)))

        return anchors, valid_mask
