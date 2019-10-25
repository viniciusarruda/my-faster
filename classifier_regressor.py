import torch
from torch import nn
import torch.nn.functional as F
from bbox_utils import bbox2offset, offset2bbox, clip_boxes, filter_boxes

from nms import nms

class ClassifierRegressor(nn.Module):

    def  __init__(self, input_img_size, input_size, n_classes):
    
        super(ClassifierRegressor, self).__init__()

        self.input_img_size = input_img_size

        self.first_layer = nn.Linear(input_size, 4096) 
        self.clss_pred = nn.Linear(4096, n_classes + 1)  # +1 for background
        self.reg_pred = nn.Linear(4096, 4)


    def forward(self, rois):

        assert rois.size(0) == 1

        rois = rois.view(rois.size(0), rois.size(1), -1)

        bi = 0

        x = self.first_layer(rois[bi, :, :])

        clss = self.clss_pred(x)
        reg = self.reg_pred(x)

        return reg.unsqueeze(0), clss.unsqueeze(0)

    
    def infer_bboxes(self, rpn_proposals, reg, clss):
        
        assert reg.size(0) == 1
        bi = 0

        clss_score = F.softmax(clss[bi, :, :], dim=1)
        clss_idxs = clss_score.argmax(dim=1)
        clss_score = clss_score[torch.arange(clss_score.size(0)), clss_idxs]

        # Filter out background
        idxs_non_background = clss_idxs != 0
        clss_score = clss_score[idxs_non_background]
        reg = reg[bi, idxs_non_background, :]
        rpn_proposals = rpn_proposals[bi, idxs_non_background, :]

        # Filter out lower scores
        # idxs_non_lower = clss_score >= 0.7 ## I am getting all clss_scores really lows
        idxs_non_lower = clss_score >= 0.01
        clss_score = clss_score[idxs_non_lower]
        reg = reg[idxs_non_lower, :]
        rpn_proposals = rpn_proposals[idxs_non_lower, :]

        # refine the bbox appling the bbox to px, py, pw and ph
        px = rpn_proposals[:, 0] + rpn_proposals[:, 2] * reg[:, 0]
        py = rpn_proposals[:, 1] + rpn_proposals[:, 3] * reg[:, 1]
        pw = rpn_proposals[:, 2] * torch.exp(reg[:, 2])
        ph = rpn_proposals[:, 3] * torch.exp(reg[:, 3])

        refined_proposals = torch.stack((px, py, pw, ph), dim=1)

        refined_proposals = refined_proposals.unsqueeze(0)
        clss_score = clss_score.unsqueeze(0)

        bboxes = offset2bbox(refined_proposals)
        bboxes = clip_boxes(bboxes, self.input_img_size)

        bboxes, clss_score = filter_boxes(bboxes, clss_score) # ??????? no rpn tem isso, fazer aqui tbm ? na verdade achei no codigo oficial que faz isso no teste sim mas no treino n.. confirmar este ultimo (treino n)

        # apply NMS
        bboxes, clss_score = nms(bboxes, clss_score)

        refined_proposals = bbox2offset(bboxes)

        return refined_proposals, clss_score
    
