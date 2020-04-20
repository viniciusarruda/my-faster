import torch
from torch import nn
import torch.nn.functional as F
from bbox_utils import bbox2offset, offset2bbox, clip_boxes, bboxes_filter_condition
import config

from nms import nms

class ClassifierRegressor(nn.Module):

    def  __init__(self, input_img_size, input_size, n_classes):
    
        super(ClassifierRegressor, self).__init__()

        self.input_img_size = input_img_size

        self.first_layer = nn.Linear(input_size, 4096) 
        self.clss_pred = nn.Linear(4096, config.n_classes) # background is already included in config.n_classes
        self.reg_pred = nn.Linear(4096, 4)


    def forward(self, rois):

        rois = rois.view(rois.size(0), -1)

        x = self.first_layer(rois)
        clss = self.clss_pred(x)
        reg = self.reg_pred(x)

        return reg, clss

    
    def infer_bboxes(self, rpn_proposals, reg, clss):

        clss_score = F.softmax(clss, dim=1)
        clss_idxs = clss_score.argmax(dim=1)
        clss_score = clss_score[torch.arange(clss_score.size(0)), clss_idxs]

        # Filter out background
        idxs_non_background = clss_idxs != 0
        clss_idxs = clss_idxs[idxs_non_background]
        clss_score = clss_score[idxs_non_background]
        reg = reg[idxs_non_background, :]
        rpn_proposals = rpn_proposals[idxs_non_background, :]

        # Filter out lower scores
        idxs_non_lower = clss_score >= 0.7 
        # idxs_non_lower = clss_score >= 0.01 ## I am getting all clss_scores really lows
        clss_idxs = clss_idxs[idxs_non_lower]
        clss_score = clss_score[idxs_non_lower]
        reg = reg[idxs_non_lower, :]
        rpn_proposals = rpn_proposals[idxs_non_lower, :]

        # refine the bbox appling the bbox to px, py, pw and ph
        px = rpn_proposals[:, 0] + rpn_proposals[:, 2] * reg[:, 0]
        py = rpn_proposals[:, 1] + rpn_proposals[:, 3] * reg[:, 1]
        pw = rpn_proposals[:, 2] * torch.exp(reg[:, 2])
        ph = rpn_proposals[:, 3] * torch.exp(reg[:, 3])

        refined_proposals = torch.stack((px, py, pw, ph), dim=1)

        bboxes = offset2bbox(refined_proposals)
        bboxes = clip_boxes(bboxes, self.input_img_size)

        # bboxes, clss_score = filter_boxes(bboxes, clss_score)
        cond = bboxes_filter_condition(bboxes) # ??????? no rpn tem isso, fazer aqui tbm ? na verdade achei no codigo oficial que faz isso no teste sim mas no treino n.. confirmar este ultimo (treino n)
        bboxes, clss_score, clss_idxs = bboxes[cond, :], clss_score[cond], clss_idxs[cond]
        # apply NMS
        # bboxes, clss_score = nms(bboxes, clss_score)
        idxs_kept = nms(bboxes, clss_score)
        bboxes = bboxes[idxs_kept, :]
        clss_score = clss_score[idxs_kept]
        clss_idxs = clss_idxs[idxs_kept]

        refined_proposals = bbox2offset(bboxes)

        return refined_proposals, clss_score, clss_idxs
    
