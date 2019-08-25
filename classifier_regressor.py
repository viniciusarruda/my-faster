import torch
from torch import nn
import torch.nn.functional as F

from nms import nms

class ClassifierRegressor(nn.Module):

    def  __init__(self, input_img_size, input_size, n_classes):
    
        super(ClassifierRegressor, self).__init__()

        self.input_img_size = input_img_size

        self.first_layer = nn.Linear(input_size, 4096) 
        self.clss_pred = nn.Linear(4096, n_classes + 1)  # +1 for background
        self.reg_pred = nn.Linear(4096, 4)

    # OLD FORWARD
    # def forward(self, rois, proposals):

    #     assert rois.size(0) == 1

    #     rois = rois.view(rois.size(0), rois.size(1), -1)

    #     bi = 0

    #     proposals = proposals[bi, :, :]

    #     x = self.first_layer(rois[bi, :, :])

    #     clss = self.clss_pred(x)
    #     reg = self.reg_pred(x)
    #     raw_reg = reg[:]

    #     clss_score = F.softmax(clss, dim=1)
    #     clss_idxs = clss_score.argmax(dim=1)
        
    #     clss_score = clss_score[torch.arange(clss_score.size(0)), clss_idxs]

    #     # Filter out background
    #     idxs_non_background = clss_idxs != 0
    #     clss_score = clss_score[idxs_non_background]
    #     reg = reg[idxs_non_background, :]
    #     proposals = proposals[idxs_non_background, :]

    #     # Filter out lower scores
    #     # idxs_non_lower = clss_score >= 0.7 ## I am getting all clss_scores really low
    #     idxs_non_lower = clss_score >= 0.01
    #     clss_score = clss_score[idxs_non_lower]
    #     reg = reg[idxs_non_lower, :]
    #     proposals = proposals[idxs_non_lower, :]

    #     # refine the bbox appling the bbox to px, py, pw and ph
    #     px = proposals[:, 0] + proposals[:, 2] * reg[:, 0]
    #     py = proposals[:, 1] + proposals[:, 3] * reg[:, 1]
    #     pw = proposals[:, 2] * torch.exp(reg[:, 2])
    #     ph = proposals[:, 3] * torch.exp(reg[:, 3])

    #     refined_proposals = torch.stack((px, py, pw, ph), dim=1)

    #     refined_proposals = refined_proposals.unsqueeze(0)
    #     clss_score = clss_score.unsqueeze(0)

    #     bboxes = self._offset2bbox(refined_proposals)
    #     bboxes = self._clip_boxes(bboxes)

    #     bboxes, clss_score = self._filter_boxes(bboxes, clss_score) # ??????? no rpn tem isso, fazer aqui tbm ?

    #     # apply NMS
    #     bboxes, clss_score = nms(bboxes, clss_score)

    #     refined_proposals = self._bbox2offset(bboxes)

    #     return refined_proposals, raw_reg.unsqueeze(0), clss_score

    def forward(self, rois, proposals):

        assert rois.size(0) == 1

        rois = rois.view(rois.size(0), rois.size(1), -1)

        bi = 0

        proposals = proposals[bi, :, :]

        x = self.first_layer(rois[bi, :, :])

        clss = self.clss_pred(x)
        reg = self.reg_pred(x)

        return reg.unsqueeze(0), clss.unsqueeze(0)


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

    
    def _clip_boxes(self, bboxes):
        """
        bboxes: batch_size, -1, 4
        bboxes: batch_size, -1, 4

        """
        # assert bboxes.size()[-1] == 4
        
        bx0 = bboxes[:, :, 0].clamp(0, self.input_img_size[0]-1)
        by0 = bboxes[:, :, 1].clamp(0, self.input_img_size[1]-1)
        bx1 = bboxes[:, :, 2].clamp(0, self.input_img_size[0]-1)
        by1 = bboxes[:, :, 3].clamp(0, self.input_img_size[1]-1)

        bboxes = torch.stack((bx0, by0, bx1, by1), dim=2)

        return bboxes


    def _filter_boxes(self, bboxes, probs_object, min_size=16.0):

        # torch.int64 -> index
        # torch.uint8 -> true or false (mask)

        assert bboxes.size()[0] == 1 # implement for batch 1 only.. todo for other batch size

        bx0 = bboxes[:, :, 0]
        by0 = bboxes[:, :, 1]
        bx1 = bboxes[:, :, 2]
        by1 = bboxes[:, :, 3]
        bw = bx1 - bx0
        bh = by1 - by0
        cond = (bw >= min_size) & (bh >= min_size)

        return bboxes[:, cond[0], :], probs_object[:, cond[0]]



if __name__ == "__main__":

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    clss_reg = ClassifierRegressor(input_img_size=(128,128), input_size=7*7*12, n_classes=10 + 1)

    rois = torch.rand(1, 5, 12, 7, 7)
    proposals = torch.rand(1, 5, 4)

    print(clss_reg.forward(rois, proposals))
