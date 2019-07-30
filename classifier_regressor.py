import torch
from torch import nn
import torch.nn.functional as F

from nms import nms

class ClassifierRegressor(nn.Module):

    def  __init__(self, input_img_size, input_size, n_classes):
    
        super(ClassifierRegressor, self).__init__()

        self.input_img_size = input_img_size

        self.first_layer = nn.Linear(input_size, 4096) 
        self.clss_pred = nn.Linear(4096, n_classes)
        self.reg_pred = nn.Linear(4096, 4)

    def forward(self, rois, proposals):

        rois = rois.view(rois.size()[0], rois.size()[1], -1)

        clss_out, proposals_out = [], []
        batch_size = rois.size()[0]

        for i in range(batch_size):

            i_proposal = proposals[i, :, :]

            x = self.first_layer(rois[i, :, :])

            clss = self.clss_pred(x)
            reg = self.reg_pred(x)

            clss_score = F.softmax(clss, dim=1)
            clss_idxs = clss_score.argmax(dim=1)
            clss_score = clss_score[torch.arange(clss_score.size()[0]), clss_idxs]

            # Filter out background
            idxs_non_background = clss_idxs != 0
            clss_score = clss_score[idxs_non_background]
            reg = reg[idxs_non_background, :]
            i_proposal = i_proposal[idxs_non_background, :]

            # Filter out lower scores
            # idxs_non_lower = clss_score >= 0.7 ## I am getting all clss_scores really low
            idxs_non_lower = clss_score >= 0.01
            clss_score = clss_score[idxs_non_lower]
            reg = reg[idxs_non_lower, :]
            i_proposal = i_proposal[idxs_non_lower, :]

            # refine the bbox appling the bbox to px, py, pw and ph
            px = i_proposal[:, 0] + i_proposal[:, 2] * reg[:, 0]
            py = i_proposal[:, 1] + i_proposal[:, 3] * reg[:, 1]
            pw = i_proposal[:, 2] * torch.exp(reg[:, 2])
            ph = i_proposal[:, 3] * torch.exp(reg[:, 3])

            i_refined_proposal = torch.stack((px, py, pw, ph), dim=1)

            # apply NMS
            i_refined_proposal, clss_score = nms(self.input_img_size, i_refined_proposal, clss_score)

            clss_out.append(clss_score)
            proposals_out.append(i_refined_proposal)

        clss_out = torch.stack(clss_out, dim=0)
        proposals_out = torch.stack(proposals_out, dim=0)

        # proposals_out = _offset2bbox(proposals_out) 
        bx0 = proposals_out[:, :, 0]
        by0 = proposals_out[:, :, 1]
        bx1 = bx0 + proposals_out[:, :, 2] - 1
        by1 = by0 + proposals_out[:, :, 3] - 1
        bboxes_out = torch.stack((bx0, by0, bx1, by1), dim=2)
        #############################################

        return clss_out, bboxes_out



if __name__ == "__main__":

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    clss_reg = ClassifierRegressor(input_img_size=(128,128), input_size=7*7*12, n_classes=10 + 1)

    rois = torch.rand(1, 5, 12, 7, 7)
    proposals = torch.rand(1, 5, 4)

    print(clss_reg.forward(rois, proposals))
