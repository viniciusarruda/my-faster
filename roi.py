import torch
from torch import nn
import torch.nn.functional as F
import torchvision

# TODO: Implement my own RoIAlign from scratch using python (dont care if become slow, just as learning)


class ROI(nn.Module):

    def __init__(self, input_img_size):
        super(ROI, self).__init__()

        self.out_dim = 7
        self.input_img_size = input_img_size

    def forward(self, proposals, features):

        # input_img_size -> (width, height)
        # fx = features.size(2) / self.input_img_size[1]  # ja sei a priori
        # fy = features.size(3) / self.input_img_size[0]  # ja sei a priori
        fx = 1.0 / 16.0

        # print(features.size(2), self.input_img_size[1])
        # print(features.size(3), self.input_img_size[0])
        # print(fx, fy)
        # assert fx == fy  # 1.0/16.0
        # LOOK example usage.. but it is really shallow: https://github.com/jwyang/faster-rcnn.pytorch/blob/31ae20687b1b3486155809a57eeb376259a5f5d4/lib/model/roi_align/modules/roi_align.py#L18

        # TODO commentint out the above and setting spatial_scale to 1 shoud be equal, according to test_nms.py file
        # NOTE Actually, I think it is not possible. I think to obtain this behavior I should also resize the features.
        # or no? todo, tothink
        # proposals[:, :, 0] *= fx
        # proposals[:, :, 1] *= fy
        # proposals[:, :, 2] *= fx
        # proposals[:, :, 3] *= fy

        rois = torchvision.ops.roi_align(features, [proposals], (14, 14), spatial_scale=fx)

        rois = F.max_pool2d(rois, kernel_size=2)  # there is avg_pool, others use stride=1

        return rois


# class ROI(nn.Module):

#     def  __init__(self, input_img_size):

#         super(ROI, self).__init__()

#         self.out_dim = 7
#         self.input_img_size = input_img_size

#     # NOTE/TODO:
#     # This is a really simple RoI implementation
#     # This is not a RoIAlign or other RoI implemented in papers
#     # It works, but is not what people are using
#     # About RoIAlign:
#     #           https://chao-ji.github.io/jekyll/update/2018/07/20/ROIAlign.html  <<<<< check it !
#     #           https://medium.com/@fractaldle/mask-r-cnn-unmasked-c029aa2f1296
#     # I coud not found implementations in plain pytorch or python.
#     # The other implementations are using pure C and Cuda and binding to python.
#     # Also, I can embed this functions here: https://pytorch.org/docs/stable/torchvision/ops.html
#     def forward(self, proposals, features):

#         fx = features.size(2) / self.input_img_size[0]  # ja sei a priori
#         fy = features.size(3) / self.input_img_size[1]  # ja sei a priori

#         x = proposals[:, :, 0] * fx
#         y = proposals[:, :, 1] * fy
#         w = proposals[:, :, 2] * fx
#         h = proposals[:, :, 3] * fy

#         # fazer um estudo do tradeoff de deixar o floor e ceil ou n
#         # I put floor and ceil to get the whole feature information, otherwise will truncate the feature size covered by the proposal
#         # roi = torch.stack((x, y, w, h), dim=2).long() # long -> torch.int64
#         roi = torch.stack((x.floor(), y.floor(), w.ceil(), h.ceil()), dim=2).long() # long -> torch.int64

#         batch_rois = []

#         for i in range(roi.size(0)):
#             rois = []
#             for k in range(roi.size(1)):

#                 x = roi[i, k, 0]
#                 y = roi[i, k, 1]
#                 w = roi[i, k, 2]
#                 h = roi[i, k, 3]

#                 roi_feature = features[i, :, x:x+w, y:y+h].unsqueeze(0)

#                 roi_feature_interpolated = F.interpolate(roi_feature, size=(14, 14),  mode='bilinear', align_corners=True)

#                 # Here, the max_pool2d is substituted for RCNN_top !
#                 roi_pooled = F.max_pool2d(roi_feature_interpolated, kernel_size=2)

#                 rois.append(roi_pooled)

#             rois = torch.cat(rois, dim=0)
#             batch_rois.append(rois)

#         rois = torch.stack(batch_rois, dim=0)

#         return rois

# NAO DESISTE !!!


if __name__ == "__main__":

    torch.manual_seed(0)
    features = torch.rand(1, 1, 10, 10, dtype=torch.float32)
    # single_roi = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
    single_roi = torch.tensor([[0, 0, 4, 4]], dtype=torch.float32)
    model = torchvision.ops.RoIAlign((5, 5), 1, 2)
    out = model(features, [single_roi])

    print(features)
    print(single_roi)

    print(out)

    # MY TRY:
    # print(single_roi.size())

    x = single_roi[:, 0] * 1.0  # fx
    y = single_roi[:, 1] * 1.0  # fy
    w = single_roi[:, 2] * 1.0  # fx
    h = single_roi[:, 3] * 1.0  # fy

    roi = torch.stack((x.floor(), y.floor(), w.ceil(), h.ceil()), dim=1).long()  # long -> torch.int64

    # print(roi.size())

    x = roi[0, 0]
    y = roi[0, 1]
    w = roi[0, 2]
    h = roi[0, 3]

    roi_feature = features[0, :, x:x + w, y:y + h].unsqueeze(0)

    # print(roi_feature.size())

    roi_feature_interpolated = F.interpolate(roi_feature, size=(5, 5), mode='bilinear', align_corners=True)

    # print(roi_feature_interpolated.size())

    print(roi_feature_interpolated)
