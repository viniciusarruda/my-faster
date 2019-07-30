import torch
import numpy as np
from dataset_loader import get_dataloader
from feature_extractor import FeatureExtractorNet
from rpn import RPN
from roi import ROI
from classifier_regressor import ClassifierRegressor
from see_results import see_results

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# TODO
# FIXME 
# BUG 

#####################################

### About the image standard ###
# The images are in the format I(n_rows, n_cols), and indexed always as (r, c)
# The height and the width of an image is handled in the code as n_rows and n_cols, respectively.
# The (0, 0) point is at the top left corner of the image

### About the tensor standard ###


### ###
# The format of bounding box is in x,y,n_rows,n_cols unless the variable name contains a bbox word.

#####################################

# TODO: Assert the forward pass.
# TODO: Implement the loss and assert its corectness
# TODO: Implement the backward and assert its correctness


if __name__ == "__main__":

    dataloader, input_img_size = get_dataloader()

    fe_net = FeatureExtractorNet()
    rpn_net = RPN(input_img_size=input_img_size, feature_extractor_out_dim=fe_net.out_dim, receptive_field_size=fe_net.receptive_field_size)
    roi_net = ROI(input_img_size=input_img_size)
    clss_reg = ClassifierRegressor(input_img_size=input_img_size, input_size=7*7*12, n_classes=10 + 1)

    for img, clss in dataloader:
        
        print('Image size: {}'.format(img.size()))

        features = fe_net.forward(img)

        print('Features size: {}'.format(features.size()))
        
        proposals, probs_object = rpn_net.forward(features)

        print('Proposals size: {}'.format(proposals.size()))
        print('Probabilities object size: {}'.format(probs_object.size()))

        rois = roi_net.forward(proposals, features)

        print('Roi size: {}'.format(rois.size()))

        # print('Ate aqui tudo certo !')

        clss_out, bbox_out = clss_reg.forward(rois, proposals)

        print('Clss size: {}'.format(clss_out.size()))
        print('Bbox size: {}'.format(bbox_out.size()))

        # clss_out: (batch_size, n_bboxes)
        # bbox_out: (batch_size, n_bboxes, 4)

        clss_out_np = clss_out[0, :].detach().numpy()
        bbox_out_np = bbox_out[0, :, :].detach().numpy()

        img_np = img[0, :, :, :].permute(1, 2, 0).numpy()

        see_results(img_np, clss_out_np, bbox_out_np)

        exit()