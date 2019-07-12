import torch
import numpy as np
from dataset_loader import get_dataloader
from feature_extractor import FeatureExtractorNet
from rpn import RPN

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

if __name__ == "__main__":

    dataloader, input_img_size = get_dataloader()

    fe_net = FeatureExtractorNet()
    rpn_net = RPN(input_img_size=input_img_size, feature_extractor_out_dim=fe_net.out_dim, receptive_field_size=fe_net.receptive_field_size)

    for img, clss in dataloader:
        
        print('Image size: {}'.format(img.size()))

        features = fe_net.forward(img)

        print('Features size: {}'.format(features.size()))

        cls_out, reg_out, proposals = rpn_net.forward(features)

        print('Class size: {}'.format(cls_out.size()))
        print('Regression size: {}'.format(reg_out.size()))
        print('Proposals size: {}'.format(proposals.size()))

        exit()