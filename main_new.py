import torch
import torch.nn.functional as F
import numpy as np
from dataset_loader import get_dataloader, inv_normalize, get_dataset
from feature_extractor import FeatureExtractorNet
from feature_extractor_complete import FeatureExtractorNetComplete
from rpn import RPN
from roi import ROI
from classifier_regressor import ClassifierRegressor
from see_results import see_rpn_results, show_training_sample, see_final_results, see_rpn_final_results, show_anchors, show_masked_anchors, LossViz
from loss import anchor_labels, get_target_distance, get_target_distance2, get_target_mask, compute_prob_loss
from PIL import Image
from tqdm import tqdm, trange
import config

from faster_rcnn import FasterRCNN

import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
    exit()

warnings.showwarning = warn_with_traceback


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# TODO
# FIXME 
# BUG 
# NOTE


def main():

    lv = LossViz()

    device = torch.device("cpu")

    model = FasterRCNN(device)

    # isso ta uma bosta (o lance de pegar o model.rpn_net.anchors... )!
    train_dataloader = get_dataloader(model.rpn_net.anchors)
    test_dataset = get_dataset(model.rpn_net.anchors)

    model.train(train_dataloader, test_dataset)


if __name__ == "__main__":
    main()