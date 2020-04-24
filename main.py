import torch
import torch.nn.functional as F
import numpy as np
from dataset_loader import get_dataloader, inv_normalize, get_dataset
from tqdm import tqdm, trange

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

# to check for nan - only to be used while testing
# https://pytorch.org/docs/stable/autograd.html#torch.autograd.detect_anomaly
# torch.autograd.set_detect_anomaly(True)


# TODO
# FIXME 
# BUG 
# NOTE


def main():

    device = torch.device("cpu")

    model = FasterRCNN(device)

    # isso ta uma bosta (o lance de pegar o model.rpn_net.anchors... )!
    train_dataloader = get_dataloader(model.rpn_net.anchors)
    test_dataset = get_dataset(model.rpn_net.anchors)

    model.train(train_dataloader, test_dataset)


if __name__ == "__main__":
    main()