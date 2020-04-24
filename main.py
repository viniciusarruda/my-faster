import config
import torch
import torch.nn.functional as F
import numpy as np
from dataset_loader import get_dataloader, inv_normalize, get_dataset
from tqdm import tqdm, trange
from visualizer import Viz
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

    viz = Viz(tensorboard=True, files=True, screen=True)

    model = FasterRCNN().to(device)

    # isso ta uma bosta (o lance de pegar o model.rpn_net.anchors... )!
    train_dataloader = get_dataloader(model.rpn_net.anchors)
    test_dataset = get_dataset(model.rpn_net.anchors)

    params = [p for p in model.parameters() if p.requires_grad == True]

    optimizer = torch.optim.Adam(params, lr=0.001)

    output = model.infer(0, test_dataset, device)
    viz.record_inference(output)

    model.train()

    l = len(train_dataloader)

    for e in trange(1, config.epochs+1):

        rpn_prob_loss_epoch, rpn_bbox_loss_epoch, rpn_loss_epoch = 0, 0, 0
        clss_reg_prob_loss_epoch, clss_reg_bbox_loss_epoch, clss_reg_loss_epoch = 0, 0, 0
        total_loss_epoch = 0

        for img, annotation, _, labels_objectness, labels_class, table_gts_positive_anchors in train_dataloader:

            # show_training_sample(inv_normalize(img[0, :, :, :].clone().detach()).permute(1, 2, 0).numpy().copy(), annotation[0].detach().numpy().copy())
            
            assert img.size(0) == annotation.size(0) == labels_objectness.size(0) == labels_class.size(0) == table_gts_positive_anchors.size(0) == 1
            img, annotation = img.to(device), annotation[0, :, :].to(device)
            labels_objectness, labels_class, table_gts_positive_anchors = labels_objectness[0, :].to(device), labels_class[0, :].to(device), table_gts_positive_anchors[0, :, :].to(device)

            # print(table_gts_positive_anchors)
            # print(labels_objectness) -> already balanced
            # print(labels_class)      -> not balanced yet
            # exit()

            optimizer.zero_grad()

            rpn_prob_loss_item, rpn_bbox_loss_item, rpn_loss_item, clss_reg_prob_loss_item, clss_reg_bbox_loss_item, clss_reg_loss_item, total_loss = model.forward(img, annotation, labels_objectness, labels_class, table_gts_positive_anchors)

            rpn_prob_loss_epoch += rpn_prob_loss_item
            rpn_bbox_loss_epoch += rpn_bbox_loss_item
            rpn_loss_epoch += rpn_loss_item
            clss_reg_prob_loss_epoch += clss_reg_prob_loss_item
            clss_reg_bbox_loss_epoch += clss_reg_bbox_loss_item
            clss_reg_loss_epoch += clss_reg_loss_item
            total_loss_epoch += total_loss.item()

            total_loss.backward()

            optimizer.step()

            viz.record(e, rpn_prob_loss_epoch / l, rpn_bbox_loss_epoch / l, rpn_loss_epoch / l, clss_reg_prob_loss_epoch / l, clss_reg_bbox_loss_epoch / l, clss_reg_loss_epoch / l, total_loss_epoch / l)


        if e % 10 == 0:
            
            output = model.infer(e, test_dataset, device)
            viz.record_inference(output)

            model.train()


if __name__ == "__main__":
    main()