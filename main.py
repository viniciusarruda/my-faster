import config
import torch
import numpy as np
from dataset_loader import get_dataloader, get_dataset
from tqdm import trange
from visualizer import Viz
from faster_rcnn import FasterRCNN

import traceback
import warnings
import sys


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file, 'write') else sys.stderr
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    viz = Viz(tensorboard=True, files=True, screen=True)

    model = FasterRCNN().to(device)

    # isso ta uma bosta (o lance de pegar o model.rpn_net.anchors... )!
    train_dataloader = get_dataloader(model.rpn_net.valid_anchors)
    test_dataset = get_dataset(model.rpn_net.valid_anchors,
                               img_dir=config.val_img_folder,
                               csv_file=config.val_annotations_file,
                               train=False)

    params = [p for p in model.parameters() if p.requires_grad is True]

    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    output = model.infer(0, test_dataset, device)
    viz.record_inference(output)

    # drawing the anchors
    viz.show_anchors(model.rpn_net.valid_anchors, config.input_img_size)
    # exit()
    tmp_dataset = get_dataset(model.rpn_net.valid_anchors,
                              img_dir=config.val_img_folder,
                              csv_file=config.val_annotations_file,
                              train=False)
    for e, (img, annotations, rpn_labels, expanded_annotations, table_annotations_dbg) in enumerate(tmp_dataset):
        img = img.unsqueeze(0)
        annotations = annotations.unsqueeze(0)
        rpn_labels = rpn_labels.unsqueeze(0)
        expanded_annotations = expanded_annotations.unsqueeze(0)
        table_annotations_dbg = table_annotations_dbg.unsqueeze(0)
        img, annotations = img.to(device), annotations[0, :, :].to(device)
        rpn_labels, expanded_annotations, table_annotations_dbg = rpn_labels[0, :].to(device), expanded_annotations[0, :, :].to(device), table_annotations_dbg[0, :].to(device)
        viz.show_masked_anchors(e, model.rpn_net.valid_anchors, rpn_labels, expanded_annotations, annotations, config.input_img_size, table_annotations_dbg)
    del tmp_dataset
    # exit()
    # end of drawing the anchors

    model.train()

    data_size = len(train_dataloader)

    for e in trange(1, config.epochs + 1):

        rpn_prob_loss_epoch, rpn_bbox_loss_epoch, rpn_loss_epoch = 0, 0, 0
        clss_reg_prob_loss_epoch, clss_reg_bbox_loss_epoch, clss_reg_loss_epoch = 0, 0, 0
        total_loss_epoch = 0

        # end_data_time = time.time()
        for img, annotation, rpn_labels, expanded_annotations in train_dataloader:
            # start_data_time = time.time()
            # print(start_data_time - end_data_time)
            # end_data_time = start_data_time

            # show_training_sample(inv_normalize(img[0, :, :, :].clone().detach()).permute(1, 2, 0).numpy().copy(), annotation[0].detach().numpy().copy())

            assert img.size(0) == annotation.size(0) == rpn_labels.size(0) == expanded_annotations.size(0) == 1
            img, annotation = img.to(device), annotation[0, :, :].to(device)
            rpn_labels, expanded_annotations = rpn_labels[0, :].to(device), expanded_annotations[0, :, :].to(device)

            # print(table_gts_positive_anchors)
            # print(labels_objectness) -> already balanced
            # print(labels_class)      -> not balanced yet
            # exit()

            optimizer.zero_grad()

            rpn_prob_loss_item, rpn_bbox_loss_item, rpn_loss_item, clss_reg_prob_loss_item, clss_reg_bbox_loss_item, clss_reg_loss_item, total_loss = model.forward(img, annotation, rpn_labels, expanded_annotations)

            rpn_prob_loss_epoch += rpn_prob_loss_item
            rpn_bbox_loss_epoch += rpn_bbox_loss_item
            rpn_loss_epoch += rpn_loss_item
            clss_reg_prob_loss_epoch += clss_reg_prob_loss_item
            clss_reg_bbox_loss_epoch += clss_reg_bbox_loss_item
            clss_reg_loss_epoch += clss_reg_loss_item
            total_loss_epoch += total_loss.item()

            total_loss.backward()

            optimizer.step()

        # the lr plotted is based in one parameter
        # if there is different lr for different parameters, it will not show them, just one: param_groups[0]
        viz.record(e, rpn_prob_loss_epoch / data_size, rpn_bbox_loss_epoch / data_size, rpn_loss_epoch / data_size, clss_reg_prob_loss_epoch / data_size, clss_reg_bbox_loss_epoch / data_size, clss_reg_loss_epoch / data_size, total_loss_epoch / data_size, optimizer.param_groups[0]['lr'])

        # if e % 10 == 0:
        output = model.infer(e, test_dataset, device)
        viz.record_inference(output)
        model.train()

        scheduler.step()


if __name__ == "__main__":
    main()
