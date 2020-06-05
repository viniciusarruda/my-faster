import config
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import DatasetWrapper
from tqdm import trange
from visualizer import Viz
from faster_rcnn import FasterRCNN
from dataset import inv_normalize

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


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    viz = Viz(tensorboard=True, files=True, screen=True)

    model = FasterRCNN().to(device)

    # format_type = 'simple'
    # train_data_info = (config.img_folder, config.annotations_file)
    # test_data_info = (config.val_img_folder, config.val_annotations_file)

    format_type = 'VOC'
    train_data_info = (config.voc_folder, config.set_type)
    test_data_info = (config.val_voc_folder, config.val_set_type)

    # isso ta uma bosta (o lance de pegar o model.rpn_net.anchors... )!
    train_dataset = DatasetWrapper(format_type=format_type,
                                   data_info=train_data_info,
                                   input_img_size=config.input_img_size,
                                   anchors=model.rpn_net.valid_anchors,
                                   train=True)

    test_dataset = DatasetWrapper(format_type=format_type,
                                  data_info=test_data_info,
                                  input_img_size=config.input_img_size,
                                  anchors=model.rpn_net.valid_anchors,
                                  train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    params = [p for p in model.parameters() if p.requires_grad is True]

    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if config.verbose:
        # drawing the anchors
        viz.show_anchors(model.rpn_net.valid_anchors, config.input_img_size)
        for idx, (_, annotations, rpn_labels, expanded_annotations, table_annotations_dbg) in test_dataloader:
            viz.show_masked_anchors(idx,
                                    model.rpn_net.valid_anchors,
                                    rpn_labels[0, :].to(device),
                                    expanded_annotations[0, :, :].to(device),
                                    annotations[0, :, :].to(device),
                                    config.input_img_size,
                                    table_annotations_dbg[0, :].to(device))
        # end of drawing the anchors

    display_times = 500
    losses_str = ['rpn_prob', 'rpn_bbox', 'rpn', 'clss_reg_prob', 'clss_reg_bbox', 'clss_reg', 'total']
    recorded_losses = {key: 0 for key in losses_str}
    iteration = 0

    for e in trange(1, config.epochs + 1):

        infer(e, model, test_dataloader, test_dataset, device, viz)
        model.train()

        # end_data_time = time.time()
        for img, annotations, rpn_labels, expanded_annotations in train_dataloader:
            # start_data_time = time.time()
            # print(start_data_time - end_data_time)
            # end_data_time = start_data_time

            # Only implemented for batch size = 1
            assert img.size(0) == annotations.size(0) == rpn_labels.size(0) == expanded_annotations.size(0) == 1
            img, annotations = img.to(device), annotations[0, :, :].to(device)
            rpn_labels, expanded_annotations = rpn_labels[0, :].to(device), expanded_annotations[0, :, :].to(device)

            optimizer.zero_grad()

            *losses_item, total_loss = model.forward(img, annotations, rpn_labels, expanded_annotations)

            for i, key in enumerate(losses_str[:-1]):
                recorded_losses[key] = losses_item[i]
            recorded_losses[losses_str[-1]] = total_loss.item()

            total_loss.backward()

            optimizer.step()

            # the lr plotted is based in one parameter
            # if there is different lr for different parameters, it will not show them, just one: param_groups[0]
            display_on = iteration % display_times == 0
            viz.record_losses(e, iteration, display_on, recorded_losses, optimizer.param_groups[0]['lr'])
            iteration += 1

        scheduler.step()

    infer(e, model, test_dataloader, test_dataset, device, viz, evaluate=True)


def infer(epoch, model, dataloader, dataset, device, viz, evaluate=False):

    output = []

    model.eval()

    with torch.no_grad():

        # for ith, (img, annotation, labels, table_gts_positive_anchors) in enumerate(dataloader):
        # there is a random number being generated inside the Dataloader: https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
        # in the final version, use the dataloader if is more fancy
        # for ith in range(len(dataset)):

        # remove this ith?
        for ith, (img, annotations, rpn_labels, expanded_annotations, table_annotations_dbg) in dataloader:

            # Only implemented for batch size = 1
            assert img.size(0) == annotations.size(0) == rpn_labels.size(0) == expanded_annotations.size(0) == table_annotations_dbg.size(0) == 1
            img, annotations = img.to(device), annotations[0, :, :].to(device)
            rpn_labels, expanded_annotations = rpn_labels[0, :].to(device), expanded_annotations[0, :, :].to(device)
            table_annotations_dbg = table_annotations_dbg[0, :].to(device)

            refined_bboxes, clss_score, pred_clss_idxs, *ret = model.forward(img, annotations, rpn_labels, expanded_annotations)

            if evaluate:
                dataset.store_prediction(ith.detach().cpu().numpy()[0], refined_bboxes, clss_score, pred_clss_idxs)

            if config.verbose:

                proposals, all_probs_object, anchors, probs_object, filtered_proposals = ret

                ith_output = [epoch,
                              inv_normalize(img[0, :, :, :].detach().clone()).cpu().numpy().transpose(1, 2, 0) * 255,
                              annotations.detach().cpu().numpy(),
                              expanded_annotations.detach().cpu().numpy(),
                              table_annotations_dbg.detach().cpu().numpy(),
                              proposals,
                              all_probs_object,
                              anchors,
                              probs_object,
                              filtered_proposals,
                              clss_score,
                              pred_clss_idxs,
                              refined_bboxes]

                output.append(ith_output)

        if evaluate:
            dataset.finish_predictions()
            dataset.evaluate_predictions()

        if config.verbose:
            viz.record_inference(output)


if __name__ == "__main__":
    main()
