import torch
import torch.nn.functional as F
import numpy as np
from dataset_loader import get_dataloader, inv_normalize
from feature_extractor import FeatureExtractorNet
from feature_extractor_complete import FeatureExtractorNetComplete
from rpn import RPN
from roi import ROI
from classifier_regressor import ClassifierRegressor
from see_results import see_rpn_results, show_training_sample, see_final_results, see_rpn_final_results, show_anchors, show_masked_anchors, LossViz
from loss import anchor_labels, get_target_distance, compute_rpn_prob_loss, get_target_distance2, get_target_mask, compute_cls_reg_prob_loss
from PIL import Image
from tqdm import tqdm, trange
import config


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

    # fe_net = FeatureExtractorNet().to(device)
    fe_net = FeatureExtractorNetComplete().to(device)
    rpn_net = RPN(input_img_size=config.input_img_size, feature_extractor_out_dim=fe_net.out_dim, feature_extractor_size=fe_net.feature_extractor_size, receptive_field_size=fe_net.receptive_field_size, device=device).to(device)
    roi_net = ROI(input_img_size=config.input_img_size).to(device)
    clss_reg = ClassifierRegressor(input_img_size=config.input_img_size, input_size=7*7*fe_net.out_dim, n_classes=1).to(device)

    dataloader = get_dataloader(rpn_net.anchors_parameters, rpn_net.valid_anchors)

    # visual debug
    # show_anchors(rpn_net.anchors_parameters.detach().numpy().copy(), rpn_net.valid_anchors.detach().numpy().copy(), config.input_img_size)

    # for e, (img, annotation, _, _) in enumerate(dataloader):
    #     img, annotation = img.to(device), annotation[0, :, :].to(device)
    #     labels, table_gts_positive_anchors = anchor_labels(rpn_net.anchors_parameters, rpn_net.valid_anchors, annotation)
    #     labels, table_gts_positive_anchors = labels.to(device), table_gts_positive_anchors.to(device)
    #     show_masked_anchors(e, rpn_net.anchors_parameters.detach().numpy().copy(), rpn_net.valid_anchors.detach().numpy().copy(), labels.detach().numpy().copy(), table_gts_positive_anchors.detach().numpy().copy(), annotation.detach().numpy().copy(), config.input_img_size)
    # exit()

    params = list(fe_net.parameters()) + list(rpn_net.parameters()) + list(roi_net.parameters()) + list(clss_reg.parameters())

    params = [p for p in params if p.requires_grad == True]

    optimizer = torch.optim.Adam(params, lr=0.001)

    for net in [fe_net, rpn_net, roi_net, clss_reg]:
        net.train()

    l = len(dataloader)

    for e in trange(1, config.epochs+1):

        rpn_prob_loss_epoch, rpn_bbox_loss_epoch, rpn_loss_epoch = 0, 0, 0
        clss_reg_prob_loss_epoch, clss_reg_bbox_loss_epoch, clss_reg_loss_epoch = 0, 0, 0
        
        for img, annotation, labels, table_gts_positive_anchors in dataloader:

            # it is just one image, however, for the image should keep the batch channel
            img, annotation = img.to(device), annotation[0, :, :].to(device)
            labels, table_gts_positive_anchors = labels[0, :].to(device), table_gts_positive_anchors[0, :, :].to(device)

            optimizer.zero_grad()

            # print('Image size: {}'.format(img.size()))
            # print('Annotation size: {}'.format(annotation.size()))
            features = fe_net.forward(img)

            # print('Features size: {}'.format(features.size()))
            proposals, cls_out, filtered_proposals, probs_object = rpn_net.forward(features)

            # print('Proposals size: {}'.format(proposals.size()))
            # print('Probabilities object size: {}'.format(probs_object.size()))

            #####
            # TODO:
            # remove here the batch channel for the above tensors
            # adapt the functions below
            #####


            ## Compute RPN loss ##
            rpn_bbox_loss = get_target_distance(proposals, rpn_net.anchors_parameters, rpn_net.valid_anchors, annotation, table_gts_positive_anchors)
            rpn_prob_loss = compute_rpn_prob_loss(cls_out, labels)
            #####

            rpn_loss = 10 * rpn_prob_loss + rpn_bbox_loss

            rpn_prob_loss_epoch += rpn_prob_loss.item()
            rpn_bbox_loss_epoch += rpn_bbox_loss.item()
            rpn_loss_epoch += rpn_loss.item()


            # assert filtered_proposals.size(1) > 0
            if filtered_proposals.size(1) > 0:

                rois = roi_net.forward(filtered_proposals, features)
                # print('Roi size: {}'.format(rois.size()))
                #
                raw_reg, raw_cls = clss_reg.forward(rois)
                # print('raw_reg size: {}'.format(raw_reg.size()))
                # print('raw_cls size: {}'.format(raw_cls.size()))

                #####
                ## Compute class_reg loss ##
                table_fgs_positive_proposals, cls_mask = get_target_mask(filtered_proposals, annotation)
                clss_reg_bbox_loss = get_target_distance2(raw_reg, filtered_proposals, annotation, table_fgs_positive_proposals)
                if (cls_mask != -1.0).sum() > 0:
                    clss_reg_prob_loss = compute_cls_reg_prob_loss(raw_cls, cls_mask)
                    clss_reg_loss = clss_reg_prob_loss + clss_reg_bbox_loss
                    clss_reg_prob_loss_epoch += clss_reg_prob_loss.item()
                else:
                    clss_reg_loss = clss_reg_bbox_loss
                #####

                clss_reg_bbox_loss_epoch += clss_reg_bbox_loss.item()
                clss_reg_loss_epoch += clss_reg_loss.item()

                total_loss = rpn_loss + clss_reg_loss
                show_all_results = True

                refined_proposals, clss_score = clss_reg.infer_bboxes(filtered_proposals, raw_reg, raw_cls)
            
            else:

                total_loss = rpn_loss
                show_all_results = False

            total_loss.backward()

            optimizer.step()

        lv.record(e, rpn_prob_loss_epoch / l, rpn_bbox_loss_epoch / l, rpn_loss_epoch / l, clss_reg_prob_loss_epoch / l, clss_reg_bbox_loss_epoch / l, clss_reg_loss_epoch / l)
        s = '\nEpoch {}: rpn_prob_loss: {} + rpn_bbox_loss: {} = {}'.format(e, rpn_prob_loss_epoch / l, rpn_bbox_loss_epoch / l, rpn_loss_epoch / l)
        s += '\n       : clss_reg_prob_loss: {} + clss_reg_bbox_loss: {} = {}'.format(clss_reg_prob_loss_epoch / l, clss_reg_bbox_loss_epoch / l, clss_reg_loss_epoch / l)
        # print((labels == -1).sum(), (labels == 0).sum(), (labels == 1).sum())
        # print()
        tqdm.write(s)

        if e % 10 == 0:
            
            for net in [fe_net, rpn_net, roi_net, clss_reg]: net.eval()
            with torch.no_grad():

                for i in range(proposals.size()[0]):
                    see_rpn_results(inv_normalize(img[i, :, :, :].clone().detach()).permute(1, 2, 0).numpy().copy(),
                                    table_gts_positive_anchors.detach().numpy().copy(), 
                                    proposals.detach().numpy().copy(), 
                                    F.softmax(cls_out, dim=2).detach().numpy().copy(),
                                    annotation.detach().numpy().copy(),
                                    rpn_net.anchors_parameters.detach().numpy().copy(),
                                    rpn_net.valid_anchors.detach().numpy().copy(), e)
                if show_all_results:
                    for i in range(proposals.size()[0]):
                        see_rpn_final_results(inv_normalize(img[i, :, :, :].clone().detach()).permute(1, 2, 0).numpy().copy(),
                                        filtered_proposals.detach().numpy().copy(), 
                                        probs_object.detach().numpy().copy(), 
                                        annotation.detach().numpy().copy(),
                                        e)

                    for i in range(refined_proposals.size()[0]):
                        see_final_results(inv_normalize(img[i, :, :, :].clone().detach()).permute(1, 2, 0).numpy().copy(),
                                        clss_score.detach().numpy().copy(), 
                                        refined_proposals.detach().numpy().copy(), 
                                        annotation.detach().numpy().copy(),
                                        e)
            for net in [fe_net, rpn_net, roi_net, clss_reg]: net.train()
    
    lv.save()


if __name__ == "__main__":
    main()