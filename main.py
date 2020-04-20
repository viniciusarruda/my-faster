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
from loss import anchor_labels, get_target_distance, get_target_distance2, get_target_mask, compute_prob_loss
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

    fe_net = FeatureExtractorNet().to(device)
    # fe_net = FeatureExtractorNetComplete().to(device)
    rpn_net = RPN(input_img_size=config.input_img_size, feature_extractor_out_dim=fe_net.out_dim, feature_extractor_size=fe_net.feature_extractor_size, receptive_field_size=fe_net.receptive_field_size, device=device).to(device)
    roi_net = ROI(input_img_size=config.input_img_size).to(device)
    clss_reg = ClassifierRegressor(input_img_size=config.input_img_size, input_size=7*7*fe_net.out_dim, n_classes=1).to(device)

    dataloader = get_dataloader(rpn_net.anchors)

    # visual debug
    # show_anchors(rpn_net.anchors.detach().numpy().copy(), config.input_img_size)

    # for e, (img, annotation, _, _) in enumerate(dataloader):
    #     img, annotation = img.to(device), annotation[0, :, :].to(device)
    #     labels, table_gts_positive_anchors = anchor_labels(rpn_net.anchors, annotation)
    #     labels, table_gts_positive_anchors = labels.to(device), table_gts_positive_anchors.to(device)
    #     show_masked_anchors(e, rpn_net.anchors.detach().numpy().copy(), labels.detach().numpy().copy(), table_gts_positive_anchors.detach().numpy().copy(), annotation.detach().numpy().copy(), config.input_img_size)
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

            # show_training_sample(inv_normalize(img[0, :, :, :].clone().detach()).permute(1, 2, 0).numpy().copy(), annotation[0].detach().numpy().copy())

            # img.size()                        -> torch.Size([1, 3, input_img_size[0], input_img_size[1]])
            # annotation.size()                 -> torch.Size([1, #bboxes_in_img, 4])
            # labels.size()                     -> torch.Size([1, #valid_anchors])
            #                                      -1, 0 and 1 for dont care, negative and positive, respectively
            # table_gts_positive_anchors.size() -> torch.Size([1, #positive_anchors, 2]) 
            #                                      [idx of gt box, idxs of its assigned anchor on labels]

            # This implemention only supports one image per batch
            # Every batch channel is removed except for the image which will be forwarded through the feature extractor
            assert img.size(0) == annotation.size(0) == labels.size(0) == table_gts_positive_anchors.size(0) == 1
            img, annotation = img.to(device), annotation[0, :, :].to(device)
            labels, table_gts_positive_anchors = labels[0, :].to(device), table_gts_positive_anchors[0, :, :].to(device)
            # img.size()                        -> torch.Size([1, 3, input_img_size[0], input_img_size[1]])
            # annotation.size()                 -> torch.Size([#bboxes_in_img, 4])
            # labels.size()                     -> torch.Size([#valid_anchors])
            # table_gts_positive_anchors.size() -> torch.Size([#positive_anchors, 2]) 

            optimizer.zero_grad()

            features = fe_net.forward(img)
            # features.size() -> torch.Size([1, fe.out_dim, fe.feature_extractor_size, fe.feature_extractor_size])

            # The RPN handles the batch channel. The input (features) has the batch channel (asserted to be 1)
            # and outputs all the objects already handled
            proposals, cls_out, filtered_proposals, probs_object = rpn_net.forward(features)
            # proposals.size()          -> torch.Size([#valid_anchors, 4])
            # cls_out.size()            -> torch.Size([#valid_anchors, 2])
            # filtered_proposals.size() -> torch.Size([#filtered_proposals, 4])
            # probs_object.size()       -> torch.Size([#filtered_proposals]) #NOTE just for visualization.. temporary
            # The features object has its batch channel kept due to later use

            ## Compute RPN loss ##
            rpn_bbox_loss = get_target_distance(proposals, rpn_net.anchors, annotation, table_gts_positive_anchors)
            rpn_prob_loss = compute_prob_loss(cls_out, labels)
            #####

            rpn_loss = 10 * rpn_prob_loss + rpn_bbox_loss

            rpn_prob_loss_epoch += rpn_prob_loss.item()
            rpn_bbox_loss_epoch += rpn_bbox_loss.item()
            rpn_loss_epoch += rpn_loss.item()

            # if there is any proposal which is classified as an object
            if filtered_proposals.size(0) > 0: 

                rois = roi_net.forward(filtered_proposals, features)
                # rois.size()      -> torch.Size([#filtered_proposals, fe.out_dim, roi_net.out_dim, roi_net.out_dim])

                raw_reg, raw_cls = clss_reg.forward(rois)
                # raw_reg.size()   -> torch.Size([#filtered_proposals, 4])
                # raw_cls.size()   -> torch.Size([#filtered_proposals, 2])

                #####
                ## Compute class_reg loss ##
                table_fgs_positive_proposals, cls_mask = get_target_mask(filtered_proposals, annotation)
                clss_reg_bbox_loss = get_target_distance2(raw_reg, filtered_proposals, annotation, table_fgs_positive_proposals)
                if (cls_mask != -1.0).sum() > 0:
                    clss_reg_prob_loss = compute_prob_loss(raw_cls, cls_mask)
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

                see_rpn_results(inv_normalize(img[0, :, :, :].clone().detach()).permute(1, 2, 0).numpy().copy(),
                                table_gts_positive_anchors.detach().numpy().copy(), 
                                proposals.detach().numpy().copy(), 
                                F.softmax(cls_out, dim=1).detach().numpy().copy(),
                                annotation.detach().numpy().copy(),
                                rpn_net.anchors.detach().numpy().copy(), e)
                
                if show_all_results:
                    see_rpn_final_results(inv_normalize(img[0, :, :, :].clone().detach()).permute(1, 2, 0).numpy().copy(),
                                    filtered_proposals.detach().numpy().copy(), 
                                    probs_object.detach().numpy().copy(), 
                                    annotation.detach().numpy().copy(),
                                    e)

                    see_final_results(inv_normalize(img[0, :, :, :].clone().detach()).permute(1, 2, 0).numpy().copy(),
                                    clss_score.detach().numpy().copy(), 
                                    refined_proposals.detach().numpy().copy(), 
                                    annotation.detach().numpy().copy(),
                                    e)
            for net in [fe_net, rpn_net, roi_net, clss_reg]: net.train()
    
    lv.save()


if __name__ == "__main__":
    main()