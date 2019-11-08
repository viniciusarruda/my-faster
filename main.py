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
from tqdm import tqdm

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# TODO
# FIXME 
# BUG 
# NOTE

#####################################

### About the image standard ###
# The images are in the format I(n_rows, n_cols), and indexed always as (r, c)
# The height and the width of an image is handled in the code as n_rows and n_cols, respectively.
# The (0, 0) point is at the top left corner of the image

### About the tensor standard ###


### ###
# The format of bounding box is in x,y,n_rows,n_cols unless the variable name contains a bbox word.

#####################################

# NOTE: For each modification, test and do not move until obtain the same anterior result (or better) 
# DONE: Assert the forward pass.
# DODE: Implement the loss and assert its corectness
# DONE: Implement the backward and assert its correctness
# TODO: Organize the code
# TODO: Increase the size of the input image
# TODO: Increase the number of images in the training set
# TODO: Implement the resnet as feature extractor
# TODO: Fix the parameters for the new feature extractor
# TODO: Implement the training strategy correctly
# TODO: Use the newest PyTorch version 

# TODO: during training, use only the valid anchors, but in test, all anchors..

# TODO: use image with different size (no multiple yet, but different sizes)
# TODO: more anchor ratios and scales
# TODO: single image with two (and then more) objects
# TODO: multiple images
# TODO: RCNN_top

# TODO: invalid anchors can be setted to -1 on labels ? (as dont care..)
# TODO: When processing the ground truths, show warnings when a certain GT box have no associated positive anchor

# TODO NOW: what is the behavior when is there a not assigned gt ?
# checar este behavior..
# eliminar esse gt caso for isso mesmo que eu estiver pensando, pois n vai servir de nada

# partir para o treino com batch balanceado!

def main():

    lv = LossViz()

    device = torch.device("cpu")
    epochs = 1000

    input_img_size = (224, 224)

    fe_net = FeatureExtractorNet().to(device)
    # fe_net = FeatureExtractorNetComplete().to(device)
    rpn_net = RPN(input_img_size=input_img_size, feature_extractor_out_dim=fe_net.out_dim, feature_extractor_size=fe_net.feature_extractor_size, receptive_field_size=fe_net.receptive_field_size, device=device).to(device)
    roi_net = ROI(input_img_size=input_img_size).to(device)
    clss_reg = ClassifierRegressor(input_img_size=input_img_size, input_size=7*7*fe_net.out_dim, n_classes=1).to(device)

    dataloader, input_img_size = get_dataloader(rpn_net.anchors_parameters, rpn_net.valid_anchors)

    # visual debug
    # show_anchors(rpn_net.anchors_parameters.detach().numpy().copy(), rpn_net.valid_anchors.detach().numpy().copy(), input_img_size)

    # for e, (img, annotation) in enumerate(dataloader):
    #     img, annotation = img.to(device), annotation[0, :, :].to(device)
    #     labels, table_gts_positive_anchors = anchor_labels(rpn_net.anchors_parameters, rpn_net.valid_anchors, annotation)
    #     labels, table_gts_positive_anchors = labels.to(device), table_gts_positive_anchors.to(device)
    #     show_masked_anchors(e, rpn_net.anchors_parameters.detach().numpy().copy(), rpn_net.valid_anchors.detach().numpy().copy(), labels.detach().numpy().copy(), table_gts_positive_anchors.detach().numpy().copy(), annotation.detach().numpy().copy(), input_img_size)
    # exit()

    params = list(fe_net.parameters()) + list(rpn_net.parameters()) + list(roi_net.parameters()) + list(clss_reg.parameters())

    params = [p for p in params if p.requires_grad == True]

    optimizer = torch.optim.Adam(params, lr=0.001)

    for net in [fe_net, rpn_net, roi_net, clss_reg]:
        net.train()

    l = len(dataloader)

    for e in tqdm(range(1, epochs+1)):

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


            ## Compute RPN loss ##
            # labels, table_gts_positive_anchors = anchor_labels(rpn_net.anchors_parameters, rpn_net.valid_anchors, annotation)
            # labels, table_gts_positive_anchors = labels.to(device), table_gts_positive_anchors.to(device)
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
                #TODO here, filter the number of positive and background
                todo here !
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
        print('\nEpoch {}: rpn_prob_loss: {} + rpn_bbox_loss: {} = {}'.format(e, rpn_prob_loss_epoch / l, rpn_bbox_loss_epoch / l, rpn_loss_epoch / l))
        print('       : clss_reg_prob_loss: {} + clss_reg_bbox_loss: {} = {}'.format(clss_reg_prob_loss_epoch / l, clss_reg_bbox_loss_epoch / l, clss_reg_loss_epoch / l))
        print((labels == -1).sum(), (labels == 0).sum(), (labels == 1).sum())
        print()

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