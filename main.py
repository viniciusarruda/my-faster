import torch
import torch.nn.functional as F
import numpy as np
from dataset_loader import get_dataloader
from feature_extractor import FeatureExtractorNet
from rpn import RPN
from roi import ROI
from classifier_regressor import ClassifierRegressor
from see_results import see_results, see_rpn_results, show_training_sample
from loss import anchor_labels, get_target_distance, compute_rpn_prob_loss, get_target_distance2
from PIL import Image
import time

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

start_time_count = None

def start_time():
    global start_time_count
    start_time_count = time.time()

def end_time(msg):
    print(msg, time.time() - start_time_count)

def main():

    device = torch.device("cpu")
    epochs = 100
    verbose = False

    dataloader, input_img_size = get_dataloader()

    fe_net = FeatureExtractorNet().to(device)
    rpn_net = RPN(input_img_size=input_img_size, feature_extractor_out_dim=fe_net.out_dim, receptive_field_size=fe_net.receptive_field_size, device=device).to(device)
    roi_net = ROI(input_img_size=input_img_size).to(device)
    clss_reg = ClassifierRegressor(input_img_size=input_img_size, input_size=7*7*12, n_classes=10 + 1).to(device)

    params = list(fe_net.parameters()) + list(rpn_net.parameters()) + list(roi_net.parameters()) + list(clss_reg.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01)

    for net in [fe_net, rpn_net, roi_net, clss_reg]:
        net.train()

    l = len(dataloader)

    for e in range(1, epochs+1):

        rpn_prob_loss_epoch, rpn_bbox_loss_epoch, rpn_loss_epoch = 0, 0, 0
        clss_reg_bbox_loss_epoch, clss_reg_loss_epoch = 0, 0
        
        for img, annotation in dataloader:

            img, annotation = img.to(device), annotation.to(device)

            optimizer.zero_grad()

            # print('Image size: {}'.format(img.size()))
            # print('Annotation size: {}'.format(annotation.size()))
            features = fe_net.forward(img)

            # print('Features size: {}'.format(features.size()))
            proposals, cls_out, filtered_proposals, probs_object = rpn_net.forward(features)

            # print('Proposals size: {}'.format(proposals.size()))
            # print('Probabilities object size: {}'.format(probs_object.size()))

            #####
            ## Compute RPN loss ##
            labels = anchor_labels(rpn_net.anchors_parameters, annotation).to(device)
            rpn_bbox_loss = get_target_distance(proposals, rpn_net.anchors_parameters, annotation, labels)
            rpn_prob_loss = compute_rpn_prob_loss(cls_out, labels)
            #####

            # ta no meio mas depois organizar.. ##############################################################
            rois = roi_net.forward(filtered_proposals, features)
            # print('Roi size: {}'.format(rois.size()))
            #
            refined_proposals, raw_reg, clss_score = clss_reg.forward(rois, filtered_proposals)
            # print('Refined proposals size: {}'.format(refined_proposals.size()))
            # print('Clss size: {}'.format(clss_score.size()))


            #####
            ## Compute .. loss ##
            clss_reg_bbox_loss = get_target_distance2(raw_reg, filtered_proposals, annotation)
            # labels = rpn_proposals_labels(filtered_proposals, annotation).to(device) # filtered mesmo ?
            # rpn_bbox_loss = get_target_distance(proposals, rpn_net.anchors_parameters, annotation, labels)
            # rpn_prob_loss = compute_rpn_prob_loss(cls_out, labels)
            #####

            clss_reg_loss = clss_reg_bbox_loss

            clss_reg_bbox_loss_epoch += clss_reg_bbox_loss.item()

            # ta no meio mas depois organizar.. ##############################################################

            rpn_loss = rpn_prob_loss + rpn_bbox_loss

            rpn_prob_loss_epoch += rpn_prob_loss.item()
            rpn_bbox_loss_epoch += rpn_bbox_loss.item()
            rpn_loss_epoch += rpn_loss.item()

            total_loss = clss_reg_loss + rpn_loss

            # rpn_loss.backward()
            total_loss.backward()
            optimizer.step()

        print('Epoch {}: rpn_prob_loss: {} + rpn_bbox_loss: {} = {}'.format(e, rpn_prob_loss_epoch / l, rpn_bbox_loss_epoch / l, rpn_loss_epoch / l))
        print('       : clss_reg_bbox_loss: {}'.format(clss_reg_bbox_loss_epoch / l))
        print()

        # with torch.no_grad():
        #     show_training_sample(img[0, :, :, :].permute(1, 2, 0).numpy().copy(), annotation.numpy().copy())

        # for net in [fe_net, rpn_net, roi_net, clss_reg]: net.eval()
        # with torch.no_grad():
        #     for i in range(proposals.size()[0]):
        #         see_rpn_results(img[i, :, :, :].permute(1, 2, 0).detach().numpy().copy(),
        #                         labels.detach().numpy().copy(), 
        #                         proposals.detach().numpy().copy(), 
        #                         F.softmax(probs_object, dim=2).detach().numpy().copy(),
        #                         annotation.detach().numpy().copy(),
        #                         rpn_net.anchors_parameters.detach().numpy().copy(), e)
        # for net in [fe_net, rpn_net, roi_net, clss_reg]: net.train()

        # rois = roi_net.forward(proposals, features)
        # print('Roi size: {}'.format(rois.size()))

        # # print('Ate aqui tudo certo !')

        # clss_out, bbox_out = clss_reg.forward(rois, proposals)

        # print('Clss size: {}'.format(clss_out.size()))
        # print('Bbox size: {}'.format(bbox_out.size()))

        # # clss_out: (batch_size, n_bboxes)
        # # bbox_out: (batch_size, n_bboxes, 4)

        # # for i in range(clss_out.size()[0]):
        # #     clss_out_np = clss_out[i, :].detach().numpy()
        # #     bbox_out_np = bbox_out[i, :, :].detach().numpy()
        # #     img_np = img[i, :, :, :].permute(1, 2, 0).numpy()
        # #     see_results(img_np, clss_out_np, bbox_out_np)


if __name__ == "__main__":
    main()