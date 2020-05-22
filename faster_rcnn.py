import config
import torch
import torch.nn as nn
from rpn import RPN
from roi import ROI
from loss import get_target_distance, get_target_distance2, get_target_mask, compute_prob_loss
from dataset_loader import inv_normalize
import torch.nn.functional as F
from backbone import ToyBackbone, ResNetBackbone
from bbox_utils import bbox2offset, offset2bbox, clip_boxes, bboxes_filter_condition, anchors_offset2bbox
from nms import nms


class FasterRCNN(nn.Module):

    def __init__(self):

        super(FasterRCNN, self).__init__()

        # define the net components
        print('\nUsing {}\n'.format(config.backbone))
        if config.backbone == 'Toy':
            self.fe_net = ToyBackbone()
        elif 'ResNet' in config.backbone:
            self.fe_net = ResNetBackbone()
        else:
            raise NotImplementedError('{} does not exist.'.format(config.backbone))
        self.rpn_net = RPN(input_img_size=config.input_img_size, feature_extractor_out_dim=self.fe_net.out_dim, feature_extractor_size=self.fe_net.feature_extractor_size, receptive_field_size=self.fe_net.receptive_field_size)
        self.roi_net = ROI(input_img_size=config.input_img_size)

    def forward(self, img, annotation, clss_idxs, labels_objectness, labels_class, table_gts_positive_anchors):

        features = self.fe_net.base(img)
        # features.size() -> torch.Size([1, fe.out_dim, fe.feature_extractor_size, fe.feature_extractor_size])

        # The RPN handles the batch channel. The input (features) has the batch channel (asserted to be 1)
        # and outputs all the objects already handled
        proposals, cls_out, filtered_proposals, probs_object, filtered_labels_class = self.rpn_net.forward(features, labels_class)
        # proposals.size()          -> torch.Size([#valid_anchors, 4])
        # cls_out.size()            -> torch.Size([#valid_anchors, 2])
        # filtered_proposals.size() -> torch.Size([#filtered_proposals, 4])
        # probs_object.size()       -> torch.Size([#filtered_proposals]) #NOTE just for visualization.. temporary
        # The features object has its batch channel kept due to later use

        #####
        # essa linha de baixo tinha quee star logo abaixo da liha do rpn forward apos a filtragem
        # e no rpn n deveria ter aquela filtragem..
        # todo (ver se eh isso msm):
        # tirar parte de filtragem do rpn e colocar aqui (depois pensa em função)
        # na filtragem, vai filtrar tbm com a COND (uma matriz de filtragem) para filtrar usando os proprios indices da table_gts_positive_anchors
        # para saber se vai pra frente ou n, ou seja, gerando uma nova table_gts_positive_anchors para o regressor. com isso, a primeira coluna consegue indexas as classes.

        table_fgs_positive_proposals, cls_mask, filtered_proposals = get_target_mask(filtered_proposals, annotation, clss_idxs, filtered_labels_class)
        # Now, the bug has been fixed.
        # The solution was to also use the gtboxes in the filtered_proposals set as seen in the original implementation (not mentioned in the paper and any other material)
        # This will add the gt as "proposals" with cls_mask == 1 to them.
        # Thus, this assertion must never fail
        assert filtered_proposals.size(0) > 0 and (cls_mask != -1.0).sum() > 0  # keep this assertion here until the code is ready

        # The filtered_proposals will act as the anchors in the RPN
        # and the table_gts_positive_proposals will act as the table_gts_positive_anchors in the RPN

        # Compute RPN loss #
        rpn_bbox_loss = get_target_distance(proposals, self.rpn_net.anchors, annotation, table_gts_positive_anchors)
        rpn_prob_loss = compute_prob_loss(cls_out, labels_objectness)
        #####

        # rpn_loss = 10 * rpn_prob_loss + rpn_bbox_loss
        rpn_loss = rpn_prob_loss + rpn_bbox_loss

        # rpn_prob_loss_epoch += rpn_prob_loss.item()
        # rpn_bbox_loss_epoch += rpn_bbox_loss.item()
        # rpn_loss_epoch += rpn_loss.item()

        # check how many times enters here to try to remove this if
        rois = self.roi_net.forward(filtered_proposals, features)
        # rois.size()      -> torch.Size([#filtered_proposals, fe.out_dim, roi_net.out_dim, roi_net.out_dim])

        raw_reg, raw_cls = self.fe_net.top_cls_reg(rois)
        # raw_reg, raw_cls = self.clss_reg.forward(tmp)
        # raw_reg.size()   -> torch.Size([#filtered_proposals, 4])
        # raw_cls.size()   -> torch.Size([#filtered_proposals, 2])

        # Compute class_reg loss ##
        clss_reg_bbox_loss = get_target_distance2(raw_reg, filtered_proposals, annotation, table_fgs_positive_proposals)
        clss_reg_prob_loss = compute_prob_loss(raw_cls, cls_mask)
        clss_reg_loss = clss_reg_prob_loss + clss_reg_bbox_loss

        # clss_reg_bbox_loss_epoch += clss_reg_bbox_loss.item()
        # clss_reg_loss_epoch += clss_reg_loss.item()

        total_loss = rpn_loss + clss_reg_loss
        # total_loss_epoch += total_loss.item() # note this shoulb below the else!

        return rpn_prob_loss.item(), rpn_bbox_loss.item(), rpn_loss.item(), clss_reg_prob_loss.item(), clss_reg_bbox_loss.item(), clss_reg_loss.item(), total_loss

    # NOTE note que este codigo eh identico ao do treino porem sem a loss e backward.. teria como fazer essa funcao funcionar para ambos treino e inferencia?
    # quero mostrar tbm na iter zero, antes de iniciar o treino
    def infer(self, epoch, dataset, device):

        output = []

        # for net in [self.fe_net, self.rpn_net, self.roi_net, self.clss_reg]: net.eval()
        # for net in [self.fe_net, self.rpn_net, self.roi_net]: net.eval()
        self.eval()

        with torch.no_grad():

            # for ith, (img, annotation, labels, table_gts_positive_anchors) in enumerate(dataloader):
            # there is a random number being generated inside the Dataloader: https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
            # in the final version, use the dataloader if is more fancy
            for ith in range(len(dataset)):

                img, annotation, clss_idxs, labels_objectness, labels_class, table_gts_positive_anchors = dataset[ith]

                img = img.unsqueeze(0)
                annotation = annotation.unsqueeze(0)
                clss_idxs = clss_idxs.unsqueeze(0)
                labels_objectness = labels_objectness.unsqueeze(0)
                labels_class = labels_class.unsqueeze(0)
                table_gts_positive_anchors = table_gts_positive_anchors.unsqueeze(0)

                assert img.size(0) == annotation.size(0) == clss_idxs.size(0) == labels_objectness.size(0) == labels_class.size(0) == table_gts_positive_anchors.size(0) == 1
                img, annotation, clss_idxs = img.to(device), annotation[0, :, :].to(device), clss_idxs[0, :].to(device)
                labels_objectness, labels_class, table_gts_positive_anchors = labels_objectness[0, :].to(device), labels_class[0, :].to(device), table_gts_positive_anchors[0, :, :].to(device)

                features = self.fe_net.base(img)
                # proposals, cls_out, filtered_proposals, probs_object = self.rpn_net.forward(features)
                proposals, cls_out, filtered_proposals, probs_object, filtered_labels_class = self.rpn_net.forward(features, labels_class)

                # if there is any proposal which is classified as an object
                if filtered_proposals.size(0) > 0:  # this if has to be implemented inside the visualization?

                    rois = self.roi_net(filtered_proposals, features)
                    raw_reg, raw_cls = self.fe_net.top_cls_reg(rois)

                    show_all_results = True

                    refined_proposals, clss_score, pred_clss_idxs = self.infer_bboxes(filtered_proposals, raw_reg, raw_cls)

                else:
                    print('Reproduce this.. got no filtered proposals when testing...')
                    print('no bbox proposals by RPN while inferring')
                    exit()
                    clss_score = None
                    pred_clss_idxs = None
                    show_all_results = False

                ith_output = [epoch]

                ith_output += [inv_normalize(img[0, :, :, :].clone().detach())]
                ith_output += [offset2bbox(annotation)]
                ith_output += [clss_idxs]

                ith_output += [table_gts_positive_anchors]
                ith_output += [offset2bbox(proposals)]
                ith_output += [F.softmax(cls_out, dim=1)]
                ith_output += [anchors_offset2bbox(self.rpn_net.anchors)]

                ith_output += [show_all_results]

                ith_output += [probs_object]
                ith_output += [offset2bbox(filtered_proposals)]

                if show_all_results:

                    ith_output += [clss_score]
                    ith_output += [pred_clss_idxs]
                    ith_output += [offset2bbox(refined_proposals)]

                else:

                    ith_output += [None]
                    ith_output += [None]
                    ith_output += [None]

                output.append(ith_output)

        return output

    def infer_bboxes(self, rpn_proposals, reg, clss):

        assert reg.size(0) == clss.size(0)

        clss_score = F.softmax(clss, dim=1)
        clss_idxs = clss_score.argmax(dim=1)
        clss_score = clss_score[torch.arange(clss_score.size(0)), clss_idxs]

        # Filter out the proposals which the net classifies as background
        idxs_non_background = clss_idxs != 0
        clss_idxs = clss_idxs[idxs_non_background]
        clss_score = clss_score[idxs_non_background]
        reg = reg[idxs_non_background, :]
        rpn_proposals = rpn_proposals[idxs_non_background, :]

        assert (clss_idxs == 0).sum() == 0

        reg = reg.view(reg.size(0), config.n_classes - 1, 4)
        reg = reg[torch.arange(reg.size(0)), clss_idxs - 1, :]

        # Filter out lower scores
        idxs_non_lower = clss_score >= 0.7
        # idxs_non_lower = clss_score >= 0.01 ## I am getting all clss_scores really lows
        clss_idxs = clss_idxs[idxs_non_lower]
        clss_score = clss_score[idxs_non_lower]
        reg = reg[idxs_non_lower, :]
        rpn_proposals = rpn_proposals[idxs_non_lower, :]

        # refine the bbox appling the bbox to px, py, pw and ph
        px = rpn_proposals[:, 0] + rpn_proposals[:, 2] * reg[:, 0]
        py = rpn_proposals[:, 1] + rpn_proposals[:, 3] * reg[:, 1]
        pw = rpn_proposals[:, 2] * torch.exp(reg[:, 2])
        ph = rpn_proposals[:, 3] * torch.exp(reg[:, 3])

        refined_proposals = torch.stack((px, py, pw, ph), dim=1)

        bboxes = offset2bbox(refined_proposals)
        bboxes = clip_boxes(bboxes, config.input_img_size)

        # bboxes, clss_score = filter_boxes(bboxes, clss_score)
        cond = bboxes_filter_condition(bboxes)  # ??????? no rpn tem isso, fazer aqui tbm ? na verdade achei no codigo oficial que faz isso no teste sim mas no treino n.. confirmar este ultimo (treino n)
        bboxes, clss_score, clss_idxs = bboxes[cond, :], clss_score[cond], clss_idxs[cond]
        # apply NMS
        # bboxes, clss_score = nms(bboxes, clss_score)
        idxs_kept = nms(bboxes, clss_score, nms_threshold=0.5)
        bboxes = bboxes[idxs_kept, :]
        clss_score = clss_score[idxs_kept]
        clss_idxs = clss_idxs[idxs_kept]

        refined_proposals = bbox2offset(bboxes)

        return refined_proposals, clss_score, clss_idxs
