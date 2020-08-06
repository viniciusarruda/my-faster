import config
import torch
import torch.nn as nn
from rpn import RPN
from roi import ROI
from loss import get_target_distance, get_target_distance2, get_target_mask, compute_prob_loss
import torch.nn.functional as F
from backbone import ToyBackbone, ResNetBackbone
from bbox_utils import clip_boxes, anchors_offset2bbox, anchors_bbox2offset
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

        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev):
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

        normal_init(self.rpn_net.conv_rpn, 0, 0.01)
        normal_init(self.rpn_net.cls_layer, 0, 0.01)
        normal_init(self.rpn_net.reg_layer, 0, 0.01)
        normal_init(self.fe_net.cls, 0, 0.01)
        normal_init(self.fe_net.reg, 0, 0.001)

    def forward(self, img, annotations, rpn_labels, expanded_annotations):

        if self.training:
            pre_nms_top_n = config.train_pre_nms_top_n
            pos_nms_top_n = config.train_pos_nms_top_n
        else:
            pre_nms_top_n = config.test_pre_nms_top_n
            pos_nms_top_n = config.test_pos_nms_top_n

        features = self.fe_net.base(img)

        proposals, cls_out, filtered_proposals, probs_object = self.rpn_net.forward(features, pre_nms_top_n, pos_nms_top_n)
        # probs_object.size()       -> torch.Size([#filtered_proposals]) #NOTE just for visualization.. temporary

        #####
        # essa linha de baixo tinha quee star logo abaixo da liha do rpn forward apos a filtragem
        # e no rpn n deveria ter aquela filtragem..
        # todo (ver se eh isso msm):
        # tirar parte de filtragem do rpn e colocar aqui (depois pensa em função)
        # na filtragem, vai filtrar tbm com a COND (uma matriz de filtragem) para filtrar usando os proprios indices da table_gts_positive_anchors
        # para saber se vai pra frente ou n, ou seja, gerando uma nova table_gts_positive_anchors para o regressor. com isso, a primeira coluna consegue indexas as classes.

        if self.training:
            proposal_annotations, filtered_proposals = get_target_mask(filtered_proposals, annotations)

        rois = self.roi_net.forward(filtered_proposals, features)

        raw_reg, raw_cls = self.fe_net.top_cls_reg(rois)

        if not self.training:
            refined_bboxes, clss_score, pred_clss_idxs = self.infer_bboxes(filtered_proposals, raw_reg, raw_cls)

            ret = [refined_bboxes, clss_score, pred_clss_idxs]

            if config.verbose:

                ret += [proposals,
                        F.softmax(cls_out, dim=1),
                        self.rpn_net.valid_anchors,
                        probs_object,
                        filtered_proposals]

            return ret

        # Compute RPN loss
        rpn_bbox_loss = get_target_distance(proposals, self.rpn_net.valid_anchors, expanded_annotations, rpn_labels)
        assert (rpn_labels == 1).sum() > 0
        assert (rpn_labels == 0).sum() > 0
        rpn_prob_loss = compute_prob_loss(cls_out, rpn_labels)

        rpn_loss = rpn_prob_loss + rpn_bbox_loss

        # Compute class_reg loss
        clss_reg_bbox_loss = get_target_distance2(raw_reg, filtered_proposals, proposal_annotations)
        clss_reg_prob_loss = compute_prob_loss(raw_cls, proposal_annotations[:, -1].long())

        clss_reg_loss = clss_reg_prob_loss + clss_reg_bbox_loss

        total_loss = rpn_loss + clss_reg_loss

        return rpn_prob_loss.item(), rpn_bbox_loss.item(), rpn_loss.item(), clss_reg_prob_loss.item(), clss_reg_bbox_loss.item(), clss_reg_loss.item(), total_loss

    def infer_bboxes(self, rpn_proposals, reg, clss):

        # assert reg.size(0) == clss.size(0)

        rpn_proposals = anchors_bbox2offset(rpn_proposals)

        clss_score = F.softmax(clss, dim=1)
        clss_idxs = clss_score.argmax(dim=1)
        clss_score = clss_score[torch.arange(clss_score.size(0)), clss_idxs]

        # Filter out the proposals which the net classifies as background
        idxs_non_background = clss_idxs != 0
        clss_idxs = clss_idxs[idxs_non_background]
        clss_score = clss_score[idxs_non_background]
        reg = reg[idxs_non_background, :]
        rpn_proposals = rpn_proposals[idxs_non_background, :]
        # enable this asserton here if uncomment this out
        assert (clss_idxs == 0).sum() == 0

        reg = reg.view(reg.size(0), config.n_classes, 4)
        reg = reg[torch.arange(reg.size(0)), clss_idxs, :]

        # De-normalize the target
        reg *= torch.tensor(config.BBOX_NORMALIZE_STDS).type_as(reg)

        #need an if visualization here! (because to compute mAP is all (i..e, >= 0.0))
        # Filter out lower scores
        if config.verbose:
            idxs_non_lower = clss_score > 0.3
            clss_idxs = clss_idxs[idxs_non_lower]
            clss_score = clss_score[idxs_non_lower]
            reg = reg[idxs_non_lower, :]
            rpn_proposals = rpn_proposals[idxs_non_lower, :]

        #make a funcion with this.. avoid repeated code.. and cetralize to avoid bugs
        # refine the bbox appling the bbox to px, py, pw and ph
        px = rpn_proposals[:, 0] + rpn_proposals[:, 2] * reg[:, 0]
        py = rpn_proposals[:, 1] + rpn_proposals[:, 3] * reg[:, 1]
        pw = rpn_proposals[:, 2] * torch.exp(reg[:, 2])
        ph = rpn_proposals[:, 3] * torch.exp(reg[:, 3])

        refined_proposals = torch.stack((px, py, pw, ph), dim=1)

        bboxes = anchors_offset2bbox(refined_proposals)
        bboxes = clip_boxes(bboxes, config.input_img_size)

        # Filter small bboxes (not implemented in that famous pytorch version)
        # cond = bboxes_filter_condition(bboxes)
        # bboxes, clss_score, clss_idxs = bboxes[cond, :], clss_score[cond], clss_idxs[cond]

        # apply NMS
        # bboxes, clss_score = nms(bboxes, clss_score)
        idxs_kept = nms(bboxes, clss_score, nms_threshold=0.3)
        refined_bboxes = bboxes[idxs_kept, :]
        clss_score = clss_score[idxs_kept]
        clss_idxs = clss_idxs[idxs_kept]

        return refined_bboxes, clss_score, clss_idxs
