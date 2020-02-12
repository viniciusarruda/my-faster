import torch
import torch.nn.functional as F
from rpn import RPN
from bbox_utils import offset2bbox
import numpy as np


def smooth_l1(x, sigma=3):

    # tentar melhorar depois com alguma funcao pronta do pytorch de if e aplicas

    sigma2 = sigma * sigma
    x_abs = torch.abs(x)

    cond = x_abs < (1.0 / sigma2)

    true_cond = sigma2 * x * x * 0.5
    false_cond = x_abs - (0.5 / sigma2)

    ret = torch.where(cond, true_cond, false_cond)

    ret = torch.sum(ret)

    return ret



# def anchor_labels(anchors, valid_anchors, gts, negative_threshold=0.3, positive_threshold=0.7): # era 0.3 no negative..
#     # tem como simplificar e otimizar..
    
#     anchors = anchors[valid_anchors, :]

#     batch_size = gts.size(0) # number of annotations for one image
#     mask = torch.zeros(batch_size, anchors.size(0)) - 1.0

#     if batch_size != 1:
#         print('WARNING: IMPLEMENT THE MASK CORRECTLY, EVEN FOR THE CASE IF ONE ANCHOR BELONGS TO TWO GTS. ALSO, ONLY ONE ARGMAX')
#         exit()
    
#     for bi in range(batch_size):

#         anchors_bbox = torch.zeros(anchors.size(), dtype=anchors.dtype, device=anchors.device)
#         anchors_bbox[:, 0] = anchors[:, 0] - 0.5 * (anchors[:, 2] - 1)  # como proceder com o lance do -1 ou +1 nesse caso ? na conversão dos bbox2offset e vice versa ?
#         anchors_bbox[:, 1] = anchors[:, 1] - 0.5 * (anchors[:, 3] - 1)  # cuidadooooooooo p anchor eh assim, mas para proposal n .. caso for gerar label para proposal..
#         anchors_bbox[:, 2] = anchors_bbox[:, 0] + anchors[:, 2] - 1
#         anchors_bbox[:, 3] = anchors_bbox[:, 1] + anchors[:, 3] - 1

#         anchors_bbox_area = anchors[:, 2] * anchors[:, 3]

#         gt_area = gts[bi, 2] * gts[bi, 3]

#         x0 = torch.max(anchors_bbox[:, 0], gts[bi, 0])
#         y0 = torch.max(anchors_bbox[:, 1], gts[bi, 1])
#         x1 = torch.min(anchors_bbox[:, 2], gts[bi, 0] + gts[bi, 2] - 1)
#         y1 = torch.min(anchors_bbox[:, 3], gts[bi, 1] + gts[bi, 3] - 1)

#         intersection = torch.clamp(x1 - x0 + 1, min=0) * torch.clamp(y1 - y0 + 1, min=0)

#         union = anchors_bbox_area + gt_area - intersection
#         iou = intersection / union

#         mask[bi, iou > positive_threshold] = 1.0
#         mask[bi, iou < negative_threshold] = 0.0
#         mask[bi, torch.argmax(iou)] = 1.0 # se mudar para fazer com bath tem que colocar dim=1 ou outra dependendo do que for
#         # else, mask = -1.0 (it is initialized with zeros - 1)    dont care

#     # print((mask == -1.0).sum(), (mask == 0.0).sum(), (mask == 1.0).sum())

#     # idx_gt, idx_positive_anchor
#     table_gts_positive_anchors = (mask == 1.0).nonzero()

#     mask = torch.squeeze(mask)

#     return mask, table_gts_positive_anchors

# TODO:
# Selecionar 128 anchoras negativas aleatoriamente
# selecionar 128 anchoras positivas aleatoriamente
# def anchor_labels(anchors, valid_anchors, gts, negative_threshold=0.3, positive_threshold=0.7): # era 0.3 no negative..
#     # tem como simplificar e otimizar..
    
#     anchors = anchors[valid_anchors, :]

#     batch_size = gts.size(0) # number of annotations for one image
#     mask = torch.zeros(batch_size, anchors.size(0)) - 1.0

#     anchors_bbox = torch.zeros(anchors.size(), dtype=anchors.dtype, device=anchors.device) # or should copy to not cause any side effects later
    
#     for bi in range(batch_size):

#         max_iou, max_iou_idx = None, None

#         for ai in range(anchors_bbox.size(0)):

#             anchors_bbox[ai, 0] = anchors[ai, 0] - 0.5 * (anchors[ai, 2] - 1)  # como proceder com o lance do -1 ou +1 nesse caso ? na conversão dos bbox2offset e vice versa ?
#             anchors_bbox[ai, 1] = anchors[ai, 1] - 0.5 * (anchors[ai, 3] - 1)  # cuidadooooooooo p anchor eh assim, mas para proposal n .. caso for gerar label para proposal..
#             anchors_bbox[ai, 2] = anchors_bbox[ai, 0] + anchors[ai, 2] - 1
#             anchors_bbox[ai, 3] = anchors_bbox[ai, 1] + anchors[ai, 3] - 1

#             anchors_bbox_area = anchors[ai, 2] * anchors[ai, 3]

#             gt_area = gts[bi, 2] * gts[bi, 3]

#             x0 = torch.max(anchors_bbox[ai, 0], gts[bi, 0])
#             y0 = torch.max(anchors_bbox[ai, 1], gts[bi, 1])
#             x1 = torch.min(anchors_bbox[ai, 2], gts[bi, 0] + gts[bi, 2] - 1)
#             y1 = torch.min(anchors_bbox[ai, 3], gts[bi, 1] + gts[bi, 3] - 1)

#             intersection = torch.clamp(x1 - x0 + 1, min=0) * torch.clamp(y1 - y0 + 1, min=0)

#             union = anchors_bbox_area + gt_area - intersection
#             iou = intersection / union

#             if iou > positive_threshold:
#                 mask[bi, ai] = 1.0
#             elif iou < negative_threshold:
#                 mask[bi, ai] = 0.0

#             if max_iou is None or iou > max_iou:
#                 max_iou = iou
#                 max_iou_idx = ai
        
#         if (mask[bi, :] == 1.0).sum() == 0:            
#             mask[bi, max_iou_idx] = 1.0

#             # mask[bi, iou > positive_threshold] = 1.0
#             # mask[bi, iou < negative_threshold] = 0.0
#             # mask[bi, torch.argmax(iou)] = 1.0 # se mudar para fazer com bath tem que colocar dim=1 ou outra dependendo do que for
#             # else, mask = -1.0 (it is initialized with zeros - 1)    dont care

#     # print((mask == -1.0).sum(), (mask == 0.0).sum(), (mask == 1.0).sum())

#     # idx_gt, idx_positive_anchor
#     table_gts_positive_anchors = (mask == 1.0).nonzero()

#     # print(table_gts_positive_anchors)

#     # generate a clean mask (rename ?)
#     clean_mask = torch.zeros(anchors_bbox.size(0)) - 1.0
#     for ai in range(anchors_bbox.size(0)):

#         if (mask[:, ai] == 1.0).sum() > 0:
#             clean_mask[ai] = 1.0
#         elif (mask[:, ai] == 0.0).sum() == batch_size:
#             clean_mask[ai] = 0.0

#     # WARNING: IT IS POSSIBLE TO ASSIGN ONE ANCHOR TO MORE THAN ONE GT,
#     # THE CORRECT IS TO ASSIGN AN ANCHOR TO A GT IN WHICH COMPUTED THE GREATEST IOU
#     # FOR THE CURRENT DATA THIS IS NOT A PROBLEM

#     # print((clean_mask == -1.0).sum(), (clean_mask == 0.0).sum(), (clean_mask == 1.0).sum())

#     return clean_mask, table_gts_positive_anchors



def anchor_labels(anchors, valid_anchors, gts, negative_threshold=0.3, positive_threshold=0.7): # era 0.3 no negative..
    # tem como simplificar e otimizar..
    
    anchors = anchors[valid_anchors, :]

    batch_size = gts.size(0) # number of annotations for one image
    mask = torch.zeros(batch_size, anchors.size(0))
    ious = torch.zeros(batch_size, anchors.size(0))
    
    for bi in range(batch_size):

        anchors_bbox = torch.zeros(anchors.size(), dtype=anchors.dtype, device=anchors.device)
        anchors_bbox[:, 0] = anchors[:, 0] - 0.5 * (anchors[:, 2] - 1)  # como proceder com o lance do -1 ou +1 nesse caso ? na conversão dos bbox2offset e vice versa ?
        anchors_bbox[:, 1] = anchors[:, 1] - 0.5 * (anchors[:, 3] - 1)  # cuidadooooooooo p anchor eh assim, mas para proposal n .. caso for gerar label para proposal..
        anchors_bbox[:, 2] = anchors_bbox[:, 0] + anchors[:, 2] - 1
        anchors_bbox[:, 3] = anchors_bbox[:, 1] + anchors[:, 3] - 1

        anchors_bbox_area = anchors[:, 2] * anchors[:, 3]

        gt_area = gts[bi, 2] * gts[bi, 3]

        x0 = torch.max(anchors_bbox[:, 0], gts[bi, 0])
        y0 = torch.max(anchors_bbox[:, 1], gts[bi, 1])
        x1 = torch.min(anchors_bbox[:, 2], gts[bi, 0] + gts[bi, 2] - 1)
        y1 = torch.min(anchors_bbox[:, 3], gts[bi, 1] + gts[bi, 3] - 1)

        intersection = torch.clamp(x1 - x0 + 1, min=0) * torch.clamp(y1 - y0 + 1, min=0)

        union = anchors_bbox_area + gt_area - intersection
        iou = intersection / union

        ious[bi, :] = iou

    # set positive anchors
    idxs = ious > positive_threshold
    idxs_cond = torch.argmax(ious, dim=0)
    cond = torch.zeros(batch_size, anchors.size(0), dtype=torch.bool) # this is to handle the possibility of an anchor to belong to more than one gt
    cond[idxs_cond, range(idxs_cond.size(0))] = True                      # it will only belong to the maximum iou
    idxs_amax = torch.argmax(ious, dim=1)  # this may introduce an anchor to belong to more than one gt
    idxs = idxs & cond                     # and to check (get the second argmax) it will be expensive
    idxs[range(idxs_amax.size(0)), idxs_amax] = True
    mask[idxs] = 1.0

    # set negative anchors
    idxs = ious < negative_threshold
    mask[idxs] = -1.0

    # mask[bi, iou > positive_threshold] = 1.0
    # mask[bi, iou < negative_threshold] = 0.0
    # mask[bi, torch.argmax(iou)] = 1.0 # se mudar para fazer com bath tem que colocar dim=1 ou outra dependendo do que for
    # else, mask = -1.0 (it is initialized with zeros - 1)    dont care

    # idx_gt, idx_positive_anchor
    table_gts_positive_anchors = (mask == 1.0).nonzero() 

    mask, _ = torch.max(mask, dim=0)

    # reverse to middle -> -1, negative -> 0 and positive -> 1
    idxs_middle = mask == 0.0
    idxs_negative = mask == -1.0

    mask[idxs_middle] = -1.0
    mask[idxs_negative] = 0.0

    return mask, table_gts_positive_anchors





# taking in consideration what that man said (about getting computing power is better), is better to find a way to get rid of
# this expensive steps instead of trying to research other methods of doing object detection
def get_target_mask(filtered_proposals, gts, low_threshold=0.1, high_threshold=0.5):
    # tem como simplificar e otimizar..

    assert filtered_proposals.size(0) == 1 # implemented for batch size 1

    batch_size = gts.size()[0]
    cls_mask = torch.zeros(batch_size, filtered_proposals.size(1))
    fg_mask = torch.zeros(batch_size, filtered_proposals.size(1))
    ious = torch.zeros(batch_size, filtered_proposals.size(1))

    filtered_bbox = offset2bbox(filtered_proposals)
    
    for bi in range(batch_size):

        proposals_bbox_area = filtered_proposals[0, :, 2] * filtered_proposals[0, :, 3]

        gt_area = gts[bi, 2] * gts[bi, 3]

        x0 = torch.max(filtered_bbox[0, :, 0], gts[bi, 0])
        y0 = torch.max(filtered_bbox[0, :, 1], gts[bi, 1])
        x1 = torch.min(filtered_bbox[0, :, 2], gts[bi, 0] + gts[bi, 2] - 1)
        y1 = torch.min(filtered_bbox[0, :, 3], gts[bi, 1] + gts[bi, 3] - 1)

        intersection = torch.clamp(x1 - x0 + 1, min=0) * torch.clamp(y1 - y0 + 1, min=0)

        union = proposals_bbox_area + gt_area - intersection
        iou = intersection / union

        ious[bi, :] = iou


    # set positive anchors
    idxs = ious > high_threshold
    idxs_cond = torch.argmax(ious, dim=0)
    cond = torch.zeros(batch_size, filtered_proposals.size(1), dtype=torch.bool) # this is to handle the possibility of an anchor to belong to more than one gt
    cond[idxs_cond, range(idxs_cond.size(0))] = True                      # it will only belong to the maximum iou
    # idxs_amax = torch.argmax(ious, dim=1)  # this may introduce an anchor to belong to more than one gt, and to check (get the second argmax) it will be expensive
    idxs = idxs & cond    
    # idxs[range(idxs_amax.size(0)), idxs_amax] = 1.0 # bellow is written "I think that this cannot be here.."

    fg_mask[idxs] = 1.0
    cls_mask[idxs] = 1.0

    # set negative anchors
    idxs = ious <= low_threshold
    cls_mask[idxs] = -1.0   # easy cases (easy background cases), irrelevant

    # idx_gt, idx_positive_anchor
    table_fgs_positive_proposals = (fg_mask == 1.0).nonzero() 

    # do not needed to reverse like the anchor_label()
    cls_mask, _ = torch.max(cls_mask, dim=0)

    #TODO here, filter the number of positive and background
    # todo here !

    # print(table_fgs_positive_proposals)
    # print((cls_mask == -1).sum(), (cls_mask == 0).sum(), (cls_mask == 1).sum())
    # exit()

    # TODO
    if (cls_mask == 1).sum() > 16:
        raise NotImplementedError('Warning, did not implemented!')

    # TODO
    if (cls_mask == 0).sum() > 48:
        raise NotImplementedError('Warning, did not implemented!')

    return table_fgs_positive_proposals, cls_mask

        # # mask[bi, iou > high_threshold] = 1.0
        # # mask[bi, torch.argmax(iou)] = 1.0 # se mudar para fazer com bath tem que colocar dim=1 ou outra dependendo do que for
        # # acredito que este passo acima ainda eh necessario pois pode haver o caso de todo o iou ser abaixo de low_threshold ?!
        # # else, mask = zero (it is initialized with zeros)bi, iou > high_thresholdbi, iou > high_thresholdbi, iou > high_thresholdbi, iou > high_threshold

        # fg_mask[bi, iou > high_threshold] = 1.0
        # # fg_mask[bi, torch.argmax(iou)] = 1.0 # I think that this cannot be here..
        
        # # logica provisoria, n eh generalizavel 
        # cls_mask[bi, iou > high_threshold] = 1.0 # carbi, iou > high_thresholdbi, iou > high_threshold
        # cls_mask[bi, iou <= low_threshold] = -1.0 # easy cases (easy background cases), irrelevant
        # # cls_mask[bi, (low_threshold < iou) & (iou <= high_threshold)] = 0.0 background -> already done when initialized

        # # I dont know how the real Faster R-CNN leads with this issue.
        # # I just handled it in this way to keep implementing, but I have not found in any materials how to handle it correctly 
        # # TODO THIS IS A MUST !
        # # if (iou > low_threshold).sum() == 0.0:
        #     # cls_mask[bi, torch.argmax(iou)] = 0.0 # put at least an easy case as a hardy background case to have a class loss
        #     # input('CLASS MASK WITHOUT ANY VALUE, TYPE ANYTHING TO CONTINUE TO SEE WHAT WILL HAPPEN:')

        # # this print is to help debug the TODO above.. in case of break of code, explore here !
        # # print((fg_mask == 0).nonzero().size())
        # # print((fg_mask == 1).nonzero().size())

        # # print((cls_mask == -1).nonzero().size())
        # # print((cls_mask == 0).nonzero().size())
        # # print((cls_mask == 1).nonzero().size())

        # # exit()



def parametrize_bbox(bbox, a_bbox):

    assert bbox.size() == a_bbox.size()

    xa, ya, wa, ha = a_bbox[:, 0], a_bbox[:, 1], a_bbox[:, 2], a_bbox[:, 3]
    x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    tx = (x - xa) / wa
    ty = (y - ya) / ha
    tw = torch.log(w / wa)
    th = torch.log(h / ha)
    return tx, ty, tw, th


def get_target_distance(proposals, anchors, valid_anchors, gts, table_gts_positive_anchors):

    anchors = anchors[valid_anchors, :]

    assert proposals.size(0) == 1 # implemented for batch size 1

    proposals = proposals[0, :, :] # batch size is 1 (just one image!)

    sum_reg = 0

    # txgt, tygt, twgt, thgt = parametrize_bbox(gts[bi, :].reshape(1, -1), anchors[idxs[0, :], :])
    txgt, tygt, twgt, thgt = parametrize_bbox(gts[table_gts_positive_anchors[:, 0], :], anchors[table_gts_positive_anchors[:, 1], :])

    # txp, typ, twp, thp = parametrize_bbox(proposals[idxs, :], anchors[idxs[0, :], :])
    txp, typ, twp, thp = parametrize_bbox(proposals[table_gts_positive_anchors[:, 1], :], anchors[table_gts_positive_anchors[:, 1], :])

    assert txp.size() == txgt.size()

    sum_reg += smooth_l1(txp - txgt, sigma=3)
    sum_reg += smooth_l1(typ - tygt, sigma=3)
    sum_reg += smooth_l1(twp - twgt, sigma=3)
    sum_reg += smooth_l1(thp - thgt, sigma=3)

    # without normalization to simplify as said in the paper
    return sum_reg # / d


def get_target_distance2(raw_reg, rpn_filtered_proposals, gts, table_fgs_positive_proposals):

    assert raw_reg.size(0) == 1 and rpn_filtered_proposals.size(0) == 1 # implemented for batch size 1

    raw_reg = raw_reg[0, :, :] # batch size is 1 (just one image!)
    rpn_filtered_proposals = rpn_filtered_proposals[0, :, :] # batch size is 1 (just one image!)
    
    sum_reg = 0

    # txgt, tygt, twgt, thgt = parametrize_bbox(gts[bi, :].reshape(1, -1), rpn_filtered_proposals[bi, idxs[0, :], :])
    txgt, tygt, twgt, thgt = parametrize_bbox(gts[table_fgs_positive_proposals[:, 0], :], rpn_filtered_proposals[table_fgs_positive_proposals[:, 1], :])

    # txp, typ, twp, thp = raw_reg[bi, idxs[0, :], 0], raw_reg[bi, idxs[0, :], 1], raw_reg[bi, idxs[0, :], 2], raw_reg[bi, idxs[0, :], 3]
    txp, typ, twp, thp = raw_reg[table_fgs_positive_proposals[:, 1], 0], raw_reg[table_fgs_positive_proposals[:, 1], 1], raw_reg[table_fgs_positive_proposals[:, 1], 2], raw_reg[table_fgs_positive_proposals[:, 1], 3]

    assert txp.size() == txgt.size()

    sum_reg += smooth_l1(txp - txgt, sigma=3)
    sum_reg += smooth_l1(typ - tygt, sigma=3)
    sum_reg += smooth_l1(twp - twgt, sigma=3)
    sum_reg += smooth_l1(thp - thgt, sigma=3)

    # without normalization to simplify as said in the paper
    return sum_reg # / d


def compute_rpn_prob_loss(probs_object, labels):

    assert probs_object.size(0) == 1 # implemented for batch size 1
    probs_object = probs_object[0, :, :] # make this filtration when producethis tensor.. to do not spread out stuff around the code

    idxs = labels != -1.0  # considering all cares ! Just positive and negative samples !

    # without normalization to simplify as said in the paper
    # todo so, reduction='mean'
    # this has effect to consider the class 0 -> negative sample
    #                             the class 1 -> positive sample
    prob_loss = F.cross_entropy(probs_object[idxs, :], labels[idxs].long(), reduction='sum') 
    return prob_loss # / d


def compute_cls_reg_prob_loss(probs_object, labels):

    assert probs_object.size(0) == 1 # implemented for batch size 1
    probs_object = probs_object[0, :, :] # make this filtration when producethis tensor.. to do not spread out stuff around the code

    idxs = labels != -1.0  # considering all cares ! Just backgrounds and cars samples !
    
    # without normalization to simplify as said in the paper
    # todo so, reduction='mean'
    # this has effect to consider the class 0 -> background
    #                             the class 1 -> car

    # if labels[idxs].long().size(0) == 0:
    #     # TODO
    #     print('CLASS MASK WITHOUT ANY VALUE, IT WILL CONTINUE BUT SHOULD SEE THIS')
    #     return torch.zeros(1)

    prob_loss = F.cross_entropy(probs_object[idxs, :], labels[idxs].long(), reduction='sum') 
    return prob_loss # / d

    
# NAO DESISTE !!!!!!!!!!!!!!!!!
