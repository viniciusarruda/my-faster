import torch
import torch.nn.functional as F
from dataset_loader import get_dataloader
from rpn import RPN
import numpy as np

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


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


def anchor_labels(anchors, gts, negative_threshold=0.3, positive_threshold=0.7):
    # tem como simplificar e otimizar..

    batch_size = gts.size()[0]
    mask = torch.zeros(batch_size, anchors.size(0))
    
    for bi in range(batch_size):

        anchors_bbox = torch.zeros(anchors.size(), dtype=anchors.dtype, device=anchors.device)
        anchors_bbox[:, 0] = anchors[:, 0] - anchors[:, 2] * 0.5  # como proceder com o lance do -1 ou +1 nesse caso ? na conversão dos bbox2offset e vice versa ?
        anchors_bbox[:, 1] = anchors[:, 1] - anchors[:, 3] * 0.5  # cuidadooooooooo p anchor eh assim, mas para proposal n .. caso for gerar label para proposal..
        anchors_bbox[:, 2] = anchors_bbox[:, 0] + anchors[:, 2]
        anchors_bbox[:, 3] = anchors_bbox[:, 1] + anchors[:, 3]

        anchors_bbox_area = anchors[:, 2] * anchors[:, 3]

        gt_area = (gts[bi, 2] - gts[bi, 0] + 1) * (gts[bi, 3] - gts[bi, 1] + 1)

        x0 = torch.max(anchors_bbox[:, 0], gts[bi, 0])
        y0 = torch.max(anchors_bbox[:, 1], gts[bi, 1])
        x1 = torch.min(anchors_bbox[:, 2], gts[bi, 2])
        y1 = torch.min(anchors_bbox[:, 3], gts[bi, 3])

        intersection = torch.clamp(x1 - x0 + 1, min=0) * torch.clamp(y1 - y0 + 1, min=0)

        union = anchors_bbox_area + gt_area - intersection
        iou = intersection / union

        mask[bi, iou > positive_threshold] = 1.0
        mask[bi, iou < negative_threshold] = -1.0
        mask[bi, torch.argmax(iou)] = 1.0 # se mudar para fazer com bath tem que colocar dim=1 ou outra dependendo do que for
        # else, mask = zero (it is initialized with zeros)

    return mask

# taking in consideration what that man said (about getting computing power is better), is better to find a way to get rid of
# this expensive steps instead of trying to research other methods of doing object detection
def get_target_mask(filtered_proposals, gts, low_threshold=0.1, high_threshold=0.5):
    # tem como simplificar e otimizar..

    batch_size = gts.size()[0]
    cls_mask = torch.zeros(batch_size, filtered_proposals.size(1))
    fg_mask = torch.zeros(batch_size, filtered_proposals.size(1))

    filtered_bbox = _offset2bbox(filtered_proposals)
    
    for bi in range(batch_size):

        proposals_bbox_area = filtered_proposals[bi, :, 2] * filtered_proposals[bi, :, 3]

        gt_area = (gts[bi, 2] - gts[bi, 0] + 1) * (gts[bi, 3] - gts[bi, 1] + 1)

        x0 = torch.max(filtered_bbox[bi, :, 0], gts[bi, 0])
        y0 = torch.max(filtered_bbox[bi, :, 1], gts[bi, 1])
        x1 = torch.min(filtered_bbox[bi, :, 2], gts[bi, 2])
        y1 = torch.min(filtered_bbox[bi, :, 3], gts[bi, 3])

        intersection = torch.clamp(x1 - x0 + 1, min=0) * torch.clamp(y1 - y0 + 1, min=0)

        union = proposals_bbox_area + gt_area - intersection
        iou = intersection / union

        # mask[bi, iou > high_threshold] = 1.0
        # mask[bi, torch.argmax(iou)] = 1.0 # se mudar para fazer com bath tem que colocar dim=1 ou outra dependendo do que for
        # acredito que este passo acima ainda eh necessario pois pode haver o caso de todo o iou ser abaixo de low_threshold ?!
        # else, mask = zero (it is initialized with zeros)

        fg_mask[bi, iou > high_threshold] = 1.0
        # fg_mask[bi, torch.argmax(iou)] = 1.0 # I think that this cannot be here..
        
        # logica provisoria, n eh generalizavel 
        cls_mask[bi, iou > high_threshold] = 1.0 # car
        cls_mask[bi, iou <= low_threshold] = -1.0 # easy cases (easy background cases), irrelevant
        # cls_mask[bi, (low_threshold < iou) & (iou <= high_threshold)] = 0.0 background -> already done when initialized

        # I dont know how the real Faster R-CNN leads with this issue.
        # I just handled it in this way to keep implementing, but I have not found in any materials how to handle it correctly 
        # TODO THIS IS A MUST !
        if (iou > low_threshold).sum() == 0.0:
            cls_mask[bi, torch.argmax(iou)] = 0.0 # put at least an easy case as a hardy background case to have a class loss

    return fg_mask, cls_mask

# already implemented, do not repeat in final code !
def _offset2bbox(proposals):
    """
    proposals: batch_size, -1, 4
    bboxes: batch_size, -1, 4

    """

    bx0 = proposals[:, :, 0]
    by0 = proposals[:, :, 1]
    bx1 = bx0 + proposals[:, :, 2] - 1
    by1 = by0 + proposals[:, :, 3] - 1

    bboxes = torch.stack((bx0, by0, bx1, by1), dim=2)

    return bboxes


def parametrize_bbox(bbox, a_bbox):

    xa, ya, wa, ha = a_bbox[:, 0], a_bbox[:, 1], a_bbox[:, 2], a_bbox[:, 3]
    x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    tx = (x - xa) / wa
    ty = (y - ya) / ha
    tw = torch.log(w / wa)
    th = torch.log(h / ha)
    return tx, ty, tw, th


def get_target_distance(proposals, anchors, gts, labels):

    assert labels.size(0) == 1 # implemented for batch size 1

    sum_reg = 0
    bi = 0 

    idxs = labels == 1.0

    txgt, tygt, twgt, thgt = parametrize_bbox(gts[bi, :].reshape(1, -1), anchors[idxs[0, :], :])
    txp, typ, twp, thp = parametrize_bbox(proposals[idxs, :], anchors[idxs[0, :], :])

    sum_reg += smooth_l1(txp - txgt, sigma=3)
    sum_reg += smooth_l1(typ - tygt, sigma=3)
    sum_reg += smooth_l1(twp - twgt, sigma=3)
    sum_reg += smooth_l1(thp - thgt, sigma=3)

    # without normalization to simplify as said in the paper
    return sum_reg # / d


def get_target_distance2(raw_reg, rpn_filtered_proposals, gts, target_mask):

    assert raw_reg.size(0) == 1 # implemented for batch size 1

    sum_reg = 0
    bi = 0 

    idxs = target_mask == 1.0 # da para gerar essa mascara ja no tipo de indice -> torch.uint64 ?

    txgt, tygt, twgt, thgt = parametrize_bbox(gts[bi, :].reshape(1, -1), rpn_filtered_proposals[bi, idxs[0, :], :])
    txp, typ, twp, thp = raw_reg[bi, idxs[0, :], 0], raw_reg[bi, idxs[0, :], 1], raw_reg[bi, idxs[0, :], 2], raw_reg[bi, idxs[0, :], 3]

    assert txp.size() == txgt.size()

    sum_reg += smooth_l1(txp - txgt, sigma=3)
    sum_reg += smooth_l1(typ - tygt, sigma=3)
    sum_reg += smooth_l1(twp - twgt, sigma=3)
    sum_reg += smooth_l1(thp - thgt, sigma=3)

    # without normalization to simplify as said in the paper
    return sum_reg # / d


def compute_rpn_prob_loss(probs_object, labels):

    assert labels.size(0) == 1 # implemented for batch size 1
    idxs = labels != -1.0  # considering all cares ! Just positive and negative samples !
    # without normalization to simplify as said in the paper
    # todo so, reduction='mean'
    # this has effect to consider the class 0 -> negative sample
    #                             the class 1 -> positive sample
    prob_loss = F.cross_entropy(probs_object[idxs, :], labels[idxs].long(), reduction='sum') 
    return prob_loss # / d


def compute_cls_reg_prob_loss(probs_object, labels):

    assert labels.size(0) == 1 # implemented for batch size 1
    idxs = labels != -1.0  # considering all cares ! Just backgrounds and cars samples !
    # without normalization to simplify as said in the paper
    # todo so, reduction='mean'
    # this has effect to consider the class 0 -> background
    #                             the class 1 -> car
    prob_loss = F.cross_entropy(probs_object[idxs, :], labels[idxs].long(), reduction='sum') 
    return prob_loss # / d


if __name__ == '__main__':

    input_img_size = (128, 128)
    feature_extractor_out_dim = 12
    receptive_field_size = 16
    dataloader, input_img_size = get_dataloader()

    rpn = RPN(input_img_size, feature_extractor_out_dim, receptive_field_size)

    for img, annotation in dataloader:

        labels = anchor_labels(rpn.anchors_parameters, annotation)

        ret = get_target_distance(rpn.anchors_parameters, annotation, labels)

        print(ret)

        exit()

    
# NAO DESISTE !!!!!!!!!!!!!!!!!

