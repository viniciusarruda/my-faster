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
        max_iou = 0
        max_idx = None

        anchors_bbox = torch.zeros(anchors.size(), dtype=anchors.dtype, device=anchors.device)
        anchors_bbox[:, 0] = anchors[:, 0] - anchors[:, 2] * 0.5
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


def get_target_distance2(raw_reg, rpn_filtered_proposals, gts):

    assert raw_reg.size(0) == 1 # implemented for batch size 1

    sum_reg = 0
    bi = 0 

    txgt, tygt, twgt, thgt = parametrize_bbox(gts[bi, :].reshape(1, -1), rpn_filtered_proposals[bi, :, :])
    txp, typ, twp, thp = raw_reg[bi, :, 0], raw_reg[bi, :, 1], raw_reg[bi, :, 2], raw_reg[bi, :, 3]

    sum_reg += smooth_l1(txp - txgt, sigma=3)
    sum_reg += smooth_l1(typ - tygt, sigma=3)
    sum_reg += smooth_l1(twp - twgt, sigma=3)
    sum_reg += smooth_l1(thp - thgt, sigma=3)

    # without normalization to simplify as said in the paper
    return sum_reg # / d


def compute_rpn_prob_loss(probs_object, labels):

    assert labels.size(0) == 1 # implemented for batch size 1
    idxs = labels != -1.0
    # without normalization to simplify as said in the paper
    # todo so, reduction='mean'
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

