import torch
import torch.nn.functional as F
from bbox_utils import anchors_bbox2offset, compute_iou
import config


# The PyTorch implementation doesn't provide a sigma parameter
# as the original implementation, so this implementation is kept.
def smooth_l1(x, sigma=3.0):

    sigma2 = sigma * sigma
    x_abs = torch.abs(x)

    cond = x_abs < (1.0 / sigma2)

    true_cond = sigma2 * x * x * 0.5
    false_cond = x_abs - (0.5 / sigma2)

    ret = torch.where(cond, true_cond, false_cond)

    # ret = torch.sum(ret) -> Other implementatios are getting the mean, not the sum!
    ret = torch.mean(ret)

    return ret


# TODO: This function is only called when generating the dataset!?
#       if so, change the location to other file as is didn't belong here.
def anchor_labels(anchors, annotations, negative_threshold=0.3, positive_threshold=0.7):

    # The anchors are associated independently of the bbox regression or classification

    ious = compute_iou(annotations, anchors)

    # RPN labels. Initializing with all anchors as don't care (i.e., -1)
    rpn_labels = torch.empty(anchors.size(0), dtype=torch.int64).fill_(-1)

    max_iou_anchors, argmax_iou_anchors = torch.max(ious, dim=0)
    max_iou_gt, _ = torch.max(ious, dim=1)

    # Set background anchors first
    # If the max iou for a given anchor is below the negative_threshold, hence all its ious
    # no matter with which gt, will be below the negative_threshold
    rpn_labels[max_iou_anchors < negative_threshold] = 0

    # Eq: Find the pair (bbox, anchor) which has the higher iou (doing this way to consider equal ious)
    # e.g., scale 0.5 with an anchor totally inside the bbox: 2 anchors for this bbox with same iou.
    # e.g., bbox inside an anchor, the other ratios of this anchor (i.e., with same area) that also
    # wraps the bbox will result in a same iou;
    # Any: Reduce the big match table to values deciding which anchor is foreground
    max_iou_gt[max_iou_gt == 0] = 1e-5  # To avoid non-overlaping bboxes to be selected (e.g. bbox near the image boarded where there is no valid anchors covering)
    match = torch.any(torch.eq(ious, max_iou_gt.view(-1, 1)), dim=0)

    # Set foreground anchors
    # If there is an anchor with iou with any bbox which is above the threshold, then mark as foreground
    rpn_labels[match | (max_iou_anchors > positive_threshold)] = 1

    # Expand the annotations to match the anchors shape
    # The background anchors (note: different thinking from rpn_labels!) are marked with
    # all zeros (including its class which already means background)
    expanded_annotations = torch.zeros(anchors.size(0), 5, dtype=annotations.dtype)
    # Taking care of only considering as foreground anchors the ones with iou > 0
    # Again, the strategy here is different from when generating the rpn_labels
    idxs = max_iou_anchors > 0.0
    assert idxs.sum() != 0  # so pq eu quero saber quando vai cair nesse caso para saber o comportamento .. acho praticamente impossivel
    expanded_annotations[idxs, :] = annotations[argmax_iou_anchors[idxs], :]

    return rpn_labels, expanded_annotations, argmax_iou_anchors[idxs] # this last is just to show (debug)

#inspect this when multiclass
def get_target_mask(rpn_filtered_proposals, annotations, low_threshold=0.1, high_threshold=0.5):

    # Add the bbox annotations into the proposals
    # IMO, the only purpose of this is to avoid empty proposals
    all_proposals = torch.cat((rpn_filtered_proposals, annotations[:, :-1]), dim=0)

    ious = compute_iou(annotations, all_proposals)
    max_iou_gt, argmax_iou_gt = torch.max(ious, dim=0)

    # ----------- Select foreground idxs ----------- #
    max_fg_idxs = int(config.fg_fraction * config.batch_size)

    # (using the max_iou to simplify as the results will be the same)
    fg_idxs = (max_iou_gt >= high_threshold).nonzero().squeeze(1)
    n_fg_idxs = fg_idxs.size(0)

    # Select up to max_fg_idxs foreground proposals
    if n_fg_idxs > max_fg_idxs:
        keep_idxs = torch.randperm(n_fg_idxs, device=annotations.device)[:n_fg_idxs - max_fg_idxs]
        fg_idxs = fg_idxs[keep_idxs]
        n_fg_idxs = max_fg_idxs
    # ---------------------------------------------- #

    # -------- Select hard background idxs --------- #
    # The easy backgroud cases (i.e., max_iou_gt < low_threshold) are ignored
    max_bg_idxs = config.batch_size - n_fg_idxs

    # (using the max_iou to simplify as the results will be the same)
    bg_idxs = ((low_threshold >= max_iou_gt) & (max_iou_gt < high_threshold)).nonzero().squeeze(1)
    n_bg_idxs = bg_idxs.size(0)

    if n_bg_idxs > max_bg_idxs:
        # Sample the bg_proposals to fill the remaining batch space
        keep_idxs = torch.randperm(n_bg_idxs, device=annotations.device)[:n_bg_idxs - max_bg_idxs]
        bg_idxs = bg_idxs[keep_idxs]
        n_bg_idxs = max_bg_idxs
    # ---------------------------------------------- #

    # Concatenate all foreground and hard background indexes
    keep_idxs = torch.cat((fg_idxs, bg_idxs), dim=0)

    # In contrast to RPN, here we also sample the target_bboxes
    # Remember: annotations have the classes attached in last dim
    #E se eu nao balancear as bbox tbm? treinar uma versao assim tbm para ver no que da
    expanded_annotations = annotations[argmax_iou_gt[keep_idxs], :]

    # Set the bg_idxs annotation labels to background class
    expanded_annotations[n_fg_idxs:, -1] = 0

    # Also filter the proposals, to lately match the target annotations in the loss
    proposals = all_proposals[keep_idxs, :]

    return expanded_annotations, proposals

#da pra passar o idx acomo parametro e colocar no lugar do : ao ives de fazer a mascara no get_target_distance
def _parametrize_bbox(bbox, a_bbox):

    assert bbox.size() == a_bbox.size()

    xa, ya, wa, ha = a_bbox[:, 0], a_bbox[:, 1], a_bbox[:, 2], a_bbox[:, 3]
    x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    tx = (x - xa) / wa
    ty = (y - ya) / ha
    tw = torch.log(w / wa)
    th = torch.log(h / ha)
    return tx, ty, tw, th

# rpn_bbox_loss = get_target_distance(proposals, self.rpn_net.valid_anchors, annotations[:, :-1], expanded_annotations)
def get_target_distance(proposals, valid_anchors, expanded_annotations):

    # Select only non-background bboxes (the background annotations are all zeros, i.e., padding)
    keep = expanded_annotations[:, -1] > 0

    proposals = anchors_bbox2offset(proposals[keep, :])
    anchors = anchors_bbox2offset(valid_anchors[keep, :])
    gts = anchors_bbox2offset(expanded_annotations[keep, :-1])

    txgt, tygt, twgt, thgt = _parametrize_bbox(gts, anchors)
    txp, typ, twp, thp = _parametrize_bbox(proposals, anchors)

    assert txp.size() == txgt.size()

    sum_reg = smooth_l1(txp - txgt, sigma=3) + \
        smooth_l1(typ - tygt, sigma=3) + \
        smooth_l1(twp - twgt, sigma=3) + \
        smooth_l1(thp - thgt, sigma=3)

    # without normalization to simplify as said in the paper  -> found that it is normalized on other implementations
    return sum_reg / 4.0


def get_target_distance2(raw_reg, rpn_filtered_proposals, proposal_annotations):

    # print(raw_reg.size())
    # print(rpn_filtered_proposals.size())
    # print(proposal_annotations.size())
    # # inspect this. raw_reg will change with multiclass
    # print('111')
    
    fg_idxs = proposal_annotations[:, -1] > 0

    rpn_filtered_proposals = anchors_bbox2offset(rpn_filtered_proposals[fg_idxs, :])
    target_bboxes = anchors_bbox2offset(proposal_annotations[fg_idxs, :-1])
    # -1 Because I did not include the prediction of background bboxes:
    target_classes = proposal_annotations[fg_idxs, -1].long() - 1


    # remove the zeros bboxes
    # agora vou ter que filtrar tbm os outros.. sera que n eh melhor deixar 
    # o background, e predizer ele como zero?

    txgt, tygt, twgt, thgt = _parametrize_bbox(target_bboxes, rpn_filtered_proposals)

    raw_reg = raw_reg.view(raw_reg.size(0), -1, 4)
    txp = raw_reg[fg_idxs, target_classes, 0]
    typ = raw_reg[fg_idxs, target_classes, 1]
    twp = raw_reg[fg_idxs, target_classes, 2]
    thp = raw_reg[fg_idxs, target_classes, 3]

    assert txp.size() == txgt.size()

    sum_reg = smooth_l1(txp - txgt, sigma=3) + \
        smooth_l1(typ - tygt, sigma=3) + \
        smooth_l1(twp - twgt, sigma=3) + \
        smooth_l1(thp - thgt, sigma=3)

    # without normalization to simplify as said in the paper  -> found that it is normalized on other implementations

    return sum_reg / 4.0


def compute_prob_loss(probs_object, labels):

    # this has effect to consider the class 0 -> negative sample (or background if is cls_reg loss)
    #                             the class 1 -> positive sample (or car if is cls_reg loss)
    # without normalization to simplify as said in the paper, todo so, reduction='mean'

    # ignore_index=-1: considering all cares ! Just positive and negative (or backgrounds and cars if is cls_reg loss) samples !

    # prob_loss = F.cross_entropy(probs_object, labels, reduction='sum', ignore_index=-1)

    prob_loss = F.cross_entropy(probs_object, labels, reduction='mean', ignore_index=-1)

    return prob_loss


# NAO DESISTE !!!!!!!!!!!!!!!!!
