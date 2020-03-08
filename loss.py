import torch
import torch.nn.functional as F
from rpn import RPN
from bbox_utils import offset2bbox
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
    ret = torch.sum(ret)

    return ret


# TODO: This function is only called when generating the dataset!?
#       if so, change the location to other file as is didn't belong here.
def anchor_labels(anchors, valid_anchors, gts, negative_threshold=0.3, positive_threshold=0.7):
    # tem como simplificar e otimizar..
    
    anchors = anchors[valid_anchors, :]

    batch_size = gts.size(0) # number of annotations for one image
    mask = torch.zeros(batch_size, anchors.size(0))
    ious = torch.zeros(batch_size, anchors.size(0))

    anchors_bbox = torch.zeros(anchors.size(), dtype=anchors.dtype, device=anchors.device)
    anchors_bbox[:, 0] = anchors[:, 0] - 0.5 * (anchors[:, 2] - 1)  # como proceder com o lance do -1 ou +1 nesse caso ? na conversão dos bbox2offset e vice versa ?
    anchors_bbox[:, 1] = anchors[:, 1] - 0.5 * (anchors[:, 3] - 1)  # cuidadooooooooo p anchor eh assim, mas para proposal n .. caso for gerar label para proposal..
    anchors_bbox[:, 2] = anchors_bbox[:, 0] + anchors[:, 2] - 1.0
    anchors_bbox[:, 3] = anchors_bbox[:, 1] + anchors[:, 3] - 1.0

    anchors_bbox_area = anchors[:, 2] * anchors[:, 3]
    gt_area = gts[:, 2] * gts[:, 3]

    # # this snippet shows that the for loop can be vectorized.. just think a little more to vectorize the rest
    # # but, optimization is the final step!
    # ## INI - not included.. just testing.. 
    # print(anchors_bbox.size(), gts.size())
    # gts[:, 0] + gts[:, 2] - 1.0
    # all_max = torch.max(anchors_bbox, gts.unsqueeze(1))
    # print(all_max.size())

    # print(gts[:, 0].unsqueeze(1).size())
    # x0_max = torch.max(anchors_bbox[:, 0], gts[:, 0].unsqueeze(1))
    # print(x0_max.size())

    # # this below was inside the for
    # print(torch.equal(x0, x0_max[bi, :]))
    # print(torch.equal(x0, all_max[bi, :, 0]))
    # print(torch.equal(y0, all_max[bi, :, 1]))
    # ## END - not included.. just testing.. 

    for bi in range(batch_size): # maybe I can vectorize this?

        x0 = torch.max(anchors_bbox[:, 0], gts[bi, 0])
        y0 = torch.max(anchors_bbox[:, 1], gts[bi, 1])
        x1 = torch.min(anchors_bbox[:, 2], gts[bi, 0] + gts[bi, 2] - 1.0)
        y1 = torch.min(anchors_bbox[:, 3], gts[bi, 1] + gts[bi, 3] - 1.0)

        intersection = torch.clamp(x1 - x0 + 1.0, min=0.0) * torch.clamp(y1 - y0 + 1.0, min=0.0)

        union = anchors_bbox_area + gt_area[bi] - intersection
        iou = intersection / union

        ious[bi, :] = iou

    # Set negative anchors
    # This should be first because the next instructions can override the mask
    idxs = ious < negative_threshold
    mask[idxs] = -1.0

    # Set positive anchors
    idxs = ious > positive_threshold
    
    # It is possible that a certain anchor is positive assigned to more than one box.
    # So to handle this issue, this snippet makes the anchor belong only to the box with maximum IoU.
    idxs_cond = torch.argmax(ious, dim=0)
    cond = torch.zeros(batch_size, anchors.size(0), dtype=torch.bool)
    cond[idxs_cond, range(idxs_cond.size(0))] = True                 
    idxs = idxs & cond                     
    
    # It is possible that a certain box has no positive anchor assigned. This snippet handles this issue.
    # First, a mask with the available anchors is built.
    # Remember, they are available because iou < positive_threshold, 
    # and yet they can be used, trying to not leave any box without an assigned anchor.
    cond = torch.ones(batch_size, anchors.size(0), dtype=torch.bool)
    cond[:, idxs.nonzero()[:, 1]] = False # Removes the already assigned anchors.
    cond[idxs.nonzero()[:, 0], :] = False # Removes the box which already has an assigned anchor.
    # Then, the max IoU is obtained, generating a mask with it.
    idxs_amax = torch.argmax(ious, dim=1)
    amax_mask = torch.zeros(batch_size, anchors.size(0), dtype=torch.bool)
    amax_mask[range(idxs_amax.size(0)), idxs_amax] = True
    # Finally, the final mask is composed.
    # Only the argmax IoU that was not already assigned that can be used to assign the remaining boxes.
    idxs = idxs | (amax_mask & cond)  # always mutual exclusive due to `cond`, i.e., (idxs & (amax_mask & cond)).sum() is always zero.

    # Here, idxs tries to assign at least one positive anchor to each box.
    # There is no anchor belonging to more than one box!
    # It is possible that a certain box doesn't have an assigned anchor because, 
    # and only because, the anchor which it has the higher IoU it was already assigned to another box with an even higher IoU, and
    # it makes no sense to assign another unused anchor with a lower IoU because the algorithm may be "confused", i.e., resulting in an unstable training.

    # Finally the mask is applied
    mask[idxs] = 1.0

    # Get a table in the following format: [box_idx, anchor_idx] relating the box with its respective positive assigned anchor
    table_gts_positive_anchors = (mask == 1.0).nonzero()

    # Get a mask with anchors which are positive, don't care or negative.
    mask, _ = torch.max(mask, dim=0)

    # Reverse the standard to later facilitate the use
    # It was: middle (don't care) -> 0, negative -> -1 and positive -> 1
    # And now: middle (don't care) -> -1, negative -> 0 and positive -> 1
    idxs_middle = mask == 0.0
    idxs_negative = mask == -1.0

    mask[idxs_middle] = -1.0
    mask[idxs_negative] = 0.0

    return mask, table_gts_positive_anchors





# taking in consideration what Rich Sutton said (about getting computing power is better), is better to find a way to get rid of
# this expensive steps instead of trying to research other methods of doing object detection
def get_target_mask(filtered_proposals, gts, low_threshold=0.1, high_threshold=0.5):
    # tem como simplificar e otimizar..

    batch_size = gts.size(0)
    cls_mask = torch.zeros(batch_size, filtered_proposals.size(0))
    ious = torch.zeros(batch_size, filtered_proposals.size(0))

    filtered_bbox = offset2bbox(filtered_proposals)

    proposals_bbox_area = filtered_proposals[:, 2] * filtered_proposals[:, 3]
    
    gt_area = gts[:, 2] * gts[:, 3]

    for bi in range(batch_size): # maybe I can vectorize this?  

        x0 = torch.max(filtered_bbox[:, 0], gts[bi, 0])
        y0 = torch.max(filtered_bbox[:, 1], gts[bi, 1])
        x1 = torch.min(filtered_bbox[:, 2], gts[bi, 0] + gts[bi, 2] - 1)
        y1 = torch.min(filtered_bbox[:, 3], gts[bi, 1] + gts[bi, 3] - 1)

        intersection = torch.clamp(x1 - x0 + 1, min=0) * torch.clamp(y1 - y0 + 1, min=0)

        union = proposals_bbox_area + gt_area[bi] - intersection
        iou = intersection / union

        ious[bi, :] = iou
        

    # Set easy background cases as don't care
    idxs = ious < low_threshold
    cls_mask[idxs] = -1.0 

    # Set foreground cases
    idxs = ious > high_threshold

    # TODO: change the name filtered_proposals to rpn_proposals?

    # It is possible that a certain proposal is positive assigned to more than one box.
    # So to handle this issue, this snippet makes the proposal belong only to the box with maximum IoU.
    idxs_cond = torch.argmax(ious, dim=0)
    cond = torch.zeros(batch_size, filtered_proposals.size(0), dtype=torch.bool)
    cond[idxs_cond, range(idxs_cond.size(0))] = True
    idxs = idxs & cond    

    # TODO -> To Check! If right, keep commented.
    # It is possible that a certain box has no positive proposal assigned. 
    # But, unlike the anchor_labels() function,
    # I think that I should leave these boxes without a proposal assigned,
    # because when the RPN adjust these proposals, this function will consider as positive organically.

    cls_mask[idxs] = 1.0

    # idx_gt, idx_positive_proposal
    table_fgs_positive_proposals = (cls_mask == 1.0).nonzero() 

    # Do not needed to reverse like the anchor_label()
    cls_mask, _ = torch.max(cls_mask, dim=0)

    n_fg_proposals = table_fgs_positive_proposals.size(0)

    # NOTE: The snippet below is identical to the one at the dataset_loader.py.. maybe should make a function with it

    max_fg_proposals = int(config.fg_fraction * config.batch_size) 

    ### Select up to max_fg_proposals foreground proposals
    ### The excess is marked as don't care
    if n_fg_proposals > max_fg_proposals:
        # raise NotImplementedError('Warning, did not implemented!')
        print('\n======\n')
        print('n_fg_proposals > max_fg_proposals')
        print('OBSERVE IF IT IS BEHAVING RIGHT! IT SHOULD!')
        print('\n======\n')
        exit()
        fg_proposals_idxs = table_fgs_positive_proposals[:, 1]
        tmp_idxs = torch.randperm(n_fg_proposals)[:n_fg_proposals - max_fg_proposals]
        idxs_to_suppress = fg_proposals_idxs[tmp_idxs]
        cls_mask[idxs_to_suppress] = -1 # mark them as don't care
        n_fg_proposals = max_fg_proposals
        # TODO -> To Check!
        # The table_fgs_positive_proposals is not consistent with the cls_mask.
        # Should be consistent? Or this balancing is just for the cross-entropy loss?
    
    n_proposals_to_complete_batch = config.batch_size - n_fg_proposals

    # if n_proposals_to_complete_batch >= n_bg_proposals:
    #     # TODO -> see the note below: There is less proposals than the batch size.. just use the available ones?
    #     # NOTE: I decided to use just the available ones.. since this isn't commented anywhere.

    ### Fill the remaining batch with bg proposals
    # Annalyze if the `if` below has low rate of entrance.. if so, put the below line inside it to optimize
    bg_proposals_idxs = (cls_mask == 0).nonzero().squeeze()
    n_bg_proposals = bg_proposals_idxs.size(0)

    if n_bg_proposals > n_proposals_to_complete_batch:
        # Sample the bg_proposals to fill the remaining batch space
        tmp_idxs = torch.randperm(n_bg_proposals)[:n_bg_proposals - n_proposals_to_complete_batch]
        idxs_to_suppress = bg_proposals_idxs[tmp_idxs]
        # TODO -> To Check! 
        # I think the excess is also marked as dont care too (didn't confirm - didn't find anything about it)
        cls_mask[idxs_to_suppress] = -1 # mark them as don't care
    # else, just use the available ones, which is the default behavior

    return table_fgs_positive_proposals, cls_mask


def parametrize_bbox(bbox, a_bbox):

    assert bbox.size() == a_bbox.size()

    xa, ya, wa, ha = a_bbox[:, 0], a_bbox[:, 1], a_bbox[:, 2], a_bbox[:, 3]
    x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    tx = (x - xa) / wa
    ty = (y - ya) / ha
    tw = torch.log(w / wa)
    th = torch.log(h / ha)
    return tx, ty, tw, th

# TODO
# check this.. really need the table_gts_positive_anchors ?
# comment inserting the expected input and the output and its meaning (document phase)
def get_target_distance(proposals, anchors, valid_anchors, gts, table_gts_positive_anchors):

    anchors = anchors[valid_anchors, :]

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

# Jesus, need to comment this.. I didnt understand nothing!
def get_target_distance2(raw_reg, rpn_filtered_proposals, gts, table_fgs_positive_proposals):

    # txgt, tygt, twgt, thgt = parametrize_bbox(gts[bi, :].reshape(1, -1), rpn_filtered_proposals[bi, idxs[0, :], :])
    txgt, tygt, twgt, thgt = parametrize_bbox(gts[table_fgs_positive_proposals[:, 0], :], rpn_filtered_proposals[table_fgs_positive_proposals[:, 1], :])

    # txp, typ, twp, thp = raw_reg[bi, idxs[0, :], 0], raw_reg[bi, idxs[0, :], 1], raw_reg[bi, idxs[0, :], 2], raw_reg[bi, idxs[0, :], 3]
    txp, typ, twp, thp = raw_reg[table_fgs_positive_proposals[:, 1], 0], raw_reg[table_fgs_positive_proposals[:, 1], 1], raw_reg[table_fgs_positive_proposals[:, 1], 2], raw_reg[table_fgs_positive_proposals[:, 1], 3]

    assert txp.size() == txgt.size()

    sum_reg = smooth_l1(txp - txgt, sigma=3) + \
              smooth_l1(typ - tygt, sigma=3) + \
              smooth_l1(twp - twgt, sigma=3) + \
              smooth_l1(thp - thgt, sigma=3)

    # without normalization to simplify as said in the paper
    return sum_reg # / d


def compute_rpn_prob_loss(probs_object, labels):

    idxs = labels != -1.0  # considering all cares ! Just positive and negative samples !

    # without normalization to simplify as said in the paper
    # todo so, reduction='mean'
    # this has effect to consider the class 0 -> negative sample
    #                             the class 1 -> positive sample
    prob_loss = F.cross_entropy(probs_object[idxs, :], labels[idxs].long(), reduction='sum') 
    return prob_loss # / d


def compute_cls_reg_prob_loss(probs_object, labels):

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
