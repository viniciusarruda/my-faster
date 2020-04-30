import torch
import torch.nn.functional as F
from rpn import RPN
from bbox_utils import offset2bbox, anchors_offset2bbox, compute_iou
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
# TODO: what would happen if everything below positive threshold is set to zero ? (not object)
#       maybe the work of the cls_reg could be less painful ? to think deeply
#       the function of cls_reg dos not seems to only to improve and get the classes of the bbox
#       but to help a lot the RPN! Think hard about this.. 
def anchor_labels(anchors, gts, class_idxs, negative_threshold=0.3, positive_threshold=0.7):

    anchors_bbox = anchors_offset2bbox(anchors)

    ious = compute_iou(gts, anchors_bbox, anchors)

    batch_size = gts.size(0) # number of annotations for one image
    mask = torch.zeros(batch_size, anchors.size(0), dtype=torch.int64)
    mask_class = torch.zeros(anchors.size(0), dtype=torch.int64)

    # Set negative anchors
    # This should be first because the next instructions can override the mask
    idxs = ious < negative_threshold
    mask[idxs] = -1

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
    mask[idxs] = 1

    # Get a table in the following format: [box_idx, anchor_idx] relating the box with its respective positive assigned anchor
    table_gts_positive_anchors = (mask == 1).nonzero()

    # Get a mask with anchors which are positive, don't care or negative.
    mask_objectness, _ = torch.max(mask, dim=0)

    # Reverse the standard to later facilitate the use
    # It was: middle (don't care) -> 0, negative -> -1 and positive -> 1
    # And now: middle (don't care) -> -1, negative -> 0 and positive -> 1
    idxs_middle = mask_objectness == 0
    idxs_negative = mask_objectness == -1

    mask_objectness[idxs_middle] = -1
    mask_objectness[idxs_negative] = 0

    mask_class[table_gts_positive_anchors[:, 1]] = class_idxs[table_gts_positive_anchors[:, 0]]

    return mask_objectness, mask_class, table_gts_positive_anchors


def get_target_mask(rpn_filtered_proposals, gts, clss_idxs, rpn_filtered_labels_class, low_threshold=0.1, high_threshold=0.5):

    # here I have to get a clone of the labels class ? or not because it was already filtered? i think the same is for the labes_objectness
    # por via das duvidas fazer o clone, quando for pra otimizar mexo nisso.

    all_proposals = torch.cat((rpn_filtered_proposals, gts), dim=0)
    all_labels_class = torch.cat((rpn_filtered_labels_class, clss_idxs), dim=0)
    # all_proposals = rpn_filtered_proposals
    # all_labels_class = rpn_filtered_labels_class

    rpn_filtered_bbox = offset2bbox(all_proposals)

    ious = compute_iou(gts, rpn_filtered_bbox, all_proposals)

    # print(rpn_filtered_bbox.size())
    # print(rpn_filtered_labels_class.size())
    # print(ious.size())

    batch_size = gts.size(0)
    cls_mask = torch.zeros(batch_size, all_proposals.size(0), dtype=torch.int64, device=gts.device)

    # Set easy background cases as don't care
    idxs = ious < low_threshold
    cls_mask[idxs] = -1

    # print(cls_mask)

    # Set foreground cases
    idxs = ious > high_threshold

    # It is possible that a certain proposal is positive assigned to more than one box.
    # So to handle this issue, this snippet makes the proposal belong only to the box with maximum IoU.
    idxs_cond = torch.argmax(ious, dim=0)
    cond = torch.zeros(batch_size, all_proposals.size(0), dtype=torch.bool, device=idxs.device)
    cond[idxs_cond, range(idxs_cond.size(0))] = True
    idxs = idxs & cond    

    # It is possible that a certain box has no positive proposal assigned. 
    # But, unlike the anchor_labels() function,
    # I think that I should leave these boxes without a proposal assigned,
    # because when the RPN adjust these proposals, this function will consider as positive organically.

    # Finally the mask is applied
    cls_mask[idxs] = 1

    # idx_gt, idx_positive_proposal
    table_fgs_positive_proposals = (cls_mask == 1).nonzero() 
    # TODO there is a problem here.. maybe some positive be marked as negative.. so making the table_fgs_positive_proposals not consistent.

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
    #     # NOTE: I decided to use just the available ones.. since this isn't commented anywhere. (keep reading the comments)

    ### Fill the remaining batch with bg proposals
    # Annalyze if the `if` below has low rate of entrance.. if so, put the below line inside it to optimize
    bg_proposals_idxs = (cls_mask == 0).nonzero().squeeze(1)
    n_bg_proposals = bg_proposals_idxs.size(0)

    # TODO remove the if checker if it is entering too much times here as was done in the dataset_loader.py
    if n_bg_proposals > n_proposals_to_complete_batch:
        print('there is a problem here.. maybe some positive be marked as negative.. so making the table_fgs_positive_proposals not consistent.')
        exit()
        # Sample the bg_proposals to fill the remaining batch space
        tmp_idxs = torch.randperm(n_bg_proposals)[:n_bg_proposals - n_proposals_to_complete_batch]
        idxs_to_suppress = bg_proposals_idxs[tmp_idxs]
        cls_mask[idxs_to_suppress] = -1 # mark them as don't care
    # else, just use the available ones, which is the default behavior

    # print()
    # print(cls_mask)
    # print(cls_mask.size())
    # print(rpn_filtered_labels_class)
    # print((cls_mask == -1).sum())
    # print((cls_mask == 0).sum())
    # print((cls_mask == 1).sum())
    # print()
    # print(table_fgs_positive_proposals)
    # print(table_fgs_positive_proposals.size())
    # print()

    # checar se for negativo como comentado assima nos printf
    # print(cls_mask)
    # print(cls_mask.size())
    # print(rpn_filtered_labels_class)
    cls_mask[table_fgs_positive_proposals[:, 1]] = all_labels_class[table_fgs_positive_proposals[:, 1]]
    # print(cls_mask)
    # print()
    # print(cls_mask)
    # print(cls_mask.size())

    # print()
    # table_fgs_positive_proposals2 = torch.zeros(table_fgs_positive_proposals.size(0), 3, dtype=table_fgs_positive_proposals.dtype)
    # table_fgs_positive_proposals2[:, :2] = table_fgs_positive_proposals
    # table_fgs_positive_proposals2[:, 2] = cls_mask[table_fgs_positive_proposals[:, 1]]
    table_fgs_positive_proposals = torch.cat((table_fgs_positive_proposals, cls_mask[table_fgs_positive_proposals[:, 1]].unsqueeze(1)), dim=1)
    non_background_idxs = table_fgs_positive_proposals[:, 2] != 0
    table_fgs_positive_proposals = table_fgs_positive_proposals[non_background_idxs]
    table_fgs_positive_proposals[:, 2] -= 1 # fixing the classes - removing the background because it is not predicted, just ignored
    # exit()
    # print((cls_mask == -1).sum())
    # print((cls_mask == 0).sum())
    # print((cls_mask == 1).sum())
    
    return table_fgs_positive_proposals, cls_mask, all_proposals


def _parametrize_bbox(bbox, a_bbox):

    assert bbox.size() == a_bbox.size()

    xa, ya, wa, ha = a_bbox[:, 0], a_bbox[:, 1], a_bbox[:, 2], a_bbox[:, 3]
    x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    tx = (x - xa) / wa
    ty = (y - ya) / ha
    tw = torch.log(w / wa)
    th = torch.log(h / ha)
    return tx, ty, tw, th


def get_target_distance(proposals, anchors, gts, table_gts_positive_anchors):

    gts_idxs, anchors_idxs = table_gts_positive_anchors[:, 0], table_gts_positive_anchors[:, 1]

    txgt, tygt, twgt, thgt = _parametrize_bbox(gts[gts_idxs, :], anchors[anchors_idxs, :])
    txp, typ, twp, thp = _parametrize_bbox(proposals[anchors_idxs, :], anchors[anchors_idxs, :])

    assert txp.size() == txgt.size()

    sum_reg = smooth_l1(txp - txgt, sigma=3) + \
              smooth_l1(typ - tygt, sigma=3) + \
              smooth_l1(twp - twgt, sigma=3) + \
              smooth_l1(thp - thgt, sigma=3)

    # without normalization to simplify as said in the paper
    return sum_reg # / d


def get_target_distance2(raw_reg, rpn_filtered_proposals, gts, table_fgs_positive_proposals):

    # TODO ainda n to filtrando backgroud
    # print('---')
    # print(raw_reg.size())
    # print(raw_reg.view(raw_reg.size(0), -1, 4).size())
    # print(raw_reg.view(raw_reg.size(0), -1, 4)[:, table_fgs_positive_proposals[:, 2], :].size())
    # print()
    # print(rpn_filtered_proposals)
    # print(rpn_filtered_proposals.size())
    # print()
    # print(gts)
    # print(gts.size()) #continue from here.. after modification check the diference of the saved 
    # print()
    # print(table_fgs_positive_proposals)
    # print(table_fgs_positive_proposals.size())
    # print('---')
    # gts_idxs, proposals_idxs = table_fgs_positive_proposals[:, 0], table_fgs_positive_proposals[:, 1]
    # print(gts[gts_idxs, :])
    # print()
    # print(rpn_filtered_proposals[proposals_idxs, :])
    # print('===')
    # print(raw_reg)
    # raw_reg = raw_reg.view(raw_reg.size(0), -1, 4)
    # txp = raw_reg[proposals_idxs, table_fgs_positive_proposals[:, 2], 0]
    # print(txp)
    # print(table_fgs_positive_proposals)
    # print(gts_idxs)
    # print(proposals_idxs)
    # print(table_fgs_positive_proposals[:, 2])
    # print('===')
    # exit()

    # cls_idx = 0 na tabela fgs

    # raw_reg = raw_reg.view(raw_reg.size(0), -1, 4)
    # raw_reg = raw_reg[:, idxs_from_table, :]

    gts_idxs, proposals_idxs = table_fgs_positive_proposals[:, 0], table_fgs_positive_proposals[:, 1]

    txgt, tygt, twgt, thgt = _parametrize_bbox(gts[gts_idxs, :], rpn_filtered_proposals[proposals_idxs, :])

    raw_reg = raw_reg.view(raw_reg.size(0), -1, 4)
    txp = raw_reg[proposals_idxs, table_fgs_positive_proposals[:, 2], 0]
    typ = raw_reg[proposals_idxs, table_fgs_positive_proposals[:, 2], 1]
    twp = raw_reg[proposals_idxs, table_fgs_positive_proposals[:, 2], 2]
    thp = raw_reg[proposals_idxs, table_fgs_positive_proposals[:, 2], 3]

    # txp = raw_reg[proposals_idxs, 0]
    # typ = raw_reg[proposals_idxs, 1]
    # twp = raw_reg[proposals_idxs, 2]
    # thp = raw_reg[proposals_idxs, 3]

    assert txp.size() == txgt.size()

    sum_reg = smooth_l1(txp - txgt, sigma=3) + \
              smooth_l1(typ - tygt, sigma=3) + \
              smooth_l1(twp - twgt, sigma=3) + \
              smooth_l1(thp - thgt, sigma=3)

    # without normalization to simplify as said in the paper

    return sum_reg # / d


def compute_prob_loss(probs_object, labels):

    # this has effect to consider the class 0 -> negative sample (or background if is cls_reg loss)
    #                             the class 1 -> positive sample (or car if is cls_reg loss)
    # without normalization to simplify as said in the paper, todo so, reduction='mean'

    # ignore_index=-1: considering all cares ! Just positive and negative (or backgrounds and cars if is cls_reg loss) samples !

    prob_loss = F.cross_entropy(probs_object, labels, reduction='sum', ignore_index=-1) 
    
    return prob_loss # / d

    
# NAO DESISTE !!!!!!!!!!!!!!!!!
