import torch
import torch.nn.functional as F
from bbox_utils import anchors_bbox2offset, compute_iou
import config


# The PyTorch implementation doesn't provide a sigma parameter
# as the original implementation, so this implementation is kept.
# TODO replace with https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss
def smooth_l1(proposals, targets, sigma=1.0): # sigma parece estar diferente na implementacaodo outro.. original deve ser 3! mas a implementacao esta 1

    assert proposals.size() == targets.size()

    diff = proposals - targets

    sigma2 = sigma * sigma
    x_abs = torch.abs(diff)

    cond = x_abs < (1.0 / sigma2)
    true_cond = sigma2 * diff * diff * 0.5
    false_cond = x_abs - (0.5 / sigma2)

    ret = torch.where(cond, true_cond, false_cond)

    # TODO WARNING: the sum is only in the non-batch channels.
    #               if batch > 1, there is a mean over the batch channel
    # Also, pay attention with the behavior when computing the final regression bbox loss
    return torch.sum(ret)



# TODO: This function is only called when generating the dataset!?
#       if so, change the location to other file as is didn't belong here.
# should call get rpn_target() -> call on dataloader, include here the balancing step
def anchor_labels(anchors, annotations, negative_threshold=0.3, positive_threshold=0.7):

    # The anchors are associated independently of the bbox regression or classification
    # print('gt: ', annotations, annotations.size())
    # print('anchors: ', anchors, anchors.size())
    ious = compute_iou(annotations, anchors)
    # print('ious 0: ', ious[0, ious.nonzero()[ious.nonzero()[:, 0] == 0, 1]], ious[0, ious.nonzero()[ious.nonzero()[:, 0] == 0, 1]].size())
    # print('ious 1: ', ious[1, ious.nonzero()[ious.nonzero()[:, 0] == 1, 1]], ious[1, ious.nonzero()[ious.nonzero()[:, 0] == 1, 1]].size())
    # print('ious 2: ', ious[2, ious.nonzero()[ious.nonzero()[:, 0] == 2, 1]], ious[2, ious.nonzero()[ious.nonzero()[:, 0] == 2, 1]].size())
    # print('ious 3: ', ious[3, ious.nonzero()[ious.nonzero()[:, 0] == 3, 1]], ious[3, ious.nonzero()[ious.nonzero()[:, 0] == 3, 1]].size())
    # print('ious 4: ', ious[4, ious.nonzero()[ious.nonzero()[:, 0] == 4, 1]], ious[4, ious.nonzero()[ious.nonzero()[:, 0] == 4, 1]].size())

    # RPN labels. Initializing with all anchors as don't care (i.e., -1)
    rpn_labels = torch.empty(anchors.size(0), dtype=torch.int64).fill_(-1)

    # max_iou_anchors, argmax_iou_anchors = torch.max(ious, dim=0)
    # TODO this should not make difference, transpose just to make it equal to that implementation, but i prefer mine..
    max_iou_anchors, argmax_iou_anchors = torch.max(torch.transpose(ious, 0, 1), dim=1)
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

    # # OLD VERSION ## HERE, ONLY THE ONES WITH SOME OVERLAP WITH AN ANCHOR IS CONSIDERED DO REGRESS TO IT
    # # Expand the annotations to match the anchors shape
    # # The background anchors (note: different thinking from rpn_labels!) are marked with
    # # all zeros (including its class which already means background)
    # expanded_annotations = torch.zeros(anchors.size(0), 5, dtype=annotations.dtype)
    # # Taking care of only considering as foreground anchors the ones with iou > 0
    # # Again, the strategy here is different from when generating the rpn_labels
    # idxs = max_iou_anchors > 0.0
    # assert idxs.sum() > 0  # so pq eu quero saber quando vai cair nesse caso para saber o comportamento .. acho praticamente impossivel
    # expanded_annotations[idxs, :] = annotations[argmax_iou_anchors[idxs], :]

    # # NEW VERSION ## HERE, ALL OUTPUT SHOULD TARGET AN BBOX, EVEN IF IS REALLY FAR - NOT OVERLAPING
    # THIS IS HOW IT IS IMPLEMENTED IN THAT IMPLEMENTATION.
    # Expand the annotations to match the anchors shape
    expanded_annotations = torch.zeros(anchors.size(0), 5, dtype=annotations.dtype)
    # Expand to match those with the argmax overlaping. Maybe an annotation has no overlap with certain anchor,
    # but is forced to regress to it.
    expanded_annotations[:, :] = annotations[argmax_iou_anchors, :]
    # TODO: do not need the class (last column)

    # print(argmax_iou_anchors) # ta igual a aquela implementacao
    # print(expanded_annotations)
    # print(expanded_annotations.size())
    # exit()

    assert (rpn_labels == 0).sum() > 0
    assert (rpn_labels == 1).sum() > 0

    # Continuar fazendo o tracking.. 
    # parece que n filtra as anotaçoes com max_iou_anchors > 0.. considerando todas
    # alem disso, parece que existe loss para as ancoras fora do espaço da imagem, 
    # isto eh, as non-valid-anchors.. e elas tem um peso menor na loss eu acho.. ver aquele bbox_outside_weights.
    # se n me lembro era o inverso da quantidade de nao validos..
    # o que eu estou fazendo eh ignorar..

    # entao:
    # - continuar tentando identificar se realmente naos filtra as target_bboxes(meu expanded_annotations) com apenas os de maior ious
    # - ver tbm se ele considera ate backgroud.. pois alem de filtrar aqui, esta filtrando la em baixo na hora de calcular a loss
    #   eliminando os background (> 0)
    # - ver se realmente preciso manter a loss dos nao validos.. acho que isso tem um grande impacto tbm..

    # NAO DESISTE.. esta quase! faltam detalhes!

    # da para finalizar a RPN em mais um dia de trabalho intenso.. tirar o dia para ela!

    # e por fim fazer o diff no git para ter certeza se acabou n inserindo outro bug.. paciencia!

    # TODO
    # 1 - check the final rpn_labels.
    #   - Here I am not sampling the batch with 256 fg/bg labels. Check if it is equal to that implementation.
    #   - Also, include, if needed, the outer anchors as don't cares.
    # 2 - Match the full size, considering the non valid anchors?

                                           # TODO agora esse argmax_iou_anchors perde o sentido, pois la na frente nao eh tudo que esta sendo usado e sim apenas os rpn_labels==1
    return rpn_labels, expanded_annotations, argmax_iou_anchors # old version: argmax_iou_anchors[idxs] # this last is just to show (debug)

#inspect this when multiclass
def get_target_mask(rpn_filtered_proposals, annotations, low_threshold=0.0, high_threshold=0.5): # is 0.0 mesmo!

    # DEIXAR O LANCE DO REQUIRES GRAD PARA DEPOIS!
    # FAZER O FORWARD DIREITO PRIMEIRO!!

    # TODO:
    # - entender _sample_rois_pytorch daquela implementation
    # - ver se estou fazendo certo aqui
    # - dar um replace (var[:] = const) em alguma variavel para testar aqui
    # - ver se esta tudo batendo com o meu
    # - usar a mesma estrategia que usei para debugar a RPN.
    # - n desiste.. mais um dia nisso aqui vc finaliza
    #   depois mais um dia vendo os gradientes
    #   mais um dia limpando o codigo
    #   e fim!

    # Add the bbox annotations into the proposals
    # IMO, the only purpose of this is to avoid empty proposals
    all_proposals = torch.cat((rpn_filtered_proposals, annotations[:, :-1]), dim=0)

    ious = compute_iou(annotations, all_proposals)
    # max_iou_gt, argmax_iou_gt = torch.max(ious, dim=0) TODO undo!!!!!!
    max_iou_gt, argmax_iou_gt = torch.max(torch.transpose(ious, 0, 1), dim=1)

    # ----------- Select foreground idxs ----------- #
    max_fg_idxs = max(int(config.fg_fraction * config.batch_size), 1)
    # (using the max_iou to simplify as the results will be the same)
    fg_idxs = (max_iou_gt >= high_threshold).nonzero().squeeze(1)
    bg_idxs = ((max_iou_gt >= low_threshold) & (max_iou_gt < high_threshold)).nonzero().squeeze(1)
    n_fg_idxs, n_bg_idxs = fg_idxs.size(0), bg_idxs.size(0)

    if n_fg_idxs > 0 and n_bg_idxs > 0:
        # Select up to max_fg_idxs foreground proposals
        select_n_fg_idxs = min(n_fg_idxs, max_fg_idxs)
        # keep_idxs = torch.randperm(n_fg_idxs, device=annotations.device)[:select_n_fg_idxs]
        # fg_idxs = fg_idxs[keep_idxs]

        # Sample the bg_proposals to fill the remaining batch space
        select_n_bg_idxs = config.batch_size - select_n_fg_idxs
        # keep_idxs = torch.randint(low=0, high=n_bg_idxs, size=(select_n_bg_idxs, ), device=annotations.device)
        keep_idxs = torch.arange(n_bg_idxs).repeat(30)[:select_n_bg_idxs]
        bg_idxs = bg_idxs[keep_idxs]

        n_fg_idxs, n_bg_idxs = select_n_fg_idxs, select_n_bg_idxs

    elif n_fg_idxs > 0 and n_bg_idxs == 0:
        print('Entrou aqui no: elif n_fg_idxs > 0 and n_bg_idxs == 0: ')
        exit()
        # Select up to config.batch_size foreground proposals
        select_n_fg_idxs = config.batch_size
        keep_idxs = torch.randint(low=0, high=n_fg_idxs, size=(select_n_fg_idxs, ), device=annotations.device)
        fg_idxs = fg_idxs[keep_idxs]

        n_fg_idxs = select_n_fg_idxs

    elif n_fg_idxs == 0 and n_bg_idxs > 0:
        print('Entrou aqui no: elif n_fg_idxs == 0 and n_bg_idxs > 0: ')
        exit()
        # Select up to config.batch_size background proposals
        select_n_bg_idxs = config.batch_size
        keep_idxs = torch.randint(low=0, high=n_bg_idxs, size=(select_n_bg_idxs, ), device=annotations.device)
        bg_idxs = bg_idxs[keep_idxs]

        n_bg_idxs = select_n_bg_idxs

    else:
        print('Bug!!! Impossible to occur: n_fg_idxs == 0 and n_bg_idxs == 0 due to insertion of gt_boxes! Are there any image without annotations?')
        exit()
        # assert n_fg_idxs > 0 # impossible since I added the gt, unless there is no gt? but I think dataset.py is handling this

    # Concatenate all foreground and hard background indexes
    keep_idxs = torch.cat((fg_idxs, bg_idxs), dim=0)

    # In contrast to RPN, here we also sample the target_bboxes
    # Remember: annotations have the classes attached in last dim
    expanded_annotations = annotations[argmax_iou_gt[keep_idxs], :]

    # Set the bg_idxs annotation labels to background class
    expanded_annotations[n_fg_idxs:, -1] = 0

    # Also filter the proposals, to lately match the target annotations in the loss
    proposals = all_proposals[keep_idxs, :]

    return expanded_annotations, proposals

#da pra passar o idx acomo parametro e colocar no lugar do : ao ives de fazer a mascara no get_target_distance
def _parametrize_bbox(bbox, a_bbox):

    assert bbox.size() == a_bbox.size()
    assert len(bbox.size()) == 2

    xa, ya, wa, ha = a_bbox[:, 0], a_bbox[:, 1], a_bbox[:, 2], a_bbox[:, 3]
    x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

    tx = (x - xa) / wa
    ty = (y - ya) / ha
    tw = torch.log(w / wa)
    th = torch.log(h / ha)

    return torch.stack((tx, ty, tw, th), dim=1)


# rpn_bbox_loss = get_target_distance(proposals, self.rpn_net.valid_anchors, annotations[:, :-1], expanded_annotations)
def get_target_distance(proposals, valid_anchors, expanded_annotations, rpn_labels):

    # keep only the fg. to avoid the bbox_inside_weights
    # this is smarter..
    keep = rpn_labels == 1

    # OLD..
    # Select only non-background bboxes (the background annotations are all zeros, i.e., padding)
    # keep = expanded_annotations[:, -1] > 0

    # OLD NEW
    # now all expanded_annotations[:, -1] > 0 are TRUE!
    # TODO remove this after and the above keep and the selectors below.
    # assert (expanded_annotations[:, -1] > 0).all()

    proposals = anchors_bbox2offset(proposals[keep, :])
    anchors = anchors_bbox2offset(valid_anchors[keep, :])
    gts = anchors_bbox2offset(expanded_annotations[keep, :-1]) # TODO above

    # txgt, tygt, twgt, thgt = _parametrize_bbox(gts, anchors)
    # txp, typ, twp, thp = _parametrize_bbox(proposals, anchors)
    gts = _parametrize_bbox(gts, anchors)
    proposals = _parametrize_bbox(proposals, anchors)

    return smooth_l1(proposals, gts) * config.rpn_bbox_weight


def get_target_distance2(raw_reg, rpn_filtered_proposals, proposal_annotations):

    fg_idxs = proposal_annotations[:, -1] > 0

    rpn_filtered_proposals = anchors_bbox2offset(rpn_filtered_proposals[fg_idxs, :])
    target_bboxes = anchors_bbox2offset(proposal_annotations[fg_idxs, :-1])

    # -1 Because I did not include the prediction of background bboxes:
    target_classes = proposal_annotations[fg_idxs, -1].long()# - 1  # na verdade agora posso tirar isso! pois de fato n usa a predicao do background

    gts = _parametrize_bbox(target_bboxes, rpn_filtered_proposals)

    # Normalize the target
    gts /= torch.tensor(config.BBOX_NORMALIZE_STDS).type_as(gts)

    raw_reg = raw_reg.view(raw_reg.size(0), -1, 4)
    txp = raw_reg[fg_idxs, target_classes, 0]
    typ = raw_reg[fg_idxs, target_classes, 1]
    twp = raw_reg[fg_idxs, target_classes, 2]
    thp = raw_reg[fg_idxs, target_classes, 3]

    proposals = torch.stack((txp, typ, twp, thp), dim=1)

    return smooth_l1(proposals, gts) * config.reg_bbox_weight


def compute_prob_loss(probs_object, labels):

    # this has effect to consider the class 0 -> negative sample (or background if is cls_reg loss)
    #                             the class 1 -> positive sample (or car if is cls_reg loss)
    # without normalization to simplify as said in the paper, todo so, reduction='mean'

    # ignore_index=-1: considering all cares ! Just positive and negative (or backgrounds and cars if is cls_reg loss) samples !

    # prob_loss = F.cross_entropy(probs_object, labels, reduction='sum', ignore_index=-1)

    prob_loss = F.cross_entropy(probs_object, labels, reduction='mean', ignore_index=-1)

    return prob_loss


# NAO DESISTE !!!!!!!!!!!!!!!!!
