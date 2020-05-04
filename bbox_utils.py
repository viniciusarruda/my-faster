import torch
import config

def bbox2offset(bboxes):
    """
    bboxes: #bboxes, 4
    proposals: #bboxes, 4

    """

    bx0 = bboxes[:, 0]
    by0 = bboxes[:, 1]
    bx1 = bboxes[:, 2]
    by1 = bboxes[:, 3]

    ox = bx0
    oy = by0
    ow = bx1 - bx0 + 1.0
    oh = by1 - by0 + 1.0

    offsets = torch.stack((ox, oy, ow, oh), dim=1)

    return offsets


def offset2bbox(proposals):
    """
    bboxes: #bboxes, 4
    proposals: #bboxes, 4

    """

    bx0 = proposals[:, 0]
    by0 = proposals[:, 1]
    bx1 = bx0 + proposals[:, 2] - 1.0
    by1 = by0 + proposals[:, 3] - 1.0

    bboxes = torch.stack((bx0, by0, bx1, by1), dim=1)

    return bboxes


def clip_boxes(bboxes, input_img_size):
    """
    bboxes: #bboxes, 4
    bboxes: #bboxes, 4

    """
    
    bx0 = bboxes[:, 0].clamp(0.0, input_img_size[0] - 1.0)
    by0 = bboxes[:, 1].clamp(0.0, input_img_size[1] - 1.0)
    bx1 = bboxes[:, 2].clamp(0.0, input_img_size[0] - 1.0)
    by1 = bboxes[:, 3].clamp(0.0, input_img_size[1] - 1.0)

    bboxes = torch.stack((bx0, by0, bx1, by1), dim=1)

    return bboxes
    

def bboxes_filter_condition(bboxes):

    # torch.int64 -> index
    # torch.uint8 -> true or false (mask)

    bx0 = bboxes[:, 0]
    by0 = bboxes[:, 1]
    bx1 = bboxes[:, 2]
    by1 = bboxes[:, 3]
    bw = bx1 - bx0 + 1.0
    bh = by1 - by0 + 1.0
    cond = (bw >= config.min_size) & (bh >= config.min_size)
    
    return cond


def anchors_offset2bbox(anchors):
    """
    anchors: #anchors, 4
    bboxes: #bboxes, 4

    """
    # TODO: check if is better/faster to torch.zeros or stack as this function or as the above functions..
    bboxes = torch.zeros(anchors.size(), dtype=anchors.dtype, device=anchors.device)
    bboxes[:, 0] = anchors[:, 0] - 0.5 * (anchors[:, 2] - 1)  # como proceder com o lance do -1 ou +1 nesse caso ? na conversão dos bbox2offset e vice versa ?
    bboxes[:, 1] = anchors[:, 1] - 0.5 * (anchors[:, 3] - 1)  # cuidadooooooooo p anchor eh assim, mas para proposal n .. caso for gerar label para proposal..
    bboxes[:, 2] = bboxes[:, 0] + anchors[:, 2] - 1.0
    bboxes[:, 3] = bboxes[:, 1] + anchors[:, 3] - 1.0

    return bboxes


def compute_iou(gts, base_bboxes, base_offsets):

    base_bboxes_area = base_offsets[:, 2] * base_offsets[:, 3]
    gt_area = gts[:, 2] * gts[:, 3]

    x0 = torch.max(base_bboxes[:, 0], gts[:, 0].view(-1, 1))
    y0 = torch.max(base_bboxes[:, 1], gts[:, 1].view(-1, 1))
    x1 = torch.min(base_bboxes[:, 2], gts[:, 0].view(-1, 1) + gts[:, 2].view(-1, 1) - 1.0)
    y1 = torch.min(base_bboxes[:, 3], gts[:, 1].view(-1, 1) + gts[:, 3].view(-1, 1) - 1.0)

    intersection = torch.clamp(x1 - x0 + 1.0, min=0.0) * torch.clamp(y1 - y0 + 1.0, min=0.0)
    union = base_bboxes_area.view(1, -1) + gt_area.view(-1, 1) - intersection
    iou = intersection / union

    return iou



