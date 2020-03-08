import torch

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


def filter_boxes(bboxes, probs_object, min_size=16.0):

    # torch.int64 -> index
    # torch.uint8 -> true or false (mask)

    bx0 = bboxes[:, 0]
    by0 = bboxes[:, 1]
    bx1 = bboxes[:, 2]
    by1 = bboxes[:, 3]
    bw = bx1 - bx0 + 1.0
    bh = by1 - by0 + 1.0
    cond = (bw >= min_size) & (bh >= min_size)
    
    return bboxes[cond, :], probs_object[cond]