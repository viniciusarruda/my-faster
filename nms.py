import torch

def nms(bboxes, probs_object):

    assert bboxes.size(0) == 1

    # for bi in range(bboxes.size(0)):
        # bboxes, probs_object = _nms(bboxes[bi, :, :], probs_object[bi, :])
    bi = 0
    bboxes, probs_object = _nms(bboxes[bi, :, :], probs_object[bi, :])

    return bboxes.unsqueeze(0), probs_object.unsqueeze(0) # unsqueeze for simulating a batch of 1   


def _nms(bboxes, probs_object):
    # ter certeza da minha implementacao nms.. testar com um if main em baixo com algum caso toy !
    # print(bboxes.size())
    # print(probs_object.size())

    idxs = torch.argsort(probs_object, descending=True)

    n_bboxes = 30 #100 #600
    idxs = idxs[:n_bboxes]
    # print(idxs.size())

    bboxes = bboxes[idxs, :]
    probs_object = probs_object[idxs]

    # print(bboxes.size())
    # print(probs_object.size())

    k = 0
    while k < bboxes.size(0) - 1:

        ### Remove iou > 0.7 ###
        x0_0, y0_0, x1_0, y1_0 = bboxes[k, 0], bboxes[k, 1], bboxes[k, 2], bboxes[k, 3]
        area_0 = (x1_0 - x0_0 + 1) * (y1_0 - y0_0 + 1)
        assert x1_0 > x0_0 and y1_0 > y0_0 # just to ensure.. but this is dealt before I think... I am shure !!

        # print(k+1, bboxes.size(0)-1)

        x0 = torch.max(x0_0, bboxes[k+1:, 0])
        y0 = torch.max(y0_0, bboxes[k+1:, 1])
        x1 = torch.min(x1_0, bboxes[k+1:, 2])
        y1 = torch.min(y1_0, bboxes[k+1:, 3])     

        intersection = torch.clamp(x1 - x0 + 1, min=0) * torch.clamp(y1 - y0 + 1, min=0)  
        area_j = (bboxes[k+1:, 2] - bboxes[k+1:, 0] + 1) * (bboxes[k+1:, 3] - bboxes[k+1:, 1] + 1) 

        union = area_0 + area_j - intersection
        iou = intersection / union

        # print(iou)
        keep_idxs = iou <= 0.7
        # print(keep_idxs)

        bboxes = torch.cat((bboxes[:k+1, :], bboxes[k+1:, :][keep_idxs, :]), dim=0)
        probs_object = torch.cat((probs_object[:k+1], probs_object[k+1:][keep_idxs]), dim=0)
        k += 1

    return bboxes, probs_object


if __name__ == "__main__":

    bboxes = torch.zeros(1, 3, 4)
    probs_object = torch.zeros(1, 3)

    bboxes[0, 0, 0] = 1.0
    bboxes[0, 0, 1] = 1.0
    bboxes[0, 0, 2] = 5.0
    bboxes[0, 0, 3] = 5.0

    bboxes[0, 1, 0] = 2
    bboxes[0, 1, 1] = 2
    bboxes[0, 1, 2] = 5.0
    bboxes[0, 1, 3] = 5.0

    bboxes[0, 2, 0] = 3
    bboxes[0, 2, 1] = 3
    bboxes[0, 2, 2] = 5.0
    bboxes[0, 2, 3] = 5.0

    probs_object[0, 0] = 0.5
    
    probs_object[0, 1] = 0.4

    probs_object[0, 2] = 0.3

    bboxes, probs_object = nms(bboxes, probs_object)

    print(bboxes)
    print(probs_object)
    