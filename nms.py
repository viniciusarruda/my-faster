import torch
import config

def nms(bboxes, probs_object):

    assert bboxes.size(0) == 1

    # for bi in range(bboxes.size(0)):
        # bboxes, probs_object = _nms(bboxes[bi, :, :], probs_object[bi, :])
    bi = 0
    bboxes, probs_object = _nms(bboxes[bi, :, :], probs_object[bi, :])

    return bboxes.unsqueeze(0), probs_object.unsqueeze(0) # unsqueeze for simulating a batch of 1   


def _nms(bboxes, probs_object):
    # ter certeza da minha implementacao nms.. testar com um if main em baixo com algum caso toy !

    # Alguns lugares dizem que a pre-seleção dos top proposals é depois do nms, 
    # mas para mim, faz mais sentido em ser antes, além de ter um menor custo computacional.
    idxs = torch.argsort(probs_object, descending=True)
    
    # colocando com 100 ou 600 piorou ! acredito que ao implementar a resnet como feature extractor pode melhorar aqui
    n_bboxes = config.pre_nms_top_n #100 #600
    idxs = idxs[:n_bboxes]

    bboxes = bboxes[idxs, :]
    probs_object = probs_object[idxs]

    k = 0
    while k < bboxes.size(0) - 1:

        ### Remove iou > 0.7 ###
        x0_0, y0_0, x1_0, y1_0 = bboxes[k, 0], bboxes[k, 1], bboxes[k, 2], bboxes[k, 3]
        area_0 = (x1_0 - x0_0 + 1) * (y1_0 - y0_0 + 1)
        assert x1_0 > x0_0 and y1_0 > y0_0 # just to ensure.. but this is dealt before I think... I am shure !!
        # actually can be assert x1_0 >= x0_0 and y1_0 >= y0_0

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




# def test_nms():
#     import torchvision
#     boxes = torch.rand(5, 4)
#     boxes[:, 2:] += torch.rand(5, 2)
#     scores = torch.randn(5)

#     print(boxes)
#     print(scores)

#     keep = torchvision.ops.nms(boxes, scores, 0.7)
#     print(keep)

#     boxes, scores = _nms(boxes, scores)

#     print(boxes)
#     print(scores)

    
# if __name__ == "__main__":
#     test_nms()