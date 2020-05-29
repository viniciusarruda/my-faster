import torch
import torchvision

# TODO: check if I need to check if x0 is greater than x1 e etc.. (the assertion)
# TODO: keep the +1 standard or no? https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L23

# accepts only bboxes.size() -> (#bboxes, 4) and probs_object.size() -> (#bboxes)


def nms(bboxes, probs_object, nms_threshold):

    # TODO I think I will leave this here..
    # I want to check degenerate cases.
    # How shold I handle it? Get these cases out or leave it?
    # https://github.com/pytorch/vision/issues/1870
    # If this assertion is not here:
    #    The PyTorch implementation will leave the degenerate boxes. (in a natural way)
    #    In my implementation, the degenerate boxes will be removed.  (in a natural way)
    #    To imitate the PyTorch implementation behavior, it will have a greater computational cost.

    # testing if the two code outputs the same and its time
    # for while, its output are the identical also quite similar the time spent
    # ------------------------------------------------------------- #
    # If using my own implementation (slower but still effective)
    # keep = _nms(bboxes, probs_object, nms_threshold)
    # ------------------------------------------------------------- #
    # If using the torchvision implementation
    # https://github.com/pytorch/vision/blob/e2a8b4185e2b668b50039c91cdcf81eb4175d765/torchvision/csrc/cpu/nms_cpu.cpp
    bboxes[:, 2:] += 1.0  # The implementation doesn't add +1 while computing the width/height.
    keep = torchvision.ops.nms(bboxes, probs_object, nms_threshold)
    bboxes[:, 2:] -= 1.0  # Undoing the above adjustment
    # ------------------------------------------------------------- #
    # DOCS: keep – int64 tensor with the indices of the elements that have been kept by NMS,
    # sorted in decreasing order of scores
    #

    # bboxes = bboxes[keep, :]
    # probs_object = probs_object[keep]
    # labels_class = labels_class[keep]

    return keep


# @profile # uncomment and run kernprof -lv nms.py on terminal
def _nms(bboxes, probs_object, threshold):

    idxs = torch.argsort(probs_object, descending=True)

    areas = (bboxes[:, 2] - bboxes[:, 0] + 1.0) * (bboxes[:, 3] - bboxes[:, 1] + 1.0)

    keep = []
    while idxs.size(0) > 0:

        k = idxs[0]
        keep.append(k)

        x0_0, y0_0, x1_0, y1_0 = bboxes[k, 0], bboxes[k, 1], bboxes[k, 2], bboxes[k, 3]
        
        area_0 = areas[k]
        
        x0 = torch.max(x0_0, bboxes[idxs[1:], 0])
        y0 = torch.max(y0_0, bboxes[idxs[1:], 1])
        x1 = torch.min(x1_0, bboxes[idxs[1:], 2])
        y1 = torch.min(y1_0, bboxes[idxs[1:], 3])

        intersection = torch.clamp(x1 - x0 + 1.0, min=0) * torch.clamp(y1 - y0 + 1.0, min=0)

        area_j = areas[idxs[1:]]
            
        union = area_0 + area_j - intersection
        iou = intersection / union

        idxs = idxs[1:][iou <= threshold]
    
    if keep == []:
        return torch.tensor([], dtype=torch.int64)
    else:
        return torch.stack(keep)


def test_single_nms():

    # tensor([0.0724, 0.2735, 0.6525, 0.7111]) tensor(1.2180)
    # tensor([0.8393, 0.3385, 0.3496, 0.8569]) tensor(-0.0867)

    # bboxes = [[0.0724, 0.2735, 0.6525, 0.7111],
    #          [0.8393, 0.3385, 0.3496, 0.8569]]

    # bboxes = [[0.0, 0.0, 1.0, 1.0],
    #           [2.0, 1.0, 1.0, 2.0]]

    # bboxes = [[0.0, 0.0, 0.9, 0.9],
    #           [1.0, 1.0, 2.0, 2.0]]

    ######
    # This case, iou ~ nms_threshold, and the implementations outputs different results
    # which is not an error of my implementation, just different float precision
    # that doesn't significantly influences the final result
    # My NMS implementation returns false for iou < nms_threshold
    # Torchvision NMS implementation returns true for iou < nms_threshold
    bboxes = [[0.5012, 0.2141, 0.8166, 1.1216],
              [0.6061, 0.0828, 0.6585, 0.9750]]
    scores = [0.5517, 2.6361]
    ######

    boxes = torch.tensor(bboxes)
    scores = torch.tensor(scores)

    keep = torchvision.ops.nms(boxes, scores, 0.7)
    mykeep = _nms(boxes, scores, 0.7)

    print(mykeep)
    print(keep)


def test_nms():
    from tqdm import tqdm
    import time

    mine_time = 0
    pyto_time = 0

    for i in tqdm(range(2)):
        torch.manual_seed(i)

        for k in tqdm(range(10, 6000)):

            offsets = torch.rand(k, 4) # xywh
            boxes = torch.zeros(k, 4)  # xyxy
            
            # generating fake boxes following the requirement of x2 > x1 and y2 > y1
            boxes[:, :2] = offsets[:, :2]
            boxes[:, 2:] = offsets[:, :2] + offsets[:, 2:]

            scores = torch.randn(k)

            start = time.time()
            mykeep = _nms(boxes, scores, 0.7)
            end = time.time()
            mine_time += end - start

            boxes[:, 2:] += 1.0
            start = time.time()
            keep = torchvision.ops.nms(boxes, scores, 0.7)
            end = time.time()
            pyto_time += end - start

            try:

                if not torch.allclose(mykeep, keep):
                    print(i, k)
                    print('not all close')
                    # exit()

            except:
                print(i, k)
                print('something wrong, probably, mykeep.size() != keep.size() or iou ~ nms_threshold')
                # exit()

    print(pyto_time)
    print(mine_time)

if __name__ == "__main__":
    test_nms()
