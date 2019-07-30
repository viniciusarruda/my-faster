import torch

def nms(input_img_size, i_proposals_o, i_prob_object_o):

    # i_proposals_o = _offset2bbox(i_proposals_o) 
    bx0 = i_proposals_o[:, 0]
    by0 = i_proposals_o[:, 1]
    bx1 = bx0 + i_proposals_o[:, 2] - 1
    by1 = by0 + i_proposals_o[:, 3] - 1
    i_proposals_o = torch.stack((bx0, by0, bx1, by1), dim=1)
    # print('lembrar de subtrair os -1, tirei pra testar')                 ### a partir daqui proposals eh bbox.. trocar nome da variavel..
    #############################################

    # i_proposals_o = _clip_boxes(i_proposals_o)
    bx0 = i_proposals_o[:, 0].clamp(0, input_img_size[0]-1)
    by0 = i_proposals_o[:, 1].clamp(0, input_img_size[1]-1)
    bx1 = i_proposals_o[:, 2].clamp(0, input_img_size[0]-1)
    by1 = i_proposals_o[:, 3].clamp(0, input_img_size[1]-1)
    i_proposals_o = torch.stack((bx0, by0, bx1, by1), dim=1)
    ############################################

    idxs = torch.argsort(i_prob_object_o, descending=True)
    n_proposals = 600
    idxs = idxs[:n_proposals]

    i_proposals = i_proposals_o[idxs, :]
    i_prob_object = i_prob_object_o[idxs]

    k = 0
    while k < i_proposals.size()[0]:

        ### Remove iou > 0.7 ###
        x0_0, y0_0, x1_0, y1_0 = i_proposals[k, 0], i_proposals[k, 1], i_proposals[k, 2], i_proposals[k, 3]
        area_0 = (x1_0 - x0_0 + 1) * (y1_0 - y0_0 + 1)
        assert x1_0 > x0_0 and y1_0 > y0_0 # just to ensure.. but this is dealt before I think

        marked_to_keep = []

        for j in range(k+1, i_proposals.size()[0]):

            x0_j, y0_j, x1_j, y1_j = i_proposals[j, 0], i_proposals[j, 1], i_proposals[j, 2], i_proposals[j, 3]
            
            x0 = torch.max(x0_0, x0_j)
            y0 = torch.max(y0_0, y0_j)
            x1 = torch.min(x1_0, x1_j)
            y1 = torch.min(y1_0, y1_j)
            
            intersection = torch.clamp(x1 - x0 + 1, min=0) * torch.clamp(y1 - y0 + 1, min=0)
            area_j = (x1_j - x0_j + 1) * (y1_j - y0_j + 1)

            union = area_0 + area_j - intersection
            iou = intersection / union
            
            if iou <= 0.7:
                marked_to_keep.append(j)

        # keep
        i_proposals = torch.cat((i_proposals[:k+1, :], i_proposals[marked_to_keep, :]), dim=0)
        i_prob_object = torch.cat((i_prob_object[:k+1], i_prob_object[marked_to_keep]), dim=0)
        k += 1

    # proposals = _bbox2offset(proposals)
    bx0 = i_proposals[:, 0]
    by0 = i_proposals[:, 1]
    bx1 = i_proposals[:, 2]
    by1 = i_proposals[:, 3]

    ox = bx0
    oy = by0
    ow = bx1 - bx0 + 1
    oh = by1 - by0 + 1

    i_proposals = torch.stack((ox, oy, ow, oh), dim=1)
    #####################################

    return i_proposals, i_prob_object
