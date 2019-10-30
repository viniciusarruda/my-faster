from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import os
from pprint import pprint
from loss import anchor_labels
from tqdm import tqdm

# TODO: normalizar os dados de acordo com : https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html


class MyDataset(Dataset):

    def __init__(self, img_dir, csv_file, input_img_size, anchors_parameters, valid_anchors):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        original_img_size = (256, 256)

        with open(csv_file) as f:
            lines = f.readlines()
        data = [l.strip().split(',') for l in lines]
        data = [(d[0], np.array([float(d[i]) for i in range(1, len(d)-1)])) for d in data]

        for i in range(len(data)):
            # TODO:
            # the -1 in input img_size is to ensure: [0, input_img_size-1],
            # but it depends if annotations is considering [0, input_img_size-1] or [1, input_img_size]
            data[i][1][0] = (data[i][1][0] / original_img_size[0]) * (input_img_size[0] - 1.0)
            data[i][1][1] = (data[i][1][1] / original_img_size[1]) * (input_img_size[1] - 1.0)
            data[i][1][2] = (data[i][1][2] / original_img_size[0]) * (input_img_size[0] - 1.0)
            data[i][1][3] = (data[i][1][3] / original_img_size[1]) * (input_img_size[1] - 1.0)

        _inplace_adjust_bbox2offset(data)
        data = _group_by_filename(data)

        data = _format_data(data, anchors_parameters, valid_anchors)

        self.files_annot = data
        self.img_dir = img_dir
        self.input_img_size = input_img_size

    def __len__(self):
        return len(self.files_annot)

    def __getitem__(self, idx):

        img_base_name, bboxes, labels, table_gts_positive_anchors = self.files_annot[idx]
        img_name = os.path.join(self.img_dir, img_base_name)
        image = Image.open(img_name)
        image = transforms.Resize(self.input_img_size)(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        return image, bboxes, labels, table_gts_positive_anchors


def inv_normalize(t):

    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

    return invTrans(t)


def get_dataloader(anchors_parameters, valid_anchors):

    # input_img_size = (128, 128)
    input_img_size = (224, 224)
    dataset = MyDataset(img_dir='dataset/mini_day/', csv_file='dataset/mini_day.csv', input_img_size=input_img_size, anchors_parameters=anchors_parameters, valid_anchors=valid_anchors)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    return dataloader, input_img_size


def _inplace_adjust_bbox2offset(bbox_data_list):
    """
    proposals: batch_size, -1, 4
    bboxes: batch_size, -1, 4

    """

    for _, bbox in bbox_data_list:

        bbox[2] = bbox[2] - bbox[0] + 1
        bbox[3] = bbox[3] - bbox[1] + 1    


def _group_by_filename(data_list):

    data_dict = {}

    for filename, bbox in data_list:

        try:
            data_dict[filename].append(bbox)
        except:
            data_dict[filename] = []
            data_dict[filename].append(bbox)

    return [(filename, np.stack(bboxes)) for filename, bboxes in data_dict.items()]


def _format_data(data, anchors_parameters, valid_anchors):

    data = [(filename, torch.Tensor(bboxes)) for filename, bboxes in data]
    new_data = []

    n_bboxes = 0
    n_removed_bboxes = 0

    n_imgs = 0
    n_removed_imgs = 0

    print('Processing bounding boxes')
    for filename, bboxes in tqdm(data):
        
        n_imgs += 1
        n_bboxes += bboxes.shape[0]
        bboxes = torch.Tensor(bboxes)
        labels, table_gts_positive_anchors = anchor_labels(anchors_parameters, valid_anchors, bboxes)

        new_bboxes = bboxes[torch.unique(table_gts_positive_anchors[:, 0]), :]

        if bboxes.size(0) != new_bboxes.size(0):

            n_removed_bboxes += bboxes.size(0) - new_bboxes.size(0)

            if new_bboxes.size(0) == 0:
                n_removed_imgs += 1
                continue
            
            bboxes = new_bboxes
            # doing again to get the correct bboxes indexes.. 
            labels, table_gts_positive_anchors = anchor_labels(anchors_parameters, valid_anchors, bboxes)

        new_data.append((filename, bboxes, labels, table_gts_positive_anchors))

    print('{} bounding boxes in {} images in the dataset.'.format(n_bboxes, n_imgs))
    print('{} bounding boxes not assigned with anchors.'.format(n_removed_bboxes))
    print('{} images with all its bounding boxes not assigned with anchors.'.format(n_removed_imgs))
    print('Using {} bounding boxes in {} images for training.'.format(n_bboxes - n_removed_bboxes, n_imgs - n_removed_imgs))

    return new_data



# def anchor_labels(anchors, valid_anchors, gts, negative_threshold=0.3, positive_threshold=0.7): # era 0.3 no negative..
#     # tem como simplificar e otimizar..
    
#     anchors = anchors[valid_anchors, :]

#     batch_size = gts.size(0) # number of annotations for one image
#     mask = np.zeros(batch_size, anchors.size(0))
#     ious = np.zeros(batch_size, anchors.size(0))
    
#     for bi in range(batch_size):

#         anchors_bbox = np.zeros(anchors.size())
#         anchors_bbox[:, 0] = anchors[:, 0] - 0.5 * (anchors[:, 2] - 1)  # como proceder com o lance do -1 ou +1 nesse caso ? na conversão dos bbox2offset e vice versa ?
#         anchors_bbox[:, 1] = anchors[:, 1] - 0.5 * (anchors[:, 3] - 1)  # cuidadooooooooo p anchor eh assim, mas para proposal n .. caso for gerar label para proposal..
#         anchors_bbox[:, 2] = anchors_bbox[:, 0] + anchors[:, 2] - 1
#         anchors_bbox[:, 3] = anchors_bbox[:, 1] + anchors[:, 3] - 1

#         anchors_bbox_area = anchors[:, 2] * anchors[:, 3]

#         gt_area = gts[bi, 2] * gts[bi, 3]

#         x0 = np.maximum(anchors_bbox[:, 0], gts[bi, 0])
#         y0 = np.maximum(anchors_bbox[:, 1], gts[bi, 1])
#         x1 = np.minimum(anchors_bbox[:, 2], gts[bi, 0] + gts[bi, 2] - 1)
#         y1 = np.minimum(anchors_bbox[:, 3], gts[bi, 1] + gts[bi, 3] - 1)

#         intersection = np.clip(x1 - x0 + 1, 0, None) * np.clip(y1 - y0 + 1, 0, None)

#         union = anchors_bbox_area + gt_area - intersection
#         iou = intersection / union

#         ious[bi, :] = iou

#     # set positive anchors
#     idxs = ious > positive_threshold
#     idxs_cond = np.argmax(ious, axis=0)
#     cond = np.zeros(batch_size, anchors.size(0), dtype=torch.uint8) # this is to handle the possibility of an anchor to belong to more than one gt
#     cond[idxs_cond, range(idxs_cond.size(0))] = 1                      # it will only belong to the maximum iou
#     idxs_amax = torch.argmax(ious, dim=1)  # this may introduce an anchor to belong to more than one gt
#     idxs = idxs & cond                     # and to check (get the second argmax) it will be expensive
#     idxs[range(idxs_amax.size(0)), idxs_amax] = 1.0
#     mask[idxs] = 1.0

#     # set negative anchors
#     idxs = ious < negative_threshold
#     mask[idxs] = -1.0

#     # mask[bi, iou > positive_threshold] = 1.0
#     # mask[bi, iou < negative_threshold] = 0.0
#     # mask[bi, torch.argmax(iou)] = 1.0 # se mudar para fazer com bath tem que colocar dim=1 ou outra dependendo do que for
#     # else, mask = -1.0 (it is initialized with zeros - 1)    dont care

#     # idx_gt, idx_positive_anchor
#     table_gts_positive_anchors = (mask == 1.0).nonzero() 

#     mask, _ = torch.max(mask, dim=0)

#     # reverse to middle -> -1, negative -> 0 and positive -> 1
#     idxs_middle = mask == 0.0
#     idxs_negative = mask == -1.0

#     mask[idxs_middle] = -1.0
#     mask[idxs_negative] = 0.0

#     return mask, table_gts_positive_anchors







if __name__ == "__main__":

    dataloader, input_img_size = get_dataloader()

    for img, annotation in dataloader:

        print(img.size(), annotation.size())