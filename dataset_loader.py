import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import os
from pprint import pprint
from loss import anchor_labels
from tqdm import tqdm
import config

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

        original_img_size = (256, 256)  # should get from data

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
        self.batch_size = config.rpn_batch_size # it is confusing putting this stuff in the dataloader
        self.half_batch_size = int(0.5 * self.batch_size)

    def __len__(self):
        return len(self.files_annot)

    def __getitem__(self, idx):

        img_base_name, bboxes, labels, table_gts_positive_anchors = self.files_annot[idx]
        img_name = os.path.join(self.img_dir, img_base_name)
        image = Image.open(img_name)
        image = transforms.Resize(self.input_img_size)(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        # here, randomly get the batch
        # print((labels == -1).sum(), (labels == 0).sum(), (labels == 1).sum())
        # TODO: fazer a implementação correta, testar
        # NEXT TODO: fazer o mesmo porem com a segunda parte da rede (25% 75%)

        ### Select up to self.half_batch_size positive anchors
        ### The excess is marked as dont care
        # TODO
        if (labels == 1).sum() > self.half_batch_size:
            raise NotImplementedError('Warning, did not implemented!')

        # should keep the table_gts_positive_anchors consistent with the positive labels
        # print(table_gts_positive_anchors)

        ### Select self.batch_size - positive anchors size negative anchors
        ### I think the excess is marked as dont care too (did not confirmed)
        negative = (labels == 0).nonzero()[:, 0]
        n_positive_anchors = (labels == 1).sum()
        n_anchors_to_complete_batch = self.batch_size - n_positive_anchors

        if n_anchors_to_complete_batch > negative.size(0):
            raise NotImplementedError('Warning, did not implemented! How to proceed with this?')
            # There is less anchors than the batch size.. just use the available ones ?

        idxs = torch.randperm(negative.size(0))[:negative.size(0) - n_anchors_to_complete_batch]
        negative_idxs = negative[idxs]

        new_labels = labels.detach().clone() # it is really needed to detach and clone ? (can much effort for the same effect)
        new_labels[negative_idxs] = -1 # mark them as dont care

        # print('-----')
        # print((labels == -1).sum(), (labels == 0).sum(), (labels == 1).sum())
        # print((new_labels == -1).sum(), (new_labels == 0).sum(), (new_labels == 1).sum())
        # print('---------')

        # exit()

        return image, bboxes, new_labels, table_gts_positive_anchors


def inv_normalize(t):

    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

    return invTrans(t)


def get_dataloader(anchors_parameters, valid_anchors):

    dataset = MyDataset(img_dir=config.img_folder, csv_file=config.annotations_file, input_img_size=config.input_img_size, anchors_parameters=anchors_parameters, valid_anchors=valid_anchors)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    return dataloader


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


if __name__ == "__main__":

    dataloader, input_img_size = get_dataloader()

    for img, annotation in dataloader:

        print(img.size(), annotation.size())