import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import os
from loss import anchor_labels
from tqdm import tqdm
import config

class MyDataset(Dataset):

    def __init__(self, img_dir, csv_file, input_img_size, anchors, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        with open(csv_file) as f:
            lines = f.readlines()
        data = [l.strip().split(',') for l in lines]
        data = [(d[0], np.array([np.float32(d[i]) for i in range(1, len(d)-1)]), d[-1]) for d in data]

        for i in range(len(data)):
            # TODO:
            # the -1 in input img_size is to ensure: [0, input_img_size-1],
            # but it depends if annotations is considering [0, input_img_size-1] or [1, input_img_size]
            data[i][1][0] = (data[i][1][0] / config.original_img_size[0]) * (input_img_size[0] - 1.0)
            data[i][1][1] = (data[i][1][1] / config.original_img_size[1]) * (input_img_size[1] - 1.0)
            data[i][1][2] = (data[i][1][2] / config.original_img_size[0]) * (input_img_size[0] - 1.0)
            data[i][1][3] = (data[i][1][3] / config.original_img_size[1]) * (input_img_size[1] - 1.0)

        _inplace_adjust_bbox2offset(data)
        data = _group_by_filename(data, config.class_names)
        data = _format_data(data, anchors)

        self.files_annot = data
        self.img_dir = img_dir
        self.input_img_size = input_img_size
        self.batch_size = config.rpn_batch_size # it is confusing putting this stuff in the dataloader
        self.max_positive_batch_ratio = config.max_positive_batch_ratio
        self.max_positive_batch_size = int(self.max_positive_batch_ratio * self.batch_size) # put this on config ?
        
        self.train = train


    def __len__(self):
        return len(self.files_annot)

    
    def __getitem__(self, idx):
        if self.train:
            return self._getitem_train(idx)
        else:
            return self._getitem_test(idx)


    def _getitem_train(self, idx):

        img_base_name, bboxes, clss_idxs, labels_objectness, labels_class, table_gts_positive_anchors = self.files_annot[idx]
        img_name = os.path.join(self.img_dir, img_base_name)
        image = Image.open(img_name)
        image = transforms.Resize((self.input_img_size[1], self.input_img_size[0]))(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        # Because I don't want to modify the original data
        balanced_labels_objectness = labels_objectness.clone()
        labels_class = labels_class.clone()

        # here, randomly get the batch
        # print((balanced_labels == -1).sum(), (balanced_labels == 0).sum(), (balanced_labels == 1).sum())
        # TODO -> DONE! : fazer a implementação correta, testar
        # NEXT TODO: fazer o mesmo porem com a segunda parte da rede (25% 75%)

        # n_positive_anchors = (balanced_labels == 1).sum() # the line below is faster and produce the same result
        n_positive_anchors = table_gts_positive_anchors.size(0)

        ### Select up to self.max_positive_batch_size positive anchors
        ### The excess is marked as dont care
        if n_positive_anchors > self.max_positive_batch_size:
            # raise NotImplementedError('Warning, did not implemented!')
            print('\n======\n')
            print('n_positive_anchors > self.max_positive_batch_size')
            print('OBSERVE IF IT IS BEHAVING RIGHT! IT SHOULD!')
            print('\n======\n')
            exit()
            # positive_anchors_idxs = (balanced_labels == 1).nonzero().squeeze() # the line below is faster and produce the same result
            positive_anchors_idxs = table_gts_positive_anchors[:, 1]
            tmp_idxs = torch.randperm(n_positive_anchors)[:n_positive_anchors - self.max_positive_batch_size]
            idxs_to_suppress = positive_anchors_idxs[tmp_idxs]
            balanced_labels[idxs_to_suppress] = -1 # mark them as don't care
            n_positive_anchors = self.max_positive_batch_size
            # TODO -> To Check!
            # The table_gts_positive_anchors is not consistent with the labels.
            # Should be consistent? Or this balancing is just for the cross-entropy loss?
            # [07/03/2020] - I think it is correct, the balancing is just for classification loss, as we have two labels.
            # [07/03/2020]   For the regression, we have no labels, just regressing the positive ones, so, making no sense to balance them.

        ### Fill the remaining batch with negative anchors
        negative_anchors_idxs = (balanced_labels_objectness == 0).nonzero().squeeze()
        n_negative_anchors = negative_anchors_idxs.size(0)
        n_anchors_to_complete_batch = self.batch_size - n_positive_anchors

        if n_anchors_to_complete_batch >= n_negative_anchors:
            # TODO: There is less anchors than the batch size.. just use the available ones?
            # If use the available, take care with the negative indexes in the line below
            # If so, use the same snippet as in get_target_mask() in loss.py
            raise NotImplementedError('Warning, did not implemented! How to proceed with this?')

        tmp_idxs = torch.randperm(n_negative_anchors)[:n_negative_anchors - n_anchors_to_complete_batch]
        idxs_to_suppress = negative_anchors_idxs[tmp_idxs]
        balanced_labels_objectness[idxs_to_suppress] = -1 # mark them as don't care

        # # You can see the difference before/after balancing the labels:
        # print('---------')
        # print((labels_class == -1).sum(), (labels_class == 0).sum(), (labels_class == 1).sum())
        # print((balanced_labels_objectness == -1).sum(), (balanced_labels_objectness == 0).sum(), (balanced_labels_objectness == 1).sum())
        # print('---------')
        # exit()

        return image, bboxes, clss_idxs, balanced_labels_objectness, labels_class, table_gts_positive_anchors

    
    def _getitem_test(self, idx):

        img_base_name, bboxes, clss_idxs, labels_objectness, labels_class, table_gts_positive_anchors = self.files_annot[idx]
        img_name = os.path.join(self.img_dir, img_base_name)
        image = Image.open(img_name)
        image = transforms.Resize((self.input_img_size[1], self.input_img_size[0]))(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        labels_objectness = labels_objectness.clone() # I think this is not needed
        labels_class = labels_class.clone() # I think this is not needed

        return image, bboxes, clss_idxs, labels_objectness, labels_class, table_gts_positive_anchors


def inv_normalize(t):

    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

    return invTrans(t)


def get_dataloader(anchors):

    dataset = MyDataset(img_dir=config.img_folder, csv_file=config.annotations_file, input_img_size=config.input_img_size, anchors=anchors, train=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    return dataloader

def get_dataset(anchors):

    dataset = MyDataset(img_dir=config.val_img_folder, csv_file=config.val_annotations_file, input_img_size=config.input_img_size, anchors=anchors, train=False)
    return dataset


def _inplace_adjust_bbox2offset(bbox_data_list):
    """
    proposals: batch_size, -1, 4
    bboxes: batch_size, -1, 4

    """

    for _, bbox, _ in bbox_data_list:

        bbox[2] = bbox[2] - bbox[0] + 1
        bbox[3] = bbox[3] - bbox[1] + 1    


def _group_by_filename(data_list, class_names):

    data_dict = {}

    for filename, bbox, classname in data_list:

        try:
            class_idx = class_names.index(classname)
        except ValueError:
            raise ValueError("Class '{}' not present in the list of classes {}.".format(classname, class_names))
        
        try:
            data_dict[filename]['bboxes'].append(bbox)
            data_dict[filename]['class_idxs'].append(class_idx)
        except:
            data_dict[filename] = {'bboxes': [bbox], 'class_idxs': [class_idx]}

    return [(filename, np.stack(annotations['bboxes']), np.stack(annotations['class_idxs'])) for filename, annotations in data_dict.items()]


def _format_data(data, anchors):

    anchors = anchors.to('cpu')

    data = [(filename, torch.Tensor(bboxes), torch.Tensor(class_idxs).long()) for filename, bboxes, class_idxs in data]
    new_data = []

    n_bboxes = 0
    n_removed_bboxes = 0

    n_imgs = 0
    n_removed_imgs = 0

    print('Processing bounding boxes')
    for filename, bboxes, class_idxs in tqdm(data):
        
        n_imgs += 1
        n_bboxes += bboxes.size(0)
        labels_objectness, labels_class, table_gts_positive_anchors = anchor_labels(anchors, bboxes, class_idxs)

        new_bboxes = bboxes[torch.unique(table_gts_positive_anchors[:, 0]), :]

        assert bboxes.size(0) >= new_bboxes.size(0) # remove this in the final version
        print('TEST THIS IF CASE - IMPACT OF CLASSES?!')
        if bboxes.size(0) != new_bboxes.size(0):
            exit() # test this before run!!
            n_removed_bboxes += bboxes.size(0) - new_bboxes.size(0)

            if new_bboxes.size(0) == 0:
                n_removed_imgs += 1
                continue
            
            bboxes = new_bboxes
            # doing again to get the correct bboxes indexes.. 
            # TODO understand why I did this
            labels_objectness, labels_class, table_gts_positive_anchors = anchor_labels(anchors, bboxes, class_idxs)

        new_data.append((filename, bboxes, class_idxs, labels_objectness, labels_class, table_gts_positive_anchors))

    print('SHOW STATISTICS OF CLASSES IN DATASET')
    print('{} bounding boxes in {} images in the dataset.'.format(n_bboxes, n_imgs))
    print('{} bounding boxes not assigned with anchors.'.format(n_removed_bboxes))
    print('{} images with all its bounding boxes not assigned with anchors.'.format(n_removed_imgs))
    print('Using {} bounding boxes in {} images for training.'.format(n_bboxes - n_removed_bboxes, n_imgs - n_removed_imgs))

    return new_data


if __name__ == "__main__":

    dataloader, input_img_size = get_dataloader()

    for img, annotation in dataloader:

        print(img.size(), annotation.size())
