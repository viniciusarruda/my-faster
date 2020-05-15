import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
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

        # Format data into (img_filename, annotations_x0y0x1y1, class_name)
        with open(csv_file) as f:
            lines = f.readlines()
        data = [l.strip().split(',') for l in lines]
        data = [(d[0], np.array([np.float32(d[i]) for i in range(1, len(d) - 1)]), d[-1]) for d in data]

        # Clip the annotations into the interval [0, config.original_img_size)
        # NOTE: to implement the ) I just subtracted an epsilon value.
        modified = []
        for i in range(len(data)):
            x0 = max(data[i][1][0], 0.0)
            y0 = max(data[i][1][1], 0.0)
            x1 = min(data[i][1][2], config.original_img_size[0] - np.finfo(np.float32).eps)
            y1 = min(data[i][1][3], config.original_img_size[1] - np.finfo(np.float32).eps)

            modified.append(x0 != data[i][1][0] or y0 != data[i][1][1] or x1 != data[i][1][2] or y1 != data[i][1][3])

            data[i][1][0] = x0
            data[i][1][1] = y0
            data[i][1][2] = x1
            data[i][1][3] = y1

        if any(modified):
            print('WARNING: There are {} of {} annotations that were clipped.'
                  .format(sum(modified), len(modified)))

        # Resize the annotations to the new image size
        for i in range(len(data)):
            # TODO:
            # the -1 in input img_size is to ensure: [0, input_img_size-1],
            # but it depends if annotations is considering [0, input_img_size-1] or [1, input_img_size]
            data[i][1][0] = (data[i][1][0] / config.original_img_size[0]) * (input_img_size[0] - 1.0)
            data[i][1][1] = (data[i][1][1] / config.original_img_size[1]) * (input_img_size[1] - 1.0)
            data[i][1][2] = (data[i][1][2] / config.original_img_size[0]) * (input_img_size[0] - 1.0)
            data[i][1][3] = (data[i][1][3] / config.original_img_size[1]) * (input_img_size[1] - 1.0)

        # Filter by min_size
        new_data = []
        max_width, max_height = 0, 0
        min_width, min_height = config.input_img_size
        for i in range(len(data)):
            w = data[i][1][2] - data[i][1][0] + 1.0
            h = data[i][1][3] - data[i][1][1] + 1.0

            if w >= config.min_size and h >= config.min_size:
                new_data.append(data[i])

                max_width, max_height = max(max_width, w), max(max_height, h)
                min_width, min_height = min(min_width, w), min(min_height, h)

        if len(data) > len(new_data):
            print('WARNING: {} annotations were smaller than the minimun width/height size, being removed ending with {} annotations.'.format(len(data) - len(new_data), len(new_data)))

        assert len(new_data) > 0  # if there is no annotation in this image what should I do?

        print('INFO: max_width: {}, max_height: {}'.format(max_width, max_height))
        print('INFO: min_width: {}, min_height: {}'.format(min_width, min_height))

        data = new_data

        _inplace_adjust_bbox2offset(data)
        data = _group_by_filename(data, config.class_names)
        data = _format_data(data, anchors)

        if len(data) == 0:
            print('WARNING: There is no data for this dataset. Please check your data and then the configurations (config file).')
            exit()

        self.files_annot = data
        self.img_dir = img_dir
        self.input_img_size = input_img_size
        self.batch_size = config.rpn_batch_size  # it is confusing putting this stuff in the dataloader
        self.max_positive_batch_ratio = config.max_positive_batch_ratio
        self.max_positive_batch_size = int(self.max_positive_batch_ratio * self.batch_size)  # put this on config ?

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

        # fazer o tracking de todo comentario.. tipo esse aqui, analisar se realmente preciso fazer o clone
        # e ir eliminando coisas desnecessarias e comentar o que eh necessario
        # Because I don't want to modify the original data
        balanced_labels_objectness = labels_objectness.clone()
        labels_class = labels_class.clone()

        # n_positive_anchors = (labels_objectness == 1).sum() the line below is faster and produce the same result
        n_positive_anchors = table_gts_positive_anchors.size(0)

        # Select up to self.max_positive_batch_size positive anchors
        # The excess is marked as dont care
        if n_positive_anchors > self.max_positive_batch_size:
            # positive_anchors_idxs = (balanced_labels == 1).nonzero().squeeze() # the line below is faster and produce the same result
            positive_anchors_idxs = table_gts_positive_anchors[:, 1]
            tmp_idxs = torch.randperm(n_positive_anchors, device=bboxes.device)[:n_positive_anchors - self.max_positive_batch_size]
            idxs_to_suppress = positive_anchors_idxs[tmp_idxs]
            balanced_labels_objectness[idxs_to_suppress] = -1  # mark them as don't care
            n_positive_anchors = self.max_positive_batch_size
            # To Check -> checked!:
            # The table_gts_positive_anchors is not consistent with the labels.
            # Should be consistent? Or this balancing is just for the cross-entropy loss?
            # [07/03/2020] - I think it is correct, the balancing is just for classification loss, as we have two labels.
            # [07/03/2020]   For the regression, we have no labels, just regressing the positive ones, so, making no sense to balance them.

        # Fill the remaining batch with negative anchors
        negative_anchors_idxs = (balanced_labels_objectness == 0).nonzero().squeeze()  # set the dim in squeeze
        n_negative_anchors = negative_anchors_idxs.size(0)
        n_anchors_to_complete_batch = self.batch_size - n_positive_anchors

        if n_negative_anchors > n_anchors_to_complete_batch:
            tmp_idxs = torch.randperm(n_negative_anchors, device=bboxes.device)[:n_negative_anchors - n_anchors_to_complete_batch]
            idxs_to_suppress = negative_anchors_idxs[tmp_idxs]
            balanced_labels_objectness[idxs_to_suppress] = -1  # mark them as don't care

        # You can see the difference before/after balancing the labels:
        # print('---------')
        # print((labels_objectness == -1).sum(), (labels_objectness == 0).sum(), (labels_objectness >= 1).sum())
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

        labels_objectness = labels_objectness.clone()  # I think this is not needed
        labels_class = labels_class.clone()  # I think this is not needed

        return image, bboxes, clss_idxs, labels_objectness, labels_class, table_gts_positive_anchors


def inv_normalize(t):

    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.])])
    return invTrans(t)


def get_dataloader(anchors):

    dataset = get_dataset(anchors, img_dir=config.img_folder, csv_file=config.annotations_file, train=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    return dataloader


def get_dataset(anchors, img_dir, csv_file, train):

    return MyDataset(img_dir=img_dir, csv_file=csv_file, input_img_size=config.input_img_size, anchors=anchors, train=train)


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
            class_idx = class_names.index(classname)  # starting from 1 (idx 0  is __background__ as in config)
        except ValueError:
            raise ValueError("Class '{}' not present in the list of classes {}.".format(classname, class_names))

        try:
            data_dict[filename]['bboxes'].append(bbox)
            data_dict[filename]['class_idxs'].append(class_idx)
        except KeyError:
            data_dict[filename] = {'bboxes': [bbox], 'class_idxs': [class_idx]}

    return [(filename, np.stack(annotations['bboxes']), np.stack(annotations['class_idxs'])) for filename, annotations in data_dict.items()]


def _format_data(data, anchors):

    anchors = anchors.to('cpu')

    # class_idxs is converted automatically to torch.int64 - inferred by its type
    data = [(filename, torch.tensor(bboxes), torch.tensor(class_idxs)) for filename, bboxes, class_idxs in data]
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

        # If this image has no positive assigned anchors
        if table_gts_positive_anchors.size(0) == 0:
            n_removed_bboxes += bboxes.size(0)
            n_removed_imgs += 1
            continue

        # tem como evitar esse unique sempre, com um if .. so pensar.. vai deixar o codigo mais legivel e simples ai em baixo?
        idxs = torch.unique(table_gts_positive_anchors[:, 0])
        new_bboxes = bboxes[idxs, :]
        new_class_idxs = class_idxs[idxs]

        if bboxes.size(0) == new_bboxes.size(0):
            assert torch.all(torch.eq(new_bboxes, bboxes))  # if this fails, my implementation is wrong

        # It is impossible this assertion to get failed
        assert bboxes.size(0) >= new_bboxes.size(0)  # remove this in the final version

        # If this fails.. my implementation is wrong .. but I fixed this.. I hope! (there was a bug here)
        # The maximum of possible gtboxes is the number of anchors available
        assert new_bboxes.size(0) <= anchors.size(0)  # remove this in the final version

        if bboxes.size(0) > new_bboxes.size(0):

            n_removed_bboxes += bboxes.size(0) - new_bboxes.size(0)

            bboxes, class_idxs = new_bboxes, new_class_idxs

            # Correct table_gts_positive_anchors indexes
            _, inv = torch.unique(table_gts_positive_anchors[:, 0], return_inverse=True)
            table_gts_positive_anchors[:, 0] = inv[:]

        new_data.append((filename, bboxes, class_idxs, labels_objectness, labels_class, table_gts_positive_anchors))

    print('SHOW STATISTICS OF CLASSES IN DATASET')
    print('{} bounding boxes in {} images in the dataset.'.format(n_bboxes, n_imgs))
    print('{} bounding boxes not assigned with anchors.'.format(n_removed_bboxes))
    print('{} images with all its bounding boxes not assigned with anchors.'.format(n_removed_imgs))
    print('Using {} bounding boxes in {} images.'.format(n_bboxes - n_removed_bboxes, n_imgs - n_removed_imgs))

    return new_data


if __name__ == "__main__":

    dataloader, input_img_size = get_dataloader()

    for img, annotation in dataloader:

        print(img.size(), annotation.size())
