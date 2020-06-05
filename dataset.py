import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import os
from loss import anchor_labels
from tqdm import tqdm
import config
import xml.etree.ElementTree as ET
from voc_eval import voc_eval


class DatasetWrapper(Dataset):

    def __init__(self, format_type, data_info, input_img_size, anchors, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_info = data_info
        self.format_type = format_type
        if format_type == 'simple':
            img_dir, csv_file = data_info
            data, classes = simple_format_loader(csv_file)
        elif format_type == 'VOC':
            voc_base_path, set_type = data_info
            img_dir = os.path.join(voc_base_path, 'JPEGImages')
            data, classes = VOC_format_loader(voc_base_path, set_type)
            #ao trocar para o abaixo deveria dar o mesmo resultado! (simple)
        else:
            print('Data format does not exist!')
            exit()

        for i in range(len(data)):

            assert (data[i][1][:, 0] >= 0.0).all() and (data[i][1][:, 1] >= 0.0).all()
            assert (data[i][1][:, 2] <= config.original_img_size[0]).all() and (data[i][1][:, 3] <= config.original_img_size[1]).all()
            assert (data[i][1][:, 0] < data[i][1][:, 2]).all() and (data[i][1][:, 1] < data[i][1][:, 3]).all()

            data[i][1][:, 2] -= 1.0  # np.finfo(np.float32).eps
            data[i][1][:, 3] -= 1.0  # np.finfo(np.float32).eps

        # Resize the annotations to the new image size
        # the -1 in input img_size is to ensure: [0, input_img_size-1]
        self.img_scales = ((input_img_size[0] - 1.0) / float(config.original_img_size[0]),
                           (input_img_size[1] - 1.0) / float(config.original_img_size[1]))

        for i in range(len(data)):
            # the -1 in input img_size is to ensure: [0, input_img_size-1],
            data[i][1][:, 0] *= self.img_scales[0]
            data[i][1][:, 1] *= self.img_scales[1]
            data[i][1][:, 2] *= self.img_scales[0]
            data[i][1][:, 3] *= self.img_scales[1]

        # # There is no filtering at all!!!!  # Filter small bboxes (not implemented in that famous pytorch version)
        # # Filter by min_size
        # new_data = []
        # max_width, max_height = 0, 0
        # min_width, min_height = config.input_img_size
        # for i in range(len(data)):
        #     w = data[i][1][2] - data[i][1][0] + 1.0
        #     h = data[i][1][3] - data[i][1][1] + 1.0

        #     if w >= config.min_size and h >= config.min_size:
        #         new_data.append(data[i])

        #         max_width, max_height = max(max_width, w), max(max_height, h)
        #         min_width, min_height = min(min_width, w), min(min_height, h)

        # if len(data) > len(new_data):
        #     print('WARNING: {} annotations were smaller than the minimun width/height size, being removed ending with {} annotations.'.format(len(data) - len(new_data), len(new_data)))

        # assert len(new_data) > 0  # if there is no annotation in this image what should I do?

        # print('INFO: max_width: {}, max_height: {}'.format(max_width, max_height))
        # print('INFO: min_width: {}, min_height: {}'.format(min_width, min_height))

        # data = new_data

        max_width, max_height = 0, 0
        min_width, min_height = config.input_img_size
        for i in range(len(data)):
            w = data[i][1][:, 2] - data[i][1][:, 0] + 1.0
            h = data[i][1][:, 3] - data[i][1][:, 1] + 1.0
            max_width, max_height = max(max_width, w.max()), max(max_height, h.max())
            min_width, min_height = min(min_width, w.min()), min(min_height, h.min())

        print()
        print('INFO: max_width: {}, max_height: {}'.format(max_width, max_height))
        print('INFO: min_width: {}, min_height: {}'.format(min_width, min_height))

        # data = _group_by_filename(data, config.class_names)
        # # Remove duplicated bboxes
        # for i in range(len(data)):
        #     data[i] = (data[i][0], np.unique(data[i][1], axis=0))

        data = _format_data(data, anchors)

        print('# of classes found: {}'.format({x: classes.count(x) for x in set(classes)}))

        if len(data) == 0:
            print('WARNING: There is no data for this dataset. Please check your data and then the configurations (config file).')
            exit()

        self.files_annot = data
        self.img_dir = img_dir
        self.input_img_size = input_img_size
        self.batch_size = config.rpn_batch_size  # it is confusing putting this stuff in the dataloader
        self.max_positive_batch_ratio = config.max_positive_batch_ratio
        self.max_positive_batch_size = int(self.max_positive_batch_ratio * self.batch_size)  # put this on config ?
        self.predictions = {img_idx: {cls_idx: None for cls_idx in range(1, config.n_classes)} for img_idx in range(len(self.files_annot))}
        self.train = train

    def __len__(self):
        return len(self.files_annot)

    def __getitem__(self, idx):
        if self.train:
            return self._getitem_train(idx)
        else:
            return idx, self._getitem_test(idx)

    def _getitem_train(self, idx):

        img_base_name, annotations, rpn_labels, expanded_annotations, _ = self.files_annot[idx]
        img_name = os.path.join(self.img_dir, img_base_name)
        image = Image.open(img_name)
        image = transforms.Resize((self.input_img_size[1], self.input_img_size[0]))(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        # fazer o tracking de todo comentario.. tipo esse aqui, analisar se realmente preciso fazer o clone
        # e ir eliminando coisas desnecessarias e comentar o que eh necessario
        # Because I don't want to modify the original data
        balanced_rpn_labels = rpn_labels.clone()

        n_positive_anchors = (balanced_rpn_labels == 1).sum()

        # Select up to self.max_positive_batch_size positive anchors
        # The excess is marked as dont care
        if n_positive_anchors > self.max_positive_batch_size:
            positive_anchors_idxs = (balanced_rpn_labels == 1).nonzero().squeeze(1)
            tmp_idxs = torch.randperm(n_positive_anchors, device=annotations.device)[:n_positive_anchors - self.max_positive_batch_size]
            idxs_to_suppress = positive_anchors_idxs[tmp_idxs]
            balanced_rpn_labels[idxs_to_suppress] = -1  # mark them as don't care
            n_positive_anchors = self.max_positive_batch_size
            # To Check -> checked!:
            # The table_gts_positive_anchors is not consistent with the labels.
            # Should be consistent? Or this balancing is just for the cross-entropy loss?
            # [07/03/2020] - I think it is correct, the balancing is just for classification loss, as we have two labels.
            # [07/03/2020]   For the regression, we have no labels, just regressing the positive ones, so, making no sense to balance them.

        # Fill the remaining batch with negative anchors
        negative_anchors_idxs = (balanced_rpn_labels == 0).nonzero().squeeze(1)
        n_negative_anchors = negative_anchors_idxs.size(0)
        n_anchors_to_complete_batch = self.batch_size - n_positive_anchors

        if n_negative_anchors > n_anchors_to_complete_batch:
            tmp_idxs = torch.randperm(n_negative_anchors, device=annotations.device)[:n_negative_anchors - n_anchors_to_complete_batch]
            idxs_to_suppress = negative_anchors_idxs[tmp_idxs]
            balanced_rpn_labels[idxs_to_suppress] = -1  # mark them as don't care

        # You can see the difference before/after balancing the labels:
        # print('---------')
        # print((rpn_labels == -1).sum(), (rpn_labels == 0).sum(), (rpn_labels == 1).sum())
        # print((balanced_rpn_labels == -1).sum(), (balanced_rpn_labels == 0).sum(), (balanced_rpn_labels == 1).sum())
        # print('---------')
        # exit()

        return image, annotations, balanced_rpn_labels, expanded_annotations

    def _getitem_test(self, idx):

        img_base_name, annotations, rpn_labels, expanded_annotations, table_annotations_dbg = self.files_annot[idx]
        img_name = os.path.join(self.img_dir, img_base_name)
        image = Image.open(img_name)
        image = transforms.Resize((self.input_img_size[1], self.input_img_size[0]))(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        rpn_labels = rpn_labels.clone()  # I think this is not needed

        return image, annotations, rpn_labels, expanded_annotations, table_annotations_dbg

    def store_prediction(self, img_idx, bboxes, scores, clss):

        assert not self.train

        # # DEBUG
        # # convert annotations to predictions - should result in maxium AP
        # annotations = self.files_annot[img_idx][1].detach().cpu().numpy()
        # bboxes = annotations[:, :-1]
        # clss = annotations[:, -1].astype(np.int)
        # scores = np.ones(clss.shape)

        # for cls_idx in range(1, config.n_classes):

        #     idxs = cls_idx == clss
        #     if any(idxs):
        #         self.predictions[img_idx][cls_idx] = (bboxes[idxs, :], scores[idxs])
        # #######

        # Correct code:
        for cls_idx in range(1, config.n_classes):

            idxs = cls_idx == clss
            if any(idxs):
                self.predictions[img_idx][cls_idx] = (bboxes[idxs, :], scores[idxs])

    def finish_predictions(self):

        for cls_idx in range(1, config.n_classes):

            data = []
            class_name = config.class_names[cls_idx]
            # print('Writing {} VOC results file'.format(class_name))

            for img_idx in range(len(self.files_annot)):

                filename = self.files_annot[img_idx][0].replace('.jpg', '')

                preds = self.predictions[img_idx][cls_idx]

                if preds is not None:

                    bboxes, scores = preds

                    # rescale the bboxes to the original scale
                    bboxes[:, 0] /= self.img_scales[0]
                    bboxes[:, 1] /= self.img_scales[1]
                    bboxes[:, 2] /= self.img_scales[0]
                    bboxes[:, 3] /= self.img_scales[1]

                    for i in range(bboxes.shape[0]):
                        data.append('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}'.
                                    format(filename,
                                           scores[i],
                                           bboxes[i, 0], bboxes[i, 1],
                                           bboxes[i, 2] + 1, bboxes[i, 3] + 1))
                                           #the VOCdevkit expects 1-based indices ??????????????????????????
                                           #eu deixei so +1 so na segunda parte pq assim bateu.. 
                                           # mas devo checar isso aqui e em todo o codigo pq ta uma emboleira so!
                                           #talvez mudar ate o voc_eval para n considerar esse +1 ou -1

            with open('output/predictions/{}.txt'.format(class_name), 'w') as f:
                f.write('\n'.join(data))

        self.predictions = None  # To avoid later use

    def evaluate_predictions(self, use_old_metric=False, verbose=False):

        # not implemented for other option
        assert self.format_type == 'VOC'

        annopath = os.path.join(self.data_info[0], 'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(self.data_info[0], 'ImageSets', 'Main', '{}.txt'.format(self.data_info[1]))
        cachedir = os.path.join(self.data_info[0], 'annotations_cache')
        aps = []
        version = '(With {} VOC metric version)'.format('OLD' if use_old_metric else 'NEW')

        for cls_idx in range(1, config.n_classes):
            class_name = config.class_names[cls_idx]
            file_name = 'output/predictions/{}.txt'.format(class_name)
            rec, prec, ap = voc_eval(file_name,
                                     annopath,
                                     imagesetfile,
                                     class_name,
                                     cachedir,
                                     ovthresh=0.5,
                                     use_07_metric=use_old_metric)
            aps.append(ap)

        print('{:*<29}'.format('*'))
        print('{:^30}'.format('Results'))
        print(version)
        print('{:-<29}'.format('-'))
        print('Class APs:')
        for i in range(len(aps)):
            print('{:<12} {:.4f}'.format(config.class_names[i + 1] + ':', aps[i]))
        print('{:-<29}'.format('-'))
        print('Mean AP: {:.4f}'.format(np.mean(aps)))
        print('{:*<29}'.format('*'))


def inv_normalize(t):

    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.])])
    return invTrans(t)


def _group_by_filename(data_list, class_names):

    data_dict = {}

    for filename, annotation in data_list:

        try:
            data_dict[filename]['annotations'].append(annotation)
        except KeyError:
            data_dict[filename] = {'annotations': [annotation]}

    return [(filename, np.stack(annotations['annotations'])) for filename, annotations in data_dict.items()]


def _format_data(data, anchors):

    anchors = anchors.to('cpu')

    #acho que n precisa criar o tensor aqui.. deixa para criar na hora do getitem - memoria..
    data = [(filename, torch.tensor(annotations)) for filename, annotations in data]
    new_data = []

    n_annotations = 0
    n_imgs = 0
    n_annotations_directly_assigned = 0

    print('Processing bounding boxes')
    for filename, annotations in tqdm(data):

        n_imgs += 1
        n_annotations += annotations.size(0)

        rpn_labels, expanded_annotations, table_annotations_dbg = anchor_labels(anchors, annotations)

        n_annotations_directly_assigned += table_annotations_dbg.unique().size(0)
        new_data.append((filename, annotations, rpn_labels, expanded_annotations, table_annotations_dbg))

    print('{} bounding boxes in {} images in the dataset.'.format(n_annotations, n_imgs))
    print('{} bounding boxes was directly assigned with anchors.'.format(n_annotations_directly_assigned))
    print('(The not directly assigned anchors may not covered by an anchor or have a higher IoU with another bbox, but will be trained and maybe get a proposal assigned to it - low-hanging fruit here?)')

    return new_data


def simple_format_loader(csv_file):

    # Format data into (img_filename, annotations_x0y0x1y1, class_name)
    with open(csv_file) as f:
        lines = f.readlines()

    data, classes = [], []
    for l in lines:
        tmp = l.strip().split(',')
        filename = tmp[0]
        bbox = [tmp[1], tmp[2], tmp[3], tmp[4]]
        try:
            class_idx = config.class_names.index(tmp[5])  # starting from 1 (idx 0  is __background__ as in config)
        except ValueError:
            raise ValueError("Class '{}' not present in the list of classes {}.".format(tmp[5], config.class_names))
        classes.append(tmp[5])
        annotation = np.array([np.float32(e) for e in bbox + [class_idx]])
        data.append((filename, annotation))

    print('vc mudou o seu __init__.. entao tem que adaptar aqui, removendo duplicatas e agrupando by filename')
    exit()
    return data, classes


def VOC_format_loader(voc_base_path, set_type):

    set_file = os.path.join(voc_base_path, 'ImageSets/Main', '{}.txt'.format(set_type))
    with open(set_file, 'r') as f:
        image_set = f.readlines()
    image_set = [f.strip()for f in image_set]

    data, classes = [], []

    for base_file_name in image_set:

        annotations, img_classes = [], []
        annotation_file = os.path.join(voc_base_path, 'Annotations', '{}.xml'.format(base_file_name))

        tree = ET.parse(annotation_file)

        for obj in tree.findall('object'):

            class_name = obj.find('name').text

            try:
                class_idx = config.class_names.index(class_name)
            except ValueError:
                raise ValueError("Class '{}' not present in the list of classes {}.".format(class_name, config.class_names))

            bbox = obj.find('bndbox')
            xmin, ymin = np.float32(bbox.find('xmin').text), np.float32(bbox.find('ymin').text)
            xmax, ymax = np.float32(bbox.find('xmax').text), np.float32(bbox.find('ymax').text)

            img_classes.append(class_name)
            annotation = np.array([xmin, ymin, xmax, ymax, np.float32(class_idx)])
            annotations.append(annotation)

        annotations, idxs = np.unique(np.stack(annotations), axis=0, return_index=True)
        img_classes = [img_classes[i] for i in idxs]

        classes += img_classes
        data.append(('{}.jpg'.format(base_file_name), np.unique(np.stack(annotations), axis=0)))

    return data, classes
