from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import os


# TODO: normalizar os dados de acordo com : https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html


class MyDataset(Dataset):

    def __init__(self, img_dir, csv_file, input_img_size):
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
            data[i][1][0] = (data[i][1][0] / original_img_size[0]) * input_img_size[0]
            data[i][1][1] = (data[i][1][1] / original_img_size[1]) * input_img_size[1]
            data[i][1][2] = (data[i][1][2] / original_img_size[0]) * input_img_size[0]
            data[i][1][3] = (data[i][1][3] / original_img_size[1]) * input_img_size[1]

        _inplace_adjust_bbox2offset(data)

        self.files_annot = data
        self.img_dir = img_dir
        self.input_img_size = input_img_size

    def __len__(self):
        return len(self.files_annot)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir,
                                self.files_annot[idx][0])
        image = Image.open(img_name)
        image = transforms.Resize(self.input_img_size)(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        clss = torch.Tensor(self.files_annot[idx][1])

        return image, clss


def inv_normalize(t):

    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

    return invTrans(t)


def get_dataloader():

    # input_img_size = (128, 128)
    input_img_size = (224, 224)
    dataset = MyDataset(img_dir='dataset/one_day/', csv_file='dataset/one_day.csv', input_img_size=input_img_size)
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