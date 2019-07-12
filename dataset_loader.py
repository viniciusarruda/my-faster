from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import os

class MyDataset(Dataset):

    def __init__(self, img_dir, csv_file, input_img_size):
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
        data = [(d[0], np.array(float(d[1]))) for d in data]
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
        clss = torch.Tensor(self.files_annot[idx][1])

        return image, clss


def get_dataloader():

    input_img_size = (128, 128)
    dataset = MyDataset(img_dir='dataset/images/', csv_file='dataset/annotations.csv', input_img_size=input_img_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    return dataloader, input_img_size