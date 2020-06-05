import os
from tqdm import tqdm
import xml.etree.ElementTree as ET

import argparse

TODO

def _parse_args():

    parser = argparse.ArgumentParser(description='Convert the dataset annotations in car_2box format to Pasval VOC format.')
    parser.add_argument('--annotations', help='Folder with the dataset in the car_2box format.', required=True)
    parser.add_argument('--imagesets', help='Folder to place the dataset in the Pascal VOC format.', required=True)    
    parser.add_argument('--csv', help='Folder to place the dataset in the Pascal VOC format.', required=True)    
    return parser.parse_args()

args = _parse_args()

with open(args.imagesets, 'r') as f:
    files = f.readlines()
files = [f.strip() for f in files]

entries = []
for f in tqdm(files):
    tree = ET.parse(os.path.join(args.annotations, '{}.xml'.format(f)))
    objs = tree.findall('object')

    for obj in objs:
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        clss = obj.find('name').text
        entries.append('{}.jpg,{},{},{},{},{}'.format(f, x1, y1, x2, y2, clss))

with open(args.csv, 'w') as f:
    f.write('\n'.join(entries))

