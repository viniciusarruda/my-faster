import os
from PIL import Image
from shutil import copyfile
from tqdm import tqdm
import argparse


def _parse_args():

    parser = argparse.ArgumentParser(description='Convert the dataset annotations in simple format to Pasval VOC format.')
    parser.add_argument('--simple_imgs', help='Folder with the images of the dataset which follows the simple format.', required=True)
    parser.add_argument('--simple_csv', help='File with the annotations of the dataset in the simple format.', required=True)
    parser.add_argument('--pascalvoc', help='Folder to place the dataset in the Pascal VOC format.', required=True)
    parser.add_argument('--set_type', help='Set type e.g., train, trainval, test.', required=True)
    return parser.parse_args()


def _VOCwritexml(rec, path):

    def writexml(fid, rec, depth):

        for fn in list(rec.keys()):

            f = rec[fn]

            if f:
                if type(f) is str:
                    fid.write('{}<{}>{}</{}>\n'.format('\t' * depth, fn, f, fn))

                elif type(f) is dict:
                    fid.write('{}<{}>\n'.format('\t' * depth, fn))
                    writexml(fid, f, depth + 1)
                    fid.write('{}</{}>\n'.format('\t' * depth, fn))

                elif type(f) is list:
                    for elem in f:  # each elem should be a dict
                        fid.write('{}<{}>\n'.format('\t' * depth, fn))
                        writexml(fid, elem, depth + 1)
                        fid.write('{}</{}>\n'.format('\t' * depth, fn))

    with open(path, 'w') as fid:
        writexml(fid, rec, 0)


args = _parse_args()

target_jpeg_images = os.path.join(args.pascalvoc, 'VOC', 'JPEGImages')
target_image_set = os.path.join(args.pascalvoc, 'VOC', 'ImageSets/Main')
target_annotations = os.path.join(args.pascalvoc, 'VOC', 'Annotations')

for f in [target_jpeg_images, target_image_set, target_annotations]:
    os.makedirs(f)


print('Copying images...')
for f in tqdm(os.listdir(args.simple_imgs)):
    assert f[-4:] == '.jpg'
    copyfile(os.path.join(args.simple_imgs, f), os.path.join(target_jpeg_images, f))


print('Processing annotations...')
with open(args.simple_csv, 'r') as f:
    lines = f.readlines()
data = [l.strip().split(',') for l in lines]

data_dict = {}
for file_name, *d in tqdm(data):
    try:
        data_dict[file_name].append(d)
    except KeyError:
        data_dict[file_name] = [d]


print('Conveting format...')
for file_name in tqdm(data_dict):

    width, height = Image.open(os.path.join(target_jpeg_images, file_name)).size

    save_var = {'annotation': {}}
    save_var['annotation']['folder'] = 'Unspecified'
    save_var['annotation']['filename'] = file_name
    save_var['annotation']['segmented'] = '0'
    save_var['annotation']['size'] = {}
    save_var['annotation']['size']['width'] = '{}'.format(width)
    save_var['annotation']['size']['height'] = '{}'.format(height)
    save_var['annotation']['size']['depth'] = '3'
    save_var['annotation']['object'] = []

    for i_obj in range(len(data_dict[file_name])):

        xmin, ymin, xmax, ymax, class_name = data_dict[file_name][i_obj]

        save_var['annotation']['object'].append({})
        save_var['annotation']['object'][i_obj]['bndbox'] = {}
        save_var['annotation']['object'][i_obj]['bndbox']['xmin'] = xmin
        save_var['annotation']['object'][i_obj]['bndbox']['ymin'] = ymin
        save_var['annotation']['object'][i_obj]['bndbox']['xmax'] = xmax
        save_var['annotation']['object'][i_obj]['bndbox']['ymax'] = ymax
        save_var['annotation']['object'][i_obj]['name'] = class_name
        save_var['annotation']['object'][i_obj]['difficult'] = '0'
        save_var['annotation']['object'][i_obj]['truncated'] = '0'
        save_var['annotation']['object'][i_obj]['occluded'] = '0'
        save_var['annotation']['object'][i_obj]['pose'] = 'Unspecified'

    _VOCwritexml(save_var, os.path.join(target_annotations, file_name.replace('.jpg', '.xml')))


print('Saving image set...')
f = os.path.join(target_image_set, '{}.txt'.format(args.set_type))
filenames = [f.replace('.jpg', '') for f in data_dict.keys()]
filenames.sort()
with open(f, 'w') as stream:
    stream.write('\n'.join(filenames))


print('Finished!')
