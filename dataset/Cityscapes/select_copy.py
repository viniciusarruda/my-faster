import argparse
import random

random.seed(0)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--img_folder', required=True, type=str)
arg_parser.add_argument('--csv_in', required=True, type=str)
arg_parser.add_argument('--csv_base', required=True, type=str)
arg_parser.add_argument('--csv_out', required=True, type=str)
args = arg_parser.parse_args()

with open(args.csv_base, 'r') as f:
    data = f.readlines()
keys = list(set([d.strip().split(',')[0] for d in data]))

with open(args.csv_in, 'r') as f:
    data = f.readlines()
data = [d.strip().split(',') for d in data]

dict_data = {}

for d in data:

    try:
        dict_data[d[0]].append(d)
    except KeyError:
        dict_data[d[0]] = []
        dict_data[d[0]].append(d)

classes = []
data = []
for k in keys:
    for bboxes in dict_data[k]: 
        filename = bboxes[0]
        x0 = bboxes[1]
        y0 = bboxes[2]
        x1 = bboxes[3]
        y1 = bboxes[4]
        class_name = bboxes[5]
        classes.append(class_name)

        data.append('{},{},{},{},{},{}'.format(filename, x0, y0, x1, y1, class_name))

with open(args.csv_out, 'w') as f:
    f.write('\n'.join(data))
    f.write('\n')

print('classes found: {}'.format(list(set(classes))))

