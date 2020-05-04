import argparse
import random
import numpy as np

np.random.seed(0)
random.seed(0)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--img_filename', required=True, type=str)
arg_parser.add_argument('--csv_filename', required=True, type=str)
arg_parser.add_argument('--class_names', required=True, type=str)
arg_parser.add_argument('--number_of_objs', required=True, type=int)
args = arg_parser.parse_args()

MIN_WIDTH, MIN_HEIGHT = 17, 17
MAX_WIDTH, MAX_HEIGHT = 55, 55
IMG_WIDTH,IMG_HEIGHT = 256, 256

args.class_names = args.class_names.split(',')

data = []
for i in range(args.number_of_objs):

    class_name = random.choice(args.class_names)
    x0 = np.random.uniform(low=0, high=IMG_WIDTH, size=1)[0]
    y0 = np.random.uniform(low=0, high=IMG_HEIGHT, size=1)[0]
    w = np.random.uniform(low=MIN_WIDTH, high=MAX_WIDTH, size=1)[0]
    h = np.random.uniform(low=MIN_WIDTH, high=MAX_WIDTH, size=1)[0]

    x1 = x0 + w
    y1 = y0 + h

    data.append('{},{:.5f},{:.5f},{:.5f},{:.5f},{}'.format(args.img_filename, x0, y0, x1, y1, class_name))

with open(args.csv_filename, 'w') as f:
    f.write('\n'.join(data))
    f.write('\n')

