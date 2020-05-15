import argparse
import random

random.seed(0)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--csv_1', required=True, type=str)
arg_parser.add_argument('--csv_2', required=True, type=str)
args = arg_parser.parse_args()

with open(args.csv_1, 'r') as f:
    data = f.readlines()
data_1 = set([d.strip().split(',')[0] for d in data])

with open(args.csv_2, 'r') as f:
    data = f.readlines()
data_2 = set([d.strip().split(',')[0] for d in data])

print(data_1 == data_2)
