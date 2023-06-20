#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:30:54 2022

@author: si-lab
"""
import os

import argparse
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('experiment_dir', help='Path to the directory of the experiment containing the 5 folds')
args = parser.parse_args()

std = 0

experiment_dir = args.experiment_dir
path_to_file = os.path.join(experiment_dir, 'configuratrion_params.txt')
with open(path_to_file, 'r') as fin:
    for line in fin.readlines():
        if line.startswith('Folds_val_accu'):
            splits = line.split('[')
            splits = splits[1].split(']')
            values = splits[0].split()
            values_npy = np.array(values, dtype=np.float32)
            print(values_npy)
            np.save(os.path.join(experiment_dir,'fold_accuracies.npy'),values_npy)

            std = np.std(values_npy)
            
with open(path_to_file, 'a') as fin:
    fin.write(f'standard_deviation={std}')
