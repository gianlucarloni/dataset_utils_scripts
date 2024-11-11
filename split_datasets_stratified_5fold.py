#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 12:12:46 2021

@author: si-lab
"""

import pandas as pd
import argparse
import os
import shutil
import numpy as np
import glob

# #
# osservazione: parlando con eva 
# TODO: dovremmo tenere conto nella suddivisione train_test_split che le immagini
# appartenenti alla stessa lesione vadano TUTTE o nel training o nel test,
# invece al momento noi stiamo splittando indistintamente
# #




parse = argparse.ArgumentParser(description="At the end of this script we obtain three dataset folders: push, valid, test.\n To use original images only (not corrupted), do not pass the argument -c")

parse.add_argument('png_dir', help='Path to the input directory with original PNG images (parent folder of class folders, e.g. \n '\
                   '*dataset*, parent of *benign* and *malignant* folders')
parse.add_argument('-csv','--csv_file', help='Path to the CSV file of original images')
parse.add_argument('-d', '--dest_dir', help='Path to destination directory, parent folder of: fold0, fold1, fold2, fold3, and fold4')

args = parse.parse_args()

png_dir = args.png_dir
csv_file_path = args.csv_file
dest_dir = args.dest_dir


df = pd.read_csv(csv_file_path,sep=',', index_col='file_name')

from distutils.dir_util import copy_tree
from stratified_group_data_splitting import StratifiedGroupKFold

# Suddivisione nelle 5 fold

X = np.array(df.index)
group = np.array(df['patient_id'])
y = np.array(df['label'])

y = np.array([0 if elem=='benign' else 1 for elem in y]) #TODO modificare con la stringa opportuna

x = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) #reproducibility, split in 5 folds
folds = {}
for idx, (push_idxs, valid_idxs) in enumerate(x.split(X,y,groups=group)):
    folds[f'fold{idx}'] = valid_idxs


#TODO 7 aprile 2022
for fold in folds.keys():
    for label in ['benign', 'malignant']:
        path = os.path.join(dest_dir, fold, label)
        if not os.path.exists(path):
            os.makedirs(path)

for fold, fold_idxs in folds.items():
    for idx in fold_idxs:
        name = X[idx]
        label = df.at[name, 'label']
        shutil.copy(os.path.join(png_dir, name), os.path.join(dest_dir, fold, label, os.path.basename(name)))

    fold_names = X[fold_idxs]
    df_fold = df.loc[fold_names,('label')]
    df_fold.to_csv(os.path.join(dest_dir, f'labels_{fold}.csv'),sep=',',index=True)

print('Done.')