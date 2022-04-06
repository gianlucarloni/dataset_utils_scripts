#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:36:11 2022

@author: si-lab
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import glob
import nibabel as nib
from tqdm import tqdm

import pandas as pd

parse = argparse.ArgumentParser(description='Selezionare le slice delle immagini che non aveano maschere ippocampali')
parse.add_argument('images', help='Path to NiFti images directory')
parse.add_argument('csv', help='Path to csv file containing the idx_min and idx_max')
args = parse.parse_args()

path_imgs = args.images
path_csv = args.csv

df = pd.read_csv(path_csv, sep=',', index_col='file_name')

patient_names = df.index
# patient_names = pd.Series(['094_S_1102'])

for name in tqdm(patient_names):
    idx_min = df.loc[name, 'idx_min']
    idx_max = df.loc[name, 'idx_max']
    
    for root, dirs, files in os.walk(os.path.join(path_imgs, name), topdown=False):
        if len(files) != 0:
            img_nib = nib.load(os.path.join(root, files[0]))
            img_npy = img_nib.get_fdata()
            shape = img_npy.shape
            img_new_npy = np.copy(img_npy)
            # mask_new_npy[:, shape[1]-idx_max:shape[1]-idx_min, :] = 1
            img_new_npy[:, :idx_min+1, :] = 0
            img_new_npy[:, idx_max:, :] = 0
            img_new_nib = nib.Nifti1Image(img_new_npy, img_nib.affine, header=img_nib.header)
            img_new_nib.to_filename(os.path.join(root, files[0])[:-7]+'_masked.nii.gz')

