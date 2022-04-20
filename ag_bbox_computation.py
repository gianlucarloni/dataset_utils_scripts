#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 12:00:29 2021
PER CROPPARE EFFETIVAMENTE LE IMMAGINI IN BASE ALLA BBOX TROVATA CON ag_bbox_detection
OVVERO IL VALORE DI COLONNA SALVATO NEL FILE txt DELLA SPECIFICA CARTELLA

@author: si-lab
"""

import os
import glob

import numpy as np
import imageio
import pandas as pd
import argparse
from tqdm import tqdm
from PIL import Image
import shutil

parse = argparse.ArgumentParser(description="Calculating Boundig Box to crop original dataset")
parse.add_argument('png_dir', help='Path to the input directory with original or corrupted PNG images')
parse.add_argument('csv_file', help='Path to the CSV file of original images')
parse.add_argument('out_dir', help='Path to the output directory with cropped PNG images')

args = parse.parse_args()
png_dir = args.png_dir
csv_path = args.csv_file
out_dir = args.out_dir

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if png_dir.endswith('/'):
    png_dir = png_dir[:-1]

dir_name = os.path.dirname(png_dir)
if os.path.basename(dir_name) == 'original':
    shutil.copy(csv_path, os.path.join(out_dir,'labels_balanced.csv'))

# read the CSV
csv_df = pd.read_csv(csv_path,sep=',', index_col='File name')

# read the index of the max_column from the file
with open(os.path.join(png_dir,'max_column.txt'),'r') as in_file:
    max_col = in_file.readline()
    
max_col = int(max_col)

for im_path in tqdm(glob.glob(os.path.join(png_dir, '*.png'))):    
    patient_name = os.path.basename(im_path)
    if patient_name.startswith('histeq_'):
        patient_name_org = patient_name[7:] #numero di lettere di histeq_
    elif patient_name.startswith('gauss_'):
        patient_name_org = patient_name[6:] #
    else:
        patient_name_org = patient_name
    im = np.array(imageio.imread(im_path))
    
    # estraggo la laterality
    laterality = csv_df.loc[patient_name_org,'LeftRight']
    
    row,col= im.shape
    
    if laterality == 'L':
        im = im[:,:max_col]
    else:
        im = im[:,-max_col:]
    
    imm = Image.fromarray(im)
    imm.save(os.path.join(out_dir,patient_name), format='PNG')     