#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 08:58:58 2022

seleziona n=20 slice per ogni stack numpy di ogni soggetto, tramite np.linspace

e salva le rispettive slice in png per l'utilizzo con ppnet

@author: si-lab
"""
import numpy as np
from glob import glob
import os
from PIL import Image
from tqdm import tqdm
import argparse
import pandas as pd


parse = argparse.ArgumentParser(description='Seleziona n=18 (default) slice per ogni stack numpy di ogni soggetto, tramite np.linspace; e salva le rispettive slice in png per l''utilizzo con ppnet')
parse.add_argument('input_npy_images', help='Path to npy images directory')
parse.add_argument('output_png_images', help='Path to png images directory')
parse.add_argument('-n','--number_of_slices', help='number of slices to select from each patient\'s scan', default=18,type=int)


args = parse.parse_args()

path_npy = args.input_npy_images
path_png = args.output_png_images
n = args.number_of_slices

diagnostic_class = os.path.basename(path_npy)


df = pd.DataFrame(columns=['path_to_png','patient_name','label'],dtype=str)

toInsert_path_to_png=[]
toInsert_patient_name=[]
toInsert_label=[]

# for name in tqdm(glob(os.path.join(path_npy,'*.npy'))):
for name in ['/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ag_ADNI/COORTE1_5T_AD_CN_ADNI1_SCREENING_PREPROCESSED_3D/ADNI_NPY/CN/ADNI_016_S_0538_MR_MPR__GradWarp__B1_Correction_Br_20070926113921303_S24796_I75310_masked.npy']:
    full_name = os.path.basename(name)[:-10]
    patient_name = full_name[:15]
    
    out_path = os.path.join(path_png,patient_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    img_npy = np.load(name)
    shape = img_npy.shape
    
    #escludiamo le eventuali fette nulle di ogni scan
    nonzero=np.nonzero(img_npy)
    idx_min = int(np.amin(nonzero[1]))
    idx_max = int(np.amax(nonzero[1]))

    if idx_max-3-(idx_min+2) < n:
        print(f'Dimensioni sbagliate: {idx_min} {idx_max} --- {full_name}')
    
    slicing_range = np.round(np.linspace(idx_min+2,idx_max-3,num=n),decimals=0).astype(int)
    # slicing_range = np.round(np.linspace(idx_min+4,idx_max-4,num=n),decimals=0).astype(int)

    img_npy = img_npy[:,slicing_range,:]
    
    
    
    for s in range(n):
        s_npy = img_npy[:,s,:]
        #
        # s_npy = 255*(s_npy-np.min(s_npy))/(np.amax(s_npy)-np.amin(s_npy))
        # s_npy = s_npy.astype(np.uint8)
        
        s_pil = Image.fromarray(s_npy)
        s_pil=s_pil.rotate(90,expand=1)
        s_pil.save(os.path.join(out_path,full_name+f'slice{s}.png'),format='PNG')
        #
        toInsert_path_to_png.append(os.path.join(patient_name,full_name+f'slice{s}.png')) 
        toInsert_patient_name.append(patient_name)
        toInsert_label.append(diagnostic_class)
        
# df['path_to_png']=pd.Series(toInsert_path_to_png)
# df['patient_name']=pd.Series(toInsert_patient_name)
# df['label']=pd.Series(toInsert_label)

# df.to_csv(os.path.join(path_png,f'labels{diagnostic_class}.csv'),index=False)