#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:56:15 2022

@author: si-lab
"""

import os

import numpy as np
from pydicom import read_file
from PIL import Image
import pandas as pd


def get_bbox(input_mask_npy):
    
    # troviamo gli indici di riga e colonna dove assume valori non nulli --> determiniamo la bounding box
    nonzero = input_mask_npy.nonzero()    
    row_min = np.amin(nonzero[0])
    row_max = np.amax(nonzero[0])
    col_min = np.amin(nonzero[1])
    col_max = np.amax(nonzero[1])
    
    return row_min, row_max, col_min, col_max


cwd_path = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM'
data_path = os.path.join(cwd_path,'data')
csv_mass_train_path = os.path.join(cwd_path,'mass_case_description_train_set.csv')
csv_mass_test_path = os.path.join(cwd_path,'mass_case_description_test_set.csv')

#Folder to save CLAHE images
output_path = os.path.join(cwd_path,'dataset_png_full_imgs')
if not os.path.exists(output_path):
    os.makedirs(output_path)

output_full_path = os.path.join(output_path,'full_images')
if not os.path.exists(output_full_path):
    os.makedirs(output_full_path)


for ind,csv_path in enumerate([csv_mass_train_path, csv_mass_test_path]):
    df = pd.read_csv(csv_path,sep=',',index_col='dirname')

    idx=0 
    dirnames = df.index
    
    while idx < len(dirnames):
        print(f'Ind {ind}. Iteration: {idx} over {len(dirnames)}')
        dirname = dirnames[idx]
        #for dirname in tqdm(df.index):
        #E.g., dirname:= Mass-Training_P_00001_LEFT_MLO_1
        
        
        splits = dirname.split(sep='_')
        full_image_dir = '_'.join(splits[0:5])
        
    
        records_per_patient=[]
        for dn in dirnames:
            if dn.startswith(full_image_dir):
                records_per_patient.append(dn)
                
        num_records_per_patient=len(records_per_patient)
        df_patient = df.loc[records_per_patient]
        
        #
        w_full = os.walk(os.path.join(data_path,full_image_dir))
        for root,dirs,files in w_full:
            if len(files)==0:
                continue
            elif len(files)==1:
                full_image_path=os.path.join(root,files[0])
            else:
                print('-------------------FULL MAMMO IMAGE NOT FOUND CORRECTLY')
        #
        
        
        
        dcm_full_im = read_file(full_image_path)
        npy_full_im = dcm_full_im.pixel_array
        npy_full_im = ((npy_full_im - np.amin(npy_full_im))/(np.amax(npy_full_im) - np.amin(npy_full_im)))*255
        npy_full_im=npy_full_im.astype(np.uint8)
        pil_full_im = Image.fromarray(npy_full_im)
        pil_full_im.save(os.path.join(output_full_path,full_image_dir+'.png'),'PNG')
        

            
        idx += num_records_per_patient
            
        