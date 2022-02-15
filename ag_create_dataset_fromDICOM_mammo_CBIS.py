#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:56:15 2022

@author: si-lab
"""

import pydicom
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pydicom import read_file
import scipy.ndimage
from PIL import Image
import pandas as pd
from tqdm import tqdm


cwd_path = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM'
data_path = os.path.join(cwd_path,'data')
csv_calc_train_path = os.path.join(cwd_path,'calc_case_description_train_set.csv')
csv_calc_test_path = os.path.join(cwd_path,'calc_case_description_test_set.csv')
csv_mass_train_path = os.path.join(cwd_path,'mass_case_description_train_set.csv')
csv_mass_test_path = os.path.join(cwd_path,'mass_case_description_test_set.csv')

for idx,csv_path in enumerate([csv_calc_train_path, csv_calc_test_path, csv_mass_train_path, csv_mass_test_path]):
    df = pd.read_csv(csv_path,sep=',',index_col='dirname')
    
    # if idx==0:
    #     benign_calc_dir = os.path.join(cwd_path,'dataset_png/train/calc/benign')
    #     if not os.path.exists(benign_calc_dir):
    #         os.makedirs(benign_calc_dir)
    #     malignant_calc_dir = os.path.join(cwd_path,'dataset_png/train/calc/malignant')
    #     if not os.path.exists(malignant_calc_dir):
    #         os.makedirs(malignant_calc_dir)
    # elif idx==1:
    #     benign_calc_dir = os.path.join(cwd_path,'dataset_png/test/calc/benign')
    #     if not os.path.exists(benign_calc_dir):
    #         os.makedirs(benign_calc_dir)
    #     malignant_calc_dir = os.path.join(cwd_path,'dataset_png/test/calc/malignant')
    #     if not os.path.exists(malignant_calc_dir):
    #         os.makedirs(malignant_calc_dir)
    # elif idx==2:
    #     benign_mass_dir = os.path.join(cwd_path,'dataset_png/train/mass/benign')
    #     if not os.path.exists(benign_mass_dir):
    #         os.makedirs(benign_mass_dir)
    #     malignant_mass_dir = os.path.join(cwd_path,'dataset_png/train/mass/malignant')
    #     if not os.path.exists(malignant_mass_dir):
    #         os.makedirs(malignant_mass_dir)
    # elif idx==3:
    #     benign_mass_dir = os.path.join(cwd_path,'dataset_png/test/mass/benign')
    #     if not os.path.exists(benign_mass_dir):
    #         os.makedirs(benign_mass_dir)
    #     malignant_mass_dir = os.path.join(cwd_path,'dataset_png/test/mass/malignant')
    #     if not os.path.exists(malignant_mass_dir):
    #         os.makedirs(malignant_mass_dir)
        
    if idx==0:
        benign_dir = os.path.join(cwd_path,'dataset_png/train/calc/benign')
        malignant_dir = os.path.join(cwd_path,'dataset_png/train/calc/malignant')
        no_callback_dir = os.path.join(cwd_path,'dataset_png/train/calc/no_callback')
    elif idx==1:
        benign_dir = os.path.join(cwd_path,'dataset_png/test/calc/benign')
        malignant_dir = os.path.join(cwd_path,'dataset_png/test/calc/malignant')
        no_callback_dir = os.path.join(cwd_path,'dataset_png/test/calc/no_callback')
    elif idx==2:
        benign_dir = os.path.join(cwd_path,'dataset_png/train/mass/benign')
        malignant_dir = os.path.join(cwd_path,'dataset_png/train/mass/malignant')
        no_callback_dir = os.path.join(cwd_path,'dataset_png/train/mass/no_callback')
    elif idx==3:
        benign_dir = os.path.join(cwd_path,'dataset_png/test/mass/benign')
        malignant_dir = os.path.join(cwd_path,'dataset_png/test/mass/malignant')
        no_callback_dir = os.path.join(cwd_path,'dataset_png/test/mass/no_callback')
        
    
    if not os.path.exists(benign_dir):
        os.makedirs(benign_dir)
    if not os.path.exists(malignant_dir):
        os.makedirs(malignant_dir)
    if not os.path.exists(no_callback_dir):
        os.makedirs(no_callback_dir)

       

    
    # for dirname in glob.iglob(data_path+'/*'):
    #     if dirname[-2]=='_':
            
    #         w = os.walk(dirname)
    
    #         for root,dirs,files in w:
                
    #             if len(files)==1:
    #                 #sei nel caso sbagliato, devi prendere il .dcm che ha per basename la parola croppedimage (sottocartella)
    #                 bn = os.path.basename(root)
    #                 if 'cropped images' in bn:
                        
    #             elif len(files)==2:
    #                 #devi prendere quella che pesa meno byte, come cropped image
                    
        
              

        


    for dirname in tqdm(df.index):
        label = df.loc[dirname]['pathology']

        w = os.walk(os.path.join(data_path,dirname))
    
        for root,dirs,files in w:
            
            if len(files)==0:
                continue
            
            if len(files)==1:
                #sei nel caso sbagliato, devi prendere il .dcm che ha per basename la parola croppedimage (sottocartella)
                bn = os.path.basename(root)
                if 'cropped images' in bn:
                    cr_im_path=os.path.join(root,files[0])
                else:
                    continue
                    
            elif len(files)==2:
                #devi prendere quella che pesa meno byte, come cropped image
                file_size0 = os.path.getsize(os.path.join(root,files[0]))
                file_size1 = os.path.getsize(os.path.join(root,files[1]))
                if file_size0<file_size1:
                    cr_im_path=os.path.join(root,files[0])
                else:
                    cr_im_path=os.path.join(root,files[1])
                

            dcm_im = read_file(cr_im_path)
            npy_im = dcm_im.pixel_array
            npy_im = ((npy_im - np.amin(npy_im))/(np.amax(npy_im) - np.amin(npy_im)))*255
            npy_im=npy_im.astype(np.uint8)
            
            pil_im = Image.fromarray(npy_im)
            file_name = os.path.basename(cr_im_path)
            file_name = file_name[:-3]
            file_name = file_name + 'png'
            if label=='BENIGN':
                out_path = os.path.join(benign_dir, dirname+'_'+file_name)
            elif label=='MALIGNANT':
                out_path = os.path.join(malignant_dir, dirname+'_'+file_name)
            elif label=='BENIGN_WITHOUT_CALLBACK':
                out_path = os.path.join(no_callback_dir, dirname+'_'+file_name)
            
            pil_im.save(out_path,'PNG')
            