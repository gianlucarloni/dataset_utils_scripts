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
import imgaug.augmenters as iaa


def get_bbox(input_mask_npy):
    
    # troviamo gli indici di riga e colonna dove assume valori non nulli --> determiniamo la bounding box
    nonzero = input_mask_npy.nonzero()    
    row_min = np.amin(nonzero[0])
    row_max = np.amax(nonzero[0])
    col_min = np.amin(nonzero[1])
    col_max = np.amax(nonzero[1])
    
    return row_min, row_max, col_min, col_max

apply_clahe = iaa.AllChannelsCLAHE()


cwd_path = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM'
data_path = os.path.join(cwd_path,'data')
csv_mass_train_path = os.path.join(cwd_path,'mass_case_description_train_set.csv')
csv_mass_test_path = os.path.join(cwd_path,'mass_case_description_test_set.csv')

#Folder to save CLAHE images
output_clahe_path = os.path.join(cwd_path,'dataset_png_clahe')
if not os.path.exists(output_clahe_path):
    os.makedirs(output_clahe_path)

output_full_clahe_path = os.path.join(output_clahe_path,'full_images')
if not os.path.exists(output_full_clahe_path):
    os.makedirs(output_full_clahe_path)


for ind,csv_path in enumerate([csv_mass_train_path, csv_mass_test_path]):
    df = pd.read_csv(csv_path,sep=',',index_col='dirname')

    if ind==0:
        benign_dir = os.path.join(output_clahe_path,'train','benign')
        malignant_dir = os.path.join(output_clahe_path,'train','malignant')
    elif ind==1:
        benign_dir = os.path.join(output_clahe_path,'test','benign')
        malignant_dir = os.path.join(output_clahe_path,'test','malignant')
    
    if not os.path.exists(benign_dir):
        os.makedirs(benign_dir)
    if not os.path.exists(malignant_dir):
        os.makedirs(malignant_dir)
       
    idx=0 
    dirnames = df.index
    
    while idx < len(dirnames):
        print(f'Ind {ind}. Iteration: {idx} over {len(dirnames)}')
        dirname = dirnames[idx]
        #for dirname in tqdm(df.index):
        #E.g., dirname:= Mass-Training_P_00001_LEFT_MLO_1
        
        
        splits = dirname.split(sep='_')
        full_image_dir = '_'.join(splits[0:5])
        
        if 'MLO' in full_image_dir:
            #Select only the records of the specific patient selected
            # and count them to increment the while iterator
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
            
            # pil_im = Image.fromarray(npy_full_im)
            #CLAHE
            npy_full_clahe_im = apply_clahe(images=npy_full_im)
            
            #saving clahe image
            # npy_full_clahe_im = ((npy_full_clahe_im - np.amin(npy_full_clahe_im))/(np.amax(npy_full_clahe_im) - np.amin(npy_full_clahe_im)))*255
            # npy_full_clahe_im =npy_full_clahe_im_.astype(np.uint8)
            # pil_full_clahe_im = Image.fromarray(npy_full_clahe_im)      

            pil_full_clahe_im = Image.fromarray(npy_full_clahe_im)      
            pil_full_clahe_im.save(os.path.join(output_full_clahe_path,full_image_dir+'.png'),'PNG')
            
            for record in records_per_patient:
                w = os.walk(os.path.join(data_path,record))   
                
                label = df_patient.loc[record]['pathology']
        
        
                for root,dirs,files in w:
                    
                    if len(files)==0:
                        continue
                    
                    if len(files)==1:
                        #sei nel caso sbagliato, devi prendere il .dcm che ha per basename la parola croppedimage (sottocartella)
                        bn = os.path.basename(root)
                        if 'ROI mask images' in bn:
                            mask_path=os.path.join(root,files[0])
                        else:
                            cr_name = files[0]
                            continue
                            
                    elif len(files)==2:
                        #devi prendere quella che pesa più byte: è la maschera
                        file_size0 = os.path.getsize(os.path.join(root,files[0]))
                        file_size1 = os.path.getsize(os.path.join(root,files[1]))
                        if file_size0<file_size1:
                            mask_path=os.path.join(root,files[1])
                            # just to maintain the same names as before
                            cr_name = files[0]
                        else:
                            mask_path=os.path.join(root,files[0])
                            cr_name = files[1]
                        
        
                dcm_mask = read_file(mask_path)
                npy_mask = dcm_mask.pixel_array
                
                npy_mask = npy_mask/255.0 #TODO
                npy_mask = npy_mask.astype(np.uint8)
               
                
                assert(np.amax(npy_mask)==1)
                assert(np.amin(npy_mask)==0)
                # assert(list(np.unique(npy_mask)).sort() == [0,1])
        
                
                row_min, row_max, col_min, col_max = get_bbox(npy_mask)
                
                npy_cropped_clahe_im = npy_full_clahe_im[row_min:row_max+1, col_min:col_max+1]
                
                # Since the cropped image could not reach maximum pixel values of 255, normalize again:
                npy_cropped_clahe_im = ((npy_cropped_clahe_im - np.amin(npy_cropped_clahe_im))/(np.amax(npy_cropped_clahe_im) - np.amin(npy_cropped_clahe_im)))*255
                npy_cropped_clahe_im =npy_cropped_clahe_im.astype(np.uint8)
        
                pil_im = Image.fromarray(npy_cropped_clahe_im)
                
                cr_name = cr_name[:-4]
                
                if label=='BENIGN':
                    out_path = os.path.join(benign_dir, f'{record}_{cr_name}.png')
                elif label=='MALIGNANT':
                    out_path = os.path.join(malignant_dir, f'{record}_{cr_name}.png')
                
                pil_im.save(out_path,'PNG')
                
            idx += num_records_per_patient
            
        else:
            idx+=1

        