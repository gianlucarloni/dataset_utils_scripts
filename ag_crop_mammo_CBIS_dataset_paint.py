#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3 maggio 2022

@author: si-lab
"""

import os

import numpy as np
import argparse
from pydicom import read_file
from PIL import Image
import pandas as pd

from scipy.ndimage import zoom
import cv2
import matplotlib.pyplot as plt


def get_bbox(input_mask_npy):
    
    # troviamo gli indici di riga e colonna dove assume valori non nulli --> determiniamo la bounding box
    nonzero = input_mask_npy.nonzero()    
    row_min = np.amin(nonzero[0])
    row_max = np.amax(nonzero[0])
    col_min = np.amin(nonzero[1])
    col_max = np.amax(nonzero[1])
    
    return row_min, row_max, col_min, col_max

def correct_for_outofbounds(idx_min, delta, crop_size):
    idx_max = idx_min + crop_size
    
    idx_min_new = idx_min + delta
    idx_max_new = idx_max + delta
    
    return idx_min_new, idx_max_new


parse = argparse.ArgumentParser(description="")
parse.add_argument('data_path', help='path to breast_CBIS_DDSM dataset',type=str)
parse.add_argument('csv_mass_train_path', help='path to CSV file for training masses',type=str)
parse.add_argument('csv_mass_test_path', help='path to CSV file for test masses',type=str)
parse.add_argument('output_crop_path', help='path to parent folder of benign/malignant dirs, where cropped images are saved',type=str)

args = parse.parse_args()

data_path = args.data_path
csv_mass_train_path = args.csv_mass_train_path
csv_mass_test_path = args.csv_mass_test_path
output_crop_path = args.output_crop_path

   
path_to_names_txt = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM/dataset_ripulito_ribilanciato/push_e_valid_MLO/names.txt'
names_txt = pd.read_csv(path_to_names_txt, header=None)
names_list = list(names_txt[0])

# names_list = ['Mass-Training_P_00534_LEFT_MLO'] #TODO
names_list =['Mass-Training_P_01155_RIGHT_MLO']



for ind,csv_path in enumerate([csv_mass_train_path, csv_mass_test_path]):
    df = pd.read_csv(csv_path,sep=',',index_col='dirname')

    if ind==0:
        benign_dir = os.path.join(output_crop_path,'train','benign')
        malignant_dir = os.path.join(output_crop_path,'train','malignant')
    elif ind==1:
        benign_dir = os.path.join(output_crop_path,'test','benign')
        malignant_dir = os.path.join(output_crop_path,'test','malignant')
    
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
        
        if full_image_dir in names_list:
           
            #Select only the records of the specific patient selected
            # and count them to increment the while iterator
            records_per_patient=[]
            for dn in dirnames:
                if dn.startswith(full_image_dir):
                    records_per_patient.append(dn)
                    
            num_records_per_patient=len(records_per_patient)
            #
            # if num_records_per_patient>1: #Images with multiple lesions
            if num_records_per_patient>0:

                # with open(txt_path,'a') as fout:
                #     fout.write(f'{full_image_dir},{num_records_per_patient}\n')
            
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
                
                
                
                
                
                
                # plt.figure()
                # plt.imshow(npy_full_im, cmap='gray')
                # plt.show()
                
                
                # (click_col_center, click_row_center) = (1413,3290)
                # (click_col_bordo, click_row_bordo) = (1434,3290)
                (click_col_center, click_row_center) = (594,2105)
                (click_col_bordo, click_row_bordo) = (613,2105)
                
                #radius
                r = np.sqrt((click_col_center-click_col_bordo)**2+(click_row_center-click_row_bordo)**2)
                
                n_rows = npy_full_im.shape[0]
                n_cols = npy_full_im.shape[1]
                
                y,x = np.ogrid[0:n_rows, 0:n_cols]
                mask_circle = (x-click_col_center)**2 + (y-click_row_center)**2 <= r**2 
                
                mask_circle_outer = (x-click_col_center)**2 + (y-click_row_center)**2 <= (r+1)**2 
                mask_corona = mask_circle_outer ^ mask_circle
                tmp = np.zeros((n_rows, n_cols))
                tmp[mask_corona] = 1
                
                paint_value = np.mean(npy_full_im[tmp==1])
                # paint_value = 0 #TODO
                
                mask_to_paint = np.zeros((n_rows, n_cols))
                mask_to_paint[mask_circle] = 1
                
                
                npy_full_im[mask_to_paint == 1] = paint_value
            
                
                
                
                
                
                was_in_dataset = False
                #
                was_mask_corrupted = False
                #
                
                
                for record in records_per_patient:
                    
                    w = os.walk(os.path.join(data_path,record))   
                    
                    label = df_patient.loc[record]['pathology']
                    cr_name = ''
                    
                    crop_size = 600 #TODO
                    
                    if label == 'BENIGN' or label == 'MALIGNANT':
                        print(f'--------------Era una label={label}------------')
                        if label == 'BENIGN':
                             output_save_dir = benign_dir
                        else:
                             output_save_dir = malignant_dir
            
                    
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
                                
                        lesion_name_png = f'{record}_{cr_name[:-3]}png'
                        # if lesion_name_png in names_list: #TODO
                        if True:

                                                
                            dcm_mask = read_file(mask_path)
                            npy_mask = dcm_mask.pixel_array
                            # check if the mask and the image have the same dimensions
                            shape_i = npy_full_im.shape
                            shape_m = npy_mask.shape
                            if shape_i!=shape_m:
                                was_mask_corrupted=True
   
                                zoom_factor_rows = shape_i[0]/shape_m[0]
                                zoom_factor_cols = shape_i[1]/shape_m[1]
                                zoom_factor = (zoom_factor_rows, zoom_factor_cols)
                                npy_mask = zoom(npy_mask,zoom=zoom_factor)
                                _, npy_mask = cv2.threshold(npy_mask,thresh=np.quantile(npy_mask,0.75),maxval=255.0, type=0)
                                
                            npy_mask = npy_mask/255.0 #TODO
                            npy_mask = npy_mask.astype(np.uint8)
                           
                            
                            assert(np.amax(npy_mask)==1)
                            assert(np.amin(npy_mask)==0)
                            # assert(list(np.unique(npy_mask)).sort() == [0,1])
                    
                            
                            row_min, row_max, col_min, col_max = get_bbox(npy_mask)
                            
                                                        
                            # now take the 600x600 bbox:
                            row_center = (row_max + row_min)/2
                            col_center = (col_max + col_min)/2                           
                            
                            
                            if (row_max-row_min)>crop_size or (col_max-col_min)>crop_size:
                                crop_size = max(row_max-row_min,col_max-col_min)
                                
                            
                            row_min_600 = int(np.floor(row_center-crop_size/2))
                            row_max_600 = int(np.floor(row_center+crop_size/2))
                            col_min_600 = int(np.floor(col_center-crop_size/2))
                            col_max_600 = int(np.floor(col_center+crop_size/2))
                            
                            if row_min_600 < 0:
                                delta = 0-row_min_600
                                row_min_600, row_max_600 = correct_for_outofbounds(row_min_600, delta, crop_size)
                                                        
                            if row_max_600 > npy_full_im.shape[0] - 1: #H
                                delta = npy_full_im.shape[0] - 1 - row_max_600
                                row_min_600, row_max_600 = correct_for_outofbounds(row_min_600, delta, crop_size)

                            if col_min_600 < 0:
                                delta = 0-col_min_600
                                col_min_600, col_max_600 = correct_for_outofbounds(col_min_600, delta, crop_size)
                           
                            if col_max_600 > npy_full_im.shape[1] - 1: #W
                                delta = npy_full_im.shape[1] - 1 - col_max_600
                                col_min_600, col_max_600 = correct_for_outofbounds(col_min_600, delta, crop_size)

                            output_cropped = npy_full_im[row_min_600:row_max_600, col_min_600:col_max_600]
                            output_cropped = ((output_cropped - np.amin(output_cropped))/(np.amax(output_cropped) - np.amin(output_cropped)))*255
                            output_cropped=output_cropped.astype(np.uint8)
                            output_cropped_pil = Image.fromarray(output_cropped)
                            output_cropped_pil.save(os.path.join(output_save_dir,lesion_name_png),format='PNG')
                            print(f'Saved {lesion_name_png}')
                                              
            idx += num_records_per_patient
            
        else:
            idx+=1

        