#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:56:15 2022

@author: si-lab
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from pydicom import read_file
from PIL import Image
import pandas as pd
import imgaug.augmenters as iaa

from scipy.ndimage import zoom
import cv2

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



#
# path_to_multiple_masses_images_dir = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM/breast_images_with_multiple_masses'
# print_txt_path = os.path.join(path_to_multiple_masses_images_dir,'print.txt')

path_to_single_masses_images_dir = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM/breast_images_with_single_masses'
print_txt_path = os.path.join(path_to_single_masses_images_dir,'print.txt')
print_txt_dim_mismatch_path = os.path.join(path_to_single_masses_images_dir,'dim_mismatch.txt')

if not os.path.exists(print_txt_path):
    with open(print_txt_path,'w') as fout:
        fout.write('Print out of sanity checks for indices:\n')
if not os.path.exists(print_txt_dim_mismatch_path):
    with open(print_txt_dim_mismatch_path,'w') as fout:
        fout.write('Print out of mismatch between image shape and mask shape:\n')
        
path_to_single_masses_images_dir_was_in_dataset = os.path.join(path_to_single_masses_images_dir,'was_in_dataset')
path_to_single_masses_images_dir_wasnt_in_dataset = os.path.join(path_to_single_masses_images_dir,'wasnt_in_dataset')
if not os.path.exists(path_to_single_masses_images_dir_was_in_dataset):
    os.makedirs(path_to_single_masses_images_dir_was_in_dataset)
if not os.path.exists(path_to_single_masses_images_dir_wasnt_in_dataset):
    os.makedirs(path_to_single_masses_images_dir_wasnt_in_dataset)
#

path_to_names_txt = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM/dataset_ripulito_ribilanciato/push_e_valid_MLO/names.txt'
names_txt = pd.read_csv(path_to_names_txt, header=None)
names_list = list(names_txt[0])

#Folder to save CLAHE images
output_clahe_path = os.path.join(cwd_path,'dataset_png_clahe')
if not os.path.exists(output_clahe_path):
    os.makedirs(output_clahe_path)

output_full_clahe_path = os.path.join(output_clahe_path,'full_images')
if not os.path.exists(output_full_clahe_path):
    os.makedirs(output_full_clahe_path)

txt_path = os.path.join(cwd_path,'number_of_lesions_per_breast.txt')    
if not os.path.exists(txt_path):
    with open(txt_path,'w') as fout:
        fout.write('breast_id,number_of_masses\n')

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
                npy_full_im = ((npy_full_im - np.amin(npy_full_im))/(np.amax(npy_full_im) - np.amin(npy_full_im)))*255
                npy_full_im=npy_full_im.astype(np.uint8)
                

                plt.figure()
                plt.imshow(npy_full_im,cmap='gray')
                
                was_in_dataset = False
                #
                was_mask_corrupted = False
                #
                output_save_dir = path_to_single_masses_images_dir_wasnt_in_dataset
                
                for record in records_per_patient:
                    w = os.walk(os.path.join(data_path,record))   
                    
                    label = df_patient.loc[record]['pathology']
                    cr_name = ''
                    if label == 'BENIGN' or label == 'MALIGNANT':
            
                    
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
                        if lesion_name_png in names_list:
                            was_in_dataset=True
                            bbox_color = 'g'
                            output_save_dir = path_to_single_masses_images_dir_was_in_dataset
                        else:
                            bbox_color = 'r'

                        
                        dcm_mask = read_file(mask_path)
                        npy_mask = dcm_mask.pixel_array
                        # check if the mask and the image have the same dimensions
                        shape_i = npy_full_im.shape
                        shape_m = npy_mask.shape
                        if shape_i!=shape_m:
                            was_mask_corrupted=True
                            # with open(print_txt_dim_mismatch_path,'a') as fout:
                            #     fout.write(f'Record {record} of Image {full_image_dir} was_in_dataset={was_in_dataset}: image_shape={shape_i}, mask_shape={shape_m}\n')
                                
                            zoom_factor_rows = shape_i[0]/shape_m[0]
                            zoom_factor_cols = shape_i[1]/shape_m[1]
                            zoom_factor = (zoom_factor_rows, zoom_factor_cols)
                            print(f'{shape_i} {shape_m}')
                            npy_mask = zoom(npy_mask,zoom=zoom_factor)
                            print(npy_mask.shape)
                            _, npy_mask = cv2.threshold(npy_mask,thresh=np.quantile(npy_mask,0.75),maxval=255.0, type=0)
                            
                     #TODO deindentare le righe sotto       
                            npy_mask = npy_mask/255.0 #TODO
                            npy_mask = npy_mask.astype(np.uint8)
                           
                            
                            assert(np.amax(npy_mask)==1)
                            assert(np.amin(npy_mask)==0)
                            # assert(list(np.unique(npy_mask)).sort() == [0,1])
                    
                            
                            row_min, row_max, col_min, col_max = get_bbox(npy_mask)
                            
                            # minimal bbox:
                            plt.hlines([row_min, row_max],col_min-1,col_max+1,colors='y',linewidths=0.5)
                            plt.vlines([col_min, col_max],row_min,row_max,colors='y',linewidths=0.5)
                            
                            # now take the 600x600 bbox:
                            row_center = (row_max + row_min)/2
                            col_center = (col_max + col_min)/2
                            
                            row_min_600 = row_center-300
                            if row_min_600 < 0:
                                with open(print_txt_path,'a') as fout:
                                    fout.write(f'Record {record} of Image {full_image_dir} is out of bounds: row_min_600={row_min_600}\n')
                            
                            row_max_600 = row_center+300
                            if row_max_600 > npy_full_im.shape[0]: #H
                                with open(print_txt_path,'a') as fout:
                                    fout.write(f'Record {record} of Image {full_image_dir} is out of bounds: row_max_600={row_max_600-npy_full_im.shape[0]}\n')
                                
                            col_min_600 = col_center-300
                            if col_min_600 < 0:
                                with open(print_txt_path,'a') as fout:
                                    fout.write(f'Record {record} of Image {full_image_dir} is out of bounds: col_min_600={col_min_600}\n')
                                
                            col_max_600 = col_center+300
                            if col_max_600 > npy_full_im.shape[1]: #W
                                with open(print_txt_path,'a') as fout:
                                    fout.write(f'Record {record} of Image {full_image_dir} is out of bounds: col_max_600={col_max_600-npy_full_im.shape[1]}\n')
                            
                            plt.hlines([row_min_600, row_max_600],col_min_600-1,col_max_600+1,colors=bbox_color,linewidths=0.5)
                            plt.vlines([col_min_600, col_max_600],row_min_600,row_max_600,colors=bbox_color,linewidths=0.5)
                            
                  
                                              
                    # plt.savefig(os.path.join(path_to_multiple_masses_images_dir,full_image_dir+'.pdf'),bbox_inches='tight')
                    if was_mask_corrupted: #TODO if da togliere poi
                        plt.savefig(os.path.join(output_save_dir,'new_'+full_image_dir+'.pdf'),bbox_inches='tight') #TODO new
                    plt.close()
            idx += num_records_per_patient
            
        else:
            idx+=1

        