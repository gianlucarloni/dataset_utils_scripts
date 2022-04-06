#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:23:17 2022
CARICARE LE IMMAGINI ADNI NIFTI E APPLICARCI LE MASCHERE BINARIE PER OTTENERE IL CERVELLO SENZA CRANIO
@author: si-lab

--- SKULL STRIPPING ---

"""


import os
import numpy as np
import argparse

parse = argparse.ArgumentParser(description='CARICARE LE IMMAGINI ADNI NIFTI E APPLICARCI LE MASCHERE BINARIE PER OTTENERE IL CERVELLO')
parse.add_argument('images', help='Path to NiFti images directory')
parse.add_argument('masks', help='Path to NiFti masks directory')
parse.add_argument('output', help='Path to NUMPY masked images directory')

args = parse.parse_args()

ima_path = args.images
mas_path = args.masks
out_path = args.output

from glob import glob
patientnames=list(filter(os.path.isdir, [os.path.join(ima_path, f) for f in os.listdir(ima_path)]))
patientnames = [os.path.basename(elem) for elem in patientnames]

# patientnames = ['011_S_0183', '109_S_1157', '128_S_0805', '094_S_1164', '099_S_1144']
#patientnames = ['011_S_0008', '128_S_0863', '011_S_0016', '016_S_0538', '099_S_0040']
#patientnames = ['094_S_1102']
patientnames = ['128_S_0863']

#print(patientnames)

import nibabel as nib
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.ndimage import affine_transform
from nibabel.processing import resample_to_output
for patient in tqdm(patientnames):
# for patient in patientnames:

    tree_images = os.walk(os.path.join(ima_path,patient),topdown=False)
    images=[]
    for root,dirs,files in tree_images:
        l = len(files)
        if l!=0:
            if l==2:
                for file in files:
                    if file.endswith('masked.nii.gz'):
                        images.append(os.path.join(root,file))
                        break
                        
            elif l==1:
                       images.append(os.path.join(root,files[0])) 
        
    tree_masks = os.walk(os.path.join(mas_path,patient),topdown=False)
    masks=[]
    for root,dirs,files in tree_masks:
        if len(files)!=0:
            masks.append(os.path.join(root,files[0]))

        
        
    zippata = zip(images,masks)
    

    # immagini
    for image, mask in zippata:
        
        # img = nib.load(image)       
        
        # img_array = img.get_fdata()
        
        # # h = img.header
        # # print(list(h['pixdim'][1:4]))
        
        # mas = nib.load(mask)
        # mas_array = mas.get_fdata()
        
        # oTm = mas.affine # from global to mask coordinates 
        # oTi = img.affine # from global to image coordinates
        
        # iTo = np.linalg.inv(oTi) #inverse: from image coordinates to global coordinates
        
        # iTm = np.dot(iTo,oTm) # from Image to Mask coordinates
        
        # output = affine_transform(img_array, iTm,output_shape=mas_array.shape)
  
        # output_masked = output*mas_array
        
        
        
        
        
        
        
        
        # img = nib.load(image)       
        # img_resampled = resample_to_output(img,voxel_sizes=(1.25,1.25,1.20))
        # img_array = img_resampled.get_fdata()
                
        # mas = nib.load(mask)
        # mas_resampled = resample_to_output(mas,voxel_sizes=(1.25,1.25,1.20))
        # mas_array = mas_resampled.get_fdata()
        
        # oTm = mas_resampled.affine # from global to mask coordinates 
        # oTi = img_resampled.affine # from global to image coordinates
        
        # iTo = np.linalg.inv(oTi) #inverse: from image coordinates to global coordinates
        
        # iTm = np.dot(iTo,oTm) # from Image to Mask coordinates
        
        # output = affine_transform(img_array, iTm,output_shape=mas_array.shape)
  
        # output_masked = output*mas_array
        
        
        
                
        img = nib.load(image)       
        img_array = img.get_fdata()
            
        mas = nib.load(mask)
        
        #mas_resampled = resample_to_output(mas,voxel_sizes=(1.25,1.25,1.20))
        mas_array = mas.get_fdata()
        
        

        oTm = mas.affine # from global to mask coordinates 
        oTi = img.affine # from global to image coordinates
        
        mTo = np.linalg.inv(oTm) #inverse
        
        mTi = np.dot(mTo,oTi) # from Mask to Image coordinates
        
        mask_new_array = affine_transform(mas_array, mTi,output_shape=img_array.shape)
  
        output_masked = mask_new_array*img_array
        
        output_masked_nib = nib.Nifti1Image(output_masked, oTi, header=img.header)
        #output_masked_nib.to_filename(os.path.join(out_path,'not_resampled',os.path.basename(image)[:-3]))

        img_resampled = resample_to_output(output_masked_nib,voxel_sizes=(1.25,1.25,1.20))
        #h = img_resampled.header
        
        img_resampled_array = img_resampled.get_fdata()
        if np.amin(img_resampled_array) < 0:
            print(f'Negative value detected in image {os.path.basename(image)}!')
            img_resampled_array[img_resampled_array<0] = 0  
        
        
        # BBOX DETECTION: CORONAL VIEW
        coronal_sum = np.sum(img_resampled_array, axis = 1) #coronale 1
        coronal_sum = np.abs(coronal_sum)
        coronal_sum = 255*(coronal_sum-np.amin(coronal_sum))/(np.amax(coronal_sum)-np.amin(coronal_sum))
        coronal_sum = coronal_sum.astype(np.uint8)        
        # troviamo gli indici di riga e colonna dove assume valori non nulli --> determiniamo la bounding box
        nonzero = coronal_sum.nonzero()
        row_min = np.amin(nonzero[0])
        row_max = np.amax(nonzero[0])
        col_min = np.amin(nonzero[1])
        col_max = np.amax(nonzero[1])
        # square-ification of row-col bbox
        width = col_max - col_min
        height = row_max - row_min
        
        delta = height - width
        # if delta >= 0: #più alta che larga, va allargata
        #     col_max += np.ceil(delta/2)
        #     col_min -= np.floor(delta/2)  
        # else:
        #     row_max += np.ceil(delta/2)
        #     row_min -= np.floor(delta/2) 
        # print(f'delta = {delta}')
        # print(f'delta/2: floor = {np.floor(delta/2)}, int_floor = {int(np.floor(delta/2))}')
        # print(f'delta/2: ceil = {np.ceil(delta/2)}, int_ceil = {int(np.ceil(delta/2))}')
        # print(f'row_min = {row_min}')
        # print(f'min row, delta/2 = {min(row_min, int(np.floor(-delta/2)))}')
        if delta >= 0: #più alta che larga, va allargata
            m = min(col_min, int(np.floor(delta/2)))
            col_min -= m
            col_max += (delta - m)
        else:
            delta = -delta
            m = min(row_min, int(np.floor(delta/2)))
            row_min -= m
            row_max += (delta - m)
        # l'immagine non è centrata (non so se significativamente o no), se la vogliamo fare centrata dobbiamo fare padding o altro
            
        # BBOX DETECTION: lateral VIEW
        saggital_sum = np.sum(img_resampled_array, axis = 0) #saggitale 0
        saggital_sum = np.abs(saggital_sum)
        saggital_sum = 255*(saggital_sum-np.amin(saggital_sum))/(np.amax(saggital_sum)-np.amin(saggital_sum))
        saggital_sum = saggital_sum.astype(np.uint8)        
        # troviamo gli indici di riga e colonna dove assume valori non nulli --> determiniamo la bounding box
        nonzero = saggital_sum.nonzero()
        depth_min = np.amin(nonzero[0])
        depth_max = np.amax(nonzero[0])
        # col_min = np.amin(nonzero[1])
        # col_max = np.amax(nonzero[1])
        

        
        cropped_volume = img_resampled_array[row_min:row_max+1, depth_min:depth_max+1, col_min:col_max+1]
    
        # scaling to 0-255 range (uint8) before saving numpy
        cropped_volume = 255*(cropped_volume-np.amin(cropped_volume))/(np.amax(cropped_volume)-np.amin(cropped_volume))
        
        cropped_volume = cropped_volume.astype(np.uint8)        
               
        np.save(os.path.join(out_path,os.path.basename(image)[:-7]),cropped_volume) #NPY
        
        # cropped_volume_nib = nib.Nifti1Image(cropped_volume, img_resampled.affine, header=img_resampled.header)

        
        # cropped_volume_nib.to_filename(os.path.join(out_path,os.path.basename(image)+'.gz')) # NIFTI
        # print('Saved')
print('Done!')
        

