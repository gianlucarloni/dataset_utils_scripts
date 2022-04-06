#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import glob
import nibabel as nib
from tqdm import tqdm

#%% Create the plots for all the images for visual investigation

parse = argparse.ArgumentParser(description='Visualizzare i plot delle percentuali di elementi non-zero nelle slice')
parse.add_argument('images', help='Path to NiFti images directory')
parse.add_argument('masks', help='Path to NiFti masks directory')

args = parse.parse_args()

path_imgs = args.images
path_msks = args.masks


patientnames=list(filter(os.path.isdir, [os.path.join(path_imgs, f) for f in os.listdir(path_imgs)]))
patientnames = [os.path.basename(elem) for elem in patientnames]
#print(patientnames)

patientnames = ['024_S_0985','136_S_0086']
for patient in tqdm(patientnames):

    tree_images = os.walk(os.path.join(path_imgs,patient),topdown=False)
    images=[]
    for root,dirs,files in tree_images:
        if len(files)!=0:
            images.append(os.path.join(root,files[0]))
        
    tree_masks = os.walk(os.path.join(path_msks,patient),topdown=False)
    masks=[]
    for root,dirs,files in tree_masks:
        if len(files)!=0:
            masks.append(os.path.join(root,files[0]))

   
    zippata = zip(images,masks)

    for image, mask in zippata:
        
        name_image = os.path.basename(image)[:15]
        name_mask = os.path.basename(mask)[:15]
        
        if name_image==name_mask:
        
            image_nib = nib.load(image)
            mask_nib = nib.load(mask)
            
            image_npy = image_nib.get_fdata()
            mask_npy = mask_nib.get_fdata()
            
            mask_nonzero = np.nonzero(mask_npy)
            idx_min = np.amin(mask_nonzero[1])
            idx_max = np.amax(mask_nonzero[1])
            
            shape = image_npy.shape
            mask_new_npy = np.zeros(image_npy.shape)
            if patient != '136_S_0086':
                mask_new_npy[:, shape[1]-idx_max+10:shape[1]-idx_min+10, :] = 1
            else:
                mask_new_npy[:, shape[1]-idx_max-25:shape[1]-idx_min-25, :] = 1

            
            image_new_npy = image_npy*mask_new_npy
            image_new_nib = nib.Nifti1Image(image_new_npy, image_nib.affine, header=image_nib.header)
            image_new_nib.to_filename(image[:-7]+'_masked.nii.gz')

