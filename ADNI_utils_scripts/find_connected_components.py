#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:11:17 2022

@author: si-lab
"""
from skimage.measure import label
import os
import nibabel as nib
import argparse
import numpy as np

parse = argparse.ArgumentParser(description='Elimina i gruppi di pixel spuri dati dalla segmentazione automatica del cervello DeepBrain U-Net')
parse.add_argument('path_to_mask_dir', help='Path to NiFti mask directory')

args = parse.parse_args()

path_mask = args.path_to_mask_dir

mask_nib = nib.load(os.path.join(path_mask,'brain_mask.nii.gz'))
mask_npy = mask_nib.get_fdata()

all_conn = label(mask_npy,connectivity=3)
components = np.unique(all_conn)

m = 0
m_label=0
for c in components[components!=0]:
    n = np.sum(all_conn[all_conn==c])
    if n>m:
        m=n
        m_label=c

mask_npy_new = np.copy(mask_npy) 
mask_npy_new[all_conn!=m_label]=0

#%% RIMOZIONE  DELLE ISOLE DI PIXEL ZERI TRAMITE fill holes

from scipy.ndimage import binary_fill_holes

mask_npy_new_filled = np.zeros(mask_npy_new.shape)
for s in range(mask_npy_new.shape[1]):
    mask_s = mask_npy_new[:,s,:]
    mask_s_new = binary_fill_holes(mask_s)
    mask_npy_new_filled[:,s,:]=mask_s_new
    



mask_nib_new = nib.Nifti1Image(mask_npy_new_filled, mask_nib.affine, header=mask_nib.header)
mask_nib_new.to_filename(os.path.join(path_mask,'brain_mask_connected.nii.gz'))
