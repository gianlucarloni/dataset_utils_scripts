#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:51:43 2022

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
import imgaug.augmenters as iaa

apply_clahe = iaa.AllChannelsCLAHE()


parse = argparse.ArgumentParser(description="Plot")

parse.add_argument('dicom_image_path', help='Path to .dcm image to compute CLAHE on',type=str)
parse.add_argument('out_dir_path', help='Path to dir where we save output figure',type=str)

args = parse.parse_args()

dicom_image_path = args.dicom_image_path
out_name = dicom_image_path.split(os.sep)[-4]

out_dir_path = args.out_dir_path
dcm_image = read_file(dicom_image_path)
npy_full_im = dcm_image.pixel_array

npy_full_im_uint8 = ((npy_full_im - np.amin(npy_full_im))/(np.amax(npy_full_im) - np.amin(npy_full_im)))*255
npy_full_im_uint8=npy_full_im_uint8.astype(np.uint8)
pil_full_im = Image.fromarray(npy_full_im_uint8)
pil_full_im.save(os.path.join(out_dir_path,f'{out_name}_full_image_original.png'),'PNG')


npy_full_clahe_im_uint8 = apply_clahe(images=npy_full_im_uint8)
pil_full_clahe_im = Image.fromarray(npy_full_clahe_im_uint8)
pil_full_clahe_im.save(os.path.join(out_dir_path,f'{out_name}_full_image_clahe.png'),'PNG')


diff = npy_full_clahe_im_uint8 - npy_full_im_uint8
pil_diff = Image.fromarray(diff)
pil_diff.save(os.path.join(out_dir_path,f'{out_name}_diff.png'),'PNG')






plt.figure()
plt.subplot(131)
plt.imshow(npy_full_clahe_im_uint8,cmap='gray')
plt.title('CLAHE')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,     
    right=False,     
    left=False,     
    labelbottom=False,
    labelleft=False)
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,     
    right=False,     
    left=False,     
    labelbottom=False,
    labelleft=False)

plt.subplot(132)
plt.imshow(npy_full_im_uint8,cmap='gray')
plt.title('Original')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,  
    right=False,     
    left=False,        # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) 
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,     
    right=False,     
    left=False,     
    labelbottom=False,
    labelleft=False)

plt.subplot(133)
plt.imshow(diff,cmap='jet')
plt.title('CLAHE - Original')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False, 
    right=False,     
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False)
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,     
    right=False,     
    left=False,     
    labelbottom=False,
    labelleft=False) 

plt.colorbar(orientation='horizontal', fraction=0.3)
plt.savefig(os.path.join(out_dir_path,f'{out_name}_clahe_subplot.pdf'),bbox_inches='tight')



plt.figure()
plt.imshow(diff,cmap='jet')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False, 
    right=False,     
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False)
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,     
    right=False,     
    left=False,     
    labelbottom=False,
    labelleft=False) 
plt.colorbar(orientation='vertical', shrink=1.0)
plt.savefig(os.path.join(out_dir_path,f'{out_name}_diff_alone.pdf'),bbox_inches='tight')