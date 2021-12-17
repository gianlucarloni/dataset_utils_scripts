#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 14:49:06 2021

@author: si-lab
Creare le versione originale (non corrotta) png delle slice dicom di prostata, 
della dimensione desiderata

Ricordarsi di modificare a mano IMG_SIZE
TODO: importare in altro modo IMG_SIZE
"""

import pydicom
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pydicom.filereader import read_dicomdir
import scipy.ndimage
from PIL import Image


IMG_SIZE = 384


def crop_center(img, cropx, cropy):
    x, y = img.shape
    startx = x //2-(cropx//2)
    starty = y //2-(cropy//2)    
    return img[startx: startx + cropx, starty: starty + cropy]

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Creating a PNG dataset from DICOM directory")
    parse.add_argument('png_dir', help='Path to the output directory')
    parse.add_argument('-d', '--dicom_dir', help='Directory of the original DICOM files',
                       default='/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/prostate/prostatex2/DICOM T2')
    parse.add_argument('-c', '--cropped_dim', help='Dimension to which crop the image',type=int)
    
    args = parse.parse_args()
    png_dir = args.png_dir
    dicom_dir = args.dicom_dir
    cropped_dim = args.cropped_dim
    print('Assumption: square medical images')
    print(f'La dimensione dell\'immagine iniziale è settata a: {IMG_SIZE}')
    
    if cropped_dim is not None:
        print(f'La dimensione dell\'immagine finale sarà: {cropped_dim}')
    
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
 
    
    for slice_dcm in glob.glob(os.path.join(dicom_dir, '*.dcm')):
        name = os.path.basename(slice_dcm)
        name = name[:-4] #per togliere il suffisso .dcm
        slice_dcm = pydicom.read_file(slice_dcm)
        
        slice_npy = slice_dcm.pixel_array
        
        if slice_npy.shape[0] != IMG_SIZE:
            zoom_value = IMG_SIZE / slice_npy.shape[0]
            slice_npy = scipy.ndimage.zoom(slice_npy, zoom=zoom_value)
        if cropped_dim is not None:
            slice_npy = crop_center(slice_npy, cropped_dim, cropped_dim)
        
        slice_npy = ((slice_npy - np.amin(slice_npy))/np.amax(slice_npy))*255
        a = slice_npy.astype(np.uint8) #range:0-255
        b = Image.fromarray(a)
        
        b.save(os.path.join(png_dir, name+'.png'), 'PNG')
        
print('Done.')
            

