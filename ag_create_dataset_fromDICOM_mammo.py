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
import pandas

IMG_SIZE = 2294



# load CSV spreadsheet for labels
path_to_csv='/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CMMD/CMMD_clinicaldata_revision_ag.csv'
csv = pandas.read_csv(path_to_csv,index_col='ID1')

patients_names = list(set(csv.index))
patients_names.sort()


filename_CC = list()
filename_MLO = list()
label_CC = list()
label_MLO = list()

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
     
    if not os.path.exists(os.path.join(png_dir,'MLO')):
        os.makedirs(os.path.join(png_dir,'MLO'))
        
    if not os.path.exists(os.path.join(png_dir,'CC')):
        os.makedirs(os.path.join(png_dir,'CC'))
    
  
    for patient_name in patients_names:
        df = csv.loc[patient_name]
        
        is_one_row = False
        
        laterality = ''
        if df.shape!=(2,2):
            laterality = df['LeftRight'] #str
            is_one_row = True
            
            
        for slice_dcm in glob.glob(os.path.join(dicom_dir,patient_name+ '*.dcm')):
            name = os.path.basename(slice_dcm)
            name = name[:-4] #per togliere il suffisso .dcm
            slice_dcm = pydicom.read_file(slice_dcm)
            #lettura dei campi dicom: metadata
            temp = slice_dcm.PatientOrientation[1]
            temp1 = slice_dcm.ImageLaterality

            if laterality != '':
                if temp1 != laterality:
                    continue
            
            if not is_one_row:
                class_label = df[df['LeftRight']==temp1]
                class_label = list(class_label['classification'])
                class_label = class_label[0]
            else:
                class_label = df['classification']
            
            patient_orientation = 'CC'
            if temp.startswith('F'):
                patient_orientation = 'MLO'
            
            
            slice_npy = slice_dcm.pixel_array
            
            if slice_npy.shape[0] != IMG_SIZE:
                zoom_value = IMG_SIZE / slice_npy.shape[0]
                slice_npy = scipy.ndimage.zoom(slice_npy, zoom=zoom_value)
            if cropped_dim is not None:
                slice_npy = crop_center(slice_npy, cropped_dim, cropped_dim)
            
            slice_npy = ((slice_npy - np.amin(slice_npy))/np.amax(slice_npy))*255
            a = slice_npy.astype(np.uint8) #range:0-255
            b = Image.fromarray(a)
            
            final_name = name+'_'+patient_orientation+'.png'
            if patient_orientation == 'MLO':
                filename_MLO.append(final_name)
                label_MLO.append(class_label)
            else:
                filename_CC.append(final_name)
                label_CC.append(class_label)
            b.save(os.path.join(png_dir,patient_orientation, final_name), 'PNG')
    
    out_CC_df = pandas.DataFrame(data={'File name': filename_CC, 'Label': label_CC})
    out_MLO_df = pandas.DataFrame(data={'File name': filename_MLO, 'Label': label_MLO})
    out_CC_df.to_csv(os.path.join(png_dir,'CC', 'labels.csv'), index=False)
    out_MLO_df.to_csv(os.path.join(png_dir,'MLO', 'labels.csv'), index=False)
                
print('Done.')
            

