#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 14:49:06 2021

@author: si-lab

SCRIPT PER AUMENTARE LE IMMAGINI DI CLASSE MENO POPOLOSA AFFINCHE' SI RAGGIUNGA 50%-50%

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
from torchvision import transforms
import random

IMG_SIZE = 2294

random.seed(10)

# def crop_center(img, cropx, cropy):
#     x, y = img.shape
#     startx = x //2-(cropx//2)
#     starty = y //2-(cropy//2)    
#     return img[startx: startx + cropx, starty: starty + cropy]



if __name__ == "__main__":
    
    flip = transforms.RandomHorizontalFlip(p=1)
   
    parse = argparse.ArgumentParser(description="Balancing a PNG dataset with data augmentation offline")
    parse.add_argument('png_dir', help='Path to the output directory')
    parse.add_argument('csv_path', help='Path to CSV file of original CC/MLO png images')
    args = parse.parse_args()
    png_dir = args.png_dir
    path_to_csv = args.csv_path
    # load CSV spreadsheet for labels
    csv_df = pandas.read_csv(path_to_csv,index_col='File name')
    csv_df_index = list(csv_df.index)
    
    benign = [elem for elem in csv_df_index if csv_df.loc[elem]['Label']=='Benign']
    file_name_to_pick = random.sample(benign, 203)
    
    copy_index = csv_df_index.copy()
    copy_label = list(csv_df['Label']).copy()
    
    for file_name in benign:
    
        imm = Image.open(os.path.join(png_dir,file_name))
        a = imm.size[0]
        b = imm.size[1]
        shift = transforms.RandomAffine(degrees=0, translate=(25/a, 25/b))

        imm_new = flip(imm)
        #imm_new = shift(imm_new)

        new_name = 'aug_'+file_name
        imm_new.save(os.path.join(png_dir,new_name),'PNG')
        
        copy_index.append(new_name)
        copy_label.append('Benign')
        
        # solo per alcune salvo anche la imm3
        if file_name in file_name_to_pick:
            
            angle_range = (0, 5)
            if csv_df.loc[file_name]['LeftRight'] == 'R':
                angle_range = (-5, 0)
            
            rot = transforms.RandomRotation(angle_range, center=(0,0))
            imm_new3 = rot(imm)
            imm_new3.save(os.path.join(png_dir,'aug2_'+file_name),'PNG')
            copy_index.append('aug2_'+file_name)
            copy_label.append('Benign')
            

    out_df = pandas.DataFrame(data={
        'File name':copy_index,
        'Label':copy_label
        })
    out_df.to_csv(os.path.join(png_dir,'label_balanced.csv'),index=False)         
            
            
       
                
print('Done.')
            

