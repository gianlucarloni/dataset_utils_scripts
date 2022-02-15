#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 12:50:06 2022
per croppare le immagini del dataset ornitologia cub200

@author: si-lab
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import shutil

input_file_bb = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/birds_CUB200/CUB_200_2011/bounding_boxes.txt'
input_file_names = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/birds_CUB200/CUB_200_2011/images.txt'
input_file_tnt = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/birds_CUB200/CUB_200_2011/train_test_split.txt'

df_bb = pd.read_csv(input_file_bb, sep=" ", names=['id', 'x', 'y', 'w', 'h'])
df_names = pd.read_csv(input_file_names, sep=" ", names=['id', 'name'])
df_tnt = pd.read_csv(input_file_tnt, sep= " ",names=["idx","group"])

path_to_parent = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/birds_CUB200/CUB_200_2011/images'
new_path='/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/birds_CUB200/CUB_200_2011/images_cropped'


list_index = list(df_bb.index)
for idx in list_index:
    if idx==3190:
        break
    else:
        imname = df_names.iloc[idx]['name']
        
        im = Image.open(path_to_parent+'/'+imname)
        
        
        y = df_bb.iloc[idx]['y']
        x = df_bb.iloc[idx]['x']
        h = df_bb.iloc[idx]['h']
        w = df_bb.iloc[idx]['w']
    
        im_crop = transforms.functional.crop(im, y, x, h, w)
        
        if not os.path.exists(new_path+'/'+os.path.dirname(imname)):
            os.makedirs(new_path+'/'+os.path.dirname(imname))
            
            
        im_crop.save(new_path+'/'+imname)
    
for idx in list_index:
    if idx==3190:
        print('esco')
        break
    v = df_tnt.iloc[idx]['group']
    imname = df_names.iloc[idx]['name']
    group = 'train' if v==1 else 'valid'
    
    out_path = os.path.dirname(new_path)+'/'+group+'/'+imname
    
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
        
    shutil.copy(new_path+'/'+imname, out_path)