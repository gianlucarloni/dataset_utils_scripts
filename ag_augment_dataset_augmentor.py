#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 16:37:59 2022

@author: si-lab
"""
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import Augmentor

import pandas as pd
import argparse
import os
import shutil
import numpy as np
import glob
import PIL
import matplotlib.pyplot as plt
import random

parse = argparse.ArgumentParser(description="Final aim: to augment (36x) the original images with the Augmentor library")
parse.add_argument('input_dir', help='Path to the input directory with original PNG images of a specific class to augment')
parse.add_argument('output_dir', help='Path to the output directory for the corresponding augmented PNG images',type=str)

args = parse.parse_args()

input_dir = args.input_dir

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)




flip = transforms.RandomHorizontalFlip(p=1)
# os.mkdir(os.path.join('.','datasets','train_augmented'))

# for patient in tqdm(glob.glob(os.path.join(input_dir, '*'))):

# for imname in tqdm(glob.glob(os.path.join(input_dir,'*.png'))):
# for imname in glob.glob(os.path.join(patient,'*.png')):
    
    
# basename = os.path.basename(imname)

# versione uno: la esatta copia
# shutil.copy(imname,os.path.join(output_dir, basename))

print('Augm. n. 1: the exact copy')
for imname in glob.glob(os.path.join(input_dir,'*.png')):
    basename = os.path.basename(imname)
    shutil.copy(imname,os.path.join(output_dir, basename))
#TODO 1: la copia esatta
# shutil.copytree(input_dir,output_dir)

#TODO 2: rotation
print('Augm. n. 2: rotation')
p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
p.rotate(probability=1, max_left_rotation=10, max_right_rotation=10)
for i in range(10):
    p.process()
del p

#TODO 3: skew
print('Augm. n. 3: skew')
p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
p.skew(probability=1, magnitude=0.2)  # max 45 degrees
for i in range(10):
    p.process()
del p

#TODO 4:  shear
print('Augm. n. 4: shear')
p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
p.shear(probability=1, max_shear_left=10, max_shear_right=10)
for i in range(10):
    p.process()
del p

#TODO 5:  flip Left Right
print('Augm. n. 5: flip LR')
p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
p.flip_left_right(probability=1)
p.process()
del p 
    
#TODO 6:  flip Top Bottom
print('Augm. n. 6: flip TB')
p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
p.flip_top_bottom(probability=1)
p.process()
del p 

# #TODO 7:  hist equalization
# print('Augm. n. 7: hist equalization')
# p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
# p.histogram_equalisation(probability=1)
# p.process()
# del p 
    
# #TODO 8:  brightness
# print('Augm. n. 8: brightness')
# p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
# p.random_brightness(1, 0.8, 1.2)
# p.process()
# del p 

# #TODO 9:  contrast
# print('Augm. n. 9: contrast')
# p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
# p.random_contrast(1, 0.6, 1.5)
# p.process()
# del p 
    
    
    
    
    

# imm = PIL.Image.open(imname)
# a = imm.size[0]
    
# #versione due: la flippa
# imm_new = flip(imm)
# imm_new.save(os.path.join(output_dir,'aug_flip_'+basename),'PNG')

# #versione tre: la shifta
# shift = transforms.RandomAffine(degrees=0, translate=(10/a, 0)) #TODO modificare in base al dataset: ad esempio in ADNI fai traslazione solo lungo un asse (10/a, 0)
# imm_new2 = shift(imm)
# imm_new2.save(os.path.join(output_dir,'aug_shift_'+basename),'PNG')

# # versione quattro: la ruota
# rot = transforms.RandomRotation(10) #TODO
# imm_new3 = rot(imm)
# imm_new3.save(os.path.join(output_dir,'aug_rot_'+basename),'PNG')
    
print('Done.')