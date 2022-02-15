#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 16:37:59 2022

@author: si-lab
"""
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pandas as pd
import argparse
import os
import shutil
import numpy as np
import glob
import PIL
import matplotlib.pyplot as plt
import random

parse = argparse.ArgumentParser(description="Final aim: to augment original images with HORIZONTAL FLIP and PIXEL SHIFT")
parse.add_argument('input_dir', help='Path to the input directory with original PNG images to augment')
parse.add_argument('output_dir', help='Path to the output directory for augmented PNG images',type=str)

args = parse.parse_args()

input_dir = args.input_dir

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)




flip = transforms.RandomHorizontalFlip(p=1)
# os.mkdir(os.path.join('.','datasets','train_augmented'))
from tqdm import tqdm 
for imname in tqdm(glob.glob(os.path.join(input_dir,'*.png'))):
    basename = os.path.basename(imname)
    
    # versione uno: la esatta copia
    shutil.copy(imname,os.path.join(output_dir, basename))

    imm = PIL.Image.open(imname)
    a = imm.size[0]
    b = imm.size[1]

    #versione due: la flippa
    imm_new = flip(imm)
    imm_new.save(os.path.join(output_dir,'aug_flip_'+basename),'PNG')
    
    #versione tre: la shifta
    shift = transforms.RandomAffine(degrees=0, translate=(20/a, 20/b))
    imm_new2 = shift(imm)
    imm_new2.save(os.path.join(output_dir,'aug_shift_'+basename),'PNG')
    
    # versione quattro: la ruota
    rot = transforms.RandomRotation(5)
    imm_new3 = rot(imm)
    imm_new3.save(os.path.join(output_dir,'aug_rot_'+basename),'PNG')

print('Done.')