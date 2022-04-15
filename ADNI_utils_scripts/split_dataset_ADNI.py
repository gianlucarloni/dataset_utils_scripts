#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:08:58 2022

 -----PRIMA PARTE: ADNI & SECONDA PARTE: CBIS MAMMO----
 
 -----SPLIT push (64%)-valid (16%)-test (20%)--------
 
 La stratificazione per paziente viene quindi ottenuta per costruzione,
     poichè si splittano I nomi delle cartelle (pazienti).
 La stratificazione per label viene ottenuta per costruzione perchè,
     eseguiamo lo split separatamente su due coorti intrinsecamente al 50-50%.

@author: si-lab
"""
from glob import glob
import os
import argparse
from sklearn.model_selection import train_test_split
from distutils.dir_util import copy_tree
import shutil

parse = argparse.ArgumentParser(description='SPLIT push (64%)-valid (16%)-test (20%)')
parse.add_argument('input_png_images', help='Path to png images directory')
parse.add_argument('output_png_images', help='Path to png images directory for ppnet')


args = parse.parse_args()

path_inp = args.input_png_images
path_out = args.output_png_images

diagnostic_class = os.path.basename(path_inp)

# #%% ADNI 

# path_out_push = os.path.join(path_out,'push',diagnostic_class)
# if not os.path.exists(path_out_push):
#     os.makedirs(path_out_push)
    
# path_out_valid = os.path.join(path_out,'valid',diagnostic_class)
# if not os.path.exists(path_out_valid):
#     os.makedirs(path_out_valid)
    
# path_out_test = os.path.join(path_out,'test',diagnostic_class)
# if not os.path.exists(path_out_test):
#     os.makedirs(path_out_test)

# patient_names = list(set(glob(os.path.join(path_inp,'*'))) - set(glob(os.path.join(path_inp,'*.csv'))))
    
# #TOTAL-->PUSH temp & TEST
# names_pushtemp, names_test = train_test_split(patient_names,test_size=0.20,train_size=0.80,random_state=42,shuffle=True)
# #PUSH temp-->PUSH & VALID
# names_push, names_valid = train_test_split(names_pushtemp,test_size=0.20,train_size=0.80,random_state=42,shuffle=True)


# for name in names_push:
#     copy_tree(name,os.path.join(path_out_push,os.path.basename(name)))
    
# for name in names_valid:
#     copy_tree(name,os.path.join(path_out_valid,os.path.basename(name)))

# for name in names_test:
#     copy_tree(name,os.path.join(path_out_test,os.path.basename(name)))
    
# print('done')



#%% MAMMO CBIS 

path_out_push = os.path.join(path_out,'push',diagnostic_class)
if not os.path.exists(path_out_push):
    os.makedirs(path_out_push)
    
path_out_valid = os.path.join(path_out,'valid',diagnostic_class)
if not os.path.exists(path_out_valid):
    os.makedirs(path_out_valid)
    
patient_names = list(set(glob(os.path.join(path_inp,'*'))) - set(glob(os.path.join(path_inp,'*.csv'))))

#Total training dataset-->PUSH & VALID
names_push, names_valid = train_test_split(patient_names,test_size=0.20,train_size=0.80,random_state=42,shuffle=True)


for name in names_push:
    shutil.copy(name,os.path.join(path_out_push,os.path.basename(name)))
    
for name in names_valid:
    shutil.copy(name,os.path.join(path_out_valid,os.path.basename(name)))

print('done')