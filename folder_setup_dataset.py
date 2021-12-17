#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 11:57:29 2021

@author: si-lab
"""
import os
import shutil
import glob

path_to_cmmd = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CMMD/manifest-1616439774456/CMMD'
path_to_patient_dir = glob.glob(os.path.join(path_to_cmmd, '*'))

out_path = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CMMD/DICOM'

if not os.path.exists(out_path):
    os.mkdir(out_path)

for p in path_to_patient_dir:
    _, _, path_to_im = os.walk(p)
    name_list = path_to_im[-1]
    for im_name in name_list:
        shutil.copy(os.path.join(path_to_im[0], im_name), os.path.join(out_path, os.path.basename(p)+'_'+im_name))
    