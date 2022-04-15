#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:20:31 2022

@author: si-lab
"""

import os
import numpy as np
from PIL import Image
import argparse
import imgaug.augmenters as iaa
from scipy.signal import medfilt2d



# import sys
# sys.path.append('/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_originale_originale/ProtoPNet/')
# from preprocess import mean
# mean = mean[0]


parse = argparse.ArgumentParser(description="")
parse.add_argument('input_dir', help='Path to the input directory with original PNG images to augment')
parse.add_argument('output_dir', help='Path to the output directory with modified PNG images')
parse.add_argument('-d', '--denoise_first', help='Apply denoise before CLAHE', action='store_true')
parse.set_defaults(denoise_first=True)
args = parse.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
denoise_first = args.denoise_first

print(f'denoise first = {denoise_first}')
apply_clahe = iaa.AllChannelsCLAHE()

for root,dirs,files in os.walk(input_dir):
    if len(files)!=0:
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root,file)
                img = Image.open(img_path).convert('L')
                img_npy = np.array(img)
                if np.amax(img_npy)<255:
                    print(f'{file} --- Era una immagine RGB')
                    # np.save(f'/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM/dataset_ripulito_ribilanciato/push_e_valid_quadrate/npy/pre_{file[:-4]}.npy',img_npy)
                    img_npy = ((img_npy - np.amin(img_npy))/(np.amax(img_npy)-np.amin(img_npy)))*255
                    # np.save(f'/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM/dataset_ripulito_ribilanciato/push_e_valid_quadrate/npy/post_{file[:-4]}.npy',img_npy)
                    img_npy = np.uint8(img_npy)
                    
                if denoise_first:
                    img_denoised = medfilt2d(img_npy, 3) 
                    img_clahe = apply_clahe(images=img_denoised)
                else:
                    img_clahe = apply_clahe(images=img_npy)
                    img_denoised = medfilt2d(img_clahe, 3) 
                    img_denoised = ((img_denoised - np.amin(img_denoised)) / (np.amax(img_denoised) - np.amin(img_denoised))) * 255
                    img_denoised = np.uint8(img_denoised)
                    
                im_out = Image.fromarray(img_clahe)

          
                output_subdir = os.path.join(output_dir, 'denoise_clahe', os.path.basename(root)) #TODO cambiare subdir
                
                
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                im_out.save(os.path.join(output_subdir,file),format='PNG')

with open(os.path.join(os.path.join(output_dir, 'denoise_clahe'), 'info.txt'), 'w') as outfile: #TODO cambiare il file di testo e il suo path
    outfile.write('Prima applicato il denoise con filtro mediano (kernel 3x3) e successivamente si è applicato il CLAHE')
