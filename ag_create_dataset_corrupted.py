#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:35:39 2021
Prende in ingresso le immagini png originali (ma potenzialmente croppate) e ne corrompe
i livelli di grigio tramite una o più corruzioni -disgiunte- e le salva nelle rispettive cartelle
@author: andreaberti
"""

import os
import glob
import numpy as np
import imageio
from skimage import exposure

import argparse

#%%
# #%% Dizionario user Gianlu/Andrea
# login = os.getlogin()

# dizionario = {
#         'andreaberti': '/Users/andreaberti/Documents/work/256x256/',
#         'Gianlu': r'C:\Users\Gianlu\.spyder-py3\256x256_t2w_prostate'
#     }

parse = argparse.ArgumentParser(description="Creating a corrupted version of the original dataset")
parse.add_argument('png_dir', help='Path to the input directory with original PNG images')
parse.add_argument('corrupted_png_dir', help='Path to the output directory to group corrupted PNG images',type=str)
parse.add_argument('-g','--gauss_noise', help='Whether to corrupt with gaussian noise (mu=1, sigma=1)',type=bool, default=False)
parse.add_argument('-p','--patch', help='Whether to corrupt with blach patches',type=bool, default=False)
parse.add_argument('-hi','--histeq', help='Whether to corrupt with histogram equalization',type=bool,default=False)

args = parse.parse_args()

png_dir = args.png_dir
corrupted_png_dir = args.corrupted_png_dir
is_gauss = args.gauss_noise
is_patch = args.patch
is_histeq = args.histeq 

if not os.path.exists(corrupted_png_dir):
    os.mkdir(corrupted_png_dir)
    print('Corrupted png dir, created')
    
#%% Caricamento immagini originali da cartella PNG T2

for im_path in glob.glob(os.path.join(png_dir, '*.png')):
      
    patient_name = os.path.basename(im_path)     
    im = np.array(imageio.imread(im_path))
    row,col= im.shape
    # assert row==col, "Attenzione, righe e colonne della immagine in numero diverso!"
    
    
    if is_gauss:
        ## Aggiunta di rumore gaussiano
        mean = 1
        var = 1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = im + gauss
        #
        noisy = noisy - np.min(noisy)
        noisy = noisy / np.max(noisy)
        noisy = 255 * noisy
        
        noisy_int = noisy.astype(np.uint8)
        temp = os.path.join(corrupted_png_dir,'gaussian_noise')
        if not os.path.exists(temp):
            os.mkdir(temp)
            print('Corrupted png dir with GAUSS NOISE, created')
        imageio.imsave(os.path.join(temp, 'gauss_'+patient_name), noisy_int)
        
    
    # if is_patch:
        # TO-DO: CAPIRE SE HA SENSO USARE QUESTI PATCH ANCHE CON IMMAGINI CROPPATE
        # -- BISOGNA CERCARE DI NON COPRIRE LA PROSTATA IN IMMAGINI PICCOLE
    #     ## Aggiunta di patch scuri quadrati ad oscurare porzioni di immagine
    #     # creazione maschera
    #     mask = np.ones((row,col),dtype=np.uint8)
        
    #     patchsize = int(np.random.uniform(10,25))
    #     c1 = int(np.random.uniform(12,18))
    #     c2 = int(np.random.uniform(0,col))
    #     mask[c1:c1+patchsize,c2:c2+patchsize]=0
        
    #     patchsize = int(np.random.uniform(10,25))    
    #     c1 = int(np.random.uniform(12,18))
    #     c2 = int(np.random.uniform(0,col))
    #     mask[-c1-patchsize:-c1,c2:c2+patchsize]=0
        
    #     patchsize = int(np.random.uniform(10,25))    
    #     c1 = int(np.random.uniform(12,18))
    #     c2 = int(np.random.uniform(0,col))
    #     mask[c2:c2+patchsize,c1:c1+patchsize]=0
        
    #     patchsize = int(np.random.uniform(10,25))    
    #     c1 = int(np.random.uniform(12,18))
    #     c2 = int(np.random.uniform(0,col))
    #     mask[c2:c2+patchsize,-c1-patchsize:-c1]=0
        
    #     im_corr = im*mask
        
    #     temp = os.path.join(corrupted_png_dir,'black_patches')
    #     if not os.path.exists(temp):
    #         os.mkdir(temp)
    #         print('Corrupted png dir with BLACK PATCHES, created')
    #     imageio.imsave(os.path.join(temp, 'bpatch_'+patient_name), im_corr)
        
    
    if is_histeq:
        ## Histogram equalization
        im_eq = exposure.equalize_hist(im)
        im_eq = im_eq - np.min(im_eq)
        im_eq = im_eq / np.max(im_eq)
        im_eq = 255 * im_eq
        im_eq_int = im_eq.astype(np.uint8)
        temp = os.path.join(corrupted_png_dir,'histogram_equalized')
        if not os.path.exists(temp):
            os.mkdir(temp)
            print('Corrupted png dir with HISTOGRAM EQUALIZATION, created')
        imageio.imsave(os.path.join(temp, 'histeq_'+patient_name), im_eq_int)


#%%
