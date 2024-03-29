#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 09:50:51 2022

@author: si-lab
"""
import os
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import argparse


# import sys
# sys.path.append('/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_originale_originale/ProtoPNet/')
# from preprocess import mean
# mean = mean[0]


parse = argparse.ArgumentParser(description="Resize and pad images to retain original aspect ratios and obtain final target dimension")
parse.add_argument('input_dir', help='Path to the input directory with PNG images (parent folder of class folders), e.g. '\
                   '*dataset*, parent of *benign* and *malignant* folders')
parse.add_argument('-is','--intermediate_size',type=int,help='Target intermediate-size of small images prior to padding; default=250',default=250)
parse.add_argument('-ts','--target_size',type=int,help='Target size of final squared images; default=500',default=500)
args = parse.parse_args()

input_dir = args.input_dir
inter_size = args.intermediate_size
target_size = args.target_size
   
    
# mean_list=[]
# std_list=[]
## CARICO IL VALORE MEAN PRECEDENTEMENTE CALCOLATO E SALVATO
mean = np.load(os.path.join(input_dir,'mean.npy'))

imgsize_list=[]
list_H = []
list_W = []
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

                # #
                # mean_list.append(img_npy.mean())
                # std_list.append(img_npy.std())
                # #
                
                shape_max = max(img.size)
                H = img.size[0]
                W = img.size[1]
                
                # print(file)
                mat = np.ones((shape_max,shape_max))*mean
                
                if H>W: #piu alta che larga
                    delta = H-W
                    col_min_idx = int(np.floor(delta/2))
                    col_max_idx = col_min_idx + W
                    mat[col_min_idx:col_max_idx,:] = img_npy
                    # print('Resa quadrata da H>W')
                    
                elif W>H:#piu larga che alta
                    delta = W-H
                    row_min_idx = int(np.floor(delta/2))
                    row_max_idx = row_min_idx + H
                    mat[:,row_min_idx:row_max_idx] = img_npy
                    # print('Resa quadrata da W>H')
                    
                else:
                    mat=np.copy(img_npy)
                    # print('Già quadrata')
                
                # small images upsampled to intermediate size prior to padding
                dim = mat.shape[0]
                if dim < inter_size:
                    mat = zoom(mat, inter_size/dim, order=3)
                    dim = inter_size

                
                
                mat2 = np.full((target_size,target_size),fill_value=mean)
                # controllo se sono più piccole della target_size
                if dim < target_size:
                    delta = target_size - dim
                    min_idx = int(np.floor(delta/2))
                    max_idx = min_idx + dim
                    mat2[min_idx:max_idx, min_idx:max_idx] = mat
                    
                else:
                    mat2 = np.copy(mat)

                
                    
                
                # e Poi si salva
                # np.save(f'/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM/dataset_ripulito_ribilanciato/push_e_valid_quadrate/npy/float_{file[:-4]}.npy',mat2)
                mat2 = np.uint8(mat2)
                # np.save(f'/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM/dataset_ripulito_ribilanciato/push_e_valid_quadrate/npy/uint8_{file[:-4]}.npy',mat2)

                im_out = Image.fromarray(mat2)

                # if im_out.mode!='L':
                #     print(f'{file} : {im_out.mode}')
                    
                # print(f'{os.path.dirname(root)}')
                # print(f'{os.path.dirname(input_dir)}')
                # #
                output_dir = os.path.join(f'{input_dir}_Intermediate-{inter_size}_Target-{target_size}',os.path.basename(root)) #TODO
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                im_out.save(os.path.join(output_dir,file),format='PNG')
                # print('Immagine salvata')
 