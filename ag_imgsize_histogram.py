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
from matplotlib import pyplot as plt



def create_lists(input_dir):
    '''
    Analize images inside input_dir and return the lists of their size

    Parameters
    ----------
    input_dir : str
        Absolute path to the parent directory containing the image folders

    Returns
    -------
    mean_list : np.array
        List of the mean values of the gray-levels in the images
    std_list : np.array
        List of the std values of the gray-levels in the images
    mean : float
        mean of the values in mean_list
    std : float
        mean of the values in std_list
    imgsize_list : np.array
        List containing the max dimension of the images
    list_H : np.array
        List of the height-dimensions of the images
    list_W : np.array
        List of the width-dimensions of the images

    '''
    mean_list=[]
    std_list=[]
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
                    #
                    mean_list.append(img_npy.mean())
                    std_list.append(img_npy.std())
                    #
                    
                    shape_max = max(img.size)
                    H = img.size[0]
                    W = img.size[1]
           
                    list_H.append(H)
                    list_W.append(W)
                    imgsize_list.append(shape_max)
                    
    mean_list = np.array(mean_list)
    std_list = np.array(std_list)
    mean = mean_list.mean()
    std = std_list.mean()
    imgsize_list = np.array(imgsize_list)
    list_H = np.array(list_H)
    list_W = np.array(list_W)
    return mean_list, std_list, mean, std, imgsize_list, list_H, list_W


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="")
    parse.add_argument('input_dir', help='Path to the parent directory containing the class-directories (ex. benign and malignant)')
    args = parse.parse_args()
    
    input_dir = args.input_dir
      
    mean_list, std_list, mean, std, imgsize_list, list_H, list_W = create_lists(input_dir)  

    # for item in [mean_list, std_list, mean, std, imgsize_list, list_H, list_W]:
    for item in [mean, std]:
        np.save(os.path.join(input_dir, f'{item}.npy'),item) #TODO

    # plt.figure()
    # plt.hist(mean_list,bins=100)
    # plt.title(f'Mean Values \nMEDIAN: {np.round(np.median(mean_list),2)}; MEAN: {np.round(np.mean(mean_list),2)}')
    # plt.savefig(os.path.join(input_dir,'mean_values.pdf'))
    
    # plt.figure()
    # plt.hist(std_list,bins=100)
    # plt.title(f'Std Values \nMEDIAN: {np.round(np.median(std_list),2)}; MEAN: {np.round(np.mean(std_list),2)}')
    # plt.savefig(os.path.join(input_dir,'std_values.pdf'))
    
    
    # plt.figure()
    # plt.hist(imgsize_list,bins=100)
    # plt.title(f'Max dimension Values \nMEDIAN: {np.round(np.median(imgsize_list),2)}; MEAN: {np.round(np.mean(imgsize_list),2)}')
    # plt.savefig(os.path.join(input_dir,'max_dimension_values.pdf'))
    
    # plt.figure()
    # plt.hist(list_H,bins=100)
    # plt.title(f'H dimension Values \nMEDIAN: {np.round(np.median(list_H),2)}; MEAN: {np.round(np.mean(list_H),2)}')
    # plt.savefig(os.path.join(input_dir,'h_dimension_values.pdf'))
    
    # plt.figure()
    # plt.hist(list_W,bins=100)
    # plt.title(f'W dimension Values \nMEDIAN: {np.round(np.median(list_W),2)}; MEAN: {np.round(np.mean(list_W),2)}')
    # plt.savefig(os.path.join(input_dir,'w_dimension_values.pdf'))
