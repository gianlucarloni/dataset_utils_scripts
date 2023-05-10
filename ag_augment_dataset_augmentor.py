#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 16:37:59 2022

@author: si-lab
"""
from torchvision import transforms
import Augmentor

# import pandas as pd
import argparse
import os
import numpy as np
import shutil
import glob
import PIL

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Final aim: to augment (33x) the original images with the Augmentor library")
    parse.add_argument('input_dir', 
                       help='Path to the input directory with original PNG images of a specific class to augment')
    parse.add_argument('output_dir', 
                       help='Path to the output directory for the corresponding augmented PNG images',type=str)
    parse.add_argument('-d', '--deep_augment', 
                       help='Use additional augmentations (hist_eq, brightness, contrast), with this tag the dataset will undergo a 36x augmentation', 
                       default=False,
                       action="store_true")
    args = parse.parse_args()
    
    input_dir = args.input_dir
    
    output_dir = args.output_dir
    
    deep_augment = args.deep_augment
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    
    # flip = transforms.RandomHorizontalFlip(p=1)
    
    print('Augm. n. 0: the exact copy')
    for imname in glob.glob(os.path.join(input_dir, '*.png')):
        basename = os.path.basename(imname)
        shutil.copy(imname,os.path.join(output_dir, basename))

    # Shift
    print('Augm. n. 1: rotation')
    for img in glob.glob(os.path.join(input_dir, '*.png')):
        imm_pil = PIL.Image.open(img)
        imm_npy = np.array(imm_pil)
        mean_gray = np.mean(imm_npy)
        shift = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=mean_gray) 
        for i in range(2):
            imm_new2 = shift(imm_pil)
            imm_new2.save(os.path.join(output_dir, f'aug_shift_{i}_' + basename),'PNG')

    p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
    p.rotate(probability=1, max_left_rotation=170, max_right_rotation=170) 
    for i in range(10):
        p.process()
    del p
    
    #TODO 2: rotation
    print('Augm. n. 2: rotation')
    p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
    p.rotate(probability=1, max_left_rotation=170, max_right_rotation=170) 
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
    
    if deep_augment:
        #TODO 7:  hist equalization
        print('Augm. n. 7: hist equalization')
        p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
        p.histogram_equalisation(probability=1)
        p.process()
        del p 
            
        #TODO 8:  brightness
        print('Augm. n. 8: brightness')
        p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
        p.random_brightness(1, 0.8, 1.2)
        p.process()
        del p 
        
        #TODO 9:  contrast
        print('Augm. n. 9: contrast')
        p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
        p.random_contrast(1, 0.6, 1.5)
        p.process()
        del p 

    print('Done.')