#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 09:50:51 2022

@author: si-lab
"""
import os
import numpy as np
import PIL.Image
#1 camminare su tutte le cartelle per scorrere tutte le immagini
path = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM/dataset_png'

imgsize_list=[]
for root ,dirs,files in os.walk(path):
    if len(files)!=0:
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root,file)
                img = PIL.Image.open(img_path)
                shape_max = max(img.size)
                imgsize_list.append(shape_max)

imgsize_list = np.array(imgsize_list)

from matplotlib import pyplot as plt
plt.hist(imgsize_list,bins=100,cumulative=True)


plt.vlines(np.median(imgsize_list),0,350,colors='r')
plt.vlines(217,0,350,colors='k')

plt.show()


# print(np.median(imgsize_list))
# print(np.mean(imgsize_list))

# from scipy import stats
# print(stats.mode(imgsize_list))