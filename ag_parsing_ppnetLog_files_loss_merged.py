#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 12:42:08 2021

@author: si-lab
"""
import numpy as np
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-lp', '--log_path', help='path to Log file', nargs=1, type=str)
args = parser.parse_args()

input_log_filename = args.log_path[0]
f = open(input_log_filename,'r')

dir_name = os.path.dirname(input_log_filename)

is_train=False
is_valid=False
is_push=False

is_iteration=False

i=0

loss=0
loss_train_list=list()
loss_valid_list=list()
loss_train_iter_list=list()
loss_valid_iter_list=list()

import sys
sys.path.append(dir_name)
from settings import coefs
#    'crs_ent': 1,
    # 'clst': 0.8,
    # 'sep': -0.08,
    # 'l1': 1e-4,
    
for idx,l in enumerate(f): 

    #print(idx,l)
    l = l.replace(' ','')
    l=l.replace('\t','')
        # if l.startswith('epoch'):
        #     epochs.append(int(l[-1])) 
        #     continue
    if not is_iteration:
        if is_train:
            if l.startswith('cross'):
                    i+=1
                    splits = l.split(':')
                    temp=splits[1]
                    temp=temp.replace('%\n','0')
                    loss = loss+ coefs['crs_ent']*float(temp)
            
                    continue
            elif l.startswith('cluster'):
               
                    splits = l.split(':')
                    temp=splits[1]
                    temp=temp.replace('%\n','0')
                    loss = loss+ coefs['clst']*float(temp)               
                    continue
            elif l.startswith('separation'):
                
                    splits = l.split(':')
                    temp=splits[1]
                    temp=temp.replace('%\n','0')
                    loss = loss+ coefs['sep']*float(temp)               
                    continue
                
            elif l.startswith('l1'):
                
                    splits = l.split(':')
                    temp=splits[1]
                    temp=temp.replace('%\n','0')
                    loss = loss+ coefs['l1']*float(temp)    
                    loss_train_list.append((i,np.round(loss,decimals=2)))
                    is_train=False 
                    continue
            else:
                continue
                
               
        if is_valid:
            if l.startswith('cross'):
                if is_push:
                    continue
                else:
                    
                    splits = l.split(':')
                    temp=splits[1]
                    temp=temp.replace('%\n','0')
                    loss = loss+ coefs['crs_ent']*float(temp)            
                    continue
            elif l.startswith('cluster'):
                if is_push:
                    continue
                else:
                    splits = l.split(':')
                    temp=splits[1]
                    temp=temp.replace('%\n','0')
                    loss = loss+ coefs['clst']*float(temp)               
                    continue
            elif l.startswith('separation'):
                if is_push:
                    continue
                else:
                    splits = l.split(':')
                    temp=splits[1]
                    temp=temp.replace('%\n','0')
                    loss = loss+ coefs['sep']*float(temp)               
                    continue
                
            elif l.startswith('l1'):   
                if is_push:
                    is_push=False
                    is_valid=False
                    continue
                else:
                    splits = l.split(':')
                    temp=splits[1]
                    temp=temp.replace('%\n','0')
                    loss = loss+ coefs['l1']*float(temp)    
                    loss_valid_list.append((i,np.round(loss,decimals=2)))
                    is_valid=False 
                    continue
            else:
                continue
            
            
            
        
         
        if l.startswith('starttraining'):
            
            continue
        
        if l.startswith('train'):
            is_train=True
            loss=0
            continue
            
        if l.startswith('test'):
            is_valid=True
            loss=0
            continue
        
        if l.startswith('push'):
            is_push=True
            continue
        
        if l.startswith('iteration'):
            is_iteration=True
            continue
    else:
        if is_train:
            if l.startswith('cross'):
                    i+=1
                    splits = l.split(':')
                    temp=splits[1]
                    temp=temp.replace('%\n','0')
                    loss = loss+ coefs['crs_ent']*float(temp)
            
                    continue
            elif l.startswith('cluster'):
               
                    splits = l.split(':')
                    temp=splits[1]
                    temp=temp.replace('%\n','0')
                    loss = loss+ coefs['clst']*float(temp)               
                    continue
            elif l.startswith('separation'):
                
                    splits = l.split(':')
                    temp=splits[1]
                    temp=temp.replace('%\n','0')
                    loss = loss+ coefs['sep']*float(temp)               
                    continue
                
            elif l.startswith('l1'):
                
                    splits = l.split(':')
                    temp=splits[1]
                    temp=temp.replace('%\n','0')
                    loss = loss+ coefs['l1']*float(temp)    
                    loss_train_iter_list.append((i,np.round(loss,decimals=2)))
                    is_train=False 
                    continue
            else:
                continue
            
               
        if is_valid:
            if l.startswith('cross'):
                if is_push:
                    continue
                else:
                    
                    splits = l.split(':')
                    temp=splits[1]
                    temp=temp.replace('%\n','0')
                    loss = loss+ coefs['crs_ent']*float(temp)            
                    continue
            elif l.startswith('cluster'):
                if is_push:
                    continue
                else:
                    splits = l.split(':')
                    temp=splits[1]
                    temp=temp.replace('%\n','0')
                    loss = loss+ coefs['clst']*float(temp)               
                    continue
            elif l.startswith('separation'):
                if is_push:
                    continue
                else:
                    splits = l.split(':')
                    temp=splits[1]
                    temp=temp.replace('%\n','0')
                    loss = loss+ coefs['sep']*float(temp)               
                    continue
                
            elif l.startswith('l1'):   
                if is_push:
                    is_push=False
                    is_valid=False
                    is_iteration=False
                    continue
                else:
                    splits = l.split(':')
                    temp=splits[1]
                    temp=temp.replace('%\n','0')
                    loss = loss+ coefs['l1']*float(temp)    
                    loss_valid_iter_list.append((i,np.round(loss,decimals=2)))
                    is_valid=False 
                    is_iteration=False
                    continue
            else:
                continue

           


    
        if l.startswith('starttraining'):
            
            continue
        
        if l.startswith('train'):
            is_train=True
            loss=0
            continue
            
        if l.startswith('test'):
            is_valid=True
            loss=0
            continue
        
        if l.startswith('push'):
            is_push=True
            continue
        
        if l.startswith('iteration'):
            is_iteration=True
            continue



f.close()       

#%% plot
import os
path_to_save_img = os.path.dirname(input_log_filename)
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
params={
        'legend.fontsize':40,
        'legend.markerscale':2,
        'figure.figsize':(40,22),
        'axes.labelsize':25,
        'axes.titlesize':25,
        'xtick.labelsize':25,
        'ytick.labelsize':25
        }
pylab.rcParams.update(params)
plt.figure()
x_axis_train_iter=[elem[0] for elem in loss_train_iter_list]
x_axis_train=[elem[0] for elem in loss_train_list]
x_axis_valid_iter=[elem[0] for elem in loss_valid_iter_list]
x_axis_valid=[elem[0] for elem in loss_valid_list]

y_axis_train_iter=[elem[1] for elem in loss_train_iter_list]
y_axis_train=[elem[1] for elem in loss_train_list]
y_axis_valid_iter=[elem[1] for elem in loss_valid_iter_list]
y_axis_valid=[elem[1] for elem in loss_valid_list]

#

x_axis_train.extend(x_axis_train_iter)
y_axis_train.extend(y_axis_train_iter)
train = sorted(list(zip(x_axis_train,y_axis_train)))

x_axis_valid.extend(x_axis_valid_iter)
y_axis_valid.extend(y_axis_valid_iter)
valid = sorted(list(zip(x_axis_valid,y_axis_valid)))

plt.plot(list(zip(*train))[0], list(zip(*train))[1],'*-k', label='Training')
plt.plot(list(zip(*valid))[0], list(zip(*valid))[1],'*-b', label='Validation')

plt.xlim((0,x_axis_train[-1]+20))
plt.title('LOSS FUNCTION')
plt.legend()
plt.savefig(os.path.join(path_to_save_img,'loss_'+ os.path.basename(path_to_save_img) +'.pdf'),bbox_inches='tight')   

print('Done, LOSS FUNCTION image saved at given folder path')