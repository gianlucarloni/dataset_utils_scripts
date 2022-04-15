#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:15:02 2022

@author: si-lab
"""
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

valid_txt_path = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_originale_originale/ProtoPNet/saved_models_baseline/resnet18/CBIS_baseline_massBenignMalignant_resnet18_mar_12_apr_2022_08:13:12_binaryCrossEntr/val_metrics.txt'
# valid_txt_path ='/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_originale_originale/ProtoPNet/saved_models_baseline/resnet18/CBIS_baseline_massBenignMalignant_resnet18_Tue_12_Apr_2022_09:45:31_binaryCrossEntr/val_metrics.txt'
# valid_txt_path = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_originale_originale/ProtoPNet/saved_models_baseline/resnet34/CBIS_baseline_massBenignMalignant_resnet34_sab_09_apr_2022_21:37:22_binaryCrossEntr/val_metrics.txt'

df = pd.read_csv(valid_txt_path)
loss = list(df.loc[:]['loss'])
epochs = list(df.loc[:]['epoch'])

loss_2 = loss[1:]
loss_2.append(0)

deriv = [(e[2],(e[0]-e[1])/1) for e in zip(loss,loss_2,epochs)]
deriv = deriv[:-1]

# deriv_o25 = [(e[2],(e[0]-e[1])/25) for idx,e in enumerate(zip(loss,loss_2,epochs)) if idx%25==0]
# deriv_o25 = deriv_o25[:-1]

window = 20
loss_npy = np.array(loss,dtype=float)
windowed = [np.mean(loss_npy[i:i+window]) for i in range(0,len(loss),window)]
windowed_epochs = range(window,len(loss)+window,window)

windowed_2 = windowed[1:]
windowed_2.append(0.0)

deriv_w = [(e[2],(e[0]-e[1])/window) for e in zip(windowed,windowed_2,windowed_epochs)]
deriv_w = deriv_w[:-1]

plt.figure()

plt.subplot(411)
plt.plot(epochs,loss,'-*')
plt.title('Loss over epochs')

plt.subplot(412)
plt.plot([elem[0] for elem in deriv],[elem[1] for elem in deriv],'-*')
plt.title('Derivative of loss values')

plt.subplot(413)
# plt.plot([elem[0] for elem in deriv_o25],[elem[1] for elem in deriv_o25],'-*')
# plt.title('Derivative every-25 loss values')
plt.plot(windowed_epochs,windowed,'-*')
plt.title(f'Loss with mobile mean with window size: {window}')

plt.subplot(414)
plt.plot(windowed_epochs[:-1],[elem[1] for elem in deriv_w],'-*')
# plt.hlines(0.01/4, 0, len(loss),linestyles='dashed',colors='red')
plt.hlines(deriv_w[0][1]*0.10, 0, len(loss),linestyles='dashed',colors='red')
plt.title(f'Derivative of Loss with mobile mean with window size: {window}')


plt.show()
