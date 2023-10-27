#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:07:52 2022

@author: si-lab
"""
import numpy as np
import matplotlib.pyplot as plt

npy_accs_noiter_train_fold4 = np.load('/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_cluster/saved_models_review/resnet18/CBIS_massBenignMalignant_Fri_22_Jul_2022_16:01:08_config4/fold4/npy_accs_noiter_train_fold4.npy')
npy_accs_noiter_train_fold3 = np.load('/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_cluster/saved_models_review/resnet18/CBIS_massBenignMalignant_Fri_22_Jul_2022_16:01:08_config4/fold3/npy_accs_noiter_train_fold3.npy')
npy_accs_noiter_train_fold2 = np.load('/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_cluster/saved_models_review/resnet18/CBIS_massBenignMalignant_Fri_22_Jul_2022_16:01:08_config4/fold2/npy_accs_noiter_train_fold2.npy')
npy_accs_noiter_train_fold1 = np.load('/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_cluster/saved_models_review/resnet18/CBIS_massBenignMalignant_Fri_22_Jul_2022_16:01:08_config4/fold1/npy_accs_noiter_train_fold1.npy')
npy_accs_noiter_train_fold0 = np.load('/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_cluster/saved_models_review/resnet18/CBIS_massBenignMalignant_Fri_22_Jul_2022_16:01:08_config4/fold0/npy_accs_noiter_train_fold0.npy')


npy_accs_noiter_valid_fold4 = np.load('/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_cluster/saved_models_review/resnet18/CBIS_massBenignMalignant_Fri_22_Jul_2022_16:01:08_config4/fold4/npy_accs_noiter_valid_fold4.npy')
npy_accs_noiter_valid_fold3 = np.load('/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_cluster/saved_models_review/resnet18/CBIS_massBenignMalignant_Fri_22_Jul_2022_16:01:08_config4/fold3/npy_accs_noiter_valid_fold3.npy')
npy_accs_noiter_valid_fold2 = np.load('/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_cluster/saved_models_review/resnet18/CBIS_massBenignMalignant_Fri_22_Jul_2022_16:01:08_config4/fold2/npy_accs_noiter_valid_fold2.npy')
npy_accs_noiter_valid_fold1 = np.load('/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_cluster/saved_models_review/resnet18/CBIS_massBenignMalignant_Fri_22_Jul_2022_16:01:08_config4/fold1/npy_accs_noiter_valid_fold1.npy')
npy_accs_noiter_valid_fold0 = np.load('/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_cluster/saved_models_review/resnet18/CBIS_massBenignMalignant_Fri_22_Jul_2022_16:01:08_config4/fold0/npy_accs_noiter_valid_fold0.npy')

#%%escludiamo da un certo punto in poi per farli avere tutti della stessa lunghezza, tanto è una zona non importante del plot
npy_accs_noiter_train_fold4 = npy_accs_noiter_train_fold4[:61]
npy_accs_noiter_valid_fold4 = npy_accs_noiter_valid_fold4[:61]
#%%
plt.figure()
for acc in [npy_accs_noiter_train_fold4, npy_accs_noiter_train_fold3, npy_accs_noiter_train_fold2, npy_accs_noiter_train_fold1, npy_accs_noiter_train_fold0]:
    plt.plot(acc,'ok-')
    
for val in [npy_accs_noiter_valid_fold4, npy_accs_noiter_valid_fold3, npy_accs_noiter_valid_fold2, npy_accs_noiter_valid_fold1, npy_accs_noiter_valid_fold0]:
    plt.plot(val,'ob-') 
plt.show()


#%%
train_acc_media_array = np.array([npy_accs_noiter_train_fold4, npy_accs_noiter_train_fold3, npy_accs_noiter_train_fold2, npy_accs_noiter_train_fold1, npy_accs_noiter_train_fold0])
train_acc_media = np.mean(train_acc_media_array,axis=0)
train_acc_std = np.std(train_acc_media_array, axis=0)

valid_acc_media_array = np.array([npy_accs_noiter_valid_fold4, npy_accs_noiter_valid_fold3, npy_accs_noiter_valid_fold2, npy_accs_noiter_valid_fold1, npy_accs_noiter_valid_fold0])
valid_acc_media = np.mean(valid_acc_media_array,axis=0)
valid_acc_std = np.std(valid_acc_media_array, axis=0)


plt.figure()
plt.plot(train_acc_media,'ok-',label='Internal-training')
plt.fill_between(np.arange(len(train_acc_media)), train_acc_media-train_acc_std, train_acc_media+train_acc_std,color='k',alpha=0.25)
plt.plot(valid_acc_media,'ob-',label='Internal-validation')
plt.fill_between(np.arange(len(valid_acc_media)), valid_acc_media-valid_acc_std, valid_acc_media+valid_acc_std, color='b',alpha=0.25)
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid()
# plt.show()
plt.savefig('/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_cluster/saved_models_review/resnet18/CBIS_massBenignMalignant_Fri_22_Jul_2022_16:01:08_config4/cumulative_acc_plot.pdf',bbox_inches='tight')
