from glob import glob
import os
import pandas as pd
import numpy as np
from pydicom import read_file
from PIL import Image


path_to_data = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM/data'

csv_file_masks_train = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM/mass_case_description_train_set.csv'
csv_file_masks_test = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM/mass_case_description_test_set.csv'

out_path = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM/png_masks'

for ind,csv_path in enumerate([csv_file_masks_train, csv_file_masks_test]):
    df = pd.read_csv(csv_path,sep=',',index_col='dirname')
    if ind==0:
        out_dir = os.path.join(out_path,'train')
    elif ind==1:
        out_dir = os.path.join(out_path,'test')
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    for i in range(len(df)):
        dirname = df.index[i]

        walk_dir = os.path.join(path_to_data,dirname)
        for root, dirs, files in os.walk(walk_dir):
            if len(files)==0:
                continue
            if len(files)==1:
                #sei nel caso sbagliato, devi prendere il .dcm che ha per basename la parola croppedimage (sottocartella)
                bn = os.path.basename(root)
                if 'ROI mask images' in bn:
                    mask_path=os.path.join(root,files[0])
                else:
                    continue
            elif len(files)==2:
                #devi prendere quella che pesa più byte: è la maschera
                file_size0 = os.path.getsize(os.path.join(root,files[0]))
                file_size1 = os.path.getsize(os.path.join(root,files[1]))
                if file_size0<file_size1:
                    mask_path=os.path.join(root,files[1])
                else:
                    mask_path=os.path.join(root,files[0])
        dcm_mask = read_file(mask_path)
        mask = dcm_mask.pixel_array
        mask[mask>0] = 255
        out_path_file = os.path.join(out_dir,dirname+'.png')
        Image.fromarray(mask).save(out_path_file)
        print('Saved mask in {}'.format(out_path_file))
       