import os

import numpy as np
import argparse
from pydicom import read_file
from PIL import Image
import pandas as pd

from scipy.ndimage import zoom
import cv2

def get_bbox(input_mask_npy):
    
    # troviamo gli indici di riga e colonna dove assume valori non nulli --> determiniamo la bounding box
    nonzero = input_mask_npy.nonzero()    
    row_min = np.amin(nonzero[0])
    row_max = np.amax(nonzero[0])
    col_min = np.amin(nonzero[1])
    col_max = np.amax(nonzero[1])
    
    return row_min, row_max, col_min, col_max

def mass_slice_and_create_pil(npy_img, apply_clahe):
    npy_img = ((npy_img - np.amin(npy_img))/(np.amax(npy_img) - np.amin(npy_img)))*255
    npy_img = npy_img.astype(np.uint8)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(npy_img)
        clahe_img = ((clahe_img - np.amin(clahe_img))/(np.amax(clahe_img) - np.amin(clahe_img)))*255
        clahe_img = clahe_img.astype(np.uint8)
        pil_mass = Image.fromarray(clahe_img)
    else:    
        pil_mass = Image.fromarray(npy_img)
    return pil_mass

parse = argparse.ArgumentParser(description="")
parse.add_argument('data_path', help='path to breast_CBIS_DDSM dataset',type=str)
parse.add_argument('csv_mass_train_path', help='path to CSV file for training masses',type=str)
parse.add_argument('csv_mass_test_path', help='path to CSV file for test masses',type=str)
parse.add_argument('output_path', help='path to parent folder of train/test dirs, where the png images and the txt labels are saved',type=str)
# parse.add_argument('path_to_correct_names_txt', help='path to txt file containing the names of the correct cases',type=str)
parse.add_argument('-c', '--clahe', help='Apply CLAHE preprocessing to images', default=False, action='store_true')

args = parse.parse_args()

data_path = args.data_path
csv_mass_train_path = args.csv_mass_train_path
csv_mass_test_path = args.csv_mass_test_path
output_path = args.output_path
# path_to_correct_names_txt = args.path_to_correct_names_txt
apply_clahe = args.clahe

# corr_names_txt = pd.read_csv(path_to_correct_names_txt, header=None)
# list_of_correct_names = list(corr_names_txt[0])

for ind, csv_path in enumerate([csv_mass_train_path, csv_mass_test_path]):
    df = pd.read_csv(csv_path,sep=',',index_col='dirname')

    if ind==0:
        img_dir = os.path.join(output_path,'train','images')
        label_dir = os.path.join(output_path,'train','labels')
    elif ind==1:
        img_dir = os.path.join(output_path,'test','images')
        label_dir = os.path.join(output_path,'test','labels')
    
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
       
    idx=0 
    dirnames = df.index
    while idx < len(dirnames):
        print(f'Ind {ind}. Iteration: {idx} over {len(dirnames)}')
        dirname = dirnames[idx]
        #for dirname in tqdm(df.index):
        #E.g., dirname:= Mass-Training_P_00001_LEFT_MLO_1
        
        
        splits = dirname.split(sep='_')
        full_image_dir = '_'.join(splits[0:5]) # E.g., full_image_dir:= Mass-Training_P_00001_LEFT_MLO (without the final number)
        
        # if full_image_dir in list_of_correct_names:
        #Select only the records of the specific patient selected
        # and count them to increment the while iterator
        records_per_patient=[]
        for dn in dirnames:
            if dn.startswith(full_image_dir):
                records_per_patient.append(dn)
                
        num_records_per_patient = len(records_per_patient)
        #
        # if num_records_per_patient>1: #Images with multiple lesions
        if num_records_per_patient > 0:

            # with open(txt_path,'a') as fout:
            #     fout.write(f'{full_image_dir},{num_records_per_patient}\n')
        
            df_patient = df.loc[records_per_patient]
            
            #
            w_full = os.walk(os.path.join(data_path, full_image_dir))
            for root, dirs, files in w_full:
                if len(files)==0:
                    continue
                elif len(files)==1:
                    full_image_path=os.path.join(root,files[0])
                else:
                    print('-------------------FULL MAMMO IMAGE NOT FOUND CORRECTLY')
            #
            
            dcm_full_im = read_file(full_image_path)
            npy_full_im = dcm_full_im.pixel_array

            out_img_name = f'{full_image_dir}.png'
            out_txt_label = f'{full_image_dir}.png'
        
            
            was_in_dataset = False
            #
            was_mask_corrupted = False
            #
            
            
            for num_record, record in enumerate(records_per_patient):
                w = os.walk(os.path.join(data_path,record))   
                
                label = df_patient.loc[record]['pathology']
                # cr_name = ''
                
                crop_size = 600 #TODO
                
                if label == 'BENIGN' or label == 'MALIGNANT':
                    
                #     if label == 'BENIGN':
                #          output_save_dir = benign_dir
                #     else:
                #          output_save_dir = malignant_dir
        
                
                    for root,dirs,files in w:
                        
                        if len(files)==0:
                            continue
                        
                        if len(files)==1:
                            #sei nel caso sbagliato, devi prendere il .dcm che ha per basename la parola croppedimage (sottocartella)
                            bn = os.path.basename(root)
                            if 'ROI mask images' in bn:
                                mask_path=os.path.join(root,files[0])
                            else:
                                #cr_name = files[0] # in questo caso non dovrebbe servirmi prendere il crop
                                continue
                                
                        elif len(files)==2:
                            #devi prendere quella che pesa più byte: è la maschera
                            file_size0 = os.path.getsize(os.path.join(root,files[0]))
                            file_size1 = os.path.getsize(os.path.join(root,files[1]))
                            if file_size0<file_size1:
                                mask_path=os.path.join(root,files[1])
                                # just to maintain the same names as before
                                # cr_name = files[0]
                            else:
                                mask_path=os.path.join(root,files[0])
                                # cr_name = files[1]
                            
                    # lesion_name_png = f'{record}_{cr_name[:-3]}png'
                    # if lesion_name_png in names_list:
                    if True: #TODO

                                            
                        dcm_mask = read_file(mask_path)
                        npy_mask = dcm_mask.pixel_array
                        # check if the mask and the whole image have the same dimensions
                        shape_i = npy_full_im.shape
                        shape_m = npy_mask.shape
                        if shape_i != shape_m:
                            was_mask_corrupted=True

                            zoom_factor_rows = shape_i[0]/shape_m[0]
                            zoom_factor_cols = shape_i[1]/shape_m[1]
                            zoom_factor = (zoom_factor_rows, zoom_factor_cols)
                            npy_mask = zoom(npy_mask,zoom=zoom_factor)
                            _, npy_mask = cv2.threshold(npy_mask, thresh=np.quantile(npy_mask,0.75), maxval=255.0, type=0)
                            
                        npy_mask = npy_mask/255.0 #TODO
                        npy_mask = npy_mask.astype(np.uint8)
                        
                        
                        assert(np.amax(npy_mask)==1)
                        assert(np.amin(npy_mask)==0)
                        # assert(list(np.unique(npy_mask)).sort() == [0,1])
                
                        
                        row_min, row_max, col_min, col_max = get_bbox(npy_mask)
                        # for yolo I need x_centre, y_centre, width, height
                        
                        col_center = (col_max + col_min)/2 
                        row_center = (row_max + row_min)/2
                        width = col_max - col_min
                        height = row_max - row_min

                        img_width = shape_i[1]
                        img_height = shape_i[0]

                        # normalising the coordinates for Yolo
                        col_center /= img_width
                        row_center /= img_height

                        width /= img_width
                        height /= img_height

                        # creating the PIL file of the annotated slice
                        pil_mass = mass_slice_and_create_pil(npy_img=npy_full_im, apply_clahe=apply_clahe)

                        # ora: creare nomi e salvare immagine e label; più label?
                        
                        # se è il primo record_per_patient creo il file della label e salvo
                        # l'immagine in png, altrimenti apro solo il file in append e aggiungo il nuovo bbox
                        if num_record == 1:
                            # the structure will be dataset/train/images and dataset/train/labels (same thing for test)
                            out_path = os.path.join(img_dir, out_img_name)
                            out_path_label = os.path.join(label_dir, out_txt_label)

                            with open(out_path_label, "w") as label_file:
                                label_file.write(f'0 {col_center} {row_center} {width} {height}')

                            pil_mass.save(out_path, 'PNG')
                        else:
                            with open(out_path_label, "a") as label_file:
                                label_file.write(f'\n0 {col_center} {row_center} {width} {height}')
                        
                                            
        idx += num_records_per_patient
            
        # else:
        #     idx+=1
