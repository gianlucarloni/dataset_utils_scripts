#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:59:53 2021

DETECTION AUTOMATICA DELLA BOUNDING BOX PER CROPPARE LE IMMAGINI
MAMMOGRAFICHE SULLA MAMMELLA ESCLUDENDO IL FONDO ARIA (in CT, HU=0)

idea:
    vedere, per ogni immagine, dove il profilo dei grigi lungo una linea droppa al valore zero
    segnarsi la colonna massima per quella immagine
    prendere la colonna massima delle colonne massime di tutto il set
    
    attenzione: leggere la colonna Left/Right per capire la lateralità del seno e ribaltare l'indice'
@author: si-lab
"""

import os
import glob
import numpy as np
import imageio
import pandas as pd
import argparse
from tqdm import tqdm
from PIL import Image
import shutil

parse = argparse.ArgumentParser(description="Calculating Boundig Box to crop original dataset")
parse.add_argument('png_dir', help='Path to the input directory with original PNG images')
parse.add_argument('csv_file', help='Path to the CSV file of original images')
# parse.add_argument('out_dir', help='Path to the outpu directory with cropped PNG images')


args = parse.parse_args()
png_dir = args.png_dir
csv_path = args.csv_file
# out_dir = args.out_dir
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)


csv_df = pd.read_csv(csv_path,sep=',', index_col='File name')


MARGINE = 10 #pixel di sicurezza

estremo_massimo = 0
img_max_name = ''
for im_path in tqdm(glob.glob(os.path.join(png_dir, '*.png'))):    
    patient_name = os.path.basename(im_path)
    if patient_name in ['D2-0440_1-2_MLO.png', 'D2-0607_1-4_MLO.png']: # in queste immagini si vede il braccio del paziente  
        continue
    im = np.array(imageio.imread(im_path))
    
    # estraggo la laterality
    laterality = csv_df.loc[patient_name,'LeftRight']
    
    row,col= im.shape
    
    for r in range(row):
        where = np.where(im[r,:]==0)[0] #vettore di indici dove valore=0
        if laterality=='L':
            estremo = where[0]
            if estremo > estremo_massimo:
                estremo_massimo = estremo
                img_max_name = patient_name
        elif laterality=='R':
            estremo = where[-1]
            temp= col-estremo
            if temp > estremo_massimo:
                estremo_massimo = temp
                img_max_name = patient_name

        else:
            print('Errore: lateralità sconosciuta')

print(f'Estremo massimo è indice di colonna: {estremo_massimo}, all immagine {img_max_name}')       

estremo_massimo = estremo_massimo + MARGINE
with open(os.path.join(png_dir,'max_column.txt'),'w') as file:
    file.write(str(estremo_massimo))

