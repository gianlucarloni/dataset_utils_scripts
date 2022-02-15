#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 11:45:03 2022

@author: si-lab
"""
from PIL import Image
import numpy as np
import os

from glob import glob
from torchvision import transforms

for i in glob('/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ProtoPNet/previous_datasets/breast_mammo/original/task_CalcificationMass/CC/Setup con imageFolder/push/calcification/'+'*.png'):
    imm = Image.open(i)
    nome = os.path.basename(i)
    resize = transforms.Resize((224,224))
    
    imm2 = resize(imm)
    
    imm2.save(f'/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ProtoPNet/previous_datasets/breast_mammo/original/task_CalcificationMass/CC/Setup con imageFolder/push/calcification/resized/{nome}','PNG')