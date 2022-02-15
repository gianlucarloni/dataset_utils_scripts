#!/bin/bash

MAMMO_DIR=/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CMMD/png/original2

for x in {D1-0051,D1-0053,D1-0706,D1-0721,D1-1309,D1-1532,D2-0060}; do \
#mv $MAMMO_DIR/CC/${x}_1-1_CC.png $MAMMO_DIR/MLO/${x}_1-1_MLO.png &&\
#mv $MAMMO_DIR/MLO/${x}_1-2_MLO.png $MAMMO_DIR/CC/${x}_1-2_CC.png &&\
sed -i "s/${x}_1-1_CC.png/${x}_1-2_CC.png/g" $MAMMO_DIR/CC/labels_abnormality.csv &&\
sed -i "s/${x}_1-2_MLO.png/${x}_1-1_MLO.png/g" $MAMMO_DIR/MLO/labels_abnormality.csv; done

for x in {D1-0021,D2-0635}; do \
#mv $MAMMO_DIR/CC/${x}_1-3_CC.png $MAMMO_DIR/MLO/${x}_1-3_MLO.png &&\
#mv $MAMMO_DIR/MLO/${x}_1-4_MLO.png $MAMMO_DIR/CC/${x}_1-4_CC.png &&\
sed -i "s/${x}_1-3_CC.png/${x}_1-4_CC.png/g" $MAMMO_DIR/CC/labels_abnormality.csv &&\
sed -i "s/${x}_1-4_MLO.png/${x}_1-3_MLO.png/g" $MAMMO_DIR/MLO/labels_abnormality.csv; done

