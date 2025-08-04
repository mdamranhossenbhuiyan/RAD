#!/bin/bash

tau=0.015 
margin=0.1
noisy_rate=0.0  #0.0 0.2 0.5 0.8
select_ratio=0.3
loss=TAL
DATASET_NAME=CUHK-PEDES
# CUHK-PEDES ICFG-PEDES RSTPReid

noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy
CUDA_VISIBLE_DEVICES=5 \
    python train.py \
    --noisy_rate $noisy_rate \
    --noisy_file $noisy_file \
    --name RDE \
    --img_aug \
    --txt_aug \
    --batch_size 96 \
    --select_ratio $select_ratio \
    --tau $tau \
    --root_dir 'Path to Datasets Directory/' \
    --pretrain_choice "ViT-B/16" \
    --teacher_choice  "ViT-L/14" \
    --teacher_ckpt 'Path to Trained Model in Stage-01/best.pth' \
    --output_dir run_logs \
    --margin $margin \
    --dataset_name $DATASET_NAME \
    --loss_names ${loss}+sr${select_ratio}_tau${tau}_margin${margin}_n${noisy_rate}_RAD \
    --num_epoch 60 \
    --rel_kd_weight 1.0 \
    --lr   2e-5 \
    --distillation
