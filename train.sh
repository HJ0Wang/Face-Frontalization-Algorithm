#!/bin/bash
CUDA_VISIBLE_DEVICES=6 python train.py \
    -d mPIE_train.list  \
    -ns 1 \
    -b 32 \
    --sym_loss \
    -sym_w 1 \
    -id_w 20 \
    -c \
    --epochs 10\
    --outf ./out \
    --modelf ./out 
