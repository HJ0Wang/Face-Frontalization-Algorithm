#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2 python evaluate_outinitname2p.py \
    -d test.list \
    --outf result \
    --modelf out
