#!/bin/bash

# Runs the model on real images to generate cartoon images and then compute the FID Score between them
# Requirements: pip install pytorch_fid

ckpt_file="checkpoints/animegan/epoch=9-step=21749.ckpt"
real_path="DATASET/cvpr_test/real/"
gen_path="DATASET/cvpr_test/real_out/"
cartoon_path="DATASET/cvpr_test/cartoon/"

device="cuda"
make infer CODE=scripts/animeganv2.py CKPT=$ckpt_file EXTRA=image_path:$real_path,device:$device

echo "FID between $real_path : $gen_path"
python -m pytorch_fid $real_path $gen_path --device $device
echo ""
echo "FID between $cartoon_path : $gen_path"
python -m pytorch_fid $cartoon_path $gen_path --device $device
