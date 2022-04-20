#!/bin/bash

ckpt_file="/home/jovian/AnimeStylized/checkpoints/animegan/epoch=9-step=21749.ckpt"
img_path="/home/jovian/AnimeStylized/DATASET/test"
device="cuda"
make infer CODE=scripts/animeganv2.py CKPT=$ckpt_file EXTRA=image_path:$img_path,device:$device