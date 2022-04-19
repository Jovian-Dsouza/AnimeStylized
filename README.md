# AnimeGAN Pytorch Implementation

## Environment
```
conda create -n torch python=3.8
conda activate torch
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install pytorch-lightning==1.0.2 opencv-python matplotlib joblib scikit-image torchsummary webdataset albumentations more_itertools

pip install kornia pytorch-lightning opencv-python matplotlib joblib scikit-image torchsummary webdataset albumentations more_itertools 
```

## Download Pretrained weights for VGG19 network
[link](https://drive.google.com/u/0/uc?id=1j0jDENjdwxCDb36meP6-u5xDBzmKBOjJ&export=download)
```
cd /content/AnimeStylized

mkdir models
cd models
gdown --id 1j0jDENjdwxCDb36meP6-u5xDBzmKBOjJ
mv vgg19_no_fc.npy vgg19.npy
%cd ..
```

## How to train the model 

### Generate Smooth Edge 
```
cd misc_scripts
python smooth_edge.py 
```

### Pretrain the Generator
```
make train CODE=scripts/animegan_pretrain.py CFG=configs/animegan_pretrain_colab.yaml
```

### Train the model 
```
make train CODE=scripts/animeganv2.py CFG=configs/animeganv2_colab.yaml CKPT={ckpt_file}
```

## Inference
```
make infer CODE=scripts/animeganv2.py \
CKPT={ckpt_file} \
EXTRA=image_path:asset/animegan_test2.jpg 
```

## Download Dataset
```
gdown --id 10SGv_kbYhVLIC2hLlz2GBkHGAo0nec-3
mkdir cvpr_dataset
cd cvpr_dataset
7z x cvpr_dataset.zip 
cd ..
```