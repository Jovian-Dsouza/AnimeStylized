'''
This scipt selects random portion of the dataset and copies to separate directory
'''
from copy import copy
import os
import sys
sys.path.insert(0, os.getcwd())
from glob import glob
import pytorch_lightning as pl
import numpy as np
import argparse
import shutil
from shutil import copy2
from tqdm.auto import tqdm

def get_image_paths(folder):
    samples = []
    for ext in ['jpg', 'png', 'jpeg']:
        samples += glob(os.path.join(folder, f'*.{ext}'))
    return samples

def setup_dir(folder_name, sub_folders=[]):
    try:
        shutil.rmtree(folder_name)
    except FileNotFoundError:
        pass
    os.mkdir(folder_name)
    for sub_folder in sub_folders:
        os.mkdir(os.path.join(folder_name, sub_folder))

def copy_files_to_dir(file_list, output_dir):
    for file_path in tqdm(file_list, desc="Copying"):
        copy2(file_path, output_dir)

if __name__ == '__main__':
    np.random.seed(137)

    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset_path', 
                        default='DATASET/cvpr_dataset',
                        help='dataset path to the cvpr dataset',
                        type=str)
    parse.add_argument('--output_dir', type=str, default='DATASET/cvpr_test')
    parse.add_argument('--num_selection', type=int, default=5000)
    args = parse.parse_args()

    root = args.dataset_path
    scene_style = 'shinkai'
    # scene_style = 'hayao'
    face_style = 'pa_face'
    
    num_selection = args.num_selection
    output_dir = args.output_dir

    scenery_photo = get_image_paths(os.path.join(root, 'scenery_photo'))
    scenery_cartoon = get_image_paths(os.path.join(root, 'scenery_cartoon', scene_style))
    face_photo = get_image_paths(os.path.join(root, 'face_photo'))
    face_cartoon = get_image_paths(os.path.join(root, "face_cartoon", face_style))

    real_images = scenery_photo + face_photo
    cartoon_images = scenery_cartoon + face_cartoon

    print("No. of real images = ", len(real_images))
    print("No. of cartoon images = ", len(cartoon_images))

    real_selected = np.random.choice(real_images, num_selection, replace=(True if len(real_images) <= num_selection else False))
    cartoon_selected = np.random.choice(cartoon_images, num_selection, replace=(True if len(cartoon_images) <= num_selection else False))
    
    print("No. of real images = ", len(real_selected))
    print("No. of cartoon images = ", len(cartoon_selected))

    setup_dir(output_dir, ['real', 'cartoon'])
    copy_files_to_dir(real_selected, os.path.join(output_dir, 'real'))
    copy_files_to_dir(cartoon_selected, os.path.join(output_dir, 'cartoon'))