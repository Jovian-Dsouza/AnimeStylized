'''
This scripts applies Guassian smooth filter to all the cartoon images on the cvpr_dataset
'''
from email.policy import default
import os 
import shutil
from glob import glob
import numpy as np
import cv2
from tqdm.auto import tqdm

def setup_dir(dir):
    shutil.rmtree(dir, ignore_errors=True)
    os.mkdir(dir)

def make_edge_smooth(input_dir, save_dir, img_size=256) :
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    file_list = glob(os.path.join(input_dir, '*'))

    for f in tqdm(file_list) :
        file_name = os.path.basename(f)

        bgr_img = cv2.imread(f)
        gray_img = cv2.imread(f, 0)

        bgr_img = cv2.resize(bgr_img, (img_size, img_size))
        pad_img = np.pad(bgr_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
        gray_img = cv2.resize(gray_img, (img_size, img_size))

        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(bgr_img)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

        cv2.imwrite(os.path.join(save_dir, file_name), gauss_img)

if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset_path', 
                        default='../DATASET/cvpr_dataset',
                        help='dataset path to the cvpr dataset',
                        type=str)
    parse.add_argument('--image_size', type=int, default=256)
    args = parse.parse_args()

    root = args.dataset_path
    img_size = args.image_size
    
    for folder in glob(os.path.join(root, '*_cartoon')):
        folder_basename = os.path.basename(folder)
        save_dir = os.path.join(root, f'{folder_basename}_smooth')
        setup_dir(save_dir)

        for style in os.listdir(folder):
            style_save_dir = os.path.join(save_dir, style)
            setup_dir(style_save_dir)
            make_edge_smooth(
                input_dir=os.path.join(folder, style),
                save_dir=style_save_dir,
                img_size=img_size
            )
