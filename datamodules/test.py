import sys
import os
sys.path.insert(0, os.getcwd())
import pytorch_lightning as pl

from dataset import *

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    root = '../cvpr_dataset'
    scene_style = 'shinkai'
    face_style = 'pa_face'

    train_val_split = 0.9

    # face_photo = ImageFolder(root=os.path.join(root, 'face_photo'))
    # n_face = len(face_photo)
    # face_photo_train, face_photo_val = random_split(face_photo,
    #                                                 [int(n_face * train_val_split),
    #                                                  n_face - int(n_face * train_val_split)])
    face_cartoon = ImageFolder(root=os.path.join(root, "face_cartoon", face_style))
    print('face_cartoon', len(face_cartoon))

    # face_ds = MergeDataset(face_cartoon, face_photo_train)

    # print('face_photo_train', len(face_photo_train))
    # print('face_photo_val', len(face_photo_val))
    # print('face_ds', len(face_ds))
    # img1, img2 = face_ds.__getitem__([0, 1])
    # print(img1.shape, img2.shape)
    # # print(img.shape, img.min(), img.max())