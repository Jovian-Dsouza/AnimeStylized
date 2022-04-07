import os
import sys
sys.path.insert(0, os.getcwd())

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from datamodules.dataset import *

# dataset output
# batch => input_photo, (input_cartoon, anime_gray_data), anime_smooth_gray_data 

class AnimeGanDataModule(pl.LightningDataModule):
    def __init__(self, root: str,
        scene_style: str = 'shinkai',
        face_style: str = 'pa_face',
        sample_steps: list = [4, 1], # scenery, face
        mean = (0.5, 0.5, 0.5),
        std = (0.5, 0.5, 0.5),
        train_val_split = 0.9,
        batch_size: int = 8, 
        num_workers = None,
        pin_memory = True,
    ):
        super().__init__()
        self.root = root
        self.scene_style = scene_style
        self.face_style = face_style
        self.sample_steps = sample_steps
        self.dims = (3, 256, 256)
        self.mean = mean
        self.std = std
        self.train_val_split = train_val_split  

        self.batch_size = batch_size
        self.num_workers = os.cpu_count()-1 if num_workers is None else num_workers
        self.pin_memory = pin_memory

        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x/255.0),
            transforms.Normalize(mean=mean, std=std),
        ])

    def setup(self, stage=None):
        # Scenery Dataset
        scenery_photo = ImageFolder(root=os.path.join(self.root, 'scenery_photo'), 
                                    transform=self.transform)
        n_scenery = len(scenery_photo)
        scenery_photo_train, scenery_photo_val = random_split(scenery_photo,
                                                                [int(n_scenery * self.train_val_split),
                                                                n_scenery - int(n_scenery * self.train_val_split)])
        scenery_cartoon = CartoonFolder(root=os.path.join(self.root, "scenery_cartoon", self.scene_style),
                                      smooth=os.path.join(self.root, "scenery_cartoon_smooth", self.scene_style), 
                                      transform=self.transform)
        scenery_ds = MergeDataset(scenery_photo_train, scenery_cartoon)

        # Face Dataset
        face_photo = ImageFolder(root=os.path.join(self.root, 'face_photo'),
                                 transform=self.transform)
        n_face = len(face_photo)
        face_photo_train, face_photo_val = random_split(face_photo,
                                                        [int(n_face * self.train_val_split),
                                                            n_face - int(n_face * self.train_val_split)])
        face_cartoon = CartoonFolder(root=os.path.join(self.root, "face_cartoon", self.face_style), 
                                   smooth=os.path.join(self.root, "face_cartoon_smooth", self.face_style),
                                   transform=self.transform)
        face_ds = MergeDataset(face_photo_train, face_cartoon)

        if stage == 'fit':
            self.ds_train = MultiBatchDataset(scenery_ds, face_ds)
            self.ds_sampler = MultiBatchSampler(
                [MultiRandomSampler(scenery_ds), MultiRandomSampler(face_ds)],
                self.sample_steps, self.batch_size)
            

        self.ds_val = scenery_photo_val + face_photo_val


    def train_dataloader(self):
        return DataLoader(self.ds_train,
                        batch_sampler=self.ds_sampler,
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.ds_val, 
                        shuffle=True,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory,
                        )

    def test_dataloader(self):
        return DataLoader(self.ds_val, 
                        shuffle=True,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory,
                        )

if __name__ == '__main__':
    from utils.utils import show_img
    
    datamodule = AnimeGanDataModule(
        root = '../cvpr_dataset',
        scene_style = 'shinkai',
        # scene_style = 'hayao',
        face_style = 'pa_face',
        batch_size = 2, 
        num_workers = 0,
    )
    datamodule.setup('fit')

    dl = datamodule.train_dataloader()
    # for img, (car_img, gray, smooth_img) in dl:
    #     break
    # print(img.shape, img.min(), img.max())
    # print(car_img.shape, car_img.min(), car_img.max())
    # print(gray.shape, gray.min(), gray.max())
    # print(smooth_img.shape, smooth_img.min(), smooth_img.max())
    # show_img(img[0])
    # show_img(car_img[0])
    # show_img(gray[0])
    # show_img(smooth_img[0])

