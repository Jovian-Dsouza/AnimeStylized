from torchvision.datasets import VisionDataset
from datamodules.dsfunction import imread
from torch.utils.data import Dataset, RandomSampler, Sampler, DataLoader, TensorDataset, random_split, ConcatDataset
import os
from glob import glob
from typing import List, Sequence, Tuple
from itertools import cycle, islice
import torch
from math import ceil
from torchvision.io import read_image
import pickle
from copy import copy, deepcopy
import torchvision.transforms.functional as F
import kornia as K

from typing import *

class DataFolder(VisionDataset):
  def __init__(self, root, loader: callable, pattern: str, transforms=None, transform=None, target_transform=None):
    super().__init__(root, transforms, transform, target_transform)
    self.loader = loader
    self.samples = glob.glob(os.path.join(root, pattern))

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, index: int):
    path = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
      sample = self.transform(**sample)
    return sample

  def size(self, idx):
    return len(self.samples)


# class ImageFolder(VisionDataset):
#   def __init__(self, root, transforms=None, transform=None, target_transform=None):
#     super().__init__(root, transforms, transform, target_transform)
#     self.loader = imread
#     self.samples = os.listdir(root)

#   def __len__(self) -> int:
#     return len(self.samples)

#   def __getitem__(self, index: int):
#     path = self.samples[index]
#     sample = self.loader(self.root + '/' + path)
#     if self.transform is not None:
#       sample = self.transform(sample)

#     return sample

#   def size(self, idx):
#     return len(self.samples)


class ImageFolder(Dataset):
    def __init__(self, root, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.samples = []
        for ext in ['jpg', 'png', 'jpeg']:
            self.samples += glob(os.path.join(root, f'*.{ext}'))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path = self.samples[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img

class CartoonFolder(Dataset):
    def __init__(self, root, smooth, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.samples = os.listdir(root)
        self.root = root
        self.smooth = smooth

        assert os.listdir(root) == os.listdir(smooth), "\nNo. of image in root dir and smooth dir should be the same\n"
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_name = self.samples[index]
        img_path = os.path.join(self.root, img_name)
        smooth_path = os.path.join(self.smooth, img_name)
        img = read_image(img_path)
        smooth_img = read_image(smooth_path)

        if self.transform is not None:
            img = self.transform(img)
            smooth_img = self.transform(smooth_img)
            smooth_img = K.color.rgb_to_grayscale(smooth_img).repeat(3,1,1)
        
        gray = deepcopy(img)
        gray = K.color.rgb_to_grayscale(gray).repeat(3,1,1)
        
        return img, gray, smooth_img


class ImagePaths(VisionDataset):
  def __init__(self, paths=List[str], transforms=None, transform=None, target_transform=None):
    super().__init__('.', transforms, transform, target_transform)
    self.loader = imread
    self.samples = paths

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, index: int):
    path = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
      sample = self.transform(sample)

    return sample

  def size(self, idx):
    return len(self.samples)


class MergeDataset(Dataset):
  def __init__(self, *tensors):
    """Merge two dataset to one Dataset
    """
    self.tensors = tensors
    self.sizes = [len(tensor) for tensor in tensors]

  def __getitem__(self, indexs: List[int]):
    return tuple(tensor[idx] for idx, tensor in zip(indexs, self.tensors))

  def __len__(self):
    return max(self.sizes)


class MultiRandomSampler(RandomSampler):
  def __init__(self, data_source: MergeDataset, replacement=True, num_samples=None, generator=None):
    """ a Random Sampler for MergeDataset. NOTE will padding all dataset to same length

    Args:
        data_source (MergeDataset): MergeDataset object
        replacement (bool, optional): shuffle index use replacement. Defaults to True.
        num_samples ([type], optional): Defaults to None.
        generator ([type], optional): Defaults to None.
    """
    self.data_source: MergeDataset = data_source
    self.replacement = replacement
    self._num_samples = num_samples
    self.generator = generator
    self.maxn = len(self.data_source)

  @property
  def num_samples(self):
    # dataset size might change at runtime
    if self._num_samples is None:
      self._num_samples = self.data_source.sizes
    return self._num_samples

  def __iter__(self):
    rands = []
    for size in self.num_samples:
      if self.maxn == size:
        rands.append(torch.randperm(size, generator=self.generator).tolist())
      else:
        rands.append(torch.randint(high=size, size=(self.maxn,),
                                   dtype=torch.int64, generator=self.generator).tolist())
    return zip(*rands)

  def __len__(self):
    return len(self.data_source)


class MultiSequentialSampler(Sampler):
  r"""Samples elements sequentially, always in the same order.
      NOTE: it whill expand all dataset to same length

  Arguments:
      data_source (Dataset): dataset to sample from
  """

  def __init__(self, data_source: MergeDataset):
    self.data_source: MergeDataset = data_source
    self.num_samples = data_source.sizes
    self.maxn = len(data_source)

  def __iter__(self):
    ls = []
    for size in self.num_samples:
      if self.maxn == size:
        ls.append(range(size))
      else:
        ls.append(islice(cycle(range(size)), self.maxn))
    return zip(*ls)

  def __len__(self):
    return len(self.data_source)


class MultiBatchDataset(MergeDataset):
  """MultiBatchDataset for MultiBatchSampler
    NOTE inputs type must be MergeDataset
  """

  def __getitem__(self, indexs: List[int]):
    dataset_idxs, idxs = indexs
    return self.tensors[dataset_idxs][idxs]


class MultiBatchSampler(Sampler):
  r"""Sample another sampler by repeats times of mini-batch indices.
    NOTE always drop last !
  Args:
      samplers (Sampler or Iterable): Base sampler. Can be any iterable object
          with ``__len__`` implemented.
      repeats (list): repeats time
      batch_size (int): Size of mini-batch.
  """

  def __init__(self, samplers: list, repeats: list, batch_size):
    # Since collections.abc.Iterable does not check for `__getitem__`, which
    # is one way for an object to be an iterable, we don't do an `isinstance`
    # check here.
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
            batch_size <= 0:
      raise ValueError("batch_size should be a positive integer value, "
                       "but got batch_size={}".format(batch_size))

    assert len(samplers) == len(repeats), 'Samplers number must equal repeats number'

    minweight = min(repeats)
    minlength = len(samplers[repeats.index(minweight)])
    self.sampler_loop = cycle([i for i, w in enumerate(repeats) for _ in range(w)])
    # expand to target length
    self.repeats = repeats
    self.sizes = [minlength * ceil(w / minweight) for w in repeats]
    self.size = sum(self.sizes)
    self.batch_size = batch_size
    self.samplers: List[Sampler] = samplers
    self.new_samplers = []

  def __iter__(self):
    self.new_samplers.clear()
    self.new_samplers = [islice(cycle(smp), size)
                         for smp, size in
                         zip(self.samplers, self.sizes)]
    return self

  def __next__(self):
    # NOTE sampler_idx choice dataset
    sampler_idx = next(self.sampler_loop)
    sampler: Sampler = self.new_samplers[sampler_idx]
    return [(sampler_idx, next(sampler)) for _ in range(self.batch_size)]

  def __len__(self):
    # NOTE find min batch scale factor
    scale = ((min(self.sizes) // self.batch_size) // min(self.repeats))
    return sum([n * scale for n in self.repeats])
