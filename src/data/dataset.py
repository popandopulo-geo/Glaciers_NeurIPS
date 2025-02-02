import torch
from torch.utils.data import Dataset, DataLoader

import os
import abc

class BaseDataset(Dataset):
  def __init__(self, root, split, transforms=None):
    super(BaseDataset, self).__init__()

    self.root = root
    self.transforms = transforms

  def __len__(self):
    pass

  @abc.abstractmethod
  def __getitem__(self, idx):
    pass

  @classmethod
  def get_dataloader(cls, root, split, transforms, bacth_size=1, shuffle=False):
    dataset = cls(root, split, transforms)
    dataloder = DataLoader(
      dataset=dataset,
      bacth_size=bacth_size,
      shuffle=shuffle
    )

    return dataloder

class MalariaDataset(BaseDataset):
  def __init__(self, root, split, transforms=None):
    super(MalariaDataset, self).__init__(root, split, transforms)

  def __getitem__(self, idx):
    image_id = self.image_ids[idx]
    group = self.grouped.get_group(image_id)