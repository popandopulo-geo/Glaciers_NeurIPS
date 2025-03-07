import torch
from torch.utils.data import Dataset, DataLoader

import rasterio
from rasterio.windows import Window

import pandas as pd

import os
import abc
import json

class BaseDataset(Dataset):
  def __init__(self, config, split, transforms=None):
    super(BaseDataset, self).__init__()

    self.config = config
    self.split = split
    self.transforms = transforms

  def __len__(self):
    pass

  @abc.abstractmethod
  def __getitem__(self, idx):
    pass

  @classmethod
  def get_dataloader(cls, config, split, transforms, bacth_size=1, shuffle=False):
    dataset = cls(config, split, transforms)
    dataloder = DataLoader(
      dataset=dataset,
      bacth_size=bacth_size,
      shuffle=shuffle
    )

    return dataloder

class TiffDataset(BaseDataset):
    def __init__(self, config, split, transforms=None):
        """
        Initializes the TiffDataset.
        
        Args:
            config: Config object containing configuration parameters.
            split: String indicating which split to use ('train' or 'val').
            transforms: Optional transforms to apply to each image tensor.
        """
        super(TiffDataset, self).__init__(config, split, transforms)
        self.split = split.lower()

        # Determine the split file based on the split type.
        if self.split == "train":
            split_file = os.path.join(config.paths.splits, config.splits.train_split_file)
        elif self.split in ["val", "validation"]:
            split_file = os.path.join(config.paths.splits, config.splits.valid_split_file)
        else:
            raise ValueError("Invalid split specified. Choose 'train' or 'val'.")

        # Load patch information from the JSON split file.
        with open(split_file, 'r') as f:
            self.patches = json.load(f)
        
        # Load metadata to obtain dates for each image.
        metadata_path = os.path.join(config.paths.root, "metadata.csv")
        self.metadata_df = pd.read_csv(metadata_path)
        self.metadata_df['date'] = pd.to_datetime(self.metadata_df['date'])
        # Build a mapping from filename to its date.
        self.filename_to_date = {row['filename']: row['date'] for _, row in self.metadata_df.iterrows()}

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        """
        For the given index, loads input and target sequences.
        For each sequence, it reads the corresponding TIFF file using rasterio,
        crops the image based on the patch boundaries, selects only the NDSI band (the last band),
        converts it to a torch tensor, and computes temporal differences (in days) relative to the first image.
        
        Returns:
            A dictionary with:
                'input': list of input image tensors,
                'target': list of target image tensors,
                'input_temporal': list of day differences for input images,
                'target_temporal': list of day differences for target images,
                'date': the date (string) of the first image in the sequence.
        """
        patch_info = self.patches[idx]
        boundary = patch_info["patch_boundary"]
        x_start = boundary["x_start"]
        y_start = boundary["y_start"]
        patch_width = boundary["x_end"] - boundary["x_start"]
        patch_height = boundary["y_end"] - boundary["y_start"]

        def load_and_process(file_list):
            images = []
            temporal = []
            # Use the first image's date as the reference.
            first_file = os.path.basename(file_list[0])
            first_date = self.filename_to_date.get(first_file)
            for filepath in file_list:
                with rasterio.open(filepath) as src:
                    # Create a window using the patch boundaries.
                    window = Window(x_start, y_start, patch_width, patch_height)
                    image_array = src.read(window=window)  # shape: (channels, H, W)
                    # Select only the NDSI band (assumed to be the last band).
                    ndsi = image_array[-1, :, :][None, :, :]  # shape: (1, H, W)
                    image_tensor = torch.from_numpy(ndsi).float()
                    if self.transforms is not None:
                        image_tensor = self.transforms(image_tensor)
                    images.append(image_tensor)
                # Compute temporal difference (in days) relative to the first image.
                file_name = os.path.basename(filepath)
                current_date = self.filename_to_date.get(file_name)
                delta_days = (current_date - first_date).days if first_date is not None else 0
                temporal.append(delta_days)
            return images, temporal

        input_images, input_temporal = load_and_process(patch_info["input_filenames"])
        target_images, target_temporal = load_and_process(patch_info["target_filenames"])

        return {
            "input": input_images,
            "target": target_images,
            "input_temporal": input_temporal,
            "target_temporal": target_temporal,
            "date": patch_info["date"]
        }