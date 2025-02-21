import h5py
import numpy as np
import pandas as pd
import random
import json
from typing import List, Dict
from src.logger import ColoredLogger

# Set up logger
logger = ColoredLogger().get_logger()

class HDF5DataSplitter:
    """
    A class to split an HDF5 file containing time series data into train and validation splits,
    and to generate patch information for each time period.
    
    The HDF5 file is expected to have datasets with shape (date, channels, x, y) and an attribute 'timestamp'
    containing the dates corresponding to each slice.
    """
    
    def __init__(
        self,
        input_path: str,
        split_ratio: float = 0.8,
        patch_size: int = 128,
        num_patches_per_image: int = 10,
        sequence_length: int = 25,  # Number of images for the input (e.g., one year)
        prediction_length: int = 6,  # Number of images for the target (e.g., half a year)
    ):
        """
        Initializes the splitter with the specified parameters.
        """
        self.input_path = input_path
        self.split_ratio = split_ratio
        self.patch_size = patch_size
        self.num_patches_per_image = num_patches_per_image
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
        # Open the HDF5 file in read mode
        with h5py.File(input_path, 'r') as f:
            self.data = f['data'][:]  # Expecting shape (date, channels, x, y)
            
            # Load timestamps from HDF5 attributes
            if 'timestamp' in f.attrs:
                self.timestamps = np.array(f.attrs['timestamp'], dtype='datetime64')
            else:
                logger.error("HDF5 file must have an attribute 'timestamp' with date information.")
                raise ValueError("HDF5 file must have an attribute 'timestamp' with date information.")
        
        # Convert timestamps to pandas datetime
        self.dates = pd.to_datetime(self.timestamps)
        self.total_samples = len(self.dates)

        # Log successful initialization
        logger.info(f"Initialized HDF5DataSplitter with {self.total_samples} samples.")
        
    def split_data(self) -> None:
        """
        Splits the data indices into training and validation based on time.
        """
        logger.info("Splitting data into training and validation sets.")
        
        # Sort indices by date
        sorted_indices = np.argsort(self.dates)
        total_samples = len(sorted_indices)
        train_size = int(total_samples * self.split_ratio)
        train_indices = sorted_indices[:train_size]
        val_indices = sorted_indices[train_size:]
        
        logger.info(f"Data split. Train samples: {len(train_indices)}, Validation samples: {len(val_indices)}.")
        
        self.train_indices, self.val_indices = train_indices.tolist(), val_indices.tolist()

    def generate_patches(self) -> None:
        """
        Generates patches for training and validation sets.
        """
        self.train_patches = self._generate_patches(self.train_indices)
        self.val_patches = self._generate_patches(self.val_indices)
    
    def _generate_patches(self, indices: List[int]) -> List[Dict]:
        """
        Generates patch information for the given list of date indices.
        """
        logger.info(f"Generating patches for {len(indices)} dates.")
        
        patches = []
        patch_id = 0
        
        _, _, x_dim, y_dim = self.data.shape
        max_x = x_dim - self.patch_size
        max_y = y_dim - self.patch_size
        
        for date_idx in indices:
            if date_idx + self.sequence_length + self.prediction_length > self.total_samples:
                continue
            
            for _ in range(self.num_patches_per_image):
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                
                input_date_indices = list(range(date_idx, date_idx + self.sequence_length))
                target_date_indices = list(range(date_idx + self.sequence_length, date_idx + self.sequence_length + self.prediction_length))
                
                patch_info = {
                    'patch_id': patch_id,
                    'date_index': int(date_idx),
                    'date': str(self.dates[date_idx]),
                    'patch_boundary': {
                        'x_start': int(x),
                        'y_start': int(y),
                        'x_end': int(x + self.patch_size),
                        'y_end': int(y + self.patch_size)
                    },
                    'input_date_indices': input_date_indices,
                    'target_date_indices': target_date_indices
                }
                patches.append(patch_info)
                patch_id += 1
        
        logger.info(f"Generated {len(patches)} patches.")
        return patches

    def save_splits(self, train_output: str = 'train_splits.json', val_output: str = 'val_splits.json'):
        """
        Saves the patch splits for training and validation to JSON files.
        """
        logger.info("Saving train and validation splits to JSON files.")
        
        with open(train_output, 'w') as f:
            json.dump(self.train_patches, f, indent=4, default=str)
        with open(val_output, 'w') as f:
            json.dump(self.val_patches, f, indent=4, default=str)
        
        logger.info(f"Splits saved to {train_output} and {val_output}.")

if __name__ == "__main__":

    # Parameters for splitting
    hdf5_file_path = 'path_to_your_hdf5_file.h5'  # Replace with your actual HDF5 file path
    split_ratio = 0.8
    patch_size = 128
    num_patches_per_image = 10
    sequence_length = 25  # Number of images for the input sequence (e.g., one year)
    prediction_length = 6  # Number of images for the target sequence (e.g., next three months)
    
    # Log the start of the process
    logger.info("Initializing HDF5DataSplitter...")
    
    # Initialize the splitter
    splitter = HDF5DataSplitter(
        input_path=hdf5_file_path,
        split_ratio=split_ratio,
        patch_size=patch_size,
        num_patches_per_image=num_patches_per_image,
        sequence_length=sequence_length,
        prediction_length=prediction_length
    )

    splitter.split_data()
    splitter.generate_patches()
    splitter.save_splits(train_output='train_splits.json', val_output='val_splits.json')
