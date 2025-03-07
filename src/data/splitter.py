import os
import json
import pandas as pd
import random
from src.logger import ColoredLogger
from src.config import Config

logger = ColoredLogger().get_logger()

class TiffDataSplitter:
    """
    Class for splitting a collection of .tif images with timestamps into training and validation sequences,
    as well as for generating random crops (patches) for neural network training.
    
    Input data:
        - metadata.csv (columns: filename, date, cloud_coverage)
        - filtered_images.csv (column: filename)
    All .tif files are located in config.paths.raw.
    """
    def __init__(self, config: Config):
        self.config = config
        
        # Set seed for reproducibility
        random.seed(config.seed)

        os.makedirs(self.config.paths.splits, exist_ok=True)
        logger.info(f"Created output directory: {self.config.paths.splits}")
        
        # Paths to the CSV files (metadata.csv and filtered_images.csv are located in config.paths.root)
        metadata_path = os.path.join(config.paths.root, "metadata.csv")
        filtered_path = os.path.join(config.paths.root, "filtered_images.csv")
        logger.info(f"Loading metadata from {metadata_path} and filtered images from {filtered_path}.")

        # Load metadata and filtered image list
        self.metadata = pd.read_csv(metadata_path)
        filtered_df = pd.read_csv(filtered_path)

        # Filter the metadata by the list of images (filename)
        filtered_filenames = set(filtered_df['filename'])
        self.metadata = self.metadata[self.metadata['filename'].isin(filtered_filenames)]

        # Convert the date column to datetime and sort by date
        self.metadata['date'] = pd.to_datetime(self.metadata['date'])
        self.metadata.sort_values('date', inplace=True)
        self.metadata.reset_index(drop=True, inplace=True)
        self.total_samples = len(self.metadata)
        logger.info(f"Number of images after filtering: {self.total_samples}.")

        # Add the full path for each .tif file (files are located in config.paths.raw)
        self.metadata['filepath'] = self.metadata['filename'].apply(
            lambda f: os.path.join(config.paths.raw, f)
        )

        # Parameters for splitting and patch generation
        self.train_test_ratio = config.splits.train_test_ratio
        self.patch_size = config.splits.patch_size
        self.num_patches_per_image = config.splits.num_patches_per_image
        self.input_sequence = config.input_sequence
        self.out_sequence = config.out_sequence

        # Image dimensions defined in the config (assumed order: width, height)
        self.width = config.rasters.size[0]
        self.height = config.rasters.size[1]

    def split_data(self) -> None:
        """
        Splits the sorted images into training and validation blocks.
        All available images are used; the first part (according to the ratio) is training,
        and the rest is validation.
        """
        logger.info("Splitting data into training and validation blocks.")
        train_size = int(self.total_samples * self.train_test_ratio)
        self.train_metadata = self.metadata.iloc[:train_size].reset_index(drop=True)
        self.val_metadata = self.metadata.iloc[train_size:].reset_index(drop=True)
        logger.info(f"Training block: {len(self.train_metadata)} images, Validation block: {len(self.val_metadata)} images.")

    def generate_patches(self) -> None:
        """
        Generates patch information (random crops) for the training and validation blocks.
        For each block, for every valid starting index (i.e. where a complete sequence can be formed),
        a fixed number of patches are generated. Sequences may overlap.
        """
        logger.info("Generating patches for the training block.")
        self.train_patches = self._generate_patches_from_block(self.train_metadata)
        logger.info("Generating patches for the validation block.")
        self.val_patches = self._generate_patches_from_block(self.val_metadata)

    def _generate_patches_from_block(self, block_data: pd.DataFrame) -> list:
        patches = []
        patch_id = 0
        block_size = len(block_data)
        # Calculate the maximum valid starting index in this block
        max_valid = block_size - (self.input_sequence + self.out_sequence) + 1
        if max_valid <= 0:
            logger.warning("Block size is too small to form a complete sequence.")
            return patches

        max_x = self.width - self.patch_size
        max_y = self.height - self.patch_size

        for start_idx in range(max_valid):
            for _ in range(self.num_patches_per_image):
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                input_files = block_data.iloc[start_idx : start_idx + self.input_sequence]['filepath'].tolist()
                target_files = block_data.iloc[start_idx + self.input_sequence : start_idx + self.input_sequence + self.out_sequence]['filepath'].tolist()
                patch_info = {
                    "patch_id": patch_id,
                    "start_index": start_idx,
                    "date": str(block_data.iloc[start_idx]['date']),
                    "patch_boundary": {
                        "x_start": x,
                        "y_start": y,
                        "x_end": x + self.patch_size,
                        "y_end": y + self.patch_size,
                    },
                    "input_filenames": input_files,
                    "target_filenames": target_files
                }
                patches.append(patch_info)
                patch_id += 1

        logger.info(f"Generated patches: {len(patches)}.")
        return patches

    def save_splits(self) -> None:
        """
        Saves the generated splits (patches) for the training and validation blocks to JSON files.
        The file paths are taken from config.paths.splits.
        """
        train_output = os.path.join(self.config.paths.splits, self.config.splits.train_split_file)
        val_output = os.path.join(self.config.paths.splits, self.config.splits.valid_split_file)
        logger.info(f"Saving training patches to {train_output} and validation patches to {val_output}.")

        with open(train_output, 'w') as f:
            json.dump(self.train_patches, f, indent=4)
        with open(val_output, 'w') as f:
            json.dump(self.val_patches, f, indent=4)

        logger.info("Saving completed.")