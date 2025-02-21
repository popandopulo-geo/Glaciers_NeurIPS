import os
import numpy as np
import rasterio
import multiprocessing
from PIL import Image

from src.logger import ColoredLogger
from . import Preprocessor

# Initialize logger
logger = ColoredLogger().get_logger()

class LandsatRGBGenerator:
    def __init__(self, config, num_workers):
        """
        Initializes the RGB generator.

        Args:
            config: A configuration object with the following attributes:
                - config.paths.raw: Directory containing the raw TIFF files.
                - config.paths.rgb: Directory to save the RGB PNG files.
            num_workers: Number of worker processes for parallel processing.
        """
        self.config = config
        self.num_workers = num_workers
        self.input_dir = config.paths.raw
        
        # Use specified output directory if available, otherwise create a "rgb" subfolder inside the input directory
        if hasattr(config.paths, "rgb"):
            self.output_dir = config.paths.rgb
        else:
            self.output_dir = os.path.join(self.input_dir, "rgb")
        os.makedirs(self.output_dir, exist_ok=True)

    def process_all(self):
        """
        Processes all TIFF files in the input directory to generate RGB images.
        """
        # List all .tif files in the input directory
        tif_files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if f.lower().endswith('.tif')]
        
        # Process files in parallel if multiple workers are specified
        if self.num_workers > 1:
            with multiprocessing.Pool(self.num_workers) as pool:
                pool.map(self._process_file, tif_files)
        else:
            for file_path in tif_files:
                self._process_file(file_path)

    def _process_file(self, file_path):
        """
        Processes a single TIFF file to generate an RGB image and saves it as a PNG file.

        Args:
            file_path: Path to the TIFF file.
        """
        try:
            with rasterio.open(file_path) as src:
                descriptions = src.descriptions
                # Get indices for the desired bands (rasterio indices start at 1)
                r_index = descriptions.index('SR_B4') + 1
                r = src.read(r_index)
                # r = Preprocessor._fill_nans_cubic(r)
                r = Preprocessor._minmax_normalize(r)

                g_index = descriptions.index('SR_B3') + 1
                g = src.read(g_index)
                # g = Preprocessor._fill_nans_cubic(g)
                g = Preprocessor._minmax_normalize(g)

                b_index = descriptions.index('SR_B2') + 1
                b = src.read(b_index)
                # b = Preprocessor._fill_nans_cubic(b)
                b = Preprocessor._minmax_normalize(b)

                # Stack bands into an RGB image
                rgb = np.stack([r, g, b], axis=-1)

                # Convert normalized values to 0-255 and cast to uint8
                rgb_uint8 = (rgb * 255).astype(np.uint8)

            # Construct the output file path with .png extension
            base_name = os.path.basename(file_path)
            name_without_ext, _ = os.path.splitext(base_name)
            output_path = os.path.join(self.output_dir, f"{name_without_ext}.png")
            Image.fromarray(rgb_uint8).save(output_path)
            logger.info(f"Saved RGB image: {output_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")

