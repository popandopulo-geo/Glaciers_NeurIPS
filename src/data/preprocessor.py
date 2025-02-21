import os
import os.path as osp
import csv
import multiprocessing
import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from typing import Optional, Dict

from src.config import Config
from src.logger import ColoredLogger

# Set up logging
logger = ColoredLogger().get_logger()

import os
import os.path as osp
import csv
from typing import List, Optional
import multiprocessing

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import Resampling

from scipy.interpolate import griddata

from src.logger import ColoredLogger
from src.config import Config  # Assuming your config class is in src/config.py

# Set up logging
logger = ColoredLogger().get_logger()

class LandsatPrerocessor:
    """
    A class to process Landsat raster files by resizing, filtering bands,
    normalizing them (with cubic interpolation of NaN values),
    and computing NDSI.
    """

    def __init__(
        self,
        config: Config,
        num_workers: Optional[int] = None
    ):
        """
        Initializes the LandsatProcessor with the specified parameters.
        
        Args:
            config (Config): Configuration object with parameters.
            num_workers (Optional[int]): Number of parallel workers for processing.
        """
        self.config = config
        self.num_workers = num_workers or multiprocessing.cpu_count()

        os.makedirs(self.config.paths.processed, exist_ok=True)
        logger.info(f"Created output directory: {self.config.paths.processed}")

    def process_all(self):
        """
        Processes all TIFF files in the input directory in parallel.
        """
        logger.info("Starting parallel processing of Landsat images...")
        tif_files = [os.path.join(self.config.paths.raw, f)
                     for f in os.listdir(self.config.paths.raw) if f.endswith('.tif')]

        if self.num_workers > 1:
            with multiprocessing.Pool(self.num_workers) as pool:
                pool.map(self._process_single, tif_files)
        else:
            for tif_file in tif_files:
                self._process_single(tif_file)

    def _process_single(self, tif_file: str):
        """
        Processes a single TIFF file: resizing, interpolating NaNs with cubic interpolation,
        performing min–max normalization on each band, and computing NDSI.
        
        The normalized bands and the NDSI are saved to the output file.
        
        Args:
            tif_file (str): Path to the input raster file.
        """
        try:
            logger.info(f"Processing file: {tif_file}")
            with rasterio.open(tif_file) as src:
                # Create a dictionary mapping band description to its index (1-indexed)
                band_indexes = {desc: i + 1 for i, desc in enumerate(src.descriptions)}
                
                if not set(self.config.rasters.bands).issubset(band_indexes.keys()):
                    logger.warning(f"Skipping {tif_file}: Required bands missing.")
                    return

                # Create output transform using the bounding box and desired output size.
                transform = from_bounds(
                    *self.config.rasters.bounding_box,
                    self.config.rasters.size[0],
                    self.config.rasters.size[1]
                )
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": self.config.rasters.size[1],
                    "width": self.config.rasters.size[0],
                    "transform": transform,
                    "count": len(self.config.rasters.bands) + 1  # Adding NDSI as an additional band
                })
                
                output_path = os.path.join(self.config.paths.processed, os.path.basename(tif_file))
                with rasterio.open(output_path, "w", **out_meta) as dst:
                    # Process and write each normalized band.
                    processed_bands = {}
                    for i, band_name in enumerate(self.config.rasters.bands, start=1):
                        data = src.read(
                            band_indexes[band_name],
                            out_shape=self.config.rasters.size,
                            resampling=Resampling.bilinear
                        ).astype(np.float32)
                        # Interpolate NaN values using cubic interpolation.
                        data_filled = LandsatPrerocessor._fill_nans_cubic(data)
                        # Normalize the band using min–max normalization.
                        data_norm = LandsatPrerocessor._minmax_normalize(data_filled)
                        dst.write(data_norm, i)
                        processed_bands[band_name] = data_norm
                    
                    # Compute NDSI (Normalized Difference Snow Index) using normalized SR_B3 and SR_B6.
                    sr_b3 = src.read(
                        band_indexes['SR_B3'],
                        out_shape=self.config.rasters.size,
                        resampling=Resampling.bilinear
                    ).astype(np.float32)
                    sr_b3 =LandsatPrerocessor._fill_nans_cubic(sr_b3)
                    sr_b3_norm = LandsatPrerocessor._minmax_normalize(sr_b3)

                    sr_b6 = src.read(
                        band_indexes['SR_B6'],
                        out_shape=self.config.rasters.size,
                        resampling=Resampling.bilinear
                    ).astype(np.float32)
                    sr_b6 = LandsatPrerocessor._fill_nans_cubic(sr_b6)
                    sr_b6_norm = LandsatPrerocessor._minmax_normalize(sr_b6)
                    
                    ndsi = (sr_b3_norm - sr_b6_norm) / (sr_b3_norm + sr_b6_norm + 1e-10)
                    dst.write(ndsi, len(self.config.rasters.bands) + 1)

                logger.info(f"Successfully processed: {output_path}")
        except Exception as e:
            logger.error(f"Error processing {tif_file}: {e}")

    @staticmethod
    def _fill_nans_cubic(arr: np.ndarray) -> np.ndarray:
        """
        Fill NaN values in a 2D array using cubic interpolation.
        Only the NaN values are interpolated; any remaining NaNs (due to extrapolation)
        are filled using nearest-neighbor interpolation.
        
        Args:
            arr (np.ndarray): 2D array with potential NaN values.
        
        Returns:
            np.ndarray: Array with NaNs replaced by interpolated values.
        """
        y, x = np.indices(arr.shape)
        valid = ~np.isnan(arr)
        if np.all(valid):
            return arr
        # Coordinates where values are valid
        points = np.column_stack((x[valid], y[valid]))
        values = arr[valid]
        # Interpolate on the full grid using cubic interpolation
        arr_filled = griddata(points, values, (x, y), method='cubic')
        # For any points still NaN, use nearest neighbor interpolation
        nan_mask = np.isnan(arr_filled)
        if np.any(nan_mask):
            arr_filled[nan_mask] = griddata(points, values, (x[nan_mask], y[nan_mask]), method='nearest')
        return arr_filled

    @staticmethod
    def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
        """
        Applies min–max normalization to a 2D array.
        
        Args:
            arr (np.ndarray): Input 2D array.
        
        Returns:
            np.ndarray: Normalized array scaled to the [0, 1] range.
        """
        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)
        # Avoid division by zero
        if np.abs(arr_max - arr_min) < 1e-10:
            return arr
        return (arr - arr_min) / (arr_max - arr_min)



class LandsatStatisticsCalculator:
    """
    A class to calculate global 5th and 95th quantiles for specified bands
    across all TIFF files in a given directory.
    """

    def __init__(self, config, num_workers: Optional[int] = None):
        """
        Initializes the LandsatStatisticsCalculator with the given configuration.

        Args:
            config: A configuration object containing paths and parameters.
                    Expected attributes:
                      - config.paths.raw: Directory with input TIFF files.
                      - config.paths.statistics: Directory to save statistics CSV.
                      - config.rasters.bands: List of band names to compute statistics for.
            num_workers (Optional[int]): Number of parallel worker processes.
        """
        self.config = config
        self.num_workers = num_workers or multiprocessing.cpu_count()
        # Create the directory for storing statistics if it doesn't exist.
        os.makedirs(self.config.paths.stats, exist_ok=True)
        logger.info(f"Created output directory: {self.config.paths.stats}")

    def compute_statistics(self, num_bins: int = 1000) -> Dict[str, tuple]:
        """
        Computes global 5th and 95th quantiles for each specified band across all TIFF files
        found in the raw data directory.

        Args:
            num_bins (int): Number of bins to use for histogram computation.

        Returns:
            quantiles (Dict[str, tuple]): A dictionary where the key is the band name
                                          and the value is a tuple (q5, q95) representing
                                          the 5th and 95th quantiles.
        """
        raw_dir = self.config.paths.raw
        tif_files = [osp.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.tif')]
        if not tif_files:
            logger.error("No raster files found in the directory.")
            return {}

        global_min = {}
        global_max = {}

        # First pass: compute global minimum and maximum for each specified band.
        for tif_file in tif_files:
            try:
                with rasterio.open(tif_file) as src:
                    # Retrieve band descriptions from the TIFF file
                    descriptions = src.descriptions
                    for band in range(1, src.count + 1):
                        # Get the band name from the file's descriptions
                        band_name = descriptions[band - 1]
                        # Process only bands specified in the config
                        if band_name not in self.config.rasters.bands:
                            continue

                        data = src.read(band, out_dtype=float)
                        valid_mask = ~np.isnan(data)
                        if not np.any(valid_mask):
                            continue
                        band_min = data[valid_mask].min()
                        band_max = data[valid_mask].max()
                        if band_name in global_min:
                            global_min[band_name] = min(global_min[band_name], band_min)
                            global_max[band_name] = max(global_max[band_name], band_max)
                        else:
                            global_min[band_name] = band_min
                            global_max[band_name] = band_max
            except Exception as e:
                logger.error(f"Error processing file {tif_file} during min/max computation: {e}")
                continue

        # Initialize histograms for each specified band.
        histograms = {}
        bin_edges = {}
        for band_name, bmin in global_min.items():
            bmax = global_max[band_name]
            edges = np.linspace(bmin, bmax, num_bins + 1)
            histograms[band_name] = np.zeros(num_bins, dtype=np.int64)
            bin_edges[band_name] = edges

        # Second pass: accumulate histograms for each file.
        for tif_file in tif_files:
            try:
                with rasterio.open(tif_file) as src:
                    descriptions = src.descriptions
                    for band in range(1, src.count + 1):
                        band_name = descriptions[band - 1]
                        if band_name not in self.config.rasters.bands:
                            continue
                        data = src.read(band, out_dtype=float)
                        valid_mask = ~np.isnan(data)
                        if not np.any(valid_mask):
                            continue
                        hist, _ = np.histogram(data[valid_mask], bins=bin_edges[band_name])
                        histograms[band_name] += hist
            except Exception as e:
                logger.error(f"Error processing file {tif_file} during histogram accumulation: {e}")
                continue

        quantiles = {}
        # Compute quantiles from the accumulated histograms.
        for band_name, hist in histograms.items():
            total = hist.sum()
            if total == 0:
                logger.warning(f"No valid data for band {band_name}.")
                continue
            cum_hist = np.cumsum(hist)
            q5_index = np.searchsorted(cum_hist, total * 0.05)
            q95_index = np.searchsorted(cum_hist, total * 0.95)
            q5_value = bin_edges[band_name][q5_index]
            q95_index = min(q95_index, len(bin_edges[band_name]) - 1)
            q95_value = bin_edges[band_name][q95_index]
            quantiles[band_name] = (q5_value, q95_value)
            logger.info(f"Band {band_name}: 5th quantile = {q5_value}, 95th quantile = {q95_value}")

        return quantiles

    def save_statistics(self, quantiles: Dict[str, tuple], filename: Optional[str] = "statistics.csv"):
        """
        Saves the computed quantiles to a CSV file. Each row contains the band name,
        the 5th quantile, and the 95th quantile.

        Args:
            quantiles (Dict[str, tuple]): Dictionary with band names as keys and
                                          (q5, q95) as values.
            filename (Optional[str]): Name of the CSV file to save the statistics.
        """
        filepath = osp.join(self.config.paths.stats, filename)
        try:
            with open(filepath, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["band_name", "q5", "q95"])
                for band_name, (q5, q95) in quantiles.items():
                    writer.writerow([band_name, q5, q95])
            logger.info(f"Statistics saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving statistics to CSV: {e}")


# Note on tqdm integration:
# Yes, you can integrate tqdm progress bars with logger messages.
# For example, you can wrap loops with tqdm as follows:
#
#     from tqdm import tqdm
#
#     for tif_file in tqdm(tif_files, desc="Processing files"):
#         # Process each file and log progress as needed.
#
# Alternatively, you can use 'tqdm.contrib.logging' to attach a handler to your logger.
# Just be cautious to avoid conflicts between progress bar output and logger messages.


# TODO: Add statistics computation
# TODO: Add processign steps with statistics
# TODO: Refine Splitter class