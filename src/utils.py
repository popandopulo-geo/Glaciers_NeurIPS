from typing import List
import os
import os.path as osp

import rasterio
import numpy as np
import torch
import random

from src.config import Config
from src.logger import ColoredLogger

# Set up logger
logger = ColoredLogger().get_logger()

def setup_seed(seed: int):
    # Set seed for random module
    random.seed(seed)
    
    # Set seed for numpy
    np.random.seed(seed)
    
    # Set seed for torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For GPU
    
    # Set deterministic behavior for reproducibility in torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_global_statistics(config: Config):
    """
    Computes global statistics (min, max, percentiles) across all raster files for specified bands.
    """
    logger.info("Starting global statistics computation...")
    raster_files = [osp.join(config.paths.raw, f) for f in os.listdir(config.paths.raw) if f.endswith('.tif')]
    
    global_stats = {band: {'min': float('inf'), 'max': float('-inf'), 'percentiles': []} for band in config.rasters.bands}
    all_band_values = {band: [] for band in config.rasters.bands}  # Store data for percentile computation
    
    for file in raster_files:
        try:
            with rasterio.open(file) as src:
                band_desc = {desc: i+1 for i, desc in enumerate(src.descriptions)}
                for band in config.rasters.bands:
                    if band in band_desc:
                        data = src.read(band_desc[band]).astype(np.float32)
                        data = data[~np.isnan(data)]  # Remove NaNs
                        if data.size > 0:
                            global_stats[band]['min'] = min(global_stats[band]['min'], np.min(data))
                            global_stats[band]['max'] = max(global_stats[band]['max'], np.max(data))
                            all_band_values[band].extend(data.tolist())
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
            continue
    
    # Compute percentiles
    for band in config.rasters.bands:
        if all_band_values[band]:
            global_stats[band]['percentiles'] = np.percentile(all_band_values[band], [5, 25, 50, 75, 95]).tolist()
    
    logger.info("Global statistics computed successfully.")
    return global_stats