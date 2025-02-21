import os
import os.path as osp
import csv
from typing import List, Optional

import pystac_client
import xarray as xr
import stackstac
import planetary_computer as pc
import rasterio
import multiprocessing

from rasterio.transform import from_origin
from src.logger import ColoredLogger

# Set up logging
logger = ColoredLogger().get_logger()


class LandsatDataFetcher:
    """
    A class to fetch Landsat data over multiple years with filtering based on the proportion of NaNs in the first band,
    and to save the results into separate raster files.
    """

    def __init__(
        self,
        config,
        parallel: bool = True,
        num_workers: Optional[int] = None,
        threshold: float = 0.3,
    ):
        """
        Initializes the LandsatDataFetcher with parameters from the configuration.

        Args:
            config: A configuration object with the following expected keys:
                - config.rasters.bounding_box: List[float] with [min_x, min_y, max_x, max_y].
                - config.rasters.years: List[int] of years to fetch.
                - config.rasters.bands: List[str] of band names; the first one is used for NaN filtering.
                - config.rasters.espg_code: Integer EPSG code.
                - config.paths.raw: Directory to store the fetched raster files.
            parallel (bool): Whether to use parallel processing.
            num_workers (Optional[int]): Number of worker processes.
            threshold (float): Maximum allowed NaN proportion in the first band.
        """
        self.config = config
        self.bounding_box = config.rasters.bounding_box
        self.years = config.rasters.years
        # Use the first band from the config for NaN filtering.
        self.first_band_name = config.rasters.bands[0]
        self.threshold = threshold
        # Note: The configuration key is 'espg_code' (assumed typo for 'epsg_code')
        self.epsg_code = config.rasters.espg_code
        # Use the raw path from config to store fetched rasters.
        self.output_dir = config.paths.raw
        self.chunk_size = {'time': 8, 'band': -1, 'y': 512, 'x': 512}
        self.parallel = parallel
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.metadata_file = os.path.join(config.paths.root, "metadata.csv")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Created directory for output files: {self.output_dir}")

        self.catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        logger.info("LandsatDataFetcher initialized.")

        manager = multiprocessing.Manager()
        self.lock = manager.Lock()

    def fetch_all_years(self):
        """
        Fetches data for all specified years and saves each scene as a separate raster file.
        """
        logger.info("Starting data fetching for all years...")

        if self.parallel:
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                pool.map(self._fetch_single_year, self.years)
        else:
            for year in self.years:
                self._fetch_single_year(year)

    def _fetch_single_year(self, year: int):
        """
        Fetches and processes data for a single year, saving each image as a raster.

        Args:
            year (int): The year to fetch data for.
        """
        try:
            logger.info(f"Fetching year: {year}")

            search = self.catalog.search(
                collections=['landsat-8-c2-l2'],
                bbox=self.bounding_box,
                datetime=f"{year}-01-01/{year + 1}-01-01",
                max_items=None
            )

            items = pc.sign(search)
            metadata_entries = []

            for item in items:
                try:
                    single_stack = stackstac.stack([item], bounds_latlon=self.bounding_box, epsg=self.epsg_code)

                    if self.first_band_name not in single_stack.band.values:
                        logger.warning(f"Band '{self.first_band_name}' not found in item {item.id}. Skipping.")
                        continue

                    nan_proportion = single_stack.isnull().sel(band=self.first_band_name).mean(dim=['y', 'x']).compute().item()

                    if nan_proportion > self.threshold:
                        logger.debug(f"Skipped item {item.id} with NaN proportion: {nan_proportion:.2%}")
                        continue

                    file_name = os.path.join(self.output_dir, f"{item.id}.tif")
                    band_names = list(single_stack.band.values)
                    self._save_raster(single_stack, file_name, band_names)

                    metadata_entries.append([file_name, item.datetime, item.properties.get('eo:cloud_cover')])
                    logger.info(f"Saved raster: {file_name}")

                except Exception as e:
                    logger.error(f"Error processing item {item.id}: {e}")
                    continue

            self._save_metadata(metadata_entries)
        except Exception as e:
            logger.error(f"Error fetching year {year}: {e}")

    def _save_raster(self, data: xr.DataArray, file_name: str, band_names: List[str]):
        """
        Saves an xarray dataset as a raster file using rasterio.

        Args:
            data (xr.DataArray): The xarray dataset.
            file_name (str): Output file name.
            band_names (List[str]): List of band names to set as descriptions.
        """
        data = data.squeeze()
        transform = from_origin(
            self.bounding_box[0],
            self.bounding_box[3],
            abs(self.bounding_box[2] - self.bounding_box[0]) / data.sizes['x'],
            abs(self.bounding_box[3] - self.bounding_box[1]) / data.sizes['y']
        )

        with rasterio.open(
            file_name, "w",
            driver="GTiff",
            height=data.sizes['y'],
            width=data.sizes['x'],
            count=data.sizes['band'],
            dtype=str(data.dtype),
            crs=f"EPSG:{self.epsg_code}",
            transform=transform,
        ) as dst:
            for i, band in enumerate(data.band.values):
                dst.write(data.sel(band=band).values, i + 1)
                dst.set_band_description(i + 1, band_names[i])

    def _save_metadata(self, metadata_entries: List[List]):
        """
        Safely saves metadata about the saved raster files to a CSV file.

        Args:
            metadata_entries (List[List]): List of metadata rows.
        """
        with self.lock:
            file_exists = os.path.exists(self.metadata_file)
            with open(self.metadata_file, mode="a", newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["file_name", "date", "cloud_coverage"])
                writer.writerows(metadata_entries)

        logger.info(f"Metadata saved to {self.metadata_file}")
