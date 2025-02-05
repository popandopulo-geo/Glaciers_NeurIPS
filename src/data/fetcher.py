import os
import logging
from typing import List, Optional

import xarray as xr
import stackstac
import planetary_computer as pc
import multiprocessing


class LandsatDataProcessor:
    """
    A class to process Landsat data over multiple years with filtering based on the proportion of NaNs in the first band
    and saving the results into a single Zarr file.

    Key Features:
    - Loads data for specified years and geographic bounds.
    - Filters data based on the proportion of NaNs in the first band.
    - Parallel processing of data elements to speed up the workflow.
    - Data chunking for optimized storage in Zarr.
    - Memory management using Dask.
    - Saves filtered data into a unified Zarr file.
    """

    def __init__(
        self,
        bounding_box: List[float],
        years: List[int],
        first_band_name: str = 'SR_B1',
        threshold: float = 0.3,
        epsg_code: int = 32644,
        zarr_save_path: str = 'filtered_data.zarr',
        chunks: Optional[dict] = None,
        parallel: bool = True,
        num_workers: Optional[int] = None,
    ):
        """
        Initializes the LandsatDataProcessor with the specified parameters.

        Args:
            bounding_box (List[float]): Geographic bounds in the format [min_lon, min_lat, max_lon, max_lat].
            years (List[int]): List of years to process data for.
            first_band_name (str, optional): Name of the first band for filtering. Defaults to 'SR_B1'.
            threshold (float, optional): Threshold for the proportion of NaNs. Bands exceeding this will be excluded.
                Defaults to 0.3 (30%).
            epsg_code (int, optional): Target EPSG code for reprojection. Defaults to 32644.
            zarr_save_path (str, optional): Path to save the resulting Zarr file. Defaults to 'filtered_data.zarr'.
            chunks (dict, optional): Dictionary defining chunk sizes for Zarr storage. Defaults to {'time': 1, 'y': 500, 'x': 500, 'band': -1}.
            parallel (bool, optional): Whether to use parallel processing. Defaults to True.
            num_workers (Optional[int], optional): Number of worker processes for parallel processing. Defaults to the number of CPU cores.
        """
        self.bounding_box = bounding_box
        self.years = years
        self.first_band_name = first_band_name
        self.threshold = threshold
        self.epsg_code = epsg_code
        self.zarr_save_path = zarr_save_path
        self.chunks = chunks or {'time': 1, 'y': 500, 'x': 500, 'band': -1} 
        self.parallel = parallel
        self.num_workers = num_workers or multiprocessing.cpu_count()

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('LandsatDataProcessor')

        # Ensure the directory for the Zarr file exists
        zarr_dir = os.path.dirname(self.zarr_save_path)
        if zarr_dir and not os.path.exists(zarr_dir):
            os.makedirs(zarr_dir)
            self.logger.info(f"Created directory for Zarr file: {zarr_dir}")

        # Initialize an empty Dataset for accumulating data
        self.accumulated_data = xr.Dataset()

    def process_all_years(self):
        """
        Processes all specified years by loading, filtering, and accumulating data.
        """
        self.logger.info("Starting data processing for all years.")

        if self.parallel:
            self.logger.info(f"Using parallel processing with {self.num_workers} workers.")
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                results = pool.map(self._process_single_year, self.years)
        else:
            self.logger.info("Using sequential processing.")
            results = [self._process_single_year(year) for year in self.years]

        # Filter out any None results (years with no data)
        filtered_year_stacks = [ds for ds in results if ds is not None]

        if not filtered_year_stacks:
            self.logger.warning("No filtered data available for any of the years.")
            return

        # Concatenate all year stacks along the time dimension
        self.accumulated_data = xr.concat(filtered_year_stacks, dim='time')
        self.logger.info("Successfully concatenated all filtered data.")

    def _process_single_year(self, year: int) -> Optional[xr.Dataset]:
        """
        Processes data for a single year: searches, filters, and stacks the data.

        Args:
            year (int): The year to process.

        Returns:
            Optional[xr.Dataset]: The filtered DataArray for the year or None if no data is available.
        """
        try:
            self.logger.info(f"Processing year: {year}")

            # Open the STAC catalog and perform a search for the current year
            catalog = pc.get_stac_client()
            search = catalog.search(
                collections=['landsat-8-c2-l2'],
                bbox=self.bounding_box,
                datetime=f"{year}-01-01/{year + 1}-01-01",
                max_items=None
            )

            # Sign the items for access
            items = pc.sign(search)

            # Initialize a list to store filtered items for the current year
            filtered_items = []

            # Iterate over each item and apply filtering based on NaN proportion
            for item in items:
                try:
                    # Stack the current item. Wrap in a list as stackstac.stack expects an iterable.
                    single_stack = stackstac.stack([item], bounds_latlon=self.bounding_box, epsg=self.epsg_code)

                    # Check if the first band exists in the data
                    if self.first_band_name not in single_stack.band.values:
                        self.logger.warning(f"Band '{self.first_band_name}' not found in item {item.id}. Skipping.")
                        continue

                    # Calculate the proportion of NaNs in the first band
                    nan_proportion = single_stack.isnull().sel(band=self.first_band_name).mean(dim=['y', 'x']).compute().item()

                    # Check if the proportion of NaNs is below the threshold
                    if nan_proportion <= self.threshold:
                        filtered_items.append(item)
                        self.logger.info(f"Added item {item.id} with NaN proportion: {nan_proportion:.2%}")
                    else:
                        self.logger.info(f"Skipped item {item.id} with NaN proportion: {nan_proportion:.2%}")

                except Exception as e:
                    self.logger.error(f"Error processing item {item.id}: {e}")
                    continue  # Continue with the next item

            # Check if there are any filtered items for the current year
            if not filtered_items:
                self.logger.warning(f"No filtered items found for year {year}.")
                return None  # Move to the next year

            # Stack all filtered items for the current year
            try:
                year_stack = stackstac.stack(filtered_items, bounds_latlon=self.bounding_box, epsg=self.epsg_code)
                self.logger.info(f"Successfully stacked filtered data for year {year}.")
            except Exception as e:
                self.logger.error(f"Error stacking filtered data for year {year}: {e}")
                return None  # Move to the next year

            return year_stack

        except Exception as e:
            self.logger.error(f"Error processing year {year}: {e}")
            return None

    def save_to_zarr(self):
        """
        Saves the accumulated data to a Zarr file with specified chunking.
        """
        if not self.accumulated_data.sizes:
            self.logger.warning("Accumulated data stack is empty. Nothing to save.")
            return

        try:
            # Define chunk sizes for optimization
            encoding = {var: {'chunks': self.chunks} for var in self.accumulated_data.data_vars}

            # Save the accumulated data to a Zarr file
            self.accumulated_data.to_zarr(self.zarr_save_path, mode='w', consolidated=True, encoding=encoding)
            self.logger.info(f"Filtered data successfully saved to '{self.zarr_save_path}'.")
        except Exception as e:
            self.logger.error(f"Error saving data to Zarr: {e}")
            raise

    def visualize_first_slice(self):
        """
        Visualizes the first time slice of the filtered data.
        """
        import matplotlib.pyplot as plt

        if self.accumulated_data.sizes['time'] > 0:
            try:
                first_time_slice = self.accumulated_data.isel(time=0).sel(band=self.first_band_name)
                first_time_slice.plot()
                plt.title(f"First Time Slice After Filtering ({first_time_slice.time.values})")
                plt.show()
            except Exception as e:
                self.logger.error(f"Error visualizing data: {e}")
        else:
            self.logger.warning("No time slices available after filtering for visualization.")


# Example usage of the LandsatDataProcessor class
if __name__ == "__main__":
    # Define parameters
    bounding_box = [162585.0, 3395085.0, 396015.0, 3632415.0]
    years = list(range(2018, 20))  # From 2018 to 2022
    first_band_name = 'SR_B1'
    threshold = 0.3
    epsg_code = 32644
    zarr_save_path = os.path.join('data', 'filtered_data.zarr')
    chunks = {'time': 1, 'y': 500, 'x': 500, 'band': -1}  # Example chunk sizes

    # Initialize the data processor
    processor = LandsatDataProcessor(
        bounding_box=bounding_box,
        years=years,
        first_band_name=first_band_name,
        threshold=threshold,
        epsg_code=epsg_code,
        zarr_save_path=zarr_save_path,
        chunks=chunks,
        parallel=True,  # Enable parallel processing
        num_workers=4  # Set the number of worker processes
    )

    # Process all years
    processor.process_all_years()

    # Save the accumulated data to a Zarr file
    processor.save_to_zarr()

    # Visualize the first time slice after filtering
    processor.visualize_first_slice()
