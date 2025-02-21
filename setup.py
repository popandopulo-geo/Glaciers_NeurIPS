import os.path as osp

from config import *
from src.data import Processor, Splitter
from src.utils import setup_seed
from src.logger import ColoredLogger

def main():
    setup_seed(SEED)
    logger = ColoredLogger().get_logger()

    if osp.exists(osp.join(DATA_ROOT, HDF5_FILE)):
        logger.info("Data file already exists. Skipping data fetching.")
    else:
        logger.info("Initializing data fetching...")
        processor = Processor(
            bounding_box=BOUNDING_BOX,
            years=YEARS,
            first_band_name='SR_B1',
            threshold=0.3,
            epsg_code=ESPG_CODE,
            out_path=osp.join(DATA_ROOT, HDF5_FILE),
            parallel=False,  # Enable parallel processing
            num_workers=4  # Set the number of worker processes
        )
        processor.process_all_years()
        processor.save_to_hdf5()
        logger.info("Data fetching completed.")


    # Initialize the splitter
    logger.info("Initializing data splitting...")
    splitter = Splitter(
        input_path=processor.out_path,
        split_ratio=TRAIN_TEST_RATIO,
        patch_size=PATCH_SIZE,
        num_patches_per_image=NUM_PATCHES_PER_IMAGE,
        sequence_length=INPUT_SEQUENCE,
        prediction_length=OUT_SEQUENCE,
    )

    splitter.split_data()
    splitter.generate_patches()
    splitter.save_splits(train_output='train_splits.json', val_output='val_splits.json')
    logger.info("Data splitting completed.")

    logger.info("Setup completed.")

if __name__ == "__main__":
    main()