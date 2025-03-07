from src.config import Config
from src.data import Splitter
from src.utils import setup_seed
from src.logger import ColoredLogger

def main():
    """
    Main function for splitting TIFF data.
    """
    config = Config('config.json')
    logger = ColoredLogger().get_logger()
    setup_seed(config.seed)

    logger.info("Initializing TIFF data splitting...")
    splitter = Splitter(config=config)
    splitter.split_data()
    splitter.generate_patches()
    splitter.save_splits()
    logger.info("TIFF data splitting completed.")

if __name__ == "__main__":
    main()
