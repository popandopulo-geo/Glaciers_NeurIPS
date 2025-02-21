from src.config import Config
from src.data import Fetcher
from src.utils import setup_seed
from src.logger import ColoredLogger

def main():
    """
    Main function for data fetching.
    """
    config = Config('config.json')
    logger = ColoredLogger().get_logger()
    setup_seed(config.seed)

    logger.info("Initializing data fetching...")
    processor = Fetcher(
        config=config,
        parallel=True,  # Enable parallel processing
        num_workers=4  # Set the number of worker processes
    )
    processor.fetch_all_years()
    logger.info("Data fetching completed.")

if __name__ == "__main__":
    main()