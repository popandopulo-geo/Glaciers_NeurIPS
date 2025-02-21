from src.config import Config
from src.data import Preprocessor
from src.utils import setup_seed
from src.logger import ColoredLogger

def main():
    """
    Main function for data preprocessing.
    """
    config = Config('config.json')
    logger = ColoredLogger().get_logger()
    setup_seed(config.seed)

    logger.info("Initializing data preprocessing...")
    processor = Preprocessor(
        config=config,
        num_workers=8 # Set the number of worker processes
    )
    processor.process_all()
    logger.info("Data preprocessing completed.")

if __name__ == "__main__":
    main()