from src.config import Config
from src.data import StatsCalculator
from src.utils import setup_seed
from src.logger import ColoredLogger

def main():
    """
    Main function for data preprocessing.
    """
    config = Config('config.json')
    logger = ColoredLogger().get_logger()
    setup_seed(config.seed)

    logger.info("Initializing statistics calculation...")
    processor = StatsCalculator(
        config=config,
        num_workers=8 # Set the number of worker processes
    )
    stats = processor.compute_statistics()
    processor.save_statistics(stats)
    logger.info("Statistics calculation completed.")

if __name__ == "__main__":
    main()