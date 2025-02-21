from src.config import Config
from src.data import RGBGenerator
from src.utils import setup_seed

def main():
    config = Config('config.json')
    setup_seed(config.seed)
    
    generator = RGBGenerator(
        config=config,
        num_workers=8)
    generator.process_all()


if __name__ == '__main__':
    main()