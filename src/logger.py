import logging
import colorlog

class ColoredLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ColoredLogger, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.INFO)
            self.initialized = True

            self.set_handler()

    def get_logger(self):
        return self.logger

    def set_handler(self):
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s %(module)-20s %(reset)s %(message)s", 
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
            }
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
