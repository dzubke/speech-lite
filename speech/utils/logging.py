import os
import logging


def get_logger(name:str, log_path: str, level:str):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logis even debug messages
    fh = logging.FileHandler(log_path)
    fh.setLevel(eval("logging."+level.upper()))
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def get_logger_filename(logger:logging.Logger)->str:
    """
    Returns the filename of the logger
    """
    basename, filename = os.path.split(logger.handlers[0].baseFilename)
    filename, ext = os.path.splitext(filename)
    return filename
