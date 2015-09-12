import logging
import logging.handlers as handlers
import os

import stitching.constants as consts

DEBUG = logging.DEBUG
ERROR = logging.ERROR
INFO = logging.INFO

MEGABYTE = 1024**2

def create_file_logger(logger_name, log_filename, entry_format, level, maxMegaBytes=3, backupCount=5):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    handler = handlers.RotatingFileHandler(
       os.path.join(consts.LOGS_DIR, log_filename),
       maxBytes=MEGABYTE*maxMegaBytes,
       backupCount=backupCount
    )
    handler.setLevel(level)
    
    formatter = logging.Formatter(entry_format)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger