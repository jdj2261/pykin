import logging
import logging.handlers

LOG_LEVEL = { "INFO" : logging.INFO, 
              "DEBUG" : logging.DEBUG, 
              "WARNING" : logging.WARNING, 
              "ERROR" : logging.ERROR, 
              "CRITICAL" :logging.CRITICAL 
            }

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;21m"
    green='\033[1;32m'
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = '[%(levelname)s] [%(name)s]: %(message)s'
    # format = '[%(levelname)s|%(name)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s'

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt,"%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def create_logger(logger_name, logging_level="debug", file_name="test.log", is_save=False):
    # Create Logger

    format = '[%(levelname)s] [%(name)s]: %(message)s'
    # |%(filename)s:%(lineno)s] %(asctime)s 

    if logging_level.upper() in LOG_LEVEL.keys():
        level = LOG_LEVEL.get(logging_level.upper())

    logger = logging.getLogger(logger_name)
 
    # Check handler exists
    if len(logger.handlers) > 0:
        return logger # Logger already exists
 
    logger.setLevel(level)

    formatter = logging.Formatter(format, "%Y-%m-%d %H:%M:%S")

    # Create Handlers
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(CustomFormatter())
    logger.addHandler(stream_handler)

    if is_save:
        rotating_handler = logging.handlers.TimedRotatingFileHandler(file_name, when='h', interval=1, backupCount=0) 
        rotating_handler.setFormatter(formatter)
        logger.addHandler(rotating_handler) 

    return logger
