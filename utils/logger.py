# coding : utf-8

import logging
import coloredlogs


def setup_logger(file_name=None, control_log=True, log_level='INFO'):
    fmt = '%(asctime)s %(name)s [%(levelname)s] %(message)s'
    # DATE_FORMAT = None  # "%Y-%d-%m %H:%M:%S"

    # create logger
    logger = logging.getLogger()
    formatter = logging.Formatter(fmt)

    coloredlogs.install(level=log_level, fmt=fmt)
    coloredlogs.DEFAULT_FIELD_STYLES = {'asctime': {'color': 'green'}, 'hostname': {'color': 'magenta'},
                                        'levelname': {'color': 'green', 'bold': True}, 'request_id':{'color': 'yellow'},
                                        'name': {'color': 'blue'}, 'programname': {'color': 'cyan'},
                                        'threadName': {'color': 'yellow'}}

    if control_log:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if file_name:
        file_handler = logging.FileHandler(file_name, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
