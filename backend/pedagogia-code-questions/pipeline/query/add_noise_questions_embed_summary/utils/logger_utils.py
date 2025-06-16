import logging
import os

def setup_logger(logger_name: str, log_file: str = '',
                 level: int = logging.DEBUG) -> None:
    """
    :param logger_name: name to give to logger
    :param log_file: file to save log to
    :param level: which base level of importance to set logger to
    :return: *None*
    """
    log = logging.getLogger(logger_name)

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    if log_file == '':
        log_file = f"{logger_name}.log"

    log_file_path = os.path.join('logs', log_file)

    formatter = logging.Formatter(
        fmt="%(name)s - %(levelname)s: %(asctime)-15s %(message)s")
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log.setLevel(level)
    if not log.hasHandlers():
        log.addHandler(file_handler)
        log.addHandler(stream_handler)
