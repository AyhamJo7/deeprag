import logging
import sys

from .distributed import is_main_process


def get_logger(name: str) -> logging.Logger:
    """Get a logger that logs to stdout, but only on the main process."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create a handler
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(handler)

    # Disable propagation to the root logger
    logger.propagate = False

    # Make sure to only log on the main process
    if not is_main_process():
        logger.setLevel(logging.WARNING) # Or another level to suppress info

    return logger
