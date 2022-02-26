import logging


def create_logger() -> logging.Logger:
    """Create logger.

    Returns:
        logging.Logger: General logger.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    logger = logging.getLogger('gyomei_detection')
    return logger


LOGGER = create_logger()
