import logging
import sys
from pathlib import Path

# Create a logs directory (if it doesn’t exist)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def get_logger(name="app"):
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # File handler (rotating by day)
    fh = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    fh.setLevel(logging.INFO)

    # Unified format
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
