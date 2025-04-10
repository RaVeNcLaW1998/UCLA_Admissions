import pandas as pd
import logging
from .config import RAW_DATA_PATH

logger = logging.getLogger(__name__)


def load_data():
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        logger.info(f"Data loaded successfully with shape {df.shape}.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
