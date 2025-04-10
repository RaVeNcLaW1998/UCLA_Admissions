import logging
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

RAW_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "Admission.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "admission_model.pkl")
LOG_LEVEL = logging.INFO

RANDOM_STATE = 123
TEST_SIZE = 0.2

# Logging settings
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
LOG_FILE = os.path.join(BASE_DIR, "..", "logs", "app.log")
