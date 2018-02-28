from pathlib import Path
import os


PROJECT_DIR = os.path.realpath(__file__)


DATA_PATH = "RECOLA"

TFRECORDS_SAVE_PATH = "tf_records"

DATA_DIR = Path(DATA_PATH)

PORTION_TO_ID = dict(
    train = [45, 46, 48, 56, 58, 62, 64, 65],
    valid = [28, 30, 34, 37, 39, 41, 42, 43],
    test  = [16, 17, 19, 21, 23, 25, 26]
)

SUBJECT_LIST = [16, 17, 19, 21, 23, 25, 26, 28, 30, 34, 37, 39, 41, 42, 43, 45, 46, 48, 56, 58, 62, 64, 65]