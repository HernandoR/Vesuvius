import os
import socket
from pathlib import Path
from logger.Loggers import get_logger

logger = get_logger(__name__)


def get_saved_model_path(cp_dir, model_id, fold=None):
    if fold is None:
        return str(Path(cp_dir) / f"{model_id}_best.pth")
    else:
        return str(Path(cp_dir) / f"{model_id}_fold{fold}_best.pth")


def decide_paths():
    HOST = socket.gethostname()

    if HOST.endswith("cloudlab.us"):
        # is_kaggle = False
        HOST = "cloudlab"
    kaggle_run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE")
    if kaggle_run_type is None:
        # is_kaggle = False
        pass
    else:
        # is_kaggle = True
        HOST = "kaggle"
        print("Kaggle run type: {}".format(kaggle_run_type))

    if HOST == "cloudlab":
        ROOT_DIR = Path("/local/Codes/Vesuvius").absolute()
        DATA_DIR = Path("/vesuvius_kaggle")
        OUTPUT_DIR = ROOT_DIR / "saved"

        EXTERNAL_MODELS_DIR = ROOT_DIR / "model"

    elif HOST == "kaggle":
        ROOT_DIR = Path("/kaggle")
        DATA_DIR = ROOT_DIR / "input" / "vesuvius-challenge-ink-detection"
        OUTPUT_DIR = ROOT_DIR / "working" / "saved"

        EXTERNAL_MODELS_DIR = ROOT_DIR / "input"
    else:
        ROOT_DIR = Path("../").absolute()
        DATA_DIR = ROOT_DIR / "data" / "raw"
        OUTPUT_DIR = ROOT_DIR / "saved"

        EXTERNAL_MODELS_DIR = ROOT_DIR / "model"

    CP_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = OUTPUT_DIR / "logs"
    CACHE_DIR = OUTPUT_DIR / "cache"
    logger.info(f"ROOT_DIR: {ROOT_DIR}")
    # assert os.listdir(DATA_DIR) != [], f"Data directory {DATA_DIR} is empty"
    if os.listdir(DATA_DIR) == []:
        logger.warning(f"Data directory {DATA_DIR} is empty")

    for p in [ROOT_DIR, DATA_DIR, OUTPUT_DIR, CP_DIR, LOG_DIR, CACHE_DIR]:
        if os.path.exists(p) is False:
            os.makedirs(p)

    return HOST, {
        "ROOT_DIR": ROOT_DIR,
        "DATA_DIR": DATA_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "CP_DIR": CP_DIR,
        "LOG_DIR": LOG_DIR,
        "CACHE_DIR": CACHE_DIR,
        "EXTERNAL_MODELS_DIR": EXTERNAL_MODELS_DIR,
    }
