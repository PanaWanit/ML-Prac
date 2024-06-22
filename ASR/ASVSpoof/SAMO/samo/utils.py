from omegaconf import DictConfig, OmegaConf
import torch
import random
import numpy as np
import os
import logging
import shutil

from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, Dict, Any
from samo.data_utils import genSpoof_list, ASVspoof2019_speaker


def setup_seed(random_seed: int, cudnn_deterministic: bool = True) -> None:
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic # CUDA convolution determinism
        torch.backends.cudnn.benchmark = False  # Disabling the benchmarking feature, deterministically select an algorithm

def output_dir_setup(cfg: DictConfig) -> None:
    if cfg.continue_training:
        return
    if os.path.exists(cfg.output_folder):
        logging.warning(f"{cfg.output_folder} exists!")
        print(f"Overwrite {cfg.output_folder}")
    os.makedirs(cfg.output_folder, exist_ok=cfg.overwrite) # ensure that have output_folder exists
    if cfg.overwrite: # remove then create this directory again
        shutil.rmtree(cfg.output_folder)
        os.mkdir(cfg.output_folder)
        os.makedirs(os.path.join(cfg.output_folder, "checkpoints"))
    if not os.path.exists(cfg.path_to_database):
        raise RuntimeError(f"Path {cfg.path_to_database} does not exists!")
    if not os.path.exists(cfg.path_to_protocol):
        raise RuntimeError(f"Path {cfg.path_to_database} does not exists!")

def cuda_checker(cfg: DictConfig) -> None:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Cuda is not available on this device!")
    print(f"using device={cfg.device}")


# TODO: test this function
def get_loader(cfg: DictConfig) -> Any:  # Tuple[Dict[str, DataLoader], ]:
    db_path, seed, target_only, batch_size = cfg.path_to_database, cfg.seed, cfg.target_only, cfg.batch_size
    tasks = ["train", "dev", "eval"]
    trn_trls = ["trn", "trl", "trl"]

    database_paths = { task : os.path.join(db_path, f"ASVspoof2019_LA_{task}") for task in tasks }
    list_paths = { task : os.path.join(db_path, "ASVspoof2019_LA_cm_protocols", f"ASVspoof2019.LA.cm.{task}.{trn_trl}.txt")
                   for task, trn_trl in zip(tasks, trn_trls) }
    enroll_paths = {
        task: [ os.path.join(db_path, "ASVspoof2019_LA_asv_protocols", f"ASVspoof2019.LA.asv.{task}.{gender}.trn.txt") 
               for gender in ["female", "male"] ]
        for task in ["dev", "eval"]
    }

    spoof_list_trn = genSpoof_list(dir_meta=list_paths["train"], base_dir=database_paths["train"], enroll=False, train=True)

    print("no. training files", len(spoof_list_trn["list_IDs"]) )
    print("no. training speaker", len(set(spoof_list_trn["utt2spk"].values())) )

    train_set = ASVspoof2019_speaker(**spoof_list_trn)
