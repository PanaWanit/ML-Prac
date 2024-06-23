from omegaconf import DictConfig, OmegaConf
import torch
import random
import numpy as np
import os
import logging
import shutil

from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, Dict, Any
from samo.data_utils import get_enroll_speaker, genSpoof_list, ASVspoof2019_speaker


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

def cuda_checker(cfg: DictConfig) -> None:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Cuda is not available on this device!")
    print(f"using device={cfg.device}")


# TODO: Finished this function
def get_loader(cfg: DictConfig) -> Any:  # Tuple[Dict[str, DataLoader], ]:
    db_path, seed, target_only, batch_size = cfg.path_to_database, cfg.seed, cfg.target_only, cfg.batch_size
    # Dataloaders generator config
    database_paths = { task : os.path.join(db_path, f"ASVspoof2019_LA_{task}") for task in ["train", "dev", "eval"] }
    list_paths = { 
        "train" : os.path.join(db_path, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"),
        "dev" : os.path.join(db_path, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"),
        "eval" : os.path.join(db_path, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"),
        "dev_enroll"  : [ os.path.join(db_path, "ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.female.trn.txt") ,
                          os.path.join(db_path, "ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.male.trn.txt")],
        "eval_enroll" : [ os.path.join(db_path, "ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.female.trn.txt") ,
                         os.path.join(db_path, "ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.male.trn.txt")],
    }
    
    enroll_spk = { "dev" : get_enroll_speaker(list_paths["dev_enroll"]), 
                   "eval": get_enroll_speaker(list_paths["eval_enroll"]) }

    genSpoof_list_cfg = {
        "train"       : {"dir_meta": list_paths["train"]      , "base_dir": database_paths["train"], "enroll": False, "train": True },
        "dev_enroll"  : {"dir_meta": list_paths["dev_enroll"] , "base_dir": database_paths["dev"]  , "enroll": True , "train": False},
        "dev"         : {"dir_meta": list_paths["dev"]        , "base_dir": database_paths["dev"]  , "enroll": False, "train": False, "target_only": target_only, "enroll_spks": enroll_spk["dev"]},
        "eval_enroll" : {"dir_meta": list_paths["eval_enroll"], "base_dir": database_paths["eval"] , "enroll": True , "train": False},
        "eval"        : {"dir_meta": list_paths["eval"]       , "base_dir": database_paths["eval"] , "enroll": False, "train": False, "target_only": target_only, "enroll_spks": enroll_spk["eval"]}
    }

    # Create dataset, dataloader, etc.
    asv_cfg_list = {task: genSpoof_list(**cfg) for task, cfg in genSpoof_list_cfg.items()}
    datasets = {task: ASVspoof2019_speaker(**cfg) for task, cfg in asv_cfg_list.items()}
    num_centers = {task: len(set(cfg["utt2spk"].values())) for task, cfg in asv_cfg_list.items()}

    print(f"{' Loaders ':-^40}")
    print(40 * '=')
    for task, dataset in asv_cfg_list.items():
        num_centers[task]=len(set(dataset["utt2spk"].values()))
        print(f'{"|":<4} no. {task: <11} {"files:":^8} {len(dataset["list_IDs"]):^5} {"|":>4}')
        print(f'{"|":<4} no. {task: <11} {"speaker:":^8} {num_centers[task]:^5} {"|":>4}' )
        print(40 * '=')
    print(40 * '*')


    return asv_cfg_list, datasets, loaders ,num_centers