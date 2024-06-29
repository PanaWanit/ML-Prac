from omegaconf import DictConfig
from collections import defaultdict

from functools import wraps

import wandb

import torch
import numpy as np
import random

import logging
import os
import shutil

from torch.utils.data import DataLoader
from typing import Tuple, Dict, List
from samo.data_utils import get_enroll_speaker, genSpoof_list, subset_bonafide, ASVspoof2019_speaker

############################################################# Reproducibility ##############################################################
def setup_seed(random_seed: int, cudnn_deterministic: bool = True) -> None:
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic # CUDA convolution determinism
        torch.backends.cudnn.benchmark = False  # Disabling the benchmarking feature, deterministically select an algorithm

def _seed_worker(worker_id): # [https://pytorch.org/docs/stable/notes/randomness.html]
    """
    Used in generating seed for the worker of torch.utils.data.Dataloader
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

############################################################# Output directory #############################################################
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

    with open(os.path.join(cfg.output_folder, "train_loss.log"), "a") as f:
        f.write(f"{'Epoch':<10} {'batch':<10} {'Loss':<20}")

    if not os.path.exists(cfg.path_to_database):
        raise RuntimeError(f"Path {cfg.path_to_database} does not exists!")

################################################################### CUDA ###################################################################
def cuda_checker(cfg: DictConfig) -> None:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this device!")
    print(f"using device: {cfg.device.upper()}")

################################################################ DataLoader ################################################################
def _log_dataset(datasets):
    print(f"{' Dataset ':-^40}")
    print(40 * '=')
    for task, dataset in datasets.items():
        if task == "train_bona": # "train_bona" is a "Subset" object so it's doesn't has a "get_total_utterance" and "get_num_centers" getters
            continue
        print(f'{"|":<4} no. {task: <11} {"files:":^8} {dataset.get_total_utterances:^5} {"|":>4}')
        print(f'{"|":<4} no. {task: <11} {"speaker:":^8} {dataset.get_num_centers:^5} {"|":>4}' )
        print(40 * '=')

    print('|', f'{"bonafide speech in train set = "+str(len(datasets["train_bona"])):^36}', '|')
    print(40 * '*')

def get_loader(cfg: DictConfig) -> Dict[str, DataLoader]:
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
    # dataset
    genSpoof_list_cfg = {
        "train"       : {"dir_meta": list_paths["train"]      , "base_dir": database_paths["train"], "enroll": False, "train": True },
        "dev_enroll"  : {"dir_meta": list_paths["dev_enroll"] , "base_dir": database_paths["dev"]  , "enroll": True , "train": False},
        "dev"         : {"dir_meta": list_paths["dev"]        , "base_dir": database_paths["dev"]  , "enroll": False, "train": False, "target_only": target_only, "enroll_spks": enroll_spk["dev"]},
        "eval_enroll" : {"dir_meta": list_paths["eval_enroll"], "base_dir": database_paths["eval"] , "enroll": True , "train": False},
        "eval"        : {"dir_meta": list_paths["eval"]       , "base_dir": database_paths["eval"] , "enroll": False, "train": False, "target_only": target_only, "enroll_spks": enroll_spk["eval"]}
    }

    asv_cfg_list = {task: genSpoof_list(**cfg) for task, cfg in genSpoof_list_cfg.items()}

    datasets = {task: ASVspoof2019_speaker(**cfg) for task, cfg in asv_cfg_list.items()}
    datasets["train_bona"] = subset_bonafide(datasets["train"])

    _log_dataset(datasets)

    # dataloader
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader_cfg = defaultdict(lambda :  {"shuffle": False, "drop_last": False }, # other dataloader
                             {"train": {"shuffle": True , "drop_last": True , "worker_init_fn": _seed_worker, "generator": gen}}) # train data loader

    loaders = {
        task : DataLoader(dataset=dataset, batch_size=batch_size, pin_memory=True, **loader_cfg[task]) 
        for task, dataset  in datasets.items()
    }

    return loaders

################################################################ WANDB #################################################################

def wandb_error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f"Found Error in {func.__qualname__}.")
            raise e
        finally:
            print("Stop recording on wandb...")
            wandb.unwatch()
            wandb.finish()
    return wrapper