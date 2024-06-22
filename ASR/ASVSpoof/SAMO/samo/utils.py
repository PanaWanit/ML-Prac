from omegaconf import DictConfig, OmegaConf
import torch
import random
import numpy as np
import os
import logging
import shutil

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
    