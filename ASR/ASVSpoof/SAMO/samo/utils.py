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

############################################################## min-tDCF ###############################################################
import eval_metrics as em
def compute_eer_tdcf(cfg, cm_score_file, asv_score_path):
    asv_score_file = os.path.join(asv_score_path)

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    # asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(float)

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    # cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2].astype(int)  # label
    cm_scores = cm_data[:, 3].astype(float)

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 0]
    spoof_cm = cm_scores[cm_keys == 1]

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

    # Compute t-DCF
    tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model)

    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    # test individual attacks
    attack_types = [f'A{_id:02d}' for _id in range(7, 20)]
    eer_cm_lst = {}
    for attack in attack_types:
        if attack == "-":
            continue
        # Extract bona fide (real human) and spoof scores from the CM scores
        bona_cm = cm_scores[cm_keys == 0]
        spoof_cm = cm_scores[cm_sources == attack]

        # EERs of the standalone systems and fix ASV operating point to EER threshold
        eer_att = em.compute_eer(bona_cm, spoof_cm)[0]
        if not np.isnan(eer_att):
            eer_cm_lst[attack] = eer_att
            # eer_cm_lst.append("{:5.2f} %".format(eer_cm * 100))
        else:
            continue

    output_file = f"./{cfg.output_folder}/eer-tdcf-{cfg.test.save_score}.txt"
    with open(output_file, "w") as f_res:
        f_res.write('\nCM SYSTEM\n')
        f_res.write('\tEER\t\t= {:8.9f} % '
                    '(Equal error rate for countermeasure)\n'.format(
            eer_cm * 100))

        f_res.write('\nTANDEM\n')
        f_res.write('\tmin-tDCF\t\t= {:8.9f}\n'.format(min_tDCF))

        f_res.write('\nBREAKDOWN CM SYSTEM\n')
        for attack_type in attack_types:
            _eer = eer_cm_lst[attack_type] * 100
            f_res.write(
                f'\tEER {attack_type}\t\t= {_eer:8.9f} % \n'
            )
    os.system(f"cat {output_file}")

    return eer_cm, min_tDCF