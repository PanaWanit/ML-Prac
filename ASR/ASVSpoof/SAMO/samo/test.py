from omegaconf import DictConfig
from typing import Dict
import os

import wandb
from hydra.utils import instantiate

import torch
from torch import nn

from torch.utils.data import DataLoader
from samo.trainer import Trainer
from torch.nn import DataParallel as DP

import samo.eval_metrics as em
from samo.utils import compute_eer_tdcf

@torch.no_grad
def test(model_path:str, loaders:Dict[str, DataLoader], cfg: DictConfig):
    os.makedirs(os.path.join(cfg.output_folder, "test_result"), exist_ok=True)
    save_path = os.path.join(cfg.output_folder, "test_result", f"{cfg.test.save_score}.txt")
    asv_score_path = os.path.join(cfg.path_to_database, "ASVspoof2019_LA_asv_scores", "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt")

    if os.path.exists(save_path):
        print(f"Found score file: {save_path}\n Computer err tdcf only.")
        compute_eer_tdcf(cfg, save_path, asv_score_path)
        return

    scoring:str = cfg.test.scoring # not used. now force to samo only

    feat_model = torch.load(model_path).to(cfg.device)
    if cfg.dp and (gpu_cnt := torch.cuda.device_count()) > 1:
        print('Trainer use total', gpu_cnt, 'GPUs')
        feat_model = DP(feat_model, device_ids=list(range(gpu_cnt))).to(cfg.device)

    print(f"Model loaded: {model_path}")
    print(f"Using scoring={scoring.upper()} target-only={cfg.target_only}")
    print("Start evaluation...")

    feat_model.eval()
    loss_fn:nn.Module = instantiate(cfg.loss.fn)

    loss, labels, scores, utts, tags, spks = Trainer.dev_n_eval(task="eval",
                                                                loaders=loaders,
                                                                feat_model=feat_model,
                                                                loss_fn=loss_fn,
                                                                device=cfg.device,
                                                                target_only=cfg.target_only)
    
    with open(save_path, "w") as f:
        for utt, tag, score, label, spk in zip(utts, tags, scores, labels, spks):
            f.write(f"{utt} {tag} {label} {score} {spk}\n")
    print(f"Score saved to {save_path}")

    if wandb.run is not None: # On train.final_test
        wandb.save(model_path) 
        wandb.save(save_path)

    compute_eer_tdcf(cfg, save_path, asv_score_path)
