from omegaconf import DictConfig
from typing import Dict
import os

from hydra.utils import instantiate

import torch
from torch import nn

from torch.utils.data import DataLoader
from samo.trainer import Trainer

import samo.eval_metrics as em
from samo.utils import compute_eer_tdcf

# TODO: Test function
@torch.no_grad
def test(model_path:str, loaders:Dict[str, DataLoader], cfg: DictConfig):
    torch.set_default_tensor_type(torch.FloatTensor)
    os.makedirs(os.path.join(cfg.output_folder, "test_result"))
    save_path = os.path.join(cfg.output_folder, "test_result", "{cfg.test.save_score}.txt")

    scoring:str = cfg.test.scoring # not used now force to samo only

    feat_model = torch.load(model_path).to(cfg.device)
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

    asv_score_path = os.path.join(cfg.path_to_database, "ASVspoof2019_LA_asv_scores", "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt")
    compute_eer_tdcf(cfg, save_path, asv_score_path)
