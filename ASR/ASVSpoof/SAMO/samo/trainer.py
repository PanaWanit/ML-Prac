from omegaconf import DictConfig, OmegaConf
from typing import Dict, Sequence, List, Any

import wandb

from hydra.utils import instantiate
import os

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel
from collections import defaultdict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tqdm.notebook import tqdm # Train on kaggle notebook.

import eval_metrics as em

class Trainer(object):
    def __init__(self, 
                 cfg:DictConfig,
                 loaders:Dict[str, DataLoader], 
                ) -> None:
        self.output_dir:str = cfg.output_folder

        self._device:str = cfg.device
        self._num_epochs:int = cfg.train.num_epochs
        self._update_interval:int = cfg.train.update_interval # Reference: update embeddings every M times
        self._target_only:bool = cfg.target_only

        self._feat_model:nn.Module = instantiate(cfg.model.model).to(self._device)
        if cfg.dp and (gpu_cnt := torch.cuda.device_count()) > 1:
            print('Trainer use total', gpu_cnt, 'GPUs')
            self._feat_model = nn.DistributedDataParallel(self._feat_model, output_device=[cfg.gpu])

        self._loaders:Dict[str, DataLoader] = loaders
        self._train_num_centers:int = loaders["train"].dataset.get_num_centers

        if cfg.continue_training:
            raise NotImplementedError()
        
        self._optimizer:Optimizer = instantiate(cfg.train.optim, params=self._feat_model.parameters())
        self._optimizer_swa = AveragedModel(self._feat_model, device=self._device)
        self._scheduler:LRScheduler = instantiate(cfg.train.scheduler, optimizer=self._optimizer)

        self._loss_fn:nn.Module = instantiate(cfg.loss.fn)
        
        self._early_stop:int = 0
        self._best_val_loss:float = float("inf")


        self._feat_dim:int = cfg.enc_dim
        self._train_spks:List[str] = loaders["train"].dataset.get_unique_speaker
        self._initialize_centers = cfg.initialize_centers
        if cfg.initialize_centers == "one_hot":
            self._w_centers:Tensor = torch.eye(cfg.enc_dim)[:self._train_num_centers]
            self._train_spk2center = dict(zip(self._train_spks, self._w_centers))
        elif cfg.initialize_centers == "evenly":  # uniform_hypersphere
            raise NotImplementedError
        else:
            raise RuntimeError("There is no {cfg.initialize_centers} method.")
        
        self.__wandb_cfg_dict = OmegaConf.to_container(cfg)
    
    def _init_wandb(self) -> None:
        os.environ["WANDB_PROJECT"] = "SAMO ASVSpoof"
        wandb.init(
            job_type="SAMO",
            config=self.__wandb_cfg_dict,
            name=f"{self._device.upper()}-E{self._num_epochs}-T{int(self._target_only)}-DIM{self._feat_dim}-INIT={self._initialize_centers}"
        )
        wandb.watch(self._feat_model, log="all", log_freq=100, log_graph=False)

    # TODO: wandb log
    def _wandb_log(self, log_train, log_val, **kwargs) -> None:
        wandb.log({
            **kwargs, **log_train, **log_val
        })
    
    def train(self) -> None:
        self._init_wandb()

        try:
            for epoch in tqdm(range(1, self._num_epochs+1)): # understandable
                log_train:dict = self._train_epoch(epoch)
                log_val:dict = self._val()

                # TODO: Find best val loss + update swa

                # logging
                self._wandb_log(log_train=log_train, log_val=log_val, epoch=epoch)

        finally:
            wandb.unwatch(self.feat_model)
            wandb.finish()
    
    def _train_epoch(self, epoch:int) -> None:
        self.feat_model.train()
        print(f"\nEpoch: {epoch}")
        train_losses = []
        if epoch % self._update_interval == 0:
            self._update_embeddings() # update both "speakers's center" and "speaker to center"

        for i, (feat, labels, spk, _, _) in enumerate(tqdm(self._loaders["train"])):
            feat, labels = feat.to(self._device), labels.to(self._device)

            self.optimizer.zero_grad()

            embs, _ = self.feat_model(feat)

            w_spks = self.map_speakers_to_center(spks=spk, spk2center=self._train_spk2center)

            loss = self.loss_fn(embs, labels, self.w_centers, w_spks=w_spks)
            train_losses.append(loss)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

        train_loss = np.nanmean(train_losses)
        return {
            "train_loss": train_loss,
            "lr": self.scheduler.get_lr()
        }
        
        
    
    @torch.no_grad
    def _val(self) -> Any:
        self.feat_model.eval()
        val_centers, val_spk2center = self._get_centers_from_loader(self._loaders["dev_enroll"])
        batch_scores, batch_labels, val_losses = [], [], []
        for i, (feat, labels, spk, _, _) in enumerate(tqdm(self._loaders["dev"])):
            feat, labels = feat.to(self._device), labels.to(self._device)
            embs, _ = self.feat_model(feat)
            w_spks = self.map_speakers_to_center(spks=spk, spk2center=val_spk2center)
            if self._target_only:
                loss, score = self.loss_fn(embs, labels, w_centers=val_centers, w_spks=w_spks, get_score=True) # get_score for computer eer
            else:
                raise NotImplementedError # Not implement SAMO.inference yet
            val_losses.append(loss.item())
            batch_scores.append(score)
            batch_labels.append(labels)
        
        val_loss = np.nanmean(val_losses)
        val_scores = torch.cat(batch_scores).cpu().numpy()
        val_labels = torch.cat(batch_labels).cpu().numpy()
        # BUG: which val_labels == 0 or 1 should be placed first?
        # For computes EER, does the result remain the same?
        eer, _ = em.compute_eer(val_scores[val_labels == 0], val_scores[val_labels == 1])

        return {
            "val_loss": val_loss,
            "val_eer": eer
        }

        
    def _get_centers_from_loader(self, task:str) -> Tensor:
        enroll_emb_dict = defaultdict(list)
        with torch.no_grad():
            for i, (batch_x, _, spk, _, _) in enumerate(self._loaders[task]):
                batch_x = batch_x.to(self._device)
                batch_cm_emb, _ = self.feat_model(batch_x)
                batch_cm_emb = batch_cm_emb.detach().cpu().numpy()

                for spk, cm_emb in zip(batch_cm_emb, spk):
                    enroll_emb_dict[spk].append(cm_emb)
            
            for spk, emb_np in enroll_emb_dict.items():
                enroll_emb_dict[spk] = Tensor(emb_np.mean(axis=0))
        return torch.stack(list(enroll_emb_dict.values())), enroll_emb_dict
    

    def _update_embeddings(self) -> None:
        avg_centers, spk2center = self._get_centers_from_loader("train_bona")
        self._w_centers = avg_centers
        self._train_spk2center = spk2center
    
    @staticmethod
    def map_speakers_to_center(spks: Sequence[str],
                                spk2center:Dict[str, torch.Tensor]
                                ) -> torch.Tensor:
        return torch.stack([spk2center[spk] for spk in spks])

    @property
    def w_centers(self):
        return self._w_centers
    @property
    def get_loaders(self):
        return self._loaders
    @property
    def feat_model(self):
        return self._feat_model
    @property
    def loss_fn(self):
        return self._loss_fn
    @property
    def optimizer(self):
        return self._optimizer
    @property
    def scheduler(self):
        return self._scheduler