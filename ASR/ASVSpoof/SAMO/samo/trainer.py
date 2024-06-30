from omegaconf import DictConfig, OmegaConf
from typing import Dict, Sequence, List, Any, Optional

import wandb
import logging

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

from samo.utils import wandb_error_handler

from tqdm.auto import tqdm

import samo.eval_metrics as em

class Trainer(object):
    def __init__(self, 
                 cfg:DictConfig,
                 loaders:Dict[str, DataLoader], 
                ) -> None:
        self.output_dir:str = cfg.output_folder

        self._device:str = cfg.device
        self._num_epochs:int = cfg.train.num_epochs
        self._update_interval:int = cfg.train.update_interval # Reference: update embeddings every M times
        self._final_test:bool = cfg.train.final_test
        self._save_interval = cfg.train.save_interval
        self._target_only:bool = cfg.target_only

        self._feat_model:nn.Module = instantiate(cfg.model.model).to(self._device)
        # FIXME: DATA PARALLEL
        # if cfg.dp and (gpu_cnt := torch.cuda.device_count()) > 1:
        if False:
            print('Trainer use total', gpu_cnt, 'GPUs')
            self._feat_model = nn.parallel.DistributedDataParallel(self._feat_model, output_device=[cfg.gpu])

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
        
        self.__cfg_dict = OmegaConf.to_container(cfg)
    
    def _init_wandb(self) -> None:
        os.environ["WANDB_PROJECT"] = "SAMO ASVSpoof"
        wandb.init(
            job_type="SAMO",
            config=self.__cfg_dict,
            name=f"{self._device.upper()}-E{self._num_epochs}-T{int(self._target_only)}-DIM{self._feat_dim}-INIT={self._initialize_centers}"
        )
        wandb.watch(self._feat_model, log="all", log_freq=100, log_graph=False)

    @wandb_error_handler # Ensure that wandb will stop recording if error occur.
    def train(self) -> None:
        self._init_wandb()

        for epoch in tqdm(range(1, self._num_epochs+1), unit="epoch", desc='Epoch', position=0):
            log_train:dict = self._train_epoch(epoch)
            log_val:dict = self._val()

            if epoch % self._save_interval == 0:
                self._save_model(epoch, best_model=False)

            if self._best_val_loss > (val_loss:=log_val["val_loss"]):
                self._best_val_loss = val_loss
                self._save_model(best_model=True)
                self.feat_model.train() # IDK 
                self.optimizer_swa.update_parameters(self.feat_model)

            wandb.log({"epoch":epoch, **log_train, **log_val})
    
    def _train_epoch(self, epoch:int) -> None:
        self.feat_model.train()
        # print(f"\nEpoch: {epoch}") # tqdm progress bar bug?
        train_losses = []
        if epoch % self._update_interval == 0:
            self._update_embeddings() # update both "speakers's center" and "speaker to center"
        for i, (feat, labels, spk, _, _) in enumerate(tqdm(self._loaders["train"], unit="batch", position=1, desc='train batch', leave=False)):
            feat, labels = feat.to(self._device), labels.to(self._device)

            self.optimizer.zero_grad()

            embs, _ = self.feat_model(feat)

            w_spks = self.map_speakers_to_center(spks=spk, spk2center=self._train_spk2center)

            loss = self.loss_fn(embs, labels, self.w_centers, w_spks=w_spks)
            train_losses.append(loss.item())
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

        train_loss = np.nanmean(train_losses)
        logging.debug(f"{self.scheduler.get_last_lr()[0]=}")
        return {
            "train_loss": train_loss,
            "lr": self.scheduler.get_last_lr()[0]
        }
        
    @torch.no_grad
    def _val(self) -> Any:
        loss, labels, scores, _, _, _ = self.dev_n_eval(task="dev", 
                                                        loaders=self._loaders,
                                                        feat_model=self.feat_model, 
                                                        loss_fn=self.loss_fn, device=self._device,
                                                        target_only=self._target_only)
        
        # BUG: which val_labels == 0 or 1 should be placed first?
        # For computes EER, does the result remain the same?
        eer, _ = em.compute_eer(scores[labels == 0], scores[labels == 1])

        return {
            "val_loss": loss,
            "val_eer": eer
        }
    
    @staticmethod
    @torch.no_grad
    def dev_n_eval(task:str, loaders:Dict[str, DataLoader], feat_model:nn.Module, loss_fn:nn.Module, device:str, target_only:bool):
        assert task in ["eval", "dev"]
        feat_model.eval()
        val_centers, val_spk2center = Trainer.get_center_from_loader(loaders[task+"_enroll"], feat_model, device)
        batch_scores, batch_labels, val_losses, batch_utt, batch_tag, batch_spk = [], [], [], [], [], []
        # print(f"eval {task} set.")
        for i, (feat, labels, spk, utt, tag) in enumerate(tqdm(loaders[task], position=0, desc=f"{task} batch", leave=False)):
            feat, labels = feat.to(device), labels.to(device)
            embs, _ = feat_model(feat)
            w_spks = Trainer.map_speakers_to_center(spks=spk, spk2center=val_spk2center)
            if target_only:
                loss, score = loss_fn(embs, labels, w_centers=val_centers, w_spks=w_spks, get_score=True) # get_score for computer eer
            else: # TODO: Implement SAMO.inference to handle non-target only
                raise NotImplementedError # Not implement SAMO.inference yet
            
            val_losses.append(loss.item())
            batch_scores.append(score) # type(score) = tensor
            batch_labels.append(labels) # type(labels) = tensor
            batch_utt.extend(utt) # type(utt) = list
            batch_tag.extend(tag) # type(tag) = list
            batch_spk.extend(spk) # type(spk) = list
        
        val_loss = np.nanmean(val_losses)
        val_scores = torch.cat(batch_scores).cpu().numpy()
        val_labels = torch.cat(batch_labels).cpu().numpy()
        val_utt = batch_utt
        val_tag = batch_tag
        val_spk = batch_spk

        return val_loss, val_labels, val_scores, val_utt, val_tag, val_spk

    
    def _save_model(self, epoch:Optional[int] = None, best_model:bool = False) -> None:
        assert (epoch is not None) != best_model # not both set and not both unset
        wandb.unwatch(self._feat_model) # unhook before save

        if not best_model: # save interval (will be used for continue training)
            checkpoint = {
                "epoch": epoch,
                "feat_model": self.feat_model.state_dict(),
                "loss_fn": self.loss_fn.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "optimizer_swa": self._optimizer_swa.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "w_centers": self.w_centers,
            }
            torch.save(checkpoint, os.path.join(self.output_dir, "checkpoints", f"anti-spoofing_settings_{epoch}.pt"))
        else:
            torch.save(self._feat_model, os.path.join(self.output_dir, "anti-spoof_best_feat_model.pt"))
            torch.save(self.loss_fn, os.path.join(self.output_dir, "anti-spoof_best_loss_fn.pt"))
            torch.save(self.w_centers, os.path.join(self.output_dir, "anti-spoof_best_centers.pt"))

        wandb.watch(self._feat_model, log="all", log_freq=100, log_graph=False)
        
    @staticmethod
    @torch.no_grad
    def get_center_from_loader(loader:DataLoader, feat_model:nn.Module, device:str) -> Tensor:
        enroll_emb_dict = defaultdict(list)
        for batch_x, _, batch_spk, _, _ in loader:
            batch_x = batch_x.to(device)
            batch_cm_emb, _ = feat_model(batch_x)
            batch_cm_emb = batch_cm_emb.detach().cpu().numpy()

            for spk, cm_emb in zip(batch_spk, batch_cm_emb):
                enroll_emb_dict[spk].append(cm_emb)
        
        for spk, emb_np in enroll_emb_dict.items():
            enroll_emb_dict[spk] = Tensor(np.mean(emb_np, axis=0))
        return torch.stack(list(enroll_emb_dict.values())), enroll_emb_dict

    def _update_embeddings(self) -> None:
        avg_centers, spk2center = self.get_center_from_loader(self._loaders["train_bona"], self.feat_model, self._device)
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
    def optimizer_swa(self):
        return self._optimizer_swa
    @property
    def scheduler(self):
        return self._scheduler