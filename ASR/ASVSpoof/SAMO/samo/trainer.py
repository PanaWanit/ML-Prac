from omegaconf import DictConfig
from typing import Dict, Sequence, Any

from hydra.utils import instantiate
import os

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel
from collections import defaultdict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tqdm.notebook import tqdm # Train on kaggle notebook.

class Trainer(object):
    def __init__(self, 
                 cfg:DictConfig,
                 loaders:Dict[str, DataLoader], 
                ) -> None:
        self.output_dir:str = cfg.output_folder

        self._device:str = cfg.device
        self._num_epochs:int = cfg.train.num_epochs
        self._update_interval:int = cfg.train.update_interval # Reference: update embeddings every M times

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
        self._train_spks = loaders["train"].dataset.get_unique_speaker
        if cfg.initialize_centers == "one_hot":
            self._w_centers:Tensor = torch.eye(cfg.enc_dim)[:self._train_num_centers]
            self._train_spk2center = dict(zip(self._train_spks, self._w_centers))
        elif cfg.initialize_centers == "evenly":  # uniform_hypersphere
            raise NotImplementedError()
        else:
            raise RuntimeError("There is no {cfg.initialize_centers} method.")
    
    # TODO: train function: Continue implementation
    def train(self) -> None:
        for epoch in tqdm(range(1, self._num_epochs+1)): # understandable
            self._train_epoch(epoch)

    
    def _train_epoch(self, epoch:int) -> None:
        self.feat_model.train()
        print(f"\nEpoch: {epoch}")

        if epoch % self._update_interval == 0:
            self._update_embeddings() # update both "speakers's center" and "speaker to center"

        for i, (feat, labels, spk, _, _) in enumerate(tqdm(self._loaders["train"])):
            feat, labels = feat.to(self._device), labels.to(self._device)

            self.optimizer.zero_grad()

            embs, outputs = self.feat_model(feat)

            w_spks = self._map_speakers_to_center(spks=spk, spk2center=self._train_spk2center)

            loss = self.loss_fn(embs, labels, self.w_centers, w_spks=w_spks)
            loss.backward()

            self.optimizer.step()
            
            with open(os.path.join(self.output_dir, "train_loss.log"), "a") as f:
                f.write(f"{epoch:<10}{i:<10}{loss.item():<20}")
        
    
    # TODO: Validation and scheduling
    def _val() -> Any:
        raise NotImplementedError
        
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
    def _map_speakers_to_center(spks: Sequence[str],
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