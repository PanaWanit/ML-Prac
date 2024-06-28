from omegaconf import DictConfig
from typing import Dict, List

from hydra.utils import instantiate

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel
from collections import defaultdict

from tqdm.notebook import tqdm # Train on kaggle notebook.

class Trainer(object):
    def __init__(self, 
                 cfg:DictConfig,
                 loaders:Dict[str, DataLoader], 
                ) -> None:
        self._device = cfg.device
        self._num_epochs = cfg.train.num_epoch
        self._update_interval = cfg.train.update_interval # Reference: update embeddings every M times

        self._feat_model = instantiate(cfg.model.model, device=self._device)
        if cfg.dp and (gpu_cnt := torch.cuda.device_count()) > 1:
            print('Trainer use total', gpu_cnt, 'GPUs')
            self._feat_model = nn.DistributedDataParallel(self._feat_model, output_device=[cfg.gpu])

        self._loaders = loaders
        self._train_num_centers = loaders["train"].dataset.get_num_centers

        if cfg.continue_training:
            raise NotImplementedError()
        
        self._optimizer = instantiate(cfg.train.optim, params=self._feat_model.parameters())
        self._optimizer_swa = AveragedModel(self._feat_model, device=self._device)
        self._scheduler = instantiate(cfg.train.scheduler, optimizer=self._optimizer)

        self._loss_fn = instantiate(cfg.loss.fn)
        
        self._early_stop = 0
        self._best_val_loss = float("inf")


        self._feat_dim = cfg.enc_dim
        if cfg.initialize_centers == "one_hot":
            self._w_centers = torch.eye(cfg.enc_dim)[:self._train_num_centers]
        elif cfg.initialize_centers == "evenly":  # uniform_hypersphere
            raise NotImplementedError()
        else:
            raise RuntimeError("There is no {cfg.initialize_centers} method.")
    
    def _get_centers_from_loader(self, task:str) -> torch.Tensor:
        enroll_emb_dict = defaultdict(list)
        with torch.no_grad():
            for i, (batch_x, _, spk, _, _) in enumerate(self._loaders[task]):
                batch_x = batch_x.to(self._device)
                batch_cm_emb, _ = self._feat_model(batch_x)
                batch_cm_emb = batch_cm_emb.detach().cpu().numpy()

                for spk, cm_emb in zip(batch_cm_emb, spk):
                    enroll_emb_dict[spk].append(cm_emb)
            
            for spk, emb_np in enroll_emb_dict.items():
                enroll_emb_dict[spk] = torch.Tensor(emb_np.mean(axis=0))
        return torch.stack(list(enroll_emb_dict.values()))
    
    
    def _update_centers(self) -> None:
        self._w_centers = self._get_centers_from_loader("train")

    def train(self) -> None:
        for epoch in tqdm(range(1, self._num_epochs+1)): # understandable
            self._train_epoch(epoch)
    
    def _train_epoch(self, epoch:int) -> None:
        self._feat_model.train()
        print(f"\nEpoch: {epoch}")
        if epoch % self._update_interval == 0:
            self._update_centers()
        # TODO: Continue implementation


    @property
    def get_loaders(self):
        return self._loaders
    @property
    def get_feat_model(self):
        return self._feat_model
    @property
    def get_optimizer(self):
        return self._optimizer