from omegaconf import DictConfig
from typing import Dict, List

from hydra.utils import instantiate

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel



class Trainer(object):
    def __init__(self, 
                 cfg:DictConfig,
                 loaders:Dict[str, DataLoader], 
                 num_centers:List[int]) -> None:
        self._device = cfg.device

        self._feat_model = instantiate(cfg.model.model, device=self._device)
        if cfg.dp and (gpu_cnt := torch.cuda.device_count()) > 1:
            print('Trainer use total', gpu_cnt, 'GPUs')
            self._feat_model = nn.DistributedDataParallel(self._feat_model, output_device=[cfg.gpu])

        self._loaders, self._num_centers = loaders, num_centers

        if cfg.continue_training:
            raise NotImplementedError()
        
        self._optimizer = instantiate(cfg.train.optim, params=self._feat_model.parameters())
        self._optimizer_swa = AveragedModel(self._feat_model, device=self._device)
        self._scheduler = instantiate(cfg.train.scheduler, optimizer=self._optimizer)
        self._loss_fn = nn.CrossEntropyLoss(weight=[.9, .1])

        self._monitor_loss = instantiate(cfg.loss.fn, num_centers=num_centers["train"])
        
        self._early_stop = 0
        self._best_val_loss = float("inf")


    def train(self) -> None:
        self.feat_model.train()
        pass
    
    def _train_epoch(self, epoch:int) -> None:
        pass


    @property
    def get_loaders(self):
        return self._loaders
    @property
    def get_feat_model(self):
        return self._feat_model
    @property
    def get_optimizer(self):
        return self._optimizer