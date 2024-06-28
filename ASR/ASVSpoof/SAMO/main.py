import hydra
from omegaconf import DictConfig, OmegaConf


import logging
from collections import defaultdict

from samo.utils import setup_seed, output_dir_setup, cuda_checker
from samo.trainer import Trainer
from samo.utils import get_loader

@hydra.main(config_path="configs", config_name="cfg", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.verbose:
        print(20 *'-', f"{'Config': ^5}", 20 *'-', '\n')
        print(OmegaConf.to_yaml(cfg))
        print(48 * '-')

    setup_seed(cfg.seed)
    output_dir_setup(cfg)
    # cuda_checker(cfg)

    loaders, num_centers = get_loader(cfg)
    return 
    if cfg.test_only:
        pass
    else:
        trainer = Trainer(cfg, loaders=loaders, num_centers=num_centers)
        trainer.train()


if __name__ == '__main__':
    main()