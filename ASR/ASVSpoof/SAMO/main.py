import hydra
from omegaconf import DictConfig, OmegaConf

import os

from samo.utils import setup_seed, output_dir_setup, cuda_checker
from samo.trainer import Trainer
from samo.test import test
from samo.utils import get_loader

@hydra.main(config_path="configs", config_name="cfg", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.verbose:
        print(20 *'-', f"{'Config': ^5}", 20 *'-', '\n')
        print(OmegaConf.to_yaml(cfg))
        print(48 * '-')

    setup_seed(cfg.seed)
    output_dir_setup(cfg)
    cuda_checker(cfg)

    loaders = get_loader(cfg)
    if cfg.test_only:
        test_feat_model_path = cfg.test.test_model
        test(test_feat_model_path, loaders, cfg)
    else:
        trainer = Trainer(cfg, loaders=loaders)
        trainer.train()
        test_feat_model_path = os.path.join(cfg.output_dir, "anti-spoof_best_feat_model.pt")
        if cfg.train.final_test:
            test(test_feat_model_path, loaders, cfg)
        
if __name__ == '__main__':
    main()