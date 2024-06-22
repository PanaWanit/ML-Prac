import hydra
from omegaconf import DictConfig, OmegaConf


import logging
from collections import defaultdict

from samo.utils import setup_seed, output_dir_setup, cuda_checker

@hydra.main(config_path="configs", config_name="cfg", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.verbose:
        print(20 *'-', f"{'Config': ^5}", 20 *'-', '\n')
        print(OmegaConf.to_yaml(cfg))
        print(48 * '-')
    
    setup_seed(cfg.seed)
    output_dir_setup(cfg)
    # cuda_checker(cfg)

    # feat_model = hydra.utils.instantiate(cfg.model.model)
    
    # test(cfg) if cfg.test_only else train(cfg)


if __name__ == '__main__':
    main()