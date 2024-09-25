from omegaconf import DictConfig, OmegaConf
from utils.utils import load_cfg
import hydra
from pathlib import Path

@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg: DictConfig) -> None:
    for k in cfg.paths.keys():
        if k != 'user':
            cfg.paths[k] = Path(cfg.paths[k])
            cfg.paths[k].mkdir(parents=True, exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.save(cfg, cfg.paths.save_dir / 'run_config.yaml')

if __name__ == "__main__":
    my_app()