import subprocess
import hydra
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if_multi', lambda pred, a, b: a if pred.name=='MULTIRUN' else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

def slurm_submit(script):
    output = subprocess.check_output("sbatch", input=script, universal_newlines=True)
    job_id = output.strip().split()[-1]
    return job_id

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def submit(cfg: DictConfig) -> None:
    """Submit job to cluster."""
    script = f"""#!/bin/bash
#SBATCH --job-name=Fruitfly    
#SBATCH --partition=gpu-l40s 
#SBATCH --account=portia
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --verbose  
#SBATCH -o ./OutFiles/slurm-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eabe@uw.edu
module load cuda/12.2.2
set -x
source ~/.bashrc
module load cuda/12.2.2
nvidia-smi
conda activate stac-mjx-env
CUDA_VISIBLE_DEVICES={cfg.gpu} python -u main_run.py paths=hyak train={cfg.train.name} dataset={cfg.dataset.dname} train.note=hyak train.num_envs={cfg.train.num_envs}
            """
    print(f"Submitting job")
    print(script)
    job_id = slurm_submit(script)
    print(job_id)


if __name__ == "__main__":
    submit()
