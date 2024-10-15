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
#SBATCH --gpus={cfg.num_gpus}
#SBATCH --mem=128G
#SBATCH --verbose  
#SBATCH --open-mode=append
#SBATCH -o ./OutFiles/slurm-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eabe@uw.edu
#SBATCH --exclude=g3090,g3107
module load cuda/12.4.1
set -x
source ~/.bashrc
nvidia-smi
conda activate stac-mjx-env
python -u main_requeue.py paths=hyak train.note={cfg.train.note} version=ckpt train={cfg.train.name} dataset={cfg.dataset.dname} train.num_envs={cfg.num_gpus*cfg.train.num_envs} num_gpus={cfg.num_gpus} run_id=$SLURM_JOB_ID 
            """
    print(f"Submitting job")
    print(script)
    job_id = slurm_submit(script)
    print(job_id)


if __name__ == "__main__":
    submit()



##### Saving command ######
#  python scripts/slurm-run_bbrunton.py paths=hyak train=train_fly_run dataset=fly_run train.note=hyak train.num_envs=1024 gpu=0
# python scripts/slurm-run_bbrunton.py paths=hyak train.note=hyak train=train_fly_run_sim dataset=fly_run_sim train.num_envs=1024 num_gpus=2