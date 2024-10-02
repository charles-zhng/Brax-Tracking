import subprocess


def slurm_submit(script):
    output = subprocess.check_output("sbatch", input=script, universal_newlines=True)
    job_id = output.strip().split()[-1]
    return job_id


def submit():
    """Submit job to cluster."""
    script = f"""#!/bin/bash
#SBATCH -A portia
#SBATCH -p gpus-l40s
#SBATCH --mem=256G
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH -J fruitfly
#SBATCH --gpus=2
#SBATCH -o ./OutFiles/slurm-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elliottabe@gmail.com
source ~/.bashrc
module load cuda/12.2.2
nvidia-smi
conda activate stac-mjx-env
python3 main.py
"""
    print(f"Submitting job")
    job_id = slurm_submit(script)
    print(job_id)


submit()
