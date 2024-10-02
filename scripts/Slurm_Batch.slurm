#!/bin/bash

#SBATCH --job-name=Fruitfly     ### Job Name
#SBATCH --partition=gpu-l40s ###  gpu ### ### Quality of Service (like a queue in PBS)
#SBATCH --account=portia
#SBATCH --time=2-00:00:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=1   ### Nuber of tasks to be launched per Node
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2              ### General REServation of gpu:number of gpus
#SBATCH --mem=256G
# #SBATCH --array=10 ###0-63 ###  ### Array index
#SBATCH --verbose  
#SBATCH -o ./OutFiles/slurm-%A_%a.out

##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eabe@uw.edu

module load cuda/12.2.2
set -x

source ~/.bashrc
module load cuda/12.2.2
nvidia-smi
conda activate stac-mjx-env
python -u main.py
## python -u /home/eabe/Research/MyRepos/DopamineModulation/TrainPatterns.py --version=${SLURM_ARRAY_TASK_ID}