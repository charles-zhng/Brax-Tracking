#!/bin/bash

#SBATCH --job-name=Fruitfly     ### Job Name
#SBATCH --partition=ckpt ###  gpu ### ### Quality of Service (like a queue in PBS) gpu-l40s 
#SBATCH --account=portia
#SBATCH --time=2-00:00:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=1   ### Nuber of tasks to be launched per Node
#SBATCH --cpus-per-task=16
#SBATCH --gpus=8              ### General REServation of gpu:number of gpus
#SBATCH --mem=128G
# #SBATCH --array=10 ###0-63 ###  ### Array index
#SBATCH --verbose  
#SBATCH --exclude=g3090,g3107,g[3001-3037],n[3001-3431],z[3001-3011]
#SBATCH -o ./OutFiles/slurm-%A_%a.out

##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eabe@uw.edu

# CUDA_VISIBLE_DEVICES=1 python -u main.py paths=hyak train=train_fly_freejnt dataset=fly_freejnt train.note=hyak train.num_envs=1024
module load cuda/12.4.1
set -x
source ~/.bashrc
nvidia-smi
conda activate stac-mjx-env
python -u main_requeue.py paths=hyak train.note=hyak_ckpt version=ckpt train=train_fly dataset=fly train.num_envs=8192 num_gpus=8 run_id=$SLURM_JOB_ID 

#### cancel all jobs: squeue -u $USER -h | awk '{print $1}' | xargs scancel
### python scripts/slurm-run_bbrunton.py paths=hyak train=train_fly_run dataset=fly_run train.note=hyak train.num_envs=1024 gpu=0
