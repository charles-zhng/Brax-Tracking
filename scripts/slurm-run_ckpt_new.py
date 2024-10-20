import argparse
import subprocess
import sys

def slurm_submit(script):
    """
    Submit the SLURM script using sbatch and return the job ID.
    """
    try:
        # Use a list for the command and pass the script via stdin
        output = subprocess.check_output(["sbatch"], input=script, universal_newlines=True)
        job_id = output.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.output}", file=sys.stderr)
        sys.exit(1)

def submit(num_gpus, partition, job_name, mem, cpus, time, note, train, dataset, num_envs, load_jobid):
    """
    Construct and submit the SLURM script with the specified parameters.
    """
        # Define GPU configurations
    gpu_configs = {
        'a100': 'gpu:a100:8',
        'h100': 'nvidia_h100_80gb_hbm3',
        'a40': 'gpu:a40:8',
        'l40': 'gpu:l40:8', 
        'l40s': 'gpu:l40s:8', 
        # Add more GPU types here if needed
    }

    # gpu_resource = f"gpu:{gpu_configs[gpu_type]}:{num_gpus}"

    """Submit job to cluster."""
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}    
#SBATCH --partition={partition}
#SBATCH --account=portia
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus={num_gpus}
#SBATCH --mem={mem}G
#SBATCH --verbose  
#SBATCH --open-mode=append
#SBATCH -o ./OutFiles/slurm-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eabe@uw.edu
#SBATCH --exclude=g3090,g3107,g3097
module load cuda/12.4.1
set -x
source ~/.bashrc
nvidia-smi
conda activate stac-mjx-env
python -u main_requeue.py paths=hyak train.note={note} version=ckpt train={train} dataset={dataset} train.num_envs={num_gpus*num_envs} num_gpus={num_gpus} load_jobid={load_jobid} run_id=$SLURM_JOB_ID 
            """
    print(f"Submitting job")
    print(script)
    job_id = slurm_submit(script)
    print(job_id)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Submit a SLURM job with specified GPU type.')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPUs to request (default: 8)')
    parser.add_argument('--job_name', type=str, default='Fruitfly',
                        help='Name of the SLURM job (default: rodent)')
    parser.add_argument('--mem', type=int, default=128,
                        help='Memory in GB (default: 128)')
    parser.add_argument('--cpus', type=int, default=16,
                        help='Number of CPU cores (default: 16)')
    parser.add_argument('--time', type=str, default='2-00:00:00',
                        help='Time limit for the job day-hr-min-sec (default: 2-00:00:00)')
    parser.add_argument('--partition', type=str, default='ckpt-g2',
                        help='Partition to run job (default: ckpt-g2)')
    parser.add_argument('--note', type=str, default='hyak_ckpt',
                        help='Note for job (default: hyak_ckpt)')
    parser.add_argument('--train', type=str, default='train_fly_freejnt',
                        help='Name of train yaml (default: train_fly_freejnt)')
    parser.add_argument('--dataset', type=str, default='fly_freejnt',
                        help='Name of dataset yaml  (default: fly_freejnt)')
    parser.add_argument('--num_envs', type=int, default=2048,
                        help='Number of environments to run (default: 2048)')
    parser.add_argument('--load_jobid', type=str, default='',
                        help='JobID to resume training (default: '')')

    args = parser.parse_args()

    submit(
        num_gpus=args.num_gpus,
        job_name=args.job_name,
        mem=args.mem,
        cpus=args.cpus,
        time=args.time,
        partition=args.partition,
        note=args.note,
        train=args.train,
        dataset=args.dataset,
        num_envs=args.num_envs,
        load_jobid=args.load_jobid,
    )

if __name__ == "__main__":
    main()
    
##### Saving commands #####
#### cancel all jobs: squeue -u $USER -h | awk '{print $1}' | xargs scancel
# python scripts/slurm-run_ckpt_new.py --dataset=fly_freejnt --num_envs=2048 --note='hyak_ckpt'
# python scripts/slurm-run_ckpt_new.py --partition=gpu-l40s --train=train_fly_multiclip --dataset=fly_multiclip --num_envs=2048 --num_gpus=2 --note='hyak'

# python scripts/slurm-run_ckpt_new.py --train=train_fly_run --dataset=fly_run --num_envs=2048 --note='hyak_ckpt'
# python scripts/slurm-run_ckpt_new.py --partition=gpu-l40s --train=train_fly_multiclip --dataset=fly_multiclip --num_envs=2048 --note='hyak_ckpt' 