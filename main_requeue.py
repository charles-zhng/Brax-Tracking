import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1
import functools
import jax
# jax.config.update("jax_enable_x64", True)

n_gpus = jax.device_count(backend="gpu")
print(f"Using {n_gpus} GPUs")
from typing import Dict
import wandb
from brax import envs
import signal
import sys
import pickle
import warnings
import hydra
from brax.io import model
from omegaconf import DictConfig, OmegaConf
from brax.training.agents.ppo import networks as ppo_networks
from custom_brax import custom_ppo as ppo
from custom_brax import custom_wrappers
# from envs.rodent import RodentSingleClip
from preprocessing.preprocess import process_clip_to_train
from envs.fruitfly import Fruitfly_Tethered, Fruitfly_Tethered_Free, Fruitfly_Run
from utils.utils import *
from utils.fly_logging import log_eval_rollout
from utils.fly_logging_run import log_eval_rollout_run

warnings.filterwarnings("ignore", category=DeprecationWarning)

from absl import app
from absl import flags

FLAGS = flags.FLAGS

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true " "--xla_gpu_triton_gemm_any=True "
)

envs.register_environment("fly_single_clip", Fruitfly_Tethered)
envs.register_environment("fly_single_clip_freejnt", Fruitfly_Tethered_Free)
envs.register_environment("fly_run", Fruitfly_Run)

# Global Boolean variable that indicates that a signal has been received
interrupted = False

# Global Boolean variable that indicates then natural end of the computations
converged = False

# Definition of the signal handler. All it does is flip the 'interrupted' variable
def signal_handler(signum, frame):
    global interrupted
    interrupted = True

# Register the signal handler
signal.signal(signal.SIGTERM, signal_handler)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print('run_id:', cfg.run_id)
    # Create paths if they don't exist and Path objects
    for k in cfg.paths.keys():
        if k != "user":
            cfg.paths[k] = Path(cfg.paths[k])
            cfg.paths[k].mkdir(parents=True, exist_ok=True)
    env_cfg = cfg.dataset
    env_args = cfg.dataset.env_args
    reference_path = cfg.paths.data_dir / f"clips/{env_cfg['clip_idx']}.p"
    reference_path.parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(reference_path):
        with open(reference_path, "rb") as file:
            # Use pickle.load() to load the data from the file
            reference_clip = pickle.load(file)
    else:
        # Process rodent clip and save as pickle
        reference_clip = process_clip_to_train(
            env_cfg["stac_path"],
            start_step=env_cfg["clip_idx"] * env_args["clip_length"],
            clip_length=env_args["clip_length"],
            mjcf_path=env_args["mjcf_path"],
        )
        with open(reference_path, "wb") as file:
            # Use pickle.dump() to save the data to the file
            pickle.dump(reference_clip, file)

    ########## Handling requeuing ##########
    try:
        # Try to recover a state file with the relevant variables stored
        # from previous stop if any
        model_path = cfg.paths.ckpt_dir / f"./{cfg.run_id}"
        if model_path.exists():
            ##### Get all the checkpoint files #####
            ckpt_files = sorted(list(model_path.glob('*[!.mp4]')))
            ##### Get the latest checkpoint #####
            max_ckpt = list(model_path.glob(f'*{max([int(file.stem) for file in ckpt_files])}'))[0]
            restore_checkpoint = max_ckpt.as_posix()
        else:
            raise ValueError('Model path does not exist. Starting from scratch.')
    except (ValueError):
        # Otherwise bootstrap (start from scratch)
        print('Model path does not exist. Starting from scratch.')
        restore_checkpoint = None

    while not interrupted and not converged:
        # Init env
        env = envs.get_environment(
            cfg.train.env_name,
            reference_clip=reference_clip,
            **env_args,
        )

        # Episode length is equal to (clip length - random init range - traj length) * steps per cur frame
        # Will work on not hardcoding these values later
        episode_length = (
            env_args.clip_length - 50 - env_cfg.ref_traj_length
        )
        print(f"episode_length {episode_length}")

        train_fn = functools.partial(
            ppo.train,
            num_timesteps=cfg.train["num_timesteps"],
            num_evals=int(cfg.train["num_timesteps"] / cfg.train["eval_every"]),
            reward_scaling=1,
            episode_length=episode_length,
            normalize_observations=True,
            action_repeat=cfg.train["action_repeat"],
            unroll_length=cfg.train["unroll_length"],
            num_minibatches=cfg.train["num_minibatches"],
            num_updates_per_batch=cfg.train["num_updates_per_batch"],
            discounting=cfg.train["discounting"],
            learning_rate=cfg.train["learning_rate"],
            entropy_cost=cfg.train["entropy_cost"],
            num_envs=cfg.train["num_envs"],
            batch_size=cfg.train["batch_size"],
            seed=cfg.seed,
            network_factory=functools.partial(
                ppo_networks.make_ppo_networks,
                policy_hidden_layer_sizes=cfg.train["mlp_policy_layer_sizes"],
                value_hidden_layer_sizes=(256, 256),
            ),
            restore_checkpoint_path=restore_checkpoint
            
        )


        run = wandb.init(
            dir=cfg.paths.log_dir,
            project=cfg.train.wandb_project,
            config=OmegaConf.to_container(cfg),
            notes=cfg.train.note,
            id=f'{cfg.run_id}',
            resume="allow",
        )

        wandb.run.name = (
            f"{env_cfg['name']}_{cfg.train['task_name']}_{cfg.train['algo_name']}_{cfg.run_id}"
        )

        def wandb_progress(num_steps, metrics):
            metrics["num_steps"] = num_steps
            wandb.log(metrics, commit=False)

        # Wrap the env in the brax autoreset and episode wrappers
        if cfg.dataset.dname == "fly_run":
            rollout_env = custom_wrappers.RenderRolloutWrapperTracking_Run(env)
        else:
            rollout_env = custom_wrappers.RenderRolloutWrapperTracking(env)
        # define the jit reset/step functions
        jit_reset = jax.jit(rollout_env.reset)
        jit_step = jax.jit(rollout_env.step)

        def policy_params_fn(num_steps, make_policy, params, model_path=model_path):
            policy_params_key = jax.random.key(0)
            os.makedirs(model_path, exist_ok=True)
            model.save_params(f"{model_path}/{num_steps}", params)
            jit_inference_fn = jax.jit(make_policy(params, deterministic=True))
            _, policy_params_key = jax.random.split(policy_params_key)
            reset_rng, act_rng = jax.random.split(policy_params_key)

            state = jit_reset(reset_rng)

            rollout = [state]
            for i in range(env_args["clip_length"]):
                _, act_rng = jax.random.split(act_rng)
                obs = state.obs
                ctrl, extras = jit_inference_fn(obs, act_rng)
                state = jit_step(state, ctrl)
                rollout.append(state)
            
            ##### Log the rollout to wandb #####
            if cfg.dataset.dname == "fly_run":
                log_eval_rollout_run(cfg,rollout,state,env,reference_clip,model_path,num_steps)
            else:
                log_eval_rollout(cfg,rollout,state,env,reference_clip,model_path,num_steps)
            

        OmegaConf.save(cfg, cfg.paths.log_dir / "run_config.yaml")
        make_inference_fn, params, _ = train_fn(
            environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn
        )

        final_save_path = f"{model_path}"/f'brax_ppo_{cfg.dataset.name}_run_finished'
        model.save_params(final_save_path, params)
        print(f'Run finished. Model saved to {final_save_path}')
    
    # Save current state 
    # if interrupted:
    #     model.save_params(f"{model_path}/{num_steps}", params)
    #     sys.exit(99)
    # sys.exit(0)

if __name__ == "__main__":
    main()
