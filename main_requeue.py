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
from custom_brax import custom_ppo_networks
from orbax import checkpoint as ocp
from flax.training import orbax_utils
# from envs.rodent import RodentSingleClip
from preprocessing.preprocess import process_clip_to_train
from envs.Fly_Env_Brax import FlyTracking, FlyMultiClipTracking
# from envs.fruitfly import Fruitfly_Tethered, Fruitfly_Run, FlyRunSim, FlyStand, Fruitfly_Freejnt
from utils.utils import *
from utils.fly_logging import log_eval_rollout

warnings.filterwarnings("ignore", category=DeprecationWarning)

from absl import app
from absl import flags

FLAGS = flags.FLAGS

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true " "--xla_gpu_triton_gemm_any=True "
)
envs.register_environment("fly_freejnt_clip", FlyTracking)
envs.register_environment("fly_freejnt_multiclip", FlyMultiClipTracking)


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
    assert n_gpus == cfg.num_gpus, 'Number of GPUs missmatched'
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

    #### TODO: Need to handle this better
    with open(reference_path, "rb") as file:
        # Use pickle.load() to load the data from the file
        reference_clip = pickle.load(file)
        
    global EVAL_STEPS
    EVAL_STEPS = 0
    ########## Handling requeuing ##########
    try: ##### TODO: Need to rework to load proper config as well. 
        # Try to recover a state file with the relevant variables stored
        # from previous stop if any
        model_path = cfg.paths.ckpt_dir / f"./{cfg.run_id}"
        if model_path.exists():
            ##### Get all the checkpoint files #####
            ckpt_files = sorted(list(model_path.glob('*[!.mp4]')))
            ##### Get the latest checkpoint #####
            max_ckpt = list(model_path.glob(f'*{max([int(file.stem) for file in ckpt_files])}'))[0]
            EVAL_STEPS = int(max_ckpt.stem)
            restore_checkpoint = max_ckpt.as_posix()
            cfg = OmegaConf.load(cfg.paths.log_dir / "run_config.yaml")
            cfg.dataset = cfg.dataset
            cfg.dataset.env_args = cfg.dataset.env_args
            env_cfg = cfg.dataset
            env_args = cfg.dataset.env_args
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


        episode_length = (env_args.clip_length - 50 - env_cfg.ref_traj_length) * env._steps_for_cur_frame
        print(f"episode_length {episode_length}")


        train_fn = functools.partial(
            ppo.train,
            num_envs=cfg.train["num_envs"],
            num_timesteps=cfg.train["num_timesteps"],
            num_evals=int(cfg.train["num_timesteps"] / cfg.train["eval_every"]),
            num_resets_per_eval=cfg.train['num_resets_per_eval'],
            reward_scaling=cfg.train['reward_scaling'],
            episode_length=episode_length,
            normalize_observations=True,
            action_repeat=cfg.train['action_repeat'],
            clipping_epsilon=cfg.train["clipping_epsilon"],
            unroll_length=cfg.train['unroll_length'],
            num_minibatches=cfg.train["num_minibatches"],
            num_updates_per_batch=cfg.train["num_updates_per_batch"],
            discounting=cfg.train['discounting'],
            learning_rate=cfg.train["learning_rate"],
            kl_weight=cfg.train["kl_weight"],
            entropy_cost=cfg.train['entropy_cost'],
            batch_size=cfg.train["batch_size"],
            seed=cfg.train['seed'],
            network_factory=functools.partial(
                custom_ppo_networks.make_intention_ppo_networks,
                encoder_hidden_layer_sizes=cfg.train['encoder_hidden_layer_sizes'],
                decoder_hidden_layer_sizes=cfg.train['decoder_hidden_layer_sizes'],
                value_hidden_layer_sizes=cfg.train['value_hidden_layer_sizes'],
            ),
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
            num_steps=int(num_steps)
            metrics["num_steps"] = num_steps
            wandb.log(metrics, commit=False)

        # Wrap the env in the brax autoreset and episode wrappers
        # if cfg.dataset.dname == "fly_run":
        #     rollout_env = custom_wrappers_old.RenderRolloutWrapperTracking_Run(env)
        # elif cfg.dataset.dname == 'fly_run_sim':
        #     rollout_env = custom_wrappers_old.RenderRolloutWrapperTracking_RunSim(env)
        # # elif cfg.dataset.dname == 'fly_stand':
        # #     rollout_env = custom_wrappers.RenderRolloutWrapperTracking_Stand(env)
        # else:
        rollout_env = custom_wrappers.RenderRolloutWrapperTracking(env)
        # define the jit reset/step functions
        jit_reset = jax.jit(rollout_env.reset)
        jit_step = jax.jit(rollout_env.step)

        def policy_params_fn(num_steps, make_policy, params, policy_params_fn_key, model_path=model_path):
            global EVAL_STEPS
            EVAL_STEPS = EVAL_STEPS + 1
            print(f'Eval Step: {EVAL_STEPS}, num_steps: {num_steps}')
            ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
            save_args = orbax_utils.save_args_from_target(params)
            path = model_path / f'{EVAL_STEPS:03d}'
            os.makedirs(path, exist_ok=True)
            ckptr.save(path, params, force=True, save_args=save_args)
            policy_params = (params[0],params[1].policy)
            Env_steps = params[2]
            jit_inference_fn = jax.jit(make_policy(policy_params, deterministic=True))
            reset_rng, act_rng = jax.random.split(policy_params_fn_key)

            state = jit_reset(reset_rng)

            rollout = [state]
            # rollout_len = env_args["clip_length"]*int(rollout_env._steps_for_cur_frame)
            rollout_len = 500
            for i in range(rollout_len):
                _, act_rng = jax.random.split(act_rng)
                obs = state.obs
                ctrl, extras = jit_inference_fn(obs, act_rng)
                state = jit_step(state, ctrl)
                rollout.append(state)
            ##### Log the rollout to wandb #####
            log_eval_rollout(cfg,rollout,state,env,reference_clip,model_path,num_steps)

        if not (cfg.paths.log_dir / "run_config.yaml").exists():
            OmegaConf.save(cfg, cfg.paths.log_dir / "run_config.yaml")
        print(OmegaConf.to_yaml(cfg))
        make_inference_fn, params, _ = train_fn(
            environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn
        )

        final_save_path = Path(f"{model_path}")/f'brax_ppo_{cfg.dataset.name}_run_finished'
        model.save_params(final_save_path, params)
        print(f'Run finished. Model saved to {final_save_path}')
    
    # Save current state 
    # if interrupted:
    #     model.save_params(f"{model_path}/{num_steps}", params)
    #     sys.exit(99)
    # sys.exit(0)

if __name__ == "__main__":
    main()
