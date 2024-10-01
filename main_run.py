import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1
import functools
import jax

n_gpus = jax.device_count(backend="gpu")
print(f"Using {n_gpus} GPUs")
# jax.config.update("jax_enable_x64", True)

from typing import Dict
import wandb
import imageio
import mujoco
from brax import envs

# from brax.training.agents.ppo import train as ppo
import numpy as np

# from envs.rodent import RodentSingleClip
import pickle
import warnings
from jax import numpy as jp
import hydra
from brax.io import model
from omegaconf import DictConfig, OmegaConf
from brax.training.agents.ppo import networks as ppo_networks
from custom_brax import custom_ppo as ppo
from custom_brax import custom_wrappers
from preprocessing.preprocess import process_clip_to_train
from envs.fruitfly import Fruitfly_Tethered, Fruitfly_Tethered_Free, Fruitfly_Run
from utils.utils import *

warnings.filterwarnings("ignore", category=DeprecationWarning)

from absl import app
from absl import flags

FLAGS = flags.FLAGS

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=True " "--xla_gpu_triton_gemm_any=True "
)


envs.register_environment("fly_single_clip", Fruitfly_Tethered)
envs.register_environment("fly_single_clip_freejnt", Fruitfly_Tethered_Free)
envs.register_environment("fly_run", Fruitfly_Run)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:

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
            start_step=env_cfg["clip_idx"] * env_cfg["clip_length"],
            clip_length=env_cfg["clip_length"],
            mjcf_path=env_args["mjcf_path"],
        )
        with open(reference_path, "wb") as file:
            # Use pickle.dump() to save the data to the file
            pickle.dump(reference_clip, file)

    # Init env
    env = envs.get_environment(
        cfg.train.env_name,
        reference_clip=reference_clip,
        **env_args,
    )

    # Episode length is equal to (clip length - random init range - traj length) * steps per cur frame
    # Will work on not hardcoding these values later
    episode_length = (
        env_cfg.clip_length - 50 - env_cfg.ref_traj_length
    ) * env._steps_for_cur_frame
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
    )

    import uuid

    # Generates a completely random UUID (version 4)
    run_id = uuid.uuid4()
    model_path = cfg.paths.ckpt_dir / f"./{run_id}"

    run = wandb.init(
        dir=cfg.paths.log_dir,
        project=cfg.train.wandb_project,
        config=OmegaConf.to_container(cfg),
        notes=cfg.train.note,
    )

    wandb.run.name = (
        f"{env_cfg['name']}_{cfg.train['task_name']}_{cfg.train['algo_name']}_{run_id}"
    )

    def wandb_progress(num_steps, metrics):
        metrics["num_steps"] = num_steps
        wandb.log(metrics, commit=False)

    # Wrap the env in the brax autoreset and episode wrappers
    # rollout_env = custom_wrappers.AutoResetWrapperTracking(env)
    rollout_env = custom_wrappers.RenderRolloutWrapperTracking_Run(env)
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
        for i in range(int(250 * rollout_env._steps_for_cur_frame)):
            _, act_rng = jax.random.split(act_rng)
            obs = state.obs
            ctrl, extras = jit_inference_fn(obs, act_rng)
            state = jit_step(state, ctrl)
            rollout.append(state)

        lin_vel = [state.metrics["tracking_lin_vel"] for state in rollout]
        table = wandb.Table(
            data=[
                [x, y] for (x, y) in zip(range(len(lin_vel)), lin_vel)
            ],
            columns=["frame", "lin_vel"],
        )
        wandb.log(
            {
                "eval/rollout_lin_vel": wandb.plot.line(
                    table,
                    "frame",
                    "lin_vel",
                    title="lin_vel for each rollout frame",
                )
            },
            commit=False,
        )

        orientation = [state.metrics["orientation"] for state in rollout]
        table = wandb.Table(
            data=[[x, y] for (x, y) in zip(range(len(orientation)), orientation)],
            columns=["frame", "orientation"],
        )
        wandb.log(
            {
                "eval/rollout_orientation": wandb.plot.line(
                    table,
                    "frame",
                    "orientation",
                    title="orientation for each rollout frame",
                )
            },
            commit=False,
        )
        
        action_rate = [state.metrics["action_rate"] for state in rollout]
        table = wandb.Table(
            data=[[x, y] for (x, y) in zip(range(len(action_rate)), action_rate)],
            columns=["frame", "action_rate"],
        )
        wandb.log(
            {
                "eval/rollout_action_rate": wandb.plot.line(
                    table,
                    "frame",
                    "action_rate",
                    title="action_rate for each rollout frame",
                )
            },
            commit=False,
        )
        
        total_dist = [state.metrics["total_dist"] for state in rollout]
        table = wandb.Table(
            data=[[x, y] for (x, y) in zip(range(len(total_dist)), total_dist)],
            columns=["frame", "total_dist"],
        )
        wandb.log(
            {
                "eval/rollout_total_dist": wandb.plot.line(
                    table,
                    "frame",
                    "total_dist",
                    title="total_dist for each rollout frame",
                )
            },
            commit=False,
        )
        
        torques = [state.metrics["torques"] for state in rollout]
        table = wandb.Table(
            data=[[x, y] for (x, y) in zip(range(len(torques)), torques)],
            columns=["frame", "torques"],
        )
        wandb.log(
            {
                "eval/rollout_torques": wandb.plot.line(
                    table,
                    "frame",
                    "torques",
                    title="torques for each rollout frame",
                )
            },
            commit=False,
        )
        
        
        stand_still = [state.metrics["stand_still"] for state in rollout]
        table = wandb.Table(
            data=[[x, y] for (x, y) in zip(range(len(stand_still)), stand_still)],
            columns=["frame", "stand_still"],
        )
        wandb.log(
            {
                "eval/rollout_stand_still": wandb.plot.line(
                    table,
                    "frame",
                    "stand_still",
                    title="stand_still for each rollout frame",
                )
            },
            commit=False,
        )
        

        thorax_heights = [
            state.pipeline_state.xpos[env._thorax_idx][2] for state in rollout
        ]
        table = wandb.Table(
            data=[[x, y] for (x, y) in zip(range(len(thorax_heights)), thorax_heights)],
            columns=["frame", "thorax_heights"],
        )
        wandb.log(
            {
                "eval/rollout_thorax_heights": wandb.plot.line(
                    table,
                    "frame",
                    "thorax_heights",
                    title="thorax_heights for each rollout frame",
                )
            },
            commit=False,
        )

        # Render the walker with the reference expert demonstration trajectory
        os.environ["MUJOCO_GL"] = "osmesa"
        qposes_rollout = np.array([state.pipeline_state.qpos for state in rollout])

        def f(x):
            if len(x.shape) != 1:
                return jax.lax.dynamic_slice_in_dim(
                    x,
                    0,
                    250,
                )
            return jp.array([])

        ref_traj = jax.tree_util.tree_map(f, reference_clip)
        if env.sys.jnt_type[0] != 0:
            qposes_ref = np.repeat(
                ref_traj.joints,
                env._steps_for_cur_frame,
                axis=0,
            )
        else:
            qposes_ref = np.repeat(
                np.hstack([ref_traj.position, ref_traj.quaternion, ref_traj.joints]),
                env._steps_for_cur_frame,
                axis=0,
            )

        spec = mujoco.MjSpec()
        spec.from_file(cfg.dataset.rendering_mjcf)
        mj_model = spec.compile()
        # mj_model = mujoco.MjModel.from_xml_path(cfg.dataset.rendering_mjcf)

        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }["cg"]
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_data = mujoco.MjData(mj_model)

        site_names = [
            mj_model.site(i).name
            for i in range(mj_model.nsite)
            if "-1" in mj_model.site(i).name
        ]
        site_id = [
            mj_model.site(i).id
            for i in range(mj_model.nsite)
            if "-1" in mj_model.site(i).name
        ]
        for id in site_id:
            mj_model.site(id).rgba = [1, 0, 0, 1]

        scene_option = mujoco.MjvOption()
        scene_option.sitegroup[:] = [1, 1, 1, 1, 1, 0]

        # save rendering and log to wandb
        os.environ["MUJOCO_GL"] = "osmesa"
        mujoco.mj_kinematics(mj_model, mj_data)
        renderer = mujoco.Renderer(mj_model, height=512, width=512)

        frames = []
        # render while stepping using mujoco
        video_path = f"{model_path}/{num_steps}.mp4"

        with imageio.get_writer(video_path, fps=int((1.0 / env.dt))) as video:
            for qpos1 in qposes_rollout:
                mj_data.qpos = qpos1
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data, camera=1, scene_option=scene_option)
                pixels = renderer.render()
                video.append_data(pixels)
                frames.append(pixels)

        wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})

    OmegaConf.save(cfg, cfg.paths.log_dir / "run_config.yaml")
    make_inference_fn, params, _ = train_fn(
        environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn
    )

    final_save_path = f"{model_path}/brax_ppo_rodent_run_finished"
    model.save_params(final_save_path, params)
    print(f"Run finished. Model saved to {final_save_path}")


if __name__ == "__main__":
    main()
