import functools
import jax
from typing import Dict
import wandb
import imageio
import mujoco
from brax import envs

# from brax.training.agents.ppo import train as ppo
from custom_brax import custom_ppo as ppo
from custom_brax import custom_wrappers
from brax.io import model
import numpy as np
from envs.rodent import RodentSingleClip
import pickle
import warnings
from preprocessing.preprocess import process_clip_to_train
from jax import numpy as jp
from brax.training.agents.ppo import networks as ppo_networks
import hydra
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from absl import app
from absl import flags

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

FLAGS = flags.FLAGS

n_gpus = jax.device_count(backend="gpu")
print(f"Using {n_gpus} GPUs")

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true " "--xla_gpu_triton_gemm_any=True "
)


envs.register_environment("rodent_single_clip", RodentSingleClip)


@hydra.main(config_path="./configs", config_name="train_config", version_base=None)
def main(train_config: DictConfig):
    env_cfg = hydra.compose(config_name="env_config")
    env_cfg = OmegaConf.to_container(env_cfg, resolve=True)
    env_cfg = env_cfg[train_config.env_name]
    env_args = env_cfg["env_args"]

    reference_path = f"clips/{env_cfg['clip_idx']}.p"

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

    # Init env
    env = envs.get_environment(
        env_cfg["name"],
        reference_clip=reference_clip,
        **env_args,
    )

    # Instantiate the environment
    # env_name = config["env_name"]
    # env = envs.get_environment(
    #     env_name,
    #     track_pos=reference_clip.position,
    #     track_quat=reference_clip.quaternion,
    #     torque_actuators=config["torque_actuators"],
    #     terminate_when_unhealthy=config["terminate_when_unhealthy"],
    #     solver=config["solver"],
    #     iterations=config["iterations"],
    #     ls_iterations=config["ls_iterations"],
    #     too_far_dist=config["too_far_dist"],
    #     ctrl_cost_weight=config["ctrl_cost_weight"],
    #     pos_reward_weight=config["pos_reward_weight"],
    #     quat_reward_weight=config["quat_reward_weight"],
    #     healthy_reward=config["healthy_reward"],
    #     healthy_z_range=config["healthy_z_range"],
    #     physics_steps_per_control_step=config["physics_steps_per_control_step"],
    # )

    # Episode length is equal to (clip length - random init range - traj length) * steps per cur frame
    # Will work on not hardcoding these values later
    episode_length = (250 - 100 - 5) * env._steps_for_cur_frame
    print(f"episode_length {episode_length}")

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=config["num_timesteps"],
        num_evals=int(config["num_timesteps"] / config["eval_every"]),
        reward_scaling=1,
        episode_length=episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=16,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.99,
        learning_rate=config["learning_rate"],
        entropy_cost=1e-3,
        num_envs=config["num_envs"],
        batch_size=config["batch_size"],
        seed=0,
        network_factory=functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=(256, 256),
            value_hidden_layer_sizes=(256, 256),
        ),
    )

    import uuid

    # Generates a completely random UUID (version 4)
    run_id = uuid.uuid4()
    model_path = f"./model_checkpoints/{run_id}"

    run = wandb.init(project="vnl_debug", config=config, notes="quat + alive")

    wandb.run.name = (
        f"{config['env_name']}_{config['task_name']}_{config['algo_name']}_{run_id}"
    )

    def wandb_progress(num_steps, metrics):
        metrics["num_steps"] = num_steps
        wandb.log(metrics, commit=False)

    # Wrap the env in the brax autoreset and episode wrappers
    # rollout_env = custom_wrappers.AutoResetWrapperTracking(env)
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
        for i in range(int(250 * rollout_env._steps_for_cur_frame)):
            _, act_rng = jax.random.split(act_rng)
            obs = state.obs
            ctrl, extras = jit_inference_fn(obs, act_rng)
            state = jit_step(state, ctrl)
            rollout.append(state)

        pos_rewards = [state.metrics["pos_reward"] for state in rollout]
        table = wandb.Table(
            data=[[x, y] for (x, y) in zip(range(len(pos_rewards)), pos_rewards)],
            columns=["frame", "pos_rewards"],
        )
        wandb.log(
            {
                "eval/rollout_pos_rewards": wandb.plot.line(
                    table,
                    "frame",
                    "pos_rewards",
                    title="pos_rewards for each rollout frame",
                )
            },
            commit=False,
        )

        summed_pos_distances = [state.info["summed_pos_distance"] for state in rollout]
        table = wandb.Table(
            data=[
                [x, y]
                for (x, y) in zip(
                    range(len(summed_pos_distances)), summed_pos_distances
                )
            ],
            columns=["frame", "summed_pos_distances"],
        )
        wandb.log(
            {
                "eval/rollout_summed_pos_distances": wandb.plot.line(
                    table,
                    "frame",
                    "summed_pos_distances",
                    title="summed_pos_distances for each rollout frame",
                )
            },
            commit=False,
        )

        torso_heights = [
            state.pipeline_state.xpos[env._torso_idx][2] for state in rollout
        ]
        table = wandb.Table(
            data=[[x, y] for (x, y) in zip(range(len(torso_heights)), torso_heights)],
            columns=["frame", "torso_heights"],
        )
        wandb.log(
            {
                "eval/rollout_torso_heights": wandb.plot.line(
                    table,
                    "frame",
                    "torso_heights",
                    title="torso_heights for each rollout frame",
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
        qposes_ref = np.repeat(
            np.hstack([ref_traj.position, ref_traj.quaternion, ref_traj.joints]),
            env._steps_for_cur_frame,
            axis=0,
        )

        # Trying to align them when using the reset wrapper...
        # Doesn't work bc reset wrapper handles the done under the hood so it's always 0 :(
        # done_array = np.array([state.done for state in rollout])
        # reset_indices = np.where(done_array == 1.0)[0]
        # if reset_indices.shape[0] == 0:
        #     aligned_traj = qposes_ref
        # else:
        #     aligned_traj = np.zeros_like(qposes_rollout)
        #     # Set the first segment
        #     aligned_traj[: reset_indices[0] + 1] = qposes_ref[: reset_indices[0] + 1]

        #     # Iterate through reset points
        #     for i in range(len(reset_indices) - 1):
        #         start = reset_indices[i] + 1
        #         end = reset_indices[i + 1] + 1
        #         length = end - start
        #         aligned_traj[start:end] = qposes_ref[:length]

        #     # Set the last segment
        #     if reset_indices[-1] < len(done_array) - 1:
        #         start = reset_indices[-1] + 1
        #         length = len(done_array) - start
        #         aligned_traj[start:] = qposes_ref[:length]

        mj_model = mujoco.MjModel.from_xml_path(f"./models/rodent_pair.xml")

        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }["cg"]
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_data = mujoco.MjData(mj_model)

        # save rendering and log to wandb
        os.environ["MUJOCO_GL"] = "osmesa"
        mujoco.mj_kinematics(mj_model, mj_data)
        renderer = mujoco.Renderer(mj_model, height=512, width=512)

        frames = []
        # render while stepping using mujoco
        video_path = f"{model_path}/{num_steps}.mp4"

        with imageio.get_writer(video_path, fps=int((1.0 / env.dt))) as video:
            for qpos1, qpos2 in zip(qposes_ref, qposes_rollout):
                mj_data.qpos = np.append(qpos1, qpos2)
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data, camera=f"close_profile")
                pixels = renderer.render()
                video.append_data(pixels)
                frames.append(pixels)

        wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})

    make_inference_fn, params, _ = train_fn(
        environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn
    )

    final_save_path = f"{model_path}/brax_ppo_rodent_run_finished"
    model.save_params(final_save_path, params)
    print(f"Run finished. Model saved to {final_save_path}")
