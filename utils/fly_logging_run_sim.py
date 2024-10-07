import wandb
import jax
import mujoco
import os
import numpy as np
import imageio
import jax.numpy as jp

def log_eval_rollout_run_sim(cfg, rollout, state, env, model_path, num_steps):
    '''Log the rollout to wandb'''

    # Log the metrics for the rollout
    state_metric_list = [
       'forward_reward',
        'reward_linvel',
        'reward_quadctrl',
        'reward_alive',
        'x_position',
        'y_position',
        'distance_from_origin',
        'x_velocity',
        'y_velocity',
    ]
    for metric in state_metric_list:
        metric_values = [state.metrics[metric] for state in rollout]
        table = wandb.Table(
            data=[[x, y] for (x, y) in zip(range(len(metric_values)), metric_values)],
            columns=["frame", metric],
        )
        wandb.log(
            {
                f"eval/rollout_{metric}": wandb.plot.line(
                    table,
                    "frame",
                    metric,
                    title=f"{metric} for each rollout frame",
                )
            },
            commit=False,
        )
        
    # Log the info for the rollout
    # for info_metric in ['joint_distance','quat_distance','angvel_distance','endeff_distance']:
    #     info_metric_values = [state.info[info_metric] for state in rollout]
    #     table = wandb.Table(
    #         data=[[x, y] for (x, y) in zip(range(len(info_metric_values)), info_metric_values)],
    #         columns=["frame", info_metric],
    #     )
    #     wandb.log(
    #         {
    #             f"eval/rollout_{info_metric}": wandb.plot.line(
    #                 table,
    #                 "frame",
    #                 info_metric,
    #                 title=f"{info_metric} for each rollout frame",
    #             )
    #         },
    #         commit=False,
    #     )
        
    # thorax_heights = [
    #     state.pipeline_state.xpos[env._thorax_idx][2] for state in rollout
    # ]
    # table = wandb.Table(
    #     data=[[x, y] for (x, y) in zip(range(len(thorax_heights)), thorax_heights)],
    #     columns=["frame", "thorax_heights"],
    # )
    # wandb.log(
    #     {
    #         "eval/rollout_thorax_heights": wandb.plot.line(
    #             table,
    #             "frame",
    #             "thorax_heights",
    #             title="thorax_heights for each rollout frame",
    #         )
    #     },
    #     commit=False,
    # )
    
    # Render the walker with the reference expert demonstration trajectory
    os.environ["MUJOCO_GL"] = "osmesa"
    qposes_rollout = np.array([state.pipeline_state.qpos for state in rollout])

    
    repeats_per_frame = 1 #int(1/(env._mocap_hz*env.sys.mj_model.opt.timestep))
    spec = mujoco.MjSpec()
    spec.from_file(cfg.dataset.rendering_mjcf)
    # thorax0 = spec.find_body("thorax")
    # first_joint0 = thorax0.first_joint()
    # if (env._free_jnt == False) & ('free' in first_joint0.name):
    #     first_joint0.delete()
    #     thorax1 = spec.find_body("thorax")
    #     first_joint1 = thorax1.first_joint()
    #     first_joint1.delete()
    mj_model = spec.compile()
    spec = mujoco.MjSpec()

    mj_model.opt.solver = {
        "cg": mujoco.mjtSolver.mjSOL_CG,
        "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    }["cg"]
    mj_model.opt.iterations = cfg.dataset.env_args.iterations
    mj_model.opt.ls_iterations = cfg.dataset.env_args.ls_iterations
    mj_model.opt.timestep = env.sys.mj_model.opt.timestep
    mj_data = mujoco.MjData(mj_model)

    scene_option = mujoco.MjvOption()
    scene_option.sitegroup[:] = [1, 1, 1, 1, 1, 0]
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    
    # save rendering and log to wandb
    os.environ["MUJOCO_GL"] = "osmesa"
    mujoco.mj_kinematics(mj_model, mj_data)
    # renderer = mujoco.Renderer(mj_model, height=512, width=512)

    frames = []
    # render while stepping using mujoco
    video_path = f"{model_path}/{num_steps}.mp4"
    with mujoco.Renderer(mj_model, height=512, width=512) as renderer:
        with imageio.get_writer(video_path, fps=50) as video:
            for qpos1 in qposes_rollout:
                mj_data.qpos = qpos1
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data, camera=1, scene_option=scene_option)
                pixels = renderer.render()
                video.append_data(pixels)
                frames.append(pixels)

    wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})
