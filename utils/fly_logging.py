import wandb
import jax
import mujoco
import os
import numpy as np
import imageio
import jax.numpy as jp


##### TODO: Make this more flexible for different environments. Add metic lists in cfg to change at runtime.
def log_eval_rollout(cfg, rollout, state, env, reference_clip, model_path, num_steps):
    '''Log the rollout to wandb'''
    
    # Log the metrics for the rollout
    reward_metrics = ['pos_reward', 'quat_reward', 'joint_reward', 'angvel_reward', 'bodypos_reward', 'endeff_reward', 'reward_ctrl', 'healthy_reward']
    for metric in reward_metrics:
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
    for info_metric in ['summed_pos_distance','joint_distance','quat_distance','angvel_distance','endeff_distance']:
        info_metric_values = [state.info[info_metric] for state in rollout]
        table = wandb.Table(
            data=[[x, y] for (x, y) in zip(range(len(info_metric_values)), info_metric_values)],
            columns=["frame", info_metric],
        )
        wandb.log(
            {
                f"eval/rollout_{info_metric}": wandb.plot.line(
                    table,
                    "frame",
                    info_metric,
                    title=f"{info_metric} for each rollout frame",
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
        if (not isinstance(x,str)):
            if (len(x.shape) != 1):
                return jax.lax.dynamic_slice_in_dim(
                    x,
                    0,
                    cfg.dataset.env_args["clip_length"],
                )
        return jp.array([])


    ref_traj = jax.tree_util.tree_map(f, reference_clip)
    
    repeats_per_frame = 1 #int(1/(env._mocap_hz*env.sys.mj_model.opt.timestep))
    spec = mujoco.MjSpec()
    spec.from_file(cfg.dataset.rendering_mjcf)
    thorax0 = spec.find_body("thorax-0")
    first_joint0 = thorax0.first_joint()
    if (env._free_jnt == False) & ('free' in first_joint0.name):
        qposes_ref = np.repeat(
            ref_traj.joints,
            repeats_per_frame,
            axis=0,
        )
        # qposes_ref = ref_traj.joints.copy()

        first_joint0.delete()
        thorax1 = spec.find_body("thorax-1")
        first_joint1 = thorax1.first_joint()
        first_joint1.delete()
    elif env._free_jnt == True: 
        # qposes_ref = np.hstack([ref_traj.position, ref_traj.quaternion, ref_traj.joints])
        qposes_ref = np.repeat(
            np.hstack([ref_traj.position, ref_traj.quaternion, ref_traj.joints]),
            repeats_per_frame,
            axis=0,
        )
        
    mj_model = spec.compile()

    # mj_model = mujoco.MjModel.from_xml_path(cfg.dataset.rendering_mjcf)

    mj_model.opt.solver = {
        "cg": mujoco.mjtSolver.mjSOL_CG,
        "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    }["cg"]
    mj_model.opt.iterations = cfg.dataset.env_args.iterations
    mj_model.opt.ls_iterations = cfg.dataset.env_args.ls_iterations
    mj_model.opt.timestep = env.sys.mj_model.opt.timestep
    
    mj_data = mujoco.MjData(mj_model)
    
    site_names = [
        mj_model.site(i).name
        for i in range(mj_model.nsite)
        if "-0" in mj_model.site(i).name
    ]
    site_id = [
        mj_model.site(i).id
        for i in range(mj_model.nsite)
        if "-0" in mj_model.site(i).name
    ]
    for id in site_id:
        mj_model.site(id).rgba = [1, 0, 0, 1]

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
    with imageio.get_writer(video_path, fps=50) as video:
        with mujoco.Renderer(mj_model, height=512, width=512) as renderer:
            for qpos1, qpos2 in zip(qposes_rollout, qposes_ref):
                mj_data.qpos = np.append(qpos1, qpos2)
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data, camera=1, scene_option=scene_option)
                pixels = renderer.render()
                video.append_data(pixels)
                frames.append(pixels)


    wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})
