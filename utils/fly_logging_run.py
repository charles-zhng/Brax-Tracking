import wandb
import jax
import mujoco
import os
import numpy as np
import imageio


def log_eval_rollout_run(cfg, rollout, state, env, reference_clip, model_path, num_steps):
    '''Log the rollout to wandb'''

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
