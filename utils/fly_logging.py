import wandb
import jax
import mujoco
import os
import numpy as np
import imageio


def log_eval_rollout(cfg, rollout, state, env, reference_clip, model_path, num_steps):
    '''Log the rollout to wandb'''
    
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

    bodypos_rewards = [state.metrics["bodypos_reward"] for state in rollout]
    table = wandb.Table(
        data=[[x, y] for (x, y) in zip(range(len(bodypos_rewards)), bodypos_rewards)],
        columns=["frame", "bodypos_rewards"],
    )
    wandb.log(
        {
            "eval/rollout_bodypos_rewards": wandb.plot.line(
                table,
                "frame",
                "bodypos_rewards",
                title="bodypos_rewards for each rollout frame",
            )
        },
        commit=False,
    )

    joint_rewards = [state.metrics["joint_reward"] for state in rollout]
    table = wandb.Table(
        data=[[x, y] for (x, y) in zip(range(len(joint_rewards)), joint_rewards)],
        columns=["frame", "joint_rewards"],
    )
    wandb.log(
        {
            "eval/rollout_joint_rewards": wandb.plot.line(
                table,
                "frame",
                "joint_rewards",
                title="joint_rewards for each rollout frame",
            )
        },
        commit=False,
    )

    summed_pos_distances = [state.info["summed_pos_distance"] for state in rollout]
    table = wandb.Table(
        data=[
            [x, y]
            for (x, y) in zip(range(len(summed_pos_distances)), summed_pos_distances)
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

    joint_distances = [state.info["joint_distance"] for state in rollout]
    table = wandb.Table(
        data=[[x, y] for (x, y) in zip(range(len(joint_distances)), joint_distances)],
        columns=["frame", "joint_distances"],
    )
    wandb.log(
        {
            "eval/rollout_joint_distances": wandb.plot.line(
                table,
                "frame",
                "joint_distances",
                title="joint_distances for each rollout frame",
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
    thorax0 = spec.find_body("thorax-0")
    first_joint0 = thorax0.first_joint()
    if (env._free_jnt == False) & ('free' in first_joint0.name):
        first_joint0.delete()
        thorax1 = spec.find_body("thorax-1")
        first_joint1 = thorax1.first_joint()
        first_joint1.delete()
    mj_model = spec.compile()

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
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # save rendering and log to wandb
    os.environ["MUJOCO_GL"] = "osmesa"
    mujoco.mj_kinematics(mj_model, mj_data)
    renderer = mujoco.Renderer(mj_model, height=512, width=512)

    frames = []
    # render while stepping using mujoco
    video_path = f"{model_path}/{num_steps}.mp4"
    assert len(qposes_ref) == len(qposes_rollout), f"qposes_ref and qposes_rollout must have the same length:{qposes_ref[0].shape},{qposes_rollout[0].shape}"
    with imageio.get_writer(video_path, fps=int((1.0 / env.dt))) as video:
        for qpos1, qpos2 in zip(qposes_ref, qposes_rollout):
            mj_data.qpos = np.append(qpos1, qpos2)
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=1, scene_option=scene_option)
            pixels = renderer.render()
            video.append_data(pixels)
            frames.append(pixels)

    wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})
