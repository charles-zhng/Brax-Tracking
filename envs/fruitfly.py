import jax
from jax import numpy as jp

from brax.envs.base import PipelineEnv, State
from brax.base import Base, Motion, Transform
from brax.io import mjcf as mjcf_brax
from brax import math as brax_math
from brax import base
from typing import Any, List, Sequence

from utils import quaternions
# from dm_control.locomotion.walkers import rescale
# from dm_control import mjcf as mjcf_dm
from typing import List
import mujoco
from mujoco import mjx

import numpy as np

import os


class Fruitfly_Tethered(PipelineEnv):

    def __init__(
        self,
        reference_clip,
        center_of_mass: str,
        end_eff_names: List[str],
        appendage_names: List[str],
        body_names: List[str],
        joint_names: List[str],
        mocap_hz: int = 250,
        mjcf_path: str = "./assets/fruitfly/fruitfly_force_free.xml",
        scale_factor: float = 0.9,
        torque_actuators: bool = False,
        ref_len: int = 5,
        too_far_dist=0.1,
        bad_pose_dist=jp.inf,
        bad_quat_dist=jp.inf,
        ctrl_cost_weight=0.01,
        pos_reward_weight=0.0,
        quat_reward_weight=1.0,
        joint_reward_weight=1.0,
        angvel_reward_weight=1.0,
        bodypos_reward_weight=1.0,
        endeff_reward_weight=1.0,
        healthy_reward=0.25,
        healthy_z_range=(0.03, 0.5),
        physics_steps_per_control_step=10,
        reset_noise_scale=1e-4,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        terminate_when_unhealthy=True,
        free_jnt=True,
        **kwargs,
    ):
        spec = mujoco.MjSpec()
        spec.from_file(mjcf_path)
        thorax = spec.find_body("thorax")
        first_joint = thorax.first_joint()
        if (free_jnt == False) & (first_joint.name == "free"):
            first_joint.delete()
        root = spec.compile()
        # root = mujoco.MjModel.from_xml_path(mjcf_path)

        # Convert to torque actuators
        if torque_actuators:
            for actuator in root.find_all("actuator"):
                actuator.gainprm = [actuator.forcerange[1]]
                del actuator.biastype
                del actuator.biasprm

        mj_model = root
        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations
        mj_model.opt.timestep =2e-4
        mj_model.opt.jacobian = 0

        sys = mjcf_brax.load_model(mj_model)

        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        max_physics_steps_per_control_step = int(
            (1.0 / (mocap_hz * mj_model.opt.timestep))
        )

        super().__init__(sys, **kwargs)
        if max_physics_steps_per_control_step % physics_steps_per_control_step != 0:
            raise ValueError(
                f"physics_steps_per_control_step ({physics_steps_per_control_step}) must be a factor of ({max_physics_steps_per_control_step})"
            )

        self._steps_for_cur_frame = (
            max_physics_steps_per_control_step / physics_steps_per_control_step
        )

        print(f"self._steps_for_cur_frame: {self._steps_for_cur_frame}")

        self._thorax_idx = mujoco.mj_name2id(
            mj_model, mujoco.mju_str2Type("body"), center_of_mass
        )

        self._joint_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("joint"), joint)
                for joint in joint_names
            ]
        )

        self._body_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in body_names
            ]
        )

        # using this for appendage for now bc im to lazy to rename
        self._endeff_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in end_eff_names
            ]
        )
        self._free_jnt = free_jnt
        self._mocap_hz = mocap_hz
        self._bad_pose_dist = bad_pose_dist
        self._too_far_dist = too_far_dist
        self._bad_quat_dist = bad_quat_dist
        self._ref_traj = reference_clip
        self._ref_len = ref_len
        self._pos_reward_weight = pos_reward_weight
        self._quat_reward_weight = quat_reward_weight
        self._joint_reward_weight = joint_reward_weight
        self._angvel_reward_weight = angvel_reward_weight
        self._bodypos_reward_weight = bodypos_reward_weight
        self._endeff_reward_weight = endeff_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._terminate_when_unhealthy = terminate_when_unhealthy

    def reset(self, rng) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng_pos = jax.random.split(rng, 4)

        start_frame = jax.random.randint(rng, (), 0, 44)

        info = {
            "cur_frame": start_frame,
            "steps_taken_cur_frame": 0,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
        }

        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # Add pos (without z height)
        new_qpos = jp.array(self.sys.qpos0)

        # Add quat
        # new_qpos = qpos_with_pos.at[3:7].set(self._track_quat[start_frame])

        # Add noise
        qpos = new_qpos + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, start_frame)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "pos_reward": zero,
            "quat_reward": zero,
            "joint_reward": zero,
            "angvel_reward": zero,
            "bodypos_reward": zero,
            "endeff_reward": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "too_far": zero,
            "bad_pose": zero,
            "bad_quat": zero,
            "fall": zero,
        }
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        info = state.info.copy()
        info["steps_taken_cur_frame"] += 1
        info["cur_frame"] += jp.where(
            info["steps_taken_cur_frame"] == self._steps_for_cur_frame, 1, 0
        )
        info["steps_taken_cur_frame"] *= jp.where(
            info["steps_taken_cur_frame"] == self._steps_for_cur_frame, 0, 1
        )

        # Logic for getting current frame aligned with simulation time
        # cur_frame = (info["cur_frame"] + (data.time // (1 / self._mocap_hz))).astype(int)
        cur_frame = info["cur_frame"]
        if self._ref_traj.position is not None:
            track_pos = self._ref_traj.position
            pos_distance = data.qpos[:3] - track_pos[cur_frame]
            pos_reward = self._pos_reward_weight * jp.exp(
                -400 * jp.sum(pos_distance**2)
            )
            track_quat = self._ref_traj.quaternion
            quat_distance = jp.sum(
                self._bounded_quat_dist(data.qpos[3:7], track_quat[cur_frame]) ** 2
            )
            quat_reward = self._quat_reward_weight * jp.exp(-4.0 * quat_distance)
        else:
            pos_distance = jp.zeros(3)
            quat_distance = 0.0
            pos_reward = 0.0
            quat_reward = 0.0

        track_joints = self._ref_traj.joints
        joint_distance = jp.sum((data.qpos - track_joints[cur_frame])** 2) 
        joint_reward = self._joint_reward_weight * jp.exp(-0.5 * joint_distance)
        info["joint_distance"] = joint_distance

        track_angvel = self._ref_traj.angular_velocity
        angvel_reward = self._angvel_reward_weight * jp.exp(
            -0.5 * jp.sum((data.qvel[3:6] - track_angvel[cur_frame])** 2) 
        )
        track_bodypos = self._ref_traj.body_positions
        bodypos_reward = self._bodypos_reward_weight * jp.exp(
            -6.0
            * jp.sum(
                (
                    data.xpos[self._body_idxs]
                    - track_bodypos[cur_frame][self._body_idxs]
                ).flatten() ** 2
            )
        )

        endeff_reward = self._endeff_reward_weight * jp.exp(
            -0.75
            * jp.sum(
                (
                    data.xpos[self._endeff_idxs]
                    - track_bodypos[cur_frame][self._endeff_idxs]
                ).flatten()** 2
            )
        )

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.xpos[self._thorax_idx][2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.xpos[self._thorax_idx][2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        summed_pos_distance = jp.sum((pos_distance * jp.array([1.0, 1.0, 0.2])) ** 2)
        too_far = jp.where(summed_pos_distance > self._too_far_dist, 1.0, 0.0)
        info["summed_pos_distance"] = summed_pos_distance
        info["quat_distance"] = quat_distance
        bad_pose = jp.where(joint_distance > self._bad_pose_dist, 1.0, 0.0)
        bad_quat = jp.where(quat_distance > self._bad_quat_dist, 1.0, 0.0)
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(data, cur_frame)
        reward = (
            joint_reward
            + pos_reward
            + quat_reward
            + angvel_reward
            + bodypos_reward
            + endeff_reward
            + healthy_reward
            - ctrl_cost
        )
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        done = jp.max(jp.array([done, too_far, bad_pose, bad_quat]))

        # Handle nans during sim by resetting env
        reward = jp.nan_to_num(reward)
        obs = jp.nan_to_num(obs)

        from jax.flatten_util import ravel_pytree

        flattened_vals, _ = ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        nan = jp.where(num_nans > 0, 1.0, 0.0)
        done = jp.max(jp.array([nan, done]))

        state.metrics.update(
            pos_reward=pos_reward,
            quat_reward=quat_reward,
            joint_reward=joint_reward,
            angvel_reward=angvel_reward,
            bodypos_reward=bodypos_reward,
            endeff_reward=endeff_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            too_far=too_far,
            bad_pose=bad_pose,
            bad_quat=bad_quat,
            fall=1 - is_healthy,
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done, info=info
        )

    def _get_obs(self, data: mjx.Data, cur_frame: int) -> jp.ndarray:
        """Observes rodent body position, velocities, and angles."""

        # Get the relevant slice of the ref_traj
        def f(x):
            if len(x.shape) != 1:
                return jax.lax.dynamic_slice_in_dim(
                    x,
                    cur_frame + 1,
                    self._ref_len,
                )
            return jp.array([])

        ref_traj = jax.tree_util.tree_map(f, self._ref_traj)

        # track_pos_local = jax.vmap(
        #     lambda a, b: brax_math.rotate(a, b), in_axes=(0, None)
        # )(
        #     ref_traj.position - data.qpos[:3],
        #     data.qpos[3:7],
        # ).flatten()

        # quat_dist = jax.vmap(
        #     lambda a, b: brax_math.relative_quat(a, b), in_axes=(None, 0)
        # )(
        #     data.qpos[3:7],
        #     ref_traj.quaternion,
        # ).flatten()

        joint_dist = (ref_traj.joints - data.qpos)[:, self._joint_idxs].flatten()

        # TODO test if this works
        body_pos_dist_local = jax.vmap(
            lambda a, b: jax.vmap(brax_math.rotate, in_axes=(0, None))(a, b),
            in_axes=(0, None),
        )(
            (ref_traj.body_positions - data.xpos)[:, self._body_idxs],
            data.qpos[3:7],
        ).flatten()

        return jp.concatenate(
            [
                data.qpos,
                data.qvel,
                # data.cinert[1:].ravel(),
                # data.cvel[1:].ravel(),
                # data.qfrc_actuator,
                # track_pos_local,
                # quat_dist,
                joint_dist,
                body_pos_dist_local,
            ]
        )

    def _bounded_quat_dist(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Computes a quaternion distance limiting the difference to a max of pi/2.

        This function supports an arbitrary number of batch dimensions, B.

        Args:
          source: a quaternion, shape (B, 4).
          target: another quaternion, shape (B, 4).

        Returns:
          Quaternion distance, shape (B, 1).
        """
        source /= jp.linalg.norm(source, axis=-1, keepdims=True)
        target /= jp.linalg.norm(target, axis=-1, keepdims=True)
        # "Distance" in interval [-1, 1].
        dist = 2 * jp.einsum("...i,...i", source, target) ** 2 - 1
        # Clip at 1 to avoid occasional machine epsilon leak beyond 1.
        dist = jp.minimum(1.0, dist)
        # Divide by 2 and add an axis to ensure consistency with expected return
        # shape and magnitude.
        return 0.5 * jp.arccos(dist)[..., np.newaxis]


class Fruitfly_Tethered_Free(PipelineEnv):

    def __init__(
        self,
        reference_clip,
        center_of_mass: str,
        end_eff_names: List[str],
        appendage_names: List[str],
        body_names: List[str],
        joint_names: List[str],
        mocap_hz: int = 250,
        mjcf_path: str = "./assets/fruitfly/fruitfly_force_fast.xml",
        scale_factor: float = 0.9,
        torque_actuators: bool = False,
        ref_len: int = 5,
        too_far_dist=0.1,
        bad_pose_dist=jp.inf,
        bad_quat_dist=jp.inf,
        ctrl_cost_weight=0.01,
        pos_reward_weight=0.0,
        quat_reward_weight=1.0,
        joint_reward_weight=10.0,
        angvel_reward_weight=1.0,
        bodypos_reward_weight=1.0,
        endeff_reward_weight=1.0,
        healthy_reward=0.25,
        healthy_z_range=(0.03, 0.5),
        physics_steps_per_control_step=10,
        reset_noise_scale=1e-3,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        terminate_when_unhealthy=True,
        free_jnt=True,
        **kwargs,
    ):
        # root = mujoco.MjModel.from_xml_path(mjcf_path)
        spec = mujoco.MjSpec()
        spec.from_file(mjcf_path)
        thorax = spec.find_body("thorax")
        first_joint = thorax.first_joint()
        if (free_jnt == False) & (first_joint.name == "free"):
            first_joint.delete()
        root = spec.compile()

        # Convert to torque actuators
        if torque_actuators:
            for actuator in root.find_all("actuator"):
                actuator.gainprm = [actuator.forcerange[1]]
                del actuator.biastype
                del actuator.biasprm

        mj_model = root
        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations
        mj_model.opt.timestep = 2e-4
        mj_model.opt.jacobian = 0

        sys = mjcf_brax.load_model(mj_model)

        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        max_physics_steps_per_control_step = int(
            (1.0 / (mocap_hz * mj_model.opt.timestep))
        )

        super().__init__(sys, **kwargs)
        if max_physics_steps_per_control_step % physics_steps_per_control_step != 0:
            raise ValueError(
                f"physics_steps_per_control_step ({physics_steps_per_control_step}) must be a factor of ({max_physics_steps_per_control_step})"
            )

        self._steps_for_cur_frame = (
            max_physics_steps_per_control_step / physics_steps_per_control_step
        )

        print(f"self._steps_for_cur_frame: {self._steps_for_cur_frame}")

        self._thorax_idx = mujoco.mj_name2id(
            mj_model, mujoco.mju_str2Type("body"), center_of_mass
        )

        self._joint_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("joint"), joint)
                for joint in joint_names
            ]
        )

        self._body_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in body_names
            ]
        )

        # using this for appendage for now bc im to lazy to rename
        self._endeff_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in end_eff_names
            ]
        )

        self._free_jnt = free_jnt
        self._mocap_hz = mocap_hz
        self._bad_pose_dist = bad_pose_dist
        self._too_far_dist = too_far_dist
        self._bad_quat_dist = bad_quat_dist
        self._ref_traj = reference_clip
        self._ref_len = ref_len
        self._pos_reward_weight = pos_reward_weight
        self._quat_reward_weight = quat_reward_weight
        self._joint_reward_weight = joint_reward_weight
        self._angvel_reward_weight = angvel_reward_weight
        self._bodypos_reward_weight = bodypos_reward_weight
        self._endeff_reward_weight = endeff_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._terminate_when_unhealthy = terminate_when_unhealthy

    def reset(self, rng) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng_pos = jax.random.split(rng, 4)

        start_frame = jax.random.randint(rng, (), 0, 44)

        info = {
            "cur_frame": start_frame,
            "steps_taken_cur_frame": 0,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
        }

        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # Add pos (without z height)
        new_qpos = jp.array(self.sys.qpos0)

        # Add quat
        # new_qpos = qpos_with_pos.at[3:7].set(self._track_quat[start_frame])

        # Add noise
        qpos = new_qpos + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, start_frame)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "pos_reward": zero,
            "quat_reward": zero,
            "joint_reward": zero,
            "angvel_reward": zero,
            "bodypos_reward": zero,
            "endeff_reward": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "too_far": zero,
            "bad_pose": zero,
            "bad_quat": zero,
            "fall": zero,
        }
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        info = state.info.copy()
        info["steps_taken_cur_frame"] += 1
        info["cur_frame"] += jp.where(
            info["steps_taken_cur_frame"] == self._steps_for_cur_frame, 1, 0
        )
        info["steps_taken_cur_frame"] *= jp.where(
            info["steps_taken_cur_frame"] == self._steps_for_cur_frame, 0, 1
        )

        # Logic for getting current frame aligned with simulation time
        # cur_frame = (info["cur_frame"] + (data.time // (1 / self._mocap_hz))).astype(int)
        cur_frame = info["cur_frame"]
        if self._ref_traj.position is not None:
            track_pos = self._ref_traj.position
            pos_distance = data.qpos[:3] - track_pos[cur_frame]
            pos_reward = self._pos_reward_weight * jp.exp(
                -400 * jp.sum(pos_distance ** 2)
            )
            track_quat = self._ref_traj.quaternion
            quat_distance = jp.sum(
                self._bounded_quat_dist(data.qpos[3:7], track_quat[cur_frame]) ** 2
            )
            quat_reward = self._quat_reward_weight * jp.exp(-4.0 * quat_distance)
        else:
            pos_distance = jp.zeros(3)
            quat_distance = 0.0
            pos_reward = 0.0
            quat_reward = 0.0

        track_joints = self._ref_traj.joints
        joint_distance = jp.sum((data.qpos[7:] - track_joints[cur_frame])** 2) 
        joint_reward = self._joint_reward_weight * jp.exp(-0.5 * joint_distance)
        info["joint_distance"] = joint_distance

        track_angvel = self._ref_traj.angular_velocity
        angvel_reward = self._angvel_reward_weight * jp.exp(
            -0.5 * jp.sum(data.qvel[3:6] - track_angvel[cur_frame]) ** 2
        )
        track_bodypos = self._ref_traj.body_positions
        bodypos_reward = self._bodypos_reward_weight * jp.exp(
            -6.0
            * jp.sum(
                (
                    data.xpos[self._body_idxs]
                    - track_bodypos[cur_frame][self._body_idxs]
                ).flatten() ** 2
            )
        )

        endeff_reward = self._endeff_reward_weight * jp.exp(
            -0.75
            * jp.sum(
                (
                    data.xpos[self._endeff_idxs]
                    - track_bodypos[cur_frame][self._endeff_idxs]
                ).flatten()** 2
            )
        )

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.xpos[self._thorax_idx][2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.xpos[self._thorax_idx][2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        summed_pos_distance = jp.sum((pos_distance * jp.array([1.0, 1.0, 0.2])) ** 2)
        too_far = jp.where(summed_pos_distance > self._too_far_dist, 1.0, 0.0)
        info["summed_pos_distance"] = summed_pos_distance
        info["quat_distance"] = quat_distance
        bad_pose = jp.where(joint_distance > self._bad_pose_dist, 1.0, 0.0)
        bad_quat = jp.where(quat_distance > self._bad_quat_dist, 1.0, 0.0)
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(data, cur_frame)
        reward = (
            joint_reward
            + pos_reward
            + quat_reward
            + angvel_reward
            + bodypos_reward
            + endeff_reward
            + healthy_reward
            - ctrl_cost
        )
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        done = jp.max(jp.array([done, too_far, bad_pose, bad_quat]))

        # Handle nans during sim by resetting env
        reward = jp.nan_to_num(reward)
        obs = jp.nan_to_num(obs)

        from jax.flatten_util import ravel_pytree

        flattened_vals, _ = ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        nan = jp.where(num_nans > 0, 1.0, 0.0)
        done = jp.max(jp.array([nan, done]))

        state.metrics.update(
            pos_reward=pos_reward,
            quat_reward=quat_reward,
            joint_reward=joint_reward,
            angvel_reward=angvel_reward,
            bodypos_reward=bodypos_reward,
            endeff_reward=endeff_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            too_far=too_far,
            bad_pose=bad_pose,
            bad_quat=bad_quat,
            fall=1 - is_healthy,
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done, info=info
        )

    def _get_obs(self, data: mjx.Data, cur_frame: int) -> jp.ndarray:
        """Observes rodent body position, velocities, and angles."""

        # Get the relevant slice of the ref_traj
        def f(x):
            if len(x.shape) != 1:
                return jax.lax.dynamic_slice_in_dim(
                    x,
                    cur_frame + 1,
                    self._ref_len,
                )
            return jp.array([])

        ref_traj = jax.tree_util.tree_map(f, self._ref_traj)

        track_pos_local = jax.vmap(
            lambda a, b: brax_math.rotate(a, b), in_axes=(0, None)
        )(
            ref_traj.position - data.qpos[:3],
            data.qpos[3:7],
        ).flatten()

        quat_dist = jax.vmap(
            lambda a, b: brax_math.relative_quat(a, b), in_axes=(None, 0)
        )(
            data.qpos[3:7],
            ref_traj.quaternion,
        ).flatten()

        joint_dist = (ref_traj.joints - data.qpos[7:])[:, self._joint_idxs].flatten()

        # TODO test if this works
        body_pos_dist_local = jax.vmap(
            lambda a, b: jax.vmap(brax_math.rotate, in_axes=(0, None))(a, b),
            in_axes=(0, None),
        )(
            (ref_traj.body_positions - data.xpos)[:, self._body_idxs],
            data.qpos[3:7],
        ).flatten()

        return jp.concatenate(
            [
                data.qpos,
                data.qvel,
                # data.cinert[1:].ravel(),
                # data.cvel[1:].ravel(),
                # data.qfrc_actuator,
                track_pos_local,
                quat_dist,
                joint_dist,
                body_pos_dist_local,
            ]
        )

    def _bounded_quat_dist(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Computes a quaternion distance limiting the difference to a max of pi/2.

        This function supports an arbitrary number of batch dimensions, B.

        Args:
          source: a quaternion, shape (B, 4).
          target: another quaternion, shape (B, 4).

        Returns:
          Quaternion distance, shape (B, 1).
        """
        source /= jp.linalg.norm(source, axis=-1, keepdims=True)
        target /= jp.linalg.norm(target, axis=-1, keepdims=True)
        # "Distance" in interval [-1, 1].
        dist = 2 * jp.einsum("...i,...i", source, target) ** 2 - 1
        # Clip at 1 to avoid occasional machine epsilon leak beyond 1.
        dist = jp.minimum(1.0, dist)
        # Divide by 2 and add an axis to ensure consistency with expected return
        # shape and magnitude.
        return 0.5 * jp.arccos(dist)[..., np.newaxis]


from ml_collections import config_dict



def get_config():
    """Returns reward config for barkour quadruped environment."""

    def get_default_rewards_config():
        default_config = config_dict.ConfigDict(
            dict(
                # The coefficients for all reward terms used for training. All
                # physical quantities are in SI units, if no otherwise specified,
                # i.e. joint positions are in rad, positions are measured in meters,
                # torques in Nm, and time in seconds, and forces in Newtons.
                scales=config_dict.ConfigDict(
                    dict(
                        # Tracking rewards are computed using exp(-delta^2/sigma)
                        # sigma can be a hyperparameters to tune.
                        # Track the base x-y velocity (no z-velocity tracking.)
                        tracking_lin_vel=10,
                        # Penalize non-zero roll and pitch angles. L2 penalty.
                        orientation=-2.0,
                        # L2 regularization of joint torques, |tau|^2.
                        torques=0, # -0.0002
                        # Penalize the change in the action and encourage smooth
                        # actions. L2 regularization |action - last_action|^2
                        action_rate=-0.01,
                        # Encourage long swing steps.  However, it does not
                        stand_still=-0.5,
                        # Early termination penalty.
                        termination=-10.0,
                        # ignore position reward
                        pos_reward=0,
                        # encourage the robot to face forward
                        quat_reward=0.0,
                        # encourage the robot to keep its joints close to the reference
                        joint_reward=0.0,
                        # encourage the robot to keep its angular velocity close to the reference
                        angvel_reward=0.0,
                        # encourage the robot to keep its body positions close to the reference
                        bodypos_reward=0,
                        # encourage the robot to keep its end effectors close to the reference
                        endeff_reward=0,
                        # 
                        healthy_reward = 10.0,
                        
                        
                    )
                ),
                # Tracking reward = exp(-error^2/sigma).
                tracking_sigma=0.25,
            )
        )
        return default_config

    default_config = config_dict.ConfigDict(
        dict(
            rewards=get_default_rewards_config(),
        )
    )

    return default_config

class Fruitfly_Run(PipelineEnv):
    """Environment for training the barkour quadruped joystick policy in MJX."""

    def __init__(
        self,
        reference_clip,
        center_of_mass: str,
        end_eff_names: List[str],
        appendage_names: List[str],
        body_names: List[str],
        joint_names: List[str],
        mocap_hz: int = 250,
        mjcf_path: str = "./assets/fruitfly/fruitfly_force_fast.xml",
        obs_noise: float = 0.05,
        action_scale: float = 3,
        ref_len: int = 5,
        too_far_dist=0.1,
        bad_pose_dist=jp.inf,
        bad_quat_dist=jp.inf,
        ctrl_cost_weight=0.01,
        pos_reward_weight=0.0,
        quat_reward_weight=1.0,
        joint_reward_weight=10.0,
        angvel_reward_weight=1.0,
        bodypos_reward_weight=0.0,
        endeff_reward_weight=0.0,
        healthy_reward=0.25,
        healthy_z_range=(0.03, 0.5),
        physics_steps_per_control_step=10,
        reset_noise_scale=1e-3,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        terminate_when_unhealthy=True,
        free_jnt=True,
        **kwargs,
    ):
        spec = mujoco.MjSpec()
        spec.from_file(mjcf_path)
        thorax = spec.find_body("thorax")
        first_joint = thorax.first_joint()
        if (free_jnt == False) & (first_joint.name == "free"):
            first_joint.delete()
        root = spec.compile()

        
        mj_model = root
        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations
        mj_model.opt.timestep =2e-4
        mj_model.opt.jacobian = 0

        sys = mjcf_brax.load_model(mj_model)

        self._dt = 0.002  # this environment is 50 fps
        # sys = sys.tree_replace({"opt.timestep": 0.002})

        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)

        print(physics_steps_per_control_step)
        super().__init__(sys, backend="mjx", n_frames=physics_steps_per_control_step, debug=True)

        self.reward_config = get_config()
        # set custom from kwargs
        for k, v in kwargs.items():
            if k.endswith("_scale"):
                self.reward_config.rewards.scales[k[:-6]] = v

        self._thorax_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "thorax"
        )
        max_physics_steps_per_control_step = int(
            (1.0 / (mocap_hz * sys.opt.timestep))
        )
        self._steps_for_cur_frame = (
            max_physics_steps_per_control_step / physics_steps_per_control_step
        )
        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._reset_noise_scale = reset_noise_scale
        self._init_q = jp.array(sys.mj_model.keyframe("home").qpos)
        self._default_pose = sys.mj_model.keyframe("home").qpos[7:]
        self.lowers = jp.array(
            [
                -2,
            ]
            * 36
        )
        self.uppers = jp.array(
            [
                2,
            ]
            * 36
        )
        
        self._thorax_idx = mujoco.mj_name2id(
            mj_model, mujoco.mju_str2Type("body"), center_of_mass
        )

        self._joint_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("joint"), joint)
                for joint in joint_names
            ]
        )

        self._body_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in body_names
            ]
        )
        _endeff_idxs = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in end_eff_names
        ]
        assert not any(id_ == -1 for id_ in _endeff_idxs), "Site not found."
        self._endeff_idxs = jp.array(_endeff_idxs)
        lower_leg_body = [
            "tarsus_T1_left",
            "tarsus_T1_right",
            "tarsus_T2_left",
            "tarsus_T2_right",
            "tarsus_T3_left",
            "tarsus_T3_right",
        ]
        lower_leg_body_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
            for l in lower_leg_body
        ]
        assert not any(id_ == -1 for id_ in lower_leg_body_id), "Body not found."
        self._lower_leg_body_id = jp.array(lower_leg_body_id)
        self._foot_radius = 0.0175
        self._nv = sys.nv
        self._nq = sys.nq
        self._nu = sys.nu
        
        self._free_jnt = free_jnt
        self._mocap_hz = mocap_hz
        self._bad_pose_dist = bad_pose_dist
        self._too_far_dist = too_far_dist
        self._bad_quat_dist = bad_quat_dist
        self._ref_traj = reference_clip
        self._ref_len = ref_len
        self._pos_reward_weight = pos_reward_weight
        self._quat_reward_weight = quat_reward_weight
        self._joint_reward_weight = joint_reward_weight
        self._angvel_reward_weight = angvel_reward_weight
        self._bodypos_reward_weight = bodypos_reward_weight
        self._endeff_reward_weight = endeff_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._terminate_when_unhealthy = terminate_when_unhealthy

    def sample_command(self, rng: jax.Array) -> jax.Array:
        lin_vel_x = [0.01, 0.05]  # min max [m/s]
        # lin_vel_y = [-0.8, 0.8]  # min max [m/s]
        # ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]

        _, key1 = jax.random.split(rng, 2)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        # lin_vel_y = jax.random.uniform(
        #     key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        # )
        # ang_vel_yaw = jax.random.uniform(
        #     key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        # )
        new_cmd = jp.array([lin_vel_x[0], 0, 0])
        return new_cmd

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)

        data = self.pipeline_init(self._init_q, jp.zeros(self._nv))

        info = {
            "rng": rng,
            "cur_frame": 0,
            "steps_taken_cur_frame": 0,
            "last_act": jp.zeros(self._nu),
            "last_vel": jp.zeros(self._nv),
            "command": self.sample_command(key),
            "last_contact": jp.zeros(6, dtype=bool),
            "rewards": {k: 0.0 for k in self.reward_config.rewards.scales.keys()},
            "step": 0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
        }

        obs_history = jp.zeros(15 * 91)  # store 15 steps of history
        obs = self._get_obs(data, info, obs_history)
        reward, done = jp.zeros(2)
        metrics = {"total_dist": 0.0}
        for k in info["rewards"]:
            metrics[k] = info["rewards"][k]

        return State(
            data, obs, reward, done, metrics, info
        )  # pytype: disable=wrong-arg-types

    def step(
        self, state: State, action: jax.Array
    ) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)
        
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        
        state.info["steps_taken_cur_frame"] += 1
        state.info["cur_frame"] += jp.where(
            state.info["steps_taken_cur_frame"] == self._steps_for_cur_frame, 1, 0
        )
        state.info["steps_taken_cur_frame"] *= jp.where(
            state.info["steps_taken_cur_frame"] == self._steps_for_cur_frame, 0, 1
        )
        # physics step
        cur_frame = state.info["cur_frame"]
        track_pos = self._ref_traj.position
        pos_distance = data.qpos[:3] - track_pos[cur_frame]
        pos_reward = jp.exp(
            -400 * jp.sum(pos_distance** 2)
        )
        track_quat = self._ref_traj.quaternion
        quat_distance = jp.sum(
            self._bounded_quat_dist(data.qpos[3:7], track_quat[cur_frame]) ** 2
        )
        quat_reward = jp.exp(-4.0 * quat_distance)
        
        track_joints = self._ref_traj.joints
        joint_distance = jp.sum((data.qpos[7:] - track_joints[cur_frame]) ** 2)
        joint_reward = jp.exp(-0.5 * joint_distance)
        state.info["joint_distance"] = joint_distance

        track_angvel = self._ref_traj.angular_velocity
        angvel_reward = jp.exp(
            -0.5 * jp.sum((data.qvel[3:6] - track_angvel[cur_frame]) ** 2)
        )
        track_bodypos = self._ref_traj.body_positions
        bodypos_reward = jp.exp(
            -6.0
            * jp.sum(
                (
                    data.xpos[self._body_idxs]
                    - track_bodypos[cur_frame][self._body_idxs]
                ).flatten()** 2
            )
            
        )

        endeff_reward = jp.exp(
            -0.75
            * jp.sum(
                (
                    data.xpos[self._endeff_idxs]
                    - track_bodypos[cur_frame][self._endeff_idxs]
                ).flatten()** 2
            )
            
        )
        x, xd = data.x, data.xd
        
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.xpos[self._thorax_idx][2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.xpos[self._thorax_idx][2] > max_z, 0.0, is_healthy)
        healthy_reward = self._healthy_reward
        state.info["quat_distance"] = quat_distance
        bad_quat = jp.where(quat_distance > self._bad_quat_dist, 1.0, 0.0)
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        # jax.debug.print(f'is_healthy: {is_healthy}')



        # observation data
        obs = self._get_obs(data, state.info, state.obs)
        joint_angles = data.q[7:]
        joint_vel = data.qd  ##### need to restrict to only legs

        # foot contact data based on z-position
        foot_pos = data.site_xpos[
            self._endeff_idxs
        ]  # pytype: disable=attribute-error
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor


        # done if joint limits are reached or robot is falling
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.dot(brax_math.rotate(up, x.rot[self._thorax_idx]), up) < 0
        done |= jp.any(joint_angles < self.lowers)
        done |= jp.any(joint_angles > self.uppers)
        done |= data.x.pos[self._thorax_idx, 2] < -0.14
        done |= data.x.pos[self._thorax_idx, 2] > 0.1
        
        # reward
        rewards = {
            'joint_reward': joint_reward,
            'pos_reward': pos_reward,
            'quat_reward': quat_reward,
            'angvel_reward': angvel_reward,
            'bodypos_reward': bodypos_reward,
            'endeff_reward': endeff_reward,
            'healthy_reward': healthy_reward,
            "tracking_lin_vel": (
                self._reward_tracking_lin_vel(state.info["command"], x, xd)
            ),
            "orientation": self._reward_orientation(x),
            "torques": self._reward_torques(
                data.qfrc_actuator
            ),  # pytype: disable=attribute-error
            "action_rate": self._reward_action_rate(action, state.info["last_act"]),
            "stand_still": self._reward_stand_still(
                state.info["command"],
                joint_angles,
            ),
            "termination": self._reward_termination(done, state.info["step"]),
        }
        rewards = {
            k: v * self.reward_config.rewards.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        

        # Handle nans during sim by resetting env
        reward = jp.nan_to_num(reward)
        obs = jp.nan_to_num(obs)
        
        # state management
        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        state.info["last_contact"] = contact
        state.info["rewards"] = rewards
        state.info["step"] += 1
        state.info["rng"] = rng

        # sample new command if more than 500 timesteps achieved
        state.info["command"] = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        
        # reset the step counter when done
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500), 0, state.info["step"]
        )

        # log total displacement as a proxy metric
        state.metrics["total_dist"] = brax_math.normalize(x.pos[self._thorax_idx])[1]
        state.metrics.update(state.info["rewards"])

        done = jp.float32(done)

        
        state = state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )
        return state

    def _get_obs(
        self,
        data: base.State,
        info: dict[str, Any],
        obs_history: jax.Array,
    ) -> jax.Array:
        inv_torso_rot = brax_math.quat_inv(data.x.rot[0])
        local_rpyrate = brax_math.rotate(data.xd.ang[0], inv_torso_rot)

        obs = jp.concatenate(
            [
                jp.array([local_rpyrate[2]]) * 0.25,  # yaw rate
                brax_math.rotate(
                    jp.array([0, 0, -1]), inv_torso_rot
                ),  # projected gravity
                info["command"] * jp.array([2.0, 2.0, 0.25]),  # command
                data.q[7:] - self._default_pose,  # motor angles
                info["last_act"],  # last action
            ]
        )

        # clip, noise
        obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
            info["rng"], obs.shape, minval=-1, maxval=1
        )
        # stack observations through time
        obs = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        return obs

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jp.sum(jp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = brax_math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _reward_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        # Penalize changes in actions
        return jp.sum(jp.square(act - last_act))

    def _reward_tracking_lin_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = brax_math.rotate(xd.vel[0], brax_math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jp.exp(
            -lin_vel_error / self.reward_config.rewards.tracking_sigma
        )
        return lin_vel_reward

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = brax_math.rotate(xd.ang[0], brax_math.quat_inv(x.rot[0]))
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        return jp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)

    def _reward_feet_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Reward air time.
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= (
            brax_math.normalize(commands[:2])[1] > 0.05
        )  # no reward for zero command
        return rew_air_time

    def _reward_stand_still(
        self,
        commands: jax.Array,
        joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize motion at zero commands
        return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
            brax_math.normalize(commands[:2])[1] < 0.1
        )

    def _reward_foot_slip(
        self, data: base.State, contact_filt: jax.Array
    ) -> jax.Array:
        # get velocities at feet which are offset from lower legs
        # pytype: disable=attribute-error
        pos = data.site_xpos[self._endeff_idxs]  # feet position
        feet_offset = pos - data.xpos[self._lower_leg_body_id]
        # pytype: enable=attribute-error
        offset = base.Transform.create(pos=feet_offset)
        foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
        foot_vel = offset.vmap().do(data.xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 500)
    
    def _bounded_quat_dist(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Computes a quaternion distance limiting the difference to a max of pi/2.

        This function supports an arbitrary number of batch dimensions, B.

        Args:
          source: a quaternion, shape (B, 4).
          target: another quaternion, shape (B, 4).

        Returns:
          Quaternion distance, shape (B, 1).
        """
        source /= jp.linalg.norm(source, axis=-1, keepdims=True)
        target /= jp.linalg.norm(target, axis=-1, keepdims=True)
        # "Distance" in interval [-1, 1].
        dist = 2 * jp.einsum("...i,...i", source, target) ** 2 - 1
        # Clip at 1 to avoid occasional machine epsilon leak beyond 1.
        dist = jp.minimum(1.0, dist)
        # Divide by 2 and add an axis to ensure consistency with expected return
        # shape and magnitude.
        return 0.5 * jp.arccos(dist)[..., np.newaxis]

    def render(
        self,
        trajectory: List[base.State],
        camera: str | None = None,
        width: int = 480,
        height: int = 320,
        scene_option: Any = None,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(
            trajectory,
            camera=camera,
            width=width,
            height=height,
            scene_option=scene_option,
        )

# def compute_diffs(walker_features: Dict[str, jp.ndarray],
#                   reference_features: Dict[str, jp.ndarray],
#                   n: int = 2) -> Dict[str, float]:
#     """Computes sums of absolute values of differences between components of
#     model and reference features.

#     Args:
#         model_features, reference_features: Dictionaries of features to compute
#             differences of.
#         n: Exponent for differences. E.g., for squared differences use n = 2.

#     Returns:
#         Dictionary of differences, one value for each entry of input dictionary.
#     """
#     diffs = {}
#     for k in walker_features:
#         if 'quat' not in k:
#             # Regular vector differences.
#             diffs[k] = jp.sum(
#                 jp.abs(walker_features[k] - reference_features[k])**n)
#         else:
#             # Quaternion differences (always positive, no need to use jp.abs).
#             diffs[k] = jp.sum(
#                 quaternions.quat_dist_short_arc(walker_features[k],
#                                                 reference_features[k])**n)
#     return diffs


# def get_walker_features(physics, mocap_joints, mocap_sites):
#     """Returns model pose features."""

#     qpos = physics.bind(mocap_joints).qpos
#     qvel = physics.bind(mocap_joints).qvel
#     sites = physics.bind(mocap_sites).xpos
#     root2site = quaternions.get_egocentric_vec(qpos[:3], sites, qpos[3:7])

#     # Joint quaternions in local egocentric reference frame,
#     # (except root quaternion, which is in world reference frame).
#     root_quat = qpos[3:7]
#     xaxis1 = physics.bind(mocap_joints).xaxis[1:, :]
#     xaxis1 = quaternions.rotate_vec_with_quat(
#         xaxis1, quaternions.reciprocal_quat(root_quat))
#     qpos7 = qpos[7:]
#     joint_quat = quaternions.joint_orientation_quat(xaxis1, qpos7)
#     joint_quat = jp.vstack((root_quat, joint_quat))

#     model_features = {
#         'com': qpos[:3],
#         'qvel': qvel,
#         'root2site': root2site,
#         'joint_quat': joint_quat,
#     }

#     return model_features


# def get_reference_features(reference_data, step):
#     """Returns reference pose features."""

#     qpos_ref = reference_data['qpos'][step, :]
#     qvel_ref = reference_data['qvel'][step, :]
#     root2site_ref = reference_data['root2site'][step, :, :]
#     joint_quat_ref = reference_data['joint_quat'][step, :, :]
#     joint_quat_ref = jp.vstack((qpos_ref[3:7], joint_quat_ref))

#     reference_features = {
#         'com': reference_data['qpos'][step, :3],
#         'qvel': qvel_ref,
#         'root2site': root2site_ref,
#         'joint_quat': joint_quat_ref,
#     }

#     return reference_features


# def reward_factors_deep_mimic(walker_features,
#                               reference_features,
#                               std=None,
#                               weights=(1, 1, 1, 1)):
#     """Returns four reward factors, each of which is a product of individual
#     (unnormalized) Gaussian distributions evaluated for the four model
#     and reference data features:
#         1. Cartesian center-of-mass position, qpos[:3].
#         2. qvel for all joints, including the root joint.
#         3. Egocentric end-effector vectors.
#         4. All joint orientation quaternions (in egocentric local reference
#           frame), and the root quaternion.

#     The reward factors are equivalent to the ones in the DeepMimic:
#     https://arxiv.org/abs/1804.02717
#     """
#     if std is None:
#         # Default values for fruitfly walking imitation task.
#         std = {
#             'com': 0.078487,
#             'qvel': 53.7801,
#             'root2site': 0.0735,
#             'joint_quat': 1.2247
#         }

#     diffs = compute_diffs(walker_features, reference_features, n=2)
#     reward_factors = []
#     for k in walker_features.keys():
#         reward_factors.append(jp.exp(-0.5 / std[k]**2 * diffs[k]))
#     reward_factors = jp.array(reward_factors)
#     reward_factors *= jp.asarray(weights)

#     return reward_factors