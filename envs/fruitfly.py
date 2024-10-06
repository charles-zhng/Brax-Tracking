import jax
from jax import numpy as jp

from brax.envs.base import PipelineEnv, State
from brax.base import Base, Motion, Transform
from brax.io import mjcf as mjcf_brax
from brax import math as brax_math
from brax import base
from typing import Any, List, Sequence, Dict, Tuple

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
        site_names: List[str],
        scale_factor: float,
        clip_length: int,
        mocap_hz: int = 200,
        mjcf_path: str = "./assets/fruitfly/fruitfly_force_free.xml",
        ref_len: int = 5,
        too_far_dist=jp.inf,
        bad_pose_dist=jp.inf,
        bad_quat_dist=jp.inf,
        ctrl_cost_weight=0.01,
        pos_reward_weight=0.0,
        quat_reward_weight=1.0,
        joint_reward_weight=10.0,
        angvel_reward_weight=10.0,
        bodypos_reward_weight=1.0,
        endeff_reward_weight=1.0,
        healthy_reward=0.25,
        healthy_z_range=(0.03, 0.5),
        physics_steps_per_control_step=10,
        reset_noise_scale=1e-3,
        sim_timestep: float = 2e-4,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        terminate_when_unhealthy=True,
        free_jnt=True,
        inference_mode=False,
        **kwargs,
    ):
        spec = mujoco.MjSpec()
        spec.from_file(mjcf_path)
        thorax = spec.find_body("thorax")
        first_joint = thorax.first_joint()
        if (free_jnt == False) & (first_joint.name == "free"):
            first_joint.delete()
        mj_model = spec.compile()

        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations
        mj_model.opt.timestep = sim_timestep
        mj_model.opt.jacobian = 0

        sys = mjcf_brax.load_model(mj_model)

        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        max_physics_steps_per_control_step = int(
            (1.0 / (mocap_hz * mj_model.opt.timestep))
        )

        super().__init__(sys, **kwargs)
        self._dt = 0.002  # this environment is 500 fps
        
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
        
        self._site_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("site"), site)
                for site in site_names
            ]
        )
        # using this for appendage for now bc im to lazy to rename
        self._endeff_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in end_eff_names
            ]
        )
        self._sim_timestep = sim_timestep
        self._free_jnt = free_jnt
        self._inference_mode = inference_mode
        self._mocap_hz = mocap_hz
        self._bad_pose_dist = bad_pose_dist
        self._too_far_dist = too_far_dist
        self._bad_quat_dist = bad_quat_dist
        self._ref_traj = reference_clip
        self._ref_len = ref_len
        self._clip_len = clip_length
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
            "start_frame": start_frame,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "angvel_distance": 0.0,
            "bodypos_distance": 0.0,
            "endeff_distance": 0.0,
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
            "reward_ctrl": zero,
            "healthy_reward": zero,
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

        # Logic for getting current frame aligned with simulation time
        cur_frame = (
            info["start_frame"] + jp.floor(data.time * self._mocap_hz).astype(jp.int32)
        ) % self._clip_len

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
            pos_distance = 0.0
            quat_distance = 0.0
            pos_reward = 0.0
            quat_reward = 0.0

        track_joints = self._ref_traj.joints
        joint_distance = jp.sum((data.qpos[self._joint_idxs] - track_joints[cur_frame])** 2) 
        # joint_reward = self._joint_reward_weight * jp.exp(-0.1 * joint_distance)
        joint_reward = self._joint_reward_weight* jp.exp(-0.5/.8**2  * joint_distance)
        info["joint_distance"] = joint_distance

        track_angvel = self._ref_traj.angular_velocity
        angvel_distance = jp.sum((data.qvel[3:6] - track_angvel[cur_frame])** 2)
        # angvel_reward = self._angvel_reward_weight * jp.exp(-0.01 * angvel_distance)
        # angvel_reward = self._angvel_reward_weight * jp.exp(-0.5/53.7801**2 * angvel_distance)
        angvel_reward = self._angvel_reward_weight* jp.exp(-20 * angvel_distance)
        info["angvel_distance"]
        
        track_bodypos = self._ref_traj.body_positions
        bodypos_distance = jp.sum((data.xpos[self._body_idxs] - track_bodypos[cur_frame][self._body_idxs]).flatten()** 2)
        # bodypos_reward = self._bodypos_reward_weight * jp.exp(-0.1* bodypos_distance)
        bodypos_reward = self._bodypos_reward_weight* jp.exp(-50 * bodypos_distance)
        info["bodypos_distance"] = bodypos_distance
        
        endeff_distance = jp.sum((data.xpos[self._endeff_idxs] - track_bodypos[cur_frame][self._endeff_idxs]).flatten()** 2)
        # endeff_reward = self._endeff_reward_weight * jp.exp(-0.5 * endeff_distance)
        endeff_reward = self._endeff_reward_weight* jp.exp(-700 * endeff_distance)
        info["endeff_distance"] = endeff_distance

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
        rewards_temp = self.get_reward_factors(data)
        pos_reward = rewards_temp[0]
        # joint_reward = rewards_temp[1]
        quat_reward = rewards_temp[2]
        # rewards = {
        #     'pos_reward': rewards_temp[0],
        #     'joint_reward': rewards_temp[1],
        #     'quat_reward': rewards_temp[2],
        # }
        # reward = sum(rewards.values())

        reward = (
            pos_reward
            + joint_reward
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
            reward_ctrl=-ctrl_cost,
            healthy_reward=healthy_reward,
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

        joint_dist = (ref_traj.joints - data.qpos[self._joint_idxs]).flatten()

        # TODO test if this works
        body_pos_dist_local = jax.vmap(
            lambda a, b: jax.vmap(brax_math.rotate, in_axes=(0, None))(a, b),
            in_axes=(0, None),
        )(
            (ref_traj.body_positions[:,self._body_idxs] - data.xpos[self._body_idxs]),
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
    # ------------ reward functions----------------
    def get_reward_factors(self, data):
        """Returns factorized reward terms."""
        if self._inference_mode:
            return (1,)
        step = round(data.time / self._dt)
        walker_ft = self._get_walker_features(data, self._joint_idxs,
                                        self._site_idxs)
        reference_ft = self._get_reference_features(self._ref_traj, step)
        reward_factors = self._reward_factors_deep_mimic(
            walker_features=walker_ft,
            reference_features=reference_ft,
            weights=(0, 1, 1))
        return reward_factors
    
    def _compute_diffs(self, walker_features: Dict[str, jp.ndarray],
                    reference_features: Dict[str, jp.ndarray],
                    n: int = 2) -> Dict[str, float]:
        """Computes sums of absolute values of differences between components of
        model and reference features.

        Args:
            model_features, reference_features: Dictionaries of features to compute
                differences of.
            n: Exponent for differences. E.g., for squared differences use n = 2.

        Returns:
            Dictionary of differences, one value for each entry of input dictionary.
        """
        diffs = {}
        for k in walker_features:
            if 'quat' not in k:
                # Regular vector differences.
                diffs[k] = jp.sum(
                    jp.abs(walker_features[k] - reference_features[k])**n)
            else:
                # Quaternion differences (always positive, no need to use jp.abs).
                diffs[k] = jp.sum(
                    quaternions.quat_dist_short_arc(walker_features[k],
                                                    reference_features[k])**n)
        return diffs


    def _get_walker_features(self, data, mocap_joints, mocap_sites):
        """Returns model pose features."""

        qpos = data.qpos[mocap_joints]
        qvel = data.qvel[mocap_joints]
        sites =data.xpos[mocap_sites]
        # root2site = quaternions.get_egocentric_vec(qpos[:3], sites, qpos[3:7])

        # Joint quaternions in local egocentric reference frame,
        # (except root quaternion, which is in world reference frame).
        root_quat = data.qpos[3:7]
        
        xaxis1 = data.xaxis[mocap_joints]
        xaxis1 = quaternions.rotate_vec_with_quat(
            xaxis1, quaternions.reciprocal_quat(root_quat))
        
        joint_quat = quaternions.joint_orientation_quat(xaxis1, qpos)
        joint_quat = jp.vstack((root_quat, joint_quat))

        model_features = {
            'com': qpos[:3],
            'qvel': qvel,
            # 'root2site': root2site,
            'joint_quat': joint_quat,
        }

        return model_features


    def _get_reference_features(self,reference_clip, step):
        """Returns reference pose features."""
        qpos_ref = reference_clip.joints[step, :]
        qvel_ref = reference_clip.joints_velocity[step, :]
        # root2site_ref = reference_clip['root2site'][step, :, :]
        joint_quat_ref = reference_clip.body_quaternions[step, self._joint_idxs, :]
        joint_quat_ref = jp.vstack((reference_clip.quaternion[step], joint_quat_ref))

        reference_features = {
            'com': reference_clip.position[step, :],
            'qvel': qvel_ref,
            # 'root2site': root2site_ref,
            'joint_quat': joint_quat_ref,
        }
        return reference_features


    def _reward_factors_deep_mimic(self, walker_features,
                                reference_features,
                                std=None,
                                weights=(1, 1, 1)):
        """Returns four reward factors, each of which is a product of individual
        (unnormalized) Gaussian distributions evaluated for the four model
        and reference data features:
            1. Cartesian center-of-mass position, qpos[:3].
            2. qvel for all joints, including the root joint.
            3. Egocentric end-effector vectors. Deleted for now.
            4. All joint orientation quaternions (in egocentric local reference
            frame), and the root quaternion.

        The reward factors are equivalent to the ones in the DeepMimic:
        https://arxiv.org/abs/1804.02717
        """
        if std is None:
            # Default values for fruitfly walking imitation task.
            std = {
                'com': 0.078487,
                'qvel': 53.7801,
                # 'root2site': 0.0735,
                'joint_quat': 1.2247
            }

        diffs = self._compute_diffs(walker_features, reference_features, n=2)
        reward_factors = []
        for k in walker_features.keys():
            reward_factors.append(jp.exp(-0.5 / std[k]**2 * diffs[k]))
        reward_factors = jp.array(reward_factors)
        reward_factors *= jp.asarray(weights)

        return reward_factors


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
        site_names: List[str],
        scale_factor: float,
        clip_length: int,
        mocap_hz: int = 200,
        mjcf_path: str = "./assets/fruitfly/fruitfly_force_free.xml",
        ref_len: int = 5,
        obs_noise: float = 0.05,
        too_far_dist=jp.inf,
        bad_pose_dist=jp.inf,
        bad_quat_dist=jp.inf,
        ctrl_cost_weight=0.01,
        pos_reward_weight=0.0,
        quat_reward_weight=0.0,
        joint_reward_weight=1.0,
        angvel_reward_weight=1.0,
        bodypos_reward_weight=1.0,
        endeff_reward_weight=1.0,
        tracking_lin_vel_weight=1,
        lin_vel_z_weight=-2.0,
        ang_vel_xy_weight=-0.05,
        orientation_weight=-1.0,
        torques_weight=-0.0002, 
        action_rate_weight=-0.01,
        stand_still_weight=-0.5,
        termination_weight=-1.0,
        healthy_reward=0.25,
        healthy_z_range=(0.03, 0.5),
        physics_steps_per_control_step=10,
        reset_noise_scale=1e-3,
        sim_timestep: float = 2e-4,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        terminate_when_unhealthy=True,
        free_jnt=True,
        inference_mode=False,
        **kwargs,
    ):
        spec = mujoco.MjSpec()
        spec.from_file(mjcf_path)
        thorax = spec.find_body("thorax")
        first_joint = thorax.first_joint()
        if (free_jnt == False) & (first_joint.name == "free"):
            first_joint.delete()
        mj_model = spec.compile()

        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations
        mj_model.opt.timestep = sim_timestep
        mj_model.opt.jacobian = 0

        sys = mjcf_brax.load_model(mj_model)

        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        max_physics_steps_per_control_step = int(
            (1.0 / (mocap_hz * mj_model.opt.timestep))
        )

        super().__init__(sys, **kwargs)
        self._dt = 0.002  # this environment is 500 fps
        
        self._obs_noise = obs_noise
        self._reset_noise_scale = reset_noise_scale
        self._init_q = jp.array(sys.mj_model.keyframe("home").qpos)
        self._default_pose = sys.mj_model.keyframe("home").qpos[7:]
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
        
        self._site_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("site"), site)
                for site in site_names
            ]
        )
        # using this for appendage for now bc im to lazy to rename
        self._endeff_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in end_eff_names
            ]
        )
        
        self._nv = sys.nv
        self._nq = sys.nq
        self._nu = sys.nu
        self._sim_timestep = sim_timestep
        self._free_jnt = free_jnt
        self._inference_mode = inference_mode
        self._mocap_hz = mocap_hz
        self._bad_pose_dist = bad_pose_dist
        self._too_far_dist = too_far_dist
        self._bad_quat_dist = bad_quat_dist
        self._ref_traj = reference_clip
        self._ref_len = ref_len
        self._clip_len = clip_length
        self._pos_reward_weight = pos_reward_weight
        self._quat_reward_weight = quat_reward_weight
        self._joint_reward_weight = joint_reward_weight
        self._angvel_reward_weight = angvel_reward_weight
        self._bodypos_reward_weight = bodypos_reward_weight
        self._endeff_reward_weight = endeff_reward_weight
        self._tracking_lin_vel_weight = tracking_lin_vel_weight
        self._lin_vel_z_weight = lin_vel_z_weight
        self._ang_vel_xy_weight = ang_vel_xy_weight
        self._orientation_weight = orientation_weight
        self._torques_weight = torques_weight
        self._action_rate_weight = action_rate_weight
        self._stand_still_weight = stand_still_weight
        self._termination_weight = termination_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._terminate_when_unhealthy = terminate_when_unhealthy

    def sample_command(self, rng: jax.Array) -> jax.Array:
        lin_vel_x = [0.5, 1.0]  # min max [m/s]
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

        return  jp.array([lin_vel_x[0], 0, 0])

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)

        data = self.pipeline_init(self._init_q, jp.zeros(self._nv))

        info = {
            "rng": rng,
            "start_frame": 0,
            "last_act": jp.zeros(self._nu),
            "last_vel": jp.zeros(self._nv),
            "command": self.sample_command(key),
            "last_contact": jp.zeros(6, dtype=bool),
            "step": 0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "angvel_distance": 0.0,
            "bodypos_distance": 0.0,
            "endeff_distance": 0.0,
        }

        obs_history = jp.zeros(15 * 91)  # store 15 steps of history
        obs = self._get_obs(data, info, obs_history)
        reward, done, zero = jp.zeros(3)
        
        metrics = {
            "total_dist": zero,
            'pos_reward': zero,
            'joint_reward': zero,
            'quat_reward': zero,
            'angvel_reward': zero,
            'bodypos_reward': zero,
            'endeff_reward': zero,
            "tracking_lin_vel": zero,
            'ang_vel_xy': zero,
            'lin_vel_z': zero,
            "orientation": zero,
            "torques": zero,
            "action_rate": zero,
            "stand_still": zero,
            "termination": zero,
        }
        
        return State(
            data, obs, reward, done, metrics, info
        )  # pytype: disable=wrong-arg-types

    def step(
        self, state: State, action: jax.Array
    ) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)
        
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        info = state.info.copy()

        # physics step
        
        cur_frame = (
            info["start_frame"] + jp.floor(data.time * self._mocap_hz).astype(jp.int32)
        ) % self._clip_len
        
        track_joints = self._ref_traj.joints
        joint_distance = jp.sum((data.qpos[self._joint_idxs] - track_joints[cur_frame])** 2) 
        # joint_reward = self._joint_reward_weight * jp.exp(-0.1 * joint_distance)
        joint_reward = self._joint_reward_weight * jp.exp(-0.5/.95**2 * joint_distance)
        info["joint_distance"] = joint_distance

        track_angvel = self._ref_traj.angular_velocity
        angvel_distance = jp.sum((data.qvel[3:6] - track_angvel[cur_frame])** 2)
        # angvel_reward = self._angvel_reward_weight * jp.exp(-0.01 * angvel_distance)
        # angvel_reward = self._angvel_reward_weight * jp.exp(-0.5/53.7801**2 * angvel_distance)
        angvel_reward = self._angvel_reward_weight * jp.exp(-0.5 * angvel_distance)
        info["angvel_distance"]
        
        track_bodypos = self._ref_traj.body_positions
        bodypos_distance = jp.sum((data.xpos[self._body_idxs] - track_bodypos[cur_frame][self._body_idxs]).flatten()** 2)
        bodypos_reward = self._bodypos_reward_weight * jp.exp(-50* bodypos_distance)
        # bodypos_reward = self._bodypos_reward_weight * jp.exp(-0.1* bodypos_distance)
        info["bodypos_distance"] = bodypos_distance
        
        endeff_distance = jp.sum((data.xpos[self._endeff_idxs] - track_bodypos[cur_frame][self._endeff_idxs]).flatten()** 2)
        endeff_reward = self._endeff_reward_weight * jp.exp(-700 * endeff_distance)
        # endeff_reward = self._endeff_reward_weight * jp.exp(-0.5 * endeff_distance)
        info["endeff_distance"] = endeff_distance

        # observation data
        x, xd = data.x, data.xd
        obs = self._get_obs(data, state.info, state.obs)
        joint_angles = data.q[7:]
        joint_vel = data.qd  ##### need to restrict to only legs


        # done if joint limits are reached or robot is falling
        min_z, max_z = self._healthy_z_range
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.dot(brax_math.rotate(up, x.rot[self._thorax_idx]), up) < 0
        done |= data.xpos[self._thorax_idx][2] < min_z
        done |= data.xpos[self._thorax_idx][2] > max_z
        done |= joint_distance > self._bad_pose_dist
        
        # reward
        rewards_temp = self.get_reward_factors(data)
        pos_reward = rewards_temp[0]
        joint_reward = rewards_temp[1]
        quat_reward = rewards_temp[2]
        tracking_lin_vel = self._tracking_lin_vel_weight*self._reward_tracking_lin_vel(state.info["command"], x, xd)
        ang_vel_xy = self._lin_vel_z_weight * self._reward_ang_vel_xy(xd)
        lin_vel_z = self._ang_vel_xy_weight * self._reward_lin_vel_z(xd)
        orientation = self._orientation_weight*self._reward_orientation(x)
        torques = self._torques_weight*self._reward_torques(data.qfrc_actuator)
        action_rate = self._action_rate_weight*self._reward_action_rate(action, state.info["last_act"])
        stand_still = self._stand_still_weight*self._reward_stand_still(state.info["command"],joint_angles,)
        termination = self._termination_weight*self._reward_termination(done, state.info["step"])
        rewards = {
            'pos_reward': pos_reward,
            'joint_reward': joint_reward,
            'quat_reward': quat_reward,
            'angvel_reward': angvel_reward,
            'bodypos_reward': bodypos_reward,
            'endeff_reward': endeff_reward,
            "tracking_lin_vel": tracking_lin_vel,
            'ang_vel_xy': ang_vel_xy,
            'lin_vel_z': lin_vel_z,
            "orientation": orientation,
            "torques": torques,
            "action_rate": action_rate,
            "stand_still": stand_still,
            "termination": termination,
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        

        # Handle nans during sim by resetting env
        reward = jp.nan_to_num(reward)
        obs = jp.nan_to_num(obs)
        
        # state management
        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        state.info["step"] += 1
        state.info["rng"] = rng

        # sample new command if more than 500 timesteps achieved
        state.info["command"] = jp.where(
            state.info["step"] > self._clip_len,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        
        # reset the step counter when done
        state.info["step"] = jp.where(
            done | (state.info["step"] > self._clip_len), 0, state.info["step"]
        )

        # log total displacement as a proxy metric
        state.metrics.update(
                            total_dist = brax_math.normalize(x.pos[self._thorax_idx])[1],
                            pos_reward = pos_reward,
                            joint_reward = joint_reward,
                            quat_reward = quat_reward,
                            angvel_reward = angvel_reward,
                            bodypos_reward = bodypos_reward,
                            endeff_reward = endeff_reward,
                            tracking_lin_vel = tracking_lin_vel,
                            ang_vel_xy = ang_vel_xy,
                            lin_vel_z = lin_vel_z,
                            orientation = orientation,
                            torques = torques,
                            action_rate = action_rate,
                            stand_still = stand_still,
                            termination = termination,
        )

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
    
    def get_reward_factors(self, data):
        """Returns factorized reward terms."""
        if self._inference_mode:
            return (1,)
        step = round(data.time / self._dt)
        walker_ft = self._get_walker_features(data, self._joint_idxs,
                                        self._site_idxs)
        reference_ft = self._get_reference_features(self._ref_traj, step)
        reward_factors = self._reward_factors_deep_mimic(
            walker_features=walker_ft,
            reference_features=reference_ft,
            weights=(0, 1, 1))
        return reward_factors
    
    # ------------ reward functions----------------
    
    def _compute_diffs(self, walker_features: Dict[str, jp.ndarray],
                    reference_features: Dict[str, jp.ndarray],
                    n: int = 2) -> Dict[str, float]:
        """Computes sums of absolute values of differences between components of
        model and reference features.

        Args:
            model_features, reference_features: Dictionaries of features to compute
                differences of.
            n: Exponent for differences. E.g., for squared differences use n = 2.

        Returns:
            Dictionary of differences, one value for each entry of input dictionary.
        """
        diffs = {}
        for k in walker_features:
            if 'quat' not in k:
                # Regular vector differences.
                diffs[k] = jp.sum(
                    jp.abs(walker_features[k] - reference_features[k])**n)
            else:
                # Quaternion differences (always positive, no need to use jp.abs).
                diffs[k] = jp.sum(
                    quaternions.quat_dist_short_arc(walker_features[k],
                                                    reference_features[k])**n)
        return diffs


    def _get_walker_features(self, data, mocap_joints, mocap_sites):
        """Returns model pose features."""

        qpos = data.qpos[mocap_joints]
        qvel = data.qvel[mocap_joints]
        sites =data.xpos[mocap_sites]
        # root2site = quaternions.get_egocentric_vec(qpos[:3], sites, qpos[3:7])

        # Joint quaternions in local egocentric reference frame,
        # (except root quaternion, which is in world reference frame).
        root_quat = data.qpos[3:7]
        
        xaxis1 = data.xaxis[mocap_joints]
        xaxis1 = quaternions.rotate_vec_with_quat(
            xaxis1, quaternions.reciprocal_quat(root_quat))
        
        joint_quat = quaternions.joint_orientation_quat(xaxis1, qpos)
        joint_quat = jp.vstack((root_quat, joint_quat))

        model_features = {
            'com': qpos[:3],
            'qvel': qvel,
            # 'root2site': root2site,
            'joint_quat': joint_quat,
        }

        return model_features


    def _get_reference_features(self,reference_clip, step):
        """Returns reference pose features."""
        qpos_ref = reference_clip.joints[step, :]
        qvel_ref = reference_clip.joints_velocity[step, :]
        # root2site_ref = reference_clip['root2site'][step, :, :]
        joint_quat_ref = reference_clip.body_quaternions[step, self._joint_idxs, :]
        joint_quat_ref = jp.vstack((reference_clip.quaternion[step], joint_quat_ref))

        reference_features = {
            'com': reference_clip.position[step, :],
            'qvel': qvel_ref,
            # 'root2site': root2site_ref,
            'joint_quat': joint_quat_ref,
        }
        return reference_features


    def _reward_factors_deep_mimic(self, walker_features,
                                reference_features,
                                std=None,
                                weights=(1, 1, 1)):
        """Returns four reward factors, each of which is a product of individual
        (unnormalized) Gaussian distributions evaluated for the four model
        and reference data features:
            1. Cartesian center-of-mass position, qpos[:3].
            2. qvel for all joints, including the root joint.
            3. Egocentric end-effector vectors.
            4. All joint orientation quaternions (in egocentric local reference
            frame), and the root quaternion.

        The reward factors are equivalent to the ones in the DeepMimic:
        https://arxiv.org/abs/1804.02717
        """
        if std is None:
            # Default values for fruitfly walking imitation task.
            std = {
                'com': 0.078487,
                'qvel': 53.7801,
                # 'root2site': 0.0735,
                'joint_quat': 1.2247
            }

        diffs = self._compute_diffs(walker_features, reference_features, n=2)
        reward_factors = []
        for k in walker_features.keys():
            reward_factors.append(jp.exp(-0.5 / std[k]**2 * diffs[k]))
        reward_factors = jp.array(reward_factors)
        reward_factors *= jp.asarray(weights)

        return reward_factors
    
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
            -lin_vel_error / 0.25
        )
        return lin_vel_reward

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = brax_math.rotate(xd.ang[0], brax_math.quat_inv(x.rot[0]))
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        return jp.exp(-ang_vel_error / 0.25)

    def _reward_stand_still(
        self,
        commands: jax.Array,
        joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize motion at zero commands
        return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
            brax_math.normalize(commands[:2])[1] < 0.1
        )
        
    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 1000)
    
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
