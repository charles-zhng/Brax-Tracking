from brax.envs.base import State, Wrapper

import jax
from jax import numpy as jp


class RenderRolloutWrapperTracking(Wrapper):
    """Always resets to 0"""

    def reset(self, rng: jax.Array) -> State:
        _, clip_rng, rng = jax.random.split(rng, 3)

        clip_idx = jax.random.randint(clip_rng, (), 0, self._n_clips)
        info = {
            "clip_idx": 0, #clip_idx
            "start_frame":0,
            "cur_frame": 0,
            "steps_taken_cur_frame": 0,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "angvel_distance": 0.0,
            "bodypos_distance": 0.0,
            "endeff_distance": 0.0,
        }

        return self.reset_from_clip(rng, info)


class AutoResetWrapperTracking(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        state.info["first_cur_frame"] = state.info["cur_frame"]
        state.info["first_steps_taken_cur_frame"] = state.info["steps_taken_cur_frame"]
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        pipeline_state = jax.tree.map(
            where_done, state.info["first_pipeline_state"], state.pipeline_state
        )
        obs = where_done(state.info["first_obs"], state.obs)
        state.info["cur_frame"] = where_done(
            state.info["first_cur_frame"],
            state.info["cur_frame"],
        )
        state.info["steps_taken_cur_frame"] = where_done(
            state.info["first_steps_taken_cur_frame"],
            state.info["steps_taken_cur_frame"],
        )
        return state.replace(pipeline_state=pipeline_state, obs=obs)



# class RenderRolloutWrapperTracking(Wrapper):
#     """Always resets to 0"""

#     def reset(self, rng: jax.Array) -> State:
#         rng, rng1, rng2, rng_pos = jax.random.split(rng, 4)

#         start_frame = jax.random.randint(rng, (), 0, 44)

#         info = {
#             "start_frame": start_frame,
#             "cur_frame": start_frame,
#             "summed_pos_distance": 0.0,
#             "steps_taken_cur_frame": 0,
#             "quat_distance": 0.0,
#             "joint_distance": 0.0,
#             "angvel_distance": 0.0,
#             "bodypos_distance": 0.0,
#             "endeff_distance": 0.0,
#         }

#         _, rng1, rng2 = jax.random.split(rng, 3)
#         # Get reference clip and select the start frame
#         reference_frame = jax.tree_map(
#             lambda x: x[(info["cur_frame"]).astype(int)], self._get_reference_clip(info)
#         )

#         low, hi = -self._reset_noise_scale, self._reset_noise_scale

#         # Add pos
#         qpos_with_pos = jp.array(self.sys.qpos0).at[:3].set(reference_frame.position)

#         # Add quat
#         new_qpos = qpos_with_pos.at[3:7].set(reference_frame.quaternion)

#         # Add noise
#         qpos = new_qpos + jax.random.uniform(
#             rng1, (self.sys.nq,), minval=low, maxval=hi
#         )
#         qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

#         data = self.pipeline_init(qpos, qvel)

#         reference_obs, proprioceptive_obs = self._get_obs(data, info)

#         # Used to intialize our intention network
#         info["reference_obs_size"] = reference_obs.shape[-1]

#         obs = jp.concatenate([reference_obs, proprioceptive_obs])

#         reward, done, zero = jp.zeros(3)
#         metrics = {
#              "pos_reward": zero,
#             "quat_reward": zero,
#             "joint_reward": zero,
#             "angvel_reward": zero,
#             "bodypos_reward": zero,
#             "endeff_reward": zero,
#             "reward_ctrlcost": zero,
#             "too_far": zero,
#             "bad_pose": zero,
#             "bad_quat": zero,
#             "fall": zero,
#         }
#         return State(data, obs, reward, done, metrics, info)



class RenderRolloutWrapperTracking_Run(Wrapper):
    """Always resets to 0"""

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))

        state_info = {
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
        obs = self._get_obs(pipeline_state, state_info, obs_history)
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
        
        return State(pipeline_state, obs, reward, done, metrics, state_info)  



class RenderRolloutWrapperTracking_RunSim(Wrapper):
    """Always resets to 0"""

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,), minval=low, maxval=hi
        )

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, jp.zeros(self.sys.nu))
        reward, done, zero = jp.zeros(3)
        metrics = {
            'forward_reward': zero,
            'reward_linvel': zero,
            'reward_quadctrl': zero,
            'reward_alive': zero,
            'x_position': zero,
            'y_position': zero,
            'xy_ang_reward': zero,
            'orientation': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
        }
        return State(data, obs, reward, done, metrics)




class RenderRolloutWrapperTracking_Stand(Wrapper):
    """Always resets to 0"""

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self.sys.qpos0, jp.zeros(self.sys.nv))

        info = {
            "rng": rng,
            "last_act": jp.zeros(self._nu),
            "last_vel": jp.zeros(self._nv),
            "command": jp.zeros(3),
            "last_contact": jp.zeros(6, dtype=bool),
            "step": 0,
        }

        
        obs = self._get_obs(pipeline_state,jp.zeros(self._nu))
        reward, done, zero = jp.zeros(3)
        
        metrics = {
            "total_dist": zero,
            "tracking_lin_vel": zero,
            'ang_vel_xy': zero,
            'lin_vel_z': zero,
            "orientation": zero,
            "torques": zero,
            "action_rate": zero,
            "stand_still": zero,
            "termination": zero,
        }
        
        return State(pipeline_state, obs, reward, done, metrics, info)  

