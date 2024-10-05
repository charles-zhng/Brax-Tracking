from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers.training import (
    EpisodeWrapper,
    VmapWrapper,
    DomainRandomizationVmapWrapper,
)
import jax
from jax import numpy as jp


def wrap(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[Callable[[System], Tuple[System, System]]] = None,
) -> Wrapper:
    """Common wrapper pattern for all training agents.

    Args:
      env: environment to be wrapped
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized system
        and in_axes to vmap over

    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
    """
    env = EpisodeWrapper(env, episode_length, action_repeat)
    if randomization_fn is None:
        env = VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(env, randomization_fn)
    env = AutoResetWrapperTracking(env)
    return env


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

class RenderRolloutWrapperTracking(Wrapper):
    """Always resets to 0"""

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        info = {
            "cur_frame": 0,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
        }

        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # Add pos (without z height)
        new_qpos = jp.array(self.sys.qpos0)

        # Add quat
        # new_qpos = qpos_with_pos.at[3:7].set(self._track_quat[0])

        # Add noise
        qpos = new_qpos + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, 0)
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



class RenderRolloutWrapperTracking_Run(Wrapper):
    """Always resets to 0"""

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))

        state_info = {
            'rng': rng,
            "cur_frame": 0,
            "steps_taken_cur_frame": 0,
            'last_act': jp.zeros(self._nu),
            'last_vel': jp.zeros(self._nv),
            'command': self.sample_command(key),
            'last_contact': jp.zeros(6, dtype=bool),
            'rewards': {k: 0.0 for k in self.reward_config.rewards.scales.keys()},
            'step': 0,
        }

        obs_history = jp.zeros(15 * 91)  # store 15 steps of history
        obs = self._get_obs(pipeline_state, state_info, obs_history)
        reward, done = jp.zeros(2)
        metrics = {'total_dist': 0.0}
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]
        
        return State(pipeline_state, obs, reward, done, metrics, state_info)  
