__credits__ = ["Rushiv Arora"]

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 1.5,
}


class HalfCheetahEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64
            )

        MujocoEnv.__init__(
            self,
            "/Users/justinvalentine/Documents/HAL_GYM/HAL_Gymnasium/envs/assets/HAL.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost
    
    def _get_geom_id(self, geom_name):
        """Returns the ID of a geometry given its name."""
        names = self.model.names.decode('utf-8').split('\x00')
        try:
            geom_id = names.index(geom_name)
            return geom_id
        except ValueError:
            raise ValueError(f"Geometry with name {geom_name} not found")


    def _check_ground_contact(self):
        floor_id = self._get_geom_id('floor')
        fthigh_id = self._get_geom_id('fthigh_geom')
        fthigh_pully_id = self._get_geom_id('fthigh_pully_geom')
        bthigh_id = self._get_geom_id('bthigh_geom')
        bthigh_pully_id = self._get_geom_id('bthigh_pully_geom')
        torso_id = self._get_geom_id('front_torso_center_geom')

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == floor_id and contact.geom2 in [fthigh_id, fthigh_pully_id, bthigh_id, bthigh_pully_id, torso_id]) or \
            (contact.geom2 == floor_id and contact.geom1 in [fthigh_id, fthigh_pully_id, bthigh_id, bthigh_pully_id, torso_id]):
                return True
        
        return False
    
    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        robot_rotation = np.abs(self.data.qpos[2])
        ctrl_cost = self.control_cost(action)
        ground_collision = self._check_ground_contact()

        forward_reward = self._forward_reward_weight * x_velocity
        flip_cost = 1 if robot_rotation >= np.pi / 2 else 0
        collision_cost = 100 if ground_collision else 0

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost - flip_cost - collision_cost
        terminated = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation