import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import mujoco

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.5,
}


class HALEnv(MujocoEnv, utils.EzPickle):
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
        ctrl_cost_weight=1.0,
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
        weights = np.ones(action.shape)  
        spine_index = 0 
        higher_weight_for_spine = 1.0
        weights[spine_index] = higher_weight_for_spine 
        
        control_cost = self._ctrl_cost_weight * np.sum(weights * np.square(action))
        return control_cost


    def _check_ground_contact(self):
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')  
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'front_torso_center_geom')
        fthigh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'fthigh_geom') 
        fthigh_pully_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'fthigh_pully_geom') 
        bthigh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'bthigh_geom') 
        bthigh_pully_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'bthigh_pully_geom') 

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == floor_id and contact.geom2 in [fthigh_id, fthigh_pully_id, bthigh_id, bthigh_pully_id, torso_id]) or \
            (contact.geom2 == floor_id and contact.geom1 in [fthigh_id, fthigh_pully_id, bthigh_id, bthigh_pully_id, torso_id]):
                return True
        
        return False
    
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        height_reward = 2*self.data.qpos[1]


        ctrl_cost = self.control_cost(action)
        ground_collision = self._check_ground_contact()
        
        collision_cost = 1 if ground_collision else 0

        observation = self._get_obs()

        reward = height_reward - ctrl_cost - collision_cost
        terminated = False
        info = {
            "reward_ctrl": -ctrl_cost,
            "height reward": height_reward
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

        qpos = self.init_qpos = np.array([0, -0.42, 0, 0, 0.51, -1.37, 0.51, -1.37])
        qvel = self.init_qvel

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation