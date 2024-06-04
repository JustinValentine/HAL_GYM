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
        forward_reward_weight=5,
        ctrl_cost_weight=0.2,
        z_vel_weigh = 4.5,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        heighest_jump=-0.37,    
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            z_vel_weigh,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            heighest_jump,
            **kwargs,
        )

        self._height_reward_weight = forward_reward_weight

        self._z_vel_weigh = z_vel_weigh

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._heighest_jump = heighest_jump

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

        self.has_jumped = False 
        self.inital_touch = False
        self.max_height = 0

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
        f_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'fshin_geom')
        b_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'bshin_geom')

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == floor_id and contact.geom2 in [fthigh_id, fthigh_pully_id, bthigh_id, bthigh_pully_id, torso_id]) or \
            (contact.geom2 == floor_id and contact.geom1 in [fthigh_id, fthigh_pully_id, bthigh_id, bthigh_pully_id, torso_id]):
                return True
        
        return False
    
    def _target_trajectory(self):
        # stay at (0.37) for 2 seconds 

        # lower to 0.2 over 0.5 secopnds 

        # Jump 

        1/2mv^2 

        # stand at 0.37 for 2 seconds

    
    def _check_feet_on_ground(self):
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        f_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'fshin_geom')
        b_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'bshin_geom')

        f_foot, b_foot = False, False

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == floor_id and contact.geom2 == f_foot_id) or \
            (contact.geom2 == floor_id and contact.geom1 == f_foot_id):
                f_foot = True
            if (contact.geom1 == floor_id and contact.geom2 == b_foot_id) or \
            (contact.geom2 == floor_id and contact.geom1 == b_foot_id):
                b_foot = True
        
        return f_foot and b_foot

    
    def step(self, action):
        z_position_before = self.data.qpos[1] + 0.652
        self.do_simulation(action, self.frame_skip)
        z_position_after = self.data.qpos[1] + 0.652
        robot_rotation = np.abs(self.data.qpos[2])

        if z_position_after > self.max_height:
            self.max_height = z_position_after

        z_vel = (z_position_after - z_position_before)/self.dt

        jump_reward = (self._height_reward_weight * self.max_height)**2

        ground_collision = self._check_ground_contact()
        collision_cost = 5 if ground_collision else 0

        if z_vel >= 0:
            vel_reward = (self._z_vel_weigh * z_vel)**2
        else: 
            vel_reward = 0

        # if self._check_feet_on_ground() == True:
        #     self.inital_touch = True

        # if self._check_feet_on_ground() == False and self.inital_touch:
        #     air_bonus = 3
        #     self.has_jumped = True
        # else:
        #     air_bonus = 0

        # if self.has_jumped and self._check_ground_contact():
        #     terminated = True
        # else:
        #     terminated = False

        terminated = False

        robot_rotation_cost = 0.1*np.abs(self.data.qpos[2])
        ctrl_cost = self.control_cost(action)

        observation = self._get_obs()   
        reward = jump_reward + vel_reward - ctrl_cost - robot_rotation_cost - collision_cost #- collision_cost + vel_reward + air_bonus 


        info = {
            "jump_reward": jump_reward,
            "vel_reward": vel_reward,
            "ctrl_cost": ctrl_cost,
            "reward": reward,
            "robot_rotation_cost": robot_rotation_cost,
            "collision_cost":collision_cost,
        }

        if self.render_mode == "human":
            self.render()

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

        self.has_jumped = False 
        self.inital_touch = False
        self.max_height = 0

        qpos = self.init_qpos = np.array([0, -0.37, -0.314, 0.54, 0.39, -1.34, 0.96, -1.4])
        qvel = self.init_qvel

        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation
