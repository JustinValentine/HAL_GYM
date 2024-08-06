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
        ctrl_cost_weight=0.1,
        z_vel_weight=4.5,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        heighest_jump=-0.37,    
        Lower_time=0.5,
        standing_height=0.3,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            z_vel_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            heighest_jump,
            Lower_time,
            standing_height,
            **kwargs,
        )

        self._height_reward_weight = forward_reward_weight
        self._z_vel_weight = z_vel_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._heighest_jump = heighest_jump
        self._Lower_time = Lower_time
        self._standing_height = standing_height

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

        # Initialize previous accelerations and velocities for jerk calculation
        self.previous_qvel = np.zeros(self.model.nv)
        self.previous_acc = np.zeros(self.model.nv)
        self.termination_case = 0

    def control_cost(self, action):
        weights = np.ones(action.shape)  
        spine_index = 0 
        higher_weight_for_spine = 1.0
        weights[spine_index] = higher_weight_for_spine 
        
        control_cost = self._ctrl_cost_weight * np.sum(weights * np.square(action))
        return control_cost

    def jerk_cost(self):
        current_qvel = self.data.qvel.copy()
        current_acc = (current_qvel - self.previous_qvel) / self.dt
        jerk = (current_acc - self.previous_acc) / self.dt

        jerk_cost = np.sum(np.square(jerk[3:8]))  # Indices 3 to 7 inclusive
        self.previous_qvel = current_qvel
        self.previous_acc = current_acc

        return jerk_cost

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
        z_position_before = self.data.qpos[1] + 0.62
        self.do_simulation(action, self.frame_skip)
        z_position_after = self.data.qpos[1] + 0.62
        fthigh_pos = self.data.qpos[4]
        bthigh_pos = self.data.qpos[6]
        spine_pos = self.data.qpos[3]
        z_velocity = (z_position_after - z_position_before) / self.dt

        robot_rotation = np.abs(self.data.qpos[2])

        rotation_cost = 0.3 * np.abs(robot_rotation)
        ctrl_cost = 1.3 * self.control_cost(action)
        z_velocity_reward = 3.0 * (z_velocity) ** 2
        jerk_cost = 0.00000001 * self.jerk_cost()

        observation = self._get_obs()   
        reward = z_velocity_reward - ctrl_cost - rotation_cost - jerk_cost
        
        terminated = False

        info = {
            "z_velocity_reward": z_velocity_reward,
            "ctrl_cost": ctrl_cost,
            "reward": reward,
            "robot_rotation_cost": rotation_cost,
            "jerk_cost": jerk_cost,
            "termination_case": self.termination_case,
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

        self.termination_case = 0

        qpos = self.init_qpos = np.array([0, -0.62, 0.283, -0.64, -0.69, 2.27, -1.41, 2.49])
        qvel = self.init_qvel

        self.set_state(qpos, qvel)

        # Reset previous accelerations and velocities
        self.previous_qvel = np.zeros(self.model.nv)
        self.previous_acc = np.zeros(self.model.nv)

        observation = self._get_obs()

        return observation
