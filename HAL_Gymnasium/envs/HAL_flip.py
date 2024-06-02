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
        forward_reward_weight=1.2,
        ctrl_cost_weight=0.15,
        reset_noise_scale=0.005,
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
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def _check_ground_contact(self):
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        contact_force_threshold = 50  # Define a threshold for significant contact force
        total_force = 0.0

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == floor_id or contact.geom2 == floor_id:
                contact_force = np.linalg.norm(self.data.cfrc_ext[contact.geom1])
                total_force += contact_force

        return total_force

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        robot_rotation = self.data.qpos[2]  # rooty joint rotation
        spine_angle = self.data.qpos[3]     # spine joint angle
        completed_flip = np.abs(robot_rotation) >= 2 * np.pi
        ctrl_cost = self.control_cost(action)
        ground_force = self._check_ground_contact()

        # Dense Reward Components
        rotation_reward = min(abs(robot_rotation / (2 * np.pi)), 1.0)  # Progress towards a full rotation
        spine_stability_reward = 1.0 - abs(spine_angle) / np.pi       # Encourage spine to stay stable
        ground_force_penalty = 0.01 * ground_force                    # Penalize excessive ground force

        # Aggregate Reward
        reward = rotation_reward - ctrl_cost - ground_force_penalty + spine_stability_reward

        if completed_flip:
            reward += 100  # Large reward for completing a backflip

        observation = self._get_obs()
        terminated = completed_flip  # End episode after completing a backflip

        info = {
            "robot_rotation": robot_rotation,
            "completed_flip": completed_flip,
            "control_cost": ctrl_cost,
            "ground_force": ground_force,
            "rotation_reward": rotation_reward,
            "spine_stability_reward": spine_stability_reward,
            "ground_force_penalty": ground_force_penalty,
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

        qpos = self.init_qpos = np.array([0, -0.42, 0, 0, 0.51, -1.37, 0.51, -1.37])
        qvel = self.init_qvel

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
