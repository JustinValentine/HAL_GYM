from gymnasium.envs.registration import register

register(
    id="HAL_Gymnasium/HAL_run",
    entry_point="HAL_Gymnasium.envs.HAL_run:HALEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="HAL_Gymnasium/HAL_flip",
    entry_point="HAL_Gymnasium.envs.HAL_flip:HALEnv",
    max_episode_steps=50,
    reward_threshold=4800.0,
)

register(
    id="HAL_Gymnasium/HAL_jump",
    entry_point="HAL_Gymnasium.envs.HAL_jump:HALEnv",
    max_episode_steps=50,
    reward_threshold=4800.0,
)

