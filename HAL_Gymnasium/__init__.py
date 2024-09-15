from gymnasium.envs.registration import register

register(
    id="HAL_Gymnasium/HAL_run-v0",
    entry_point="HAL_Gymnasium.envs.HAL_run:HALEnv",
    max_episode_steps=1000,
    reward_threshold=5000.0,
)

register(
    id="HAL_Gymnasium/HAL_flip-v0",
    entry_point="HAL_Gymnasium.envs.HAL_flip:HALEnv",
    max_episode_steps=50,
    reward_threshold=4800.0,
)

register(
    id="HAL_Gymnasium/HAL_jump-v0",
    entry_point="HAL_Gymnasium.envs.HAL_jump:HALEnv",
    max_episode_steps=20,
    reward_threshold=4800.0,
)

register(
    id="HAL_Gymnasium/HAL_jump_play-v0",
    entry_point="HAL_Gymnasium.envs.HAL_jump_play:HALEnv",
    max_episode_steps=40,
    reward_threshold=4800.0,
)
