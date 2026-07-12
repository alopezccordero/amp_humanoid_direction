from gymnasium.envs.registration import register

print("registering environment")

register(
    id="HumanoidDirection-v0",
    entry_point="humanoid_direction_env:HumanoidDirectionEnv",
    max_episode_steps=1000,
)