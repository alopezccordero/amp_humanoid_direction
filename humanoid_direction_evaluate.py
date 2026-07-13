import gymnasium as gym
import numpy as np
import register_env

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


MODEL_PATH = (
    "models_exp2/"
    "ppo_humanoid_direction_amp_fixed.zip"
)

VECNORMALIZE_PATH = (
    "models_exp2/"
    "vecnormalize_amp_fixed.pkl"
)


def make_env():
    return gym.make(
        "HumanoidDirection-v0",
        render_mode="human",
    )


# Create the same vectorized environment structure used during training
env = DummyVecEnv([make_env])

# Load the saved observation-normalization statistics
env = VecNormalize.load(
    VECNORMALIZE_PATH,
    env,
)

# Evaluation settings
env.training = False
env.norm_reward = False

# Load the matching PPO model and attach the environment
model = PPO.load(
    MODEL_PATH,
    env=env,
    device="cpu",
)

# DummyVecEnv.reset() returns only observations
obs = env.reset()

episode_reward = 0.0
episode = 0
num_episodes = 10
episode_rewards = []

while episode < num_episodes:
    action, _ = model.predict(
        obs,
        deterministic=True,
    )

    # VecEnv.step() returns four values, not five
    obs, rewards, dones, infos = env.step(action)

    # rewards and dones are arrays because this is a vectorized environment
    episode_reward += float(rewards[0])

    if dones[0]:
        episode += 1
        episode_rewards.append(episode_reward)

        print(f"Episode: {episode}")
        print(f"Episode reward: {episode_reward:.2f}")
        print(f"Episode info: {infos[0]}")
        print()

        episode_reward = 0.0

        # DummyVecEnv normally resets automatically after done.
        # The returned obs is already the next episode's initial observation.

env.close()

print(f"Episodes evaluated: {len(episode_rewards)}")
print(
    f"mean_reward = {np.mean(episode_rewards):.2f} "
    f"+/- {np.std(episode_rewards):.2f}"
)

