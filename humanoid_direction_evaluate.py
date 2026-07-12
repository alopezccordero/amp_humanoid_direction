import gymnasium as gym
import register_env
from stable_baselines3 import PPO
model = PPO.load("models/ppo_humanoid_direction_real_amp_filtered.ZIP", device="cpu")
env = gym.make("HumanoidDirection-v0", render_mode="human")

obs, info = env.reset()

episode_reward = 0

episode = 0
num_episode = 10000

while episode < num_episode:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    episode_reward += reward
    
    if terminated or truncated:
        episode +=1
        print(f"episode: {episode}\n")
        print(f"episode reward: {episode_reward}")
        

        if episode < num_episode:
            obs, info = env.reset()
            episode_reward = 0
        else:
            break

env.close()

