import os
os.environ["MUJOCO_GL"] = "glfw"

import imageio
import gymnasium as gym
import register_env
from stable_baselines3 import PPO

model = PPO.load("models/ppo_humanoid_direction_20m.zip", device="cpu")

env = gym.make("HumanoidDirection-v0", render_mode="rgb_array")
obs, info = env.reset()

frames = []
num_episodes = 5
episode_count = 0

while episode_count < num_episodes:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()
    frames.append(frame)
    if terminated or truncated:
        episode_count += 1
        print(f"Episode: {episode_count}")
        last_frame = frames[-1]

        #pause using last frame
        for _ in range(15):
            frames.append(last_frame)
        if episode_count < num_episodes:
            obs, info = env.reset()
            

env.close()

print(f"number of frames: {len(frames)}")
print(f"frames shape: {frames[0].shape}")
writer = imageio.get_writer("humanoid_direction_20m.mp4", fps=30)

for frame in frames:
    writer.append_data(frame)

writer.close()

print(f"saved humanoid_direction.mp4 with episodes: {num_episodes}")