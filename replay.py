import pickle
import time
import gymnasium as gym
import mujoco


PKL_FILE = "retargeted_pkl/Female1Running_c3d/C2 - Run to stand_poses.pkl"

with open(PKL_FILE, "rb") as f:
    motion = pickle.load(f)

qpos_seq = motion["qpos"]
fps = motion["fps"]

env = gym.make("Humanoid-v5", render_mode="human")
obs, info = env.reset()

model = env.unwrapped.model
data = env.unwrapped.data

left_body = model.body("left_foot").id
right_body = model.body("right_foot").id

print("qpos sequence:", qpos_seq.shape)
print("model nq:", model.nq)

dt = 1.0 / fps

for x in range(10):
    for i, qpos in enumerate(qpos_seq):
        data.qpos[:] = qpos
        data.qvel[:] = 0.0

        mujoco.mj_forward(model, data)
        env.render()

        left_z = data.xpos[left_body][2]
        right_z = data.xpos[right_body][2]

        print(
            f"frame {i} | "
            f"root_z={data.qpos[2]:.3f} | "
            f"left_foot_z={left_z:.3f} | "
            f"right_foot_z={right_z:.3f}"
        )

        time.sleep(dt)

env.close()