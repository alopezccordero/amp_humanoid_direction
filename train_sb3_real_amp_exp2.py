from pathlib import Path

import gymnasium as gym
import torch

import register_env  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from amp_callback import AMPDiscriminatorCallback
from amp_discriminator import AMPDiscriminator
from amp_env import AMPHumanoidEnv
from motion_lib import MotionLib

# --------------------------------------------------------------------- #
TOTAL_TIMESTEPS = 50_000_000
N_ENVS = 8                      # = --cpus-per-task in your slurm file
N_STEPS = 2048
TRAIN_FREQ = N_ENVS * N_STEPS   # train the discriminator once per rollout

TASK_WEIGHT = 0.5               # both rewards are in [0, 1] now, so
AMP_WEIGHT = 0.5                # 0.5 / 0.5 mixing as in the AMP paper
REFERENCE_STATE_INIT_PROB = 0.5

DISCRIMINATOR_LR = 1e-4
DISCRIMINATOR_HIDDEN_DIM = 512
DISC_UPDATES_PER_ROLLOUT = 8
DISC_BATCH_SIZE = 512
GRADIENT_PENALTY_WEIGHT = 5.0

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models_exp2"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
TENSORBOARD_DIR = BASE_DIR / "tensorboard_real_amp_fixed"
CHECKPOINT_EVERY = 500_000
# --------------------------------------------------------------------- #


def make_env(motion_lib, amp_mean, amp_std):
    def _init():
        # Runs inside each SubprocVecEnv worker: keep torch single-threaded so
        # 8 workers don't oversubscribe the 8-CPU slurm allocation.
        torch.set_num_threads(1)
        # Each worker holds its OWN cpu copy of the discriminator for reward
        # evaluation; the callback pushes fresh weights after every update.
        disc_local = AMPDiscriminator(
            input_dim=90, hidden_dim=DISCRIMINATOR_HIDDEN_DIM
        )
        env = AMPHumanoidEnv(
            discriminator=disc_local,
            motion_lib=motion_lib,
            task_weight=TASK_WEIGHT,
            amp_weight=AMP_WEIGHT,
            device="cpu",
            amp_mean=amp_mean,
            amp_std=amp_std,
            reference_state_init_prob=REFERENCE_STATE_INIT_PROB,
        )
        return Monitor(env)

    return _init


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    TENSORBOARD_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Expert transition spacing must match the env control timestep exactly.
    _tmp = gym.make("HumanoidDirection-v0")
    env_dt = float(_tmp.unwrapped.dt)
    _tmp.close()
    print("Environment dt:", env_dt)

    motion_lib = MotionLib(
        str(BASE_DIR / "retargeted_pkl"),
        transition_dt=env_dt,
    )
    amp_mean, amp_std = motion_lib.compute_amp_stats(num_samples=10000)

    # Master discriminator (the one actually trained, on GPU if available).
    disc = AMPDiscriminator(
        input_dim=90, hidden_dim=DISCRIMINATOR_HIDDEN_DIM
    ).to(device)
    disc_optimizer = torch.optim.Adam(
        disc.parameters(), lr=DISCRIMINATOR_LR, weight_decay=1e-4
    )

    # SubprocVecEnv: envs step in parallel processes. With DummyVecEnv all 8
    # envs ran serially in one process - roughly an 8x throughput loss.
    env = SubprocVecEnv(
        [make_env(motion_lib, amp_mean, amp_std) for _ in range(N_ENVS)]
    )
    # Observation normalization matters a lot for Humanoid + PPO.
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    amp_callback = AMPDiscriminatorCallback(
        motion_lib=motion_lib,
        discriminator=disc,
        optimizer=disc_optimizer,
        batch_size=DISC_BATCH_SIZE,
        updates_per_call=DISC_UPDATES_PER_ROLLOUT,
        train_freq=TRAIN_FREQ,
        save_freq=CHECKPOINT_EVERY,
        save_path=str(CHECKPOINT_DIR),
        device=device,
        amp_mean=amp_mean,
        amp_std=amp_std,
        fake_replay_size=100_000,
        gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT,
        score_reg_weight=1e-4,
        max_grad_norm=1.0,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_EVERY // N_ENVS, 1),
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_amp_fixed",
        save_replay_buffer=False,
        save_vecnormalize=True,   # needed to evaluate the model later!
    )

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[1024, 512], vf=[1024, 512]),
    )

    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        learning_rate=1e-4,      # 5e-5 was very slow for 20M steps
        n_steps=N_STEPS,
        batch_size=512,
        n_epochs=5,
        target_kl=0.02,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,            # 0.01 keeps the gaussian noisy -> jittery gait
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(TENSORBOARD_DIR),
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([amp_callback, checkpoint_callback]),
    )

    model.save(str(MODEL_DIR / "ppo_humanoid_direction_amp_fixed"))
    env.save(str(MODEL_DIR / "vecnormalize_amp_fixed.pkl"))
    torch.save(disc.state_dict(), MODEL_DIR / "amp_discriminator_fixed.pt")
    env.close()
    print("Saved final PPO model, VecNormalize stats, and discriminator.")


if __name__ == "__main__":
    main()
