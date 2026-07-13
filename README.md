# AMP Humanoid Direction

MuJoCo `Humanoid-v5` that walks in a random commanded direction with a natural, human-like gait, using **Adversarial Motion Priors (AMP)** ([Peng et al., 2021](https://arxiv.org/abs/2104.02180)) + **Stable-Baselines3 PPO**.

Each episode the humanoid gets a random 2D target heading. Reward is `0.5 · task + 0.5 · style`: the task reward (bounded in [0, 1]) pays full credit for walking at ~1.4 m/s along the target, and the style reward comes from an LSGAN discriminator trained to tell policy transitions from retargeted mocap transitions. See `replay.mp4` for a sample rollout.

Eval (10 episodes, deterministic): **971.74 ± 13.51** episode reward.

## Files

- `humanoid_direction_env.py` / `register_env.py` — `HumanoidDirection-v0` (Humanoid-v5 + target heading, bounded directional reward)
- `amp_obs.py` — shared heading-invariant 45-D AMP features (used by both policy and mocap sides)
- `amp_env.py` — AMP reward wrapper (task/style mixing, reference state init)
- `amp_discriminator.py` / `amp_callback.py` — LSGAN discriminator + training callback (syncs weights into `SubprocVecEnv` workers)
- `motion_lib.py` — loads mocap `.pkl` clips, filters bad contacts, samples expert transitions interpolated at the exact env control dt
- `train_sb3_real_amp.py` — training (exp 1); `train_sb3_real_amp_exp2.py` — more conservative discriminator (exp 2)
- `humanoid_direction_evaluate.py` — roll out a trained policy, print per-episode and mean ± std reward
- `models/`, `models_exp2/` — trained PPO policies, `VecNormalize` stats, discriminators

### Training data

~60 motion-capture clips (stand, walk, run, turns 45°–135°, backwards locomotion, gait transitions) retargeted to the Humanoid-v5 skeleton, stored as `qpos`/`qvel`/`fps` pickles. Clips originate from AMASS-format mocap (ACCAD subject sets, among others). **The original mocap datasets carry their own licenses (typically non-commercial for AMASS subsets) — verify before redistribution or commercial use.**


## Usage

```bash
pip install "gymnasium[mujoco]" stable-baselines3 torch numpy mujoco

python train_sb3_real_amp.py            # train (SLURM scripts included)
python humanoid_direction_evaluate.py   # evaluate + render
```

The `VecNormalize` stats must be loaded alongside the PPO model at eval time — the policy was trained on normalized observations.

## Reference

Peng et al., *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control*, SIGGRAPH 2021. [arXiv:2104.02180](https://arxiv.org/abs/2104.02180)
