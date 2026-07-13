---
library_name: stable-baselines3
pipeline_tag: reinforcement-learning
tags:
- reinforcement-learning
- deep-reinforcement-learning
- stable-baselines3
- ppo
- mujoco
- humanoid
- adversarial-motion-priors
- amp
- character-animation
- motion-capture
model-index:
- name: ppo-amp-humanoid-direction
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: HumanoidDirection-v0
      type: HumanoidDirection-v0
    metrics:
    - type: mean_reward
      value: "971.74 ± 13.51"   
      name: mean_reward
---

# PPO + AMP — Direction-Following MuJoCo Humanoid

A **PPO** policy (Stable-Baselines3) trained with **Adversarial Motion Priors (AMP)** ([Peng et al., 2021](https://arxiv.org/abs/2104.02180)) on a custom `HumanoidDirection-v0` environment — Gymnasium's MuJoCo `Humanoid-v5` extended with a random target heading each episode.

## Model description

Three artifacts are trained jointly and shipped together:

| File | What it is |
|---|---|
| `ppo_humanoid_direction_amp_fixed.zip` | SB3 PPO policy (MLP, π and V nets [1024, 512], ReLU) |
| `vecnormalize_amp_fixed.pkl` | `VecNormalize` observation statistics — **required at inference** |
| `amp_discriminator_fixed.pt` | AMP discriminator state dict (2×512 MLP, 90-D input) — training artifact, not needed for inference |

- **Observation space**: `Humanoid-v5` observation + 2-D target direction (unit vector).
- **Action space**: `Humanoid-v5` continuous torques (17-D).
- **Task reward** ∈ [0, 1]: `exp(-2 · max(0, 1.4 − v_proj)²)` — full credit at walking speed along the target, no bonus for sprinting.
- **Style reward** ∈ [0, 1]: `clamp(1 − 0.25(d − 1)², 0, 1)` from the discriminator, computed on heading-invariant 45-D feature transitions so it judges gait quality, not global heading.

## Training procedure

Trained with `train_sb3_real_amp.py` on 8 `SubprocVecEnv` workers (CPU) + 1 GPU for PPO/discriminator updates, up to 50M environment steps. The discriminator is updated once per rollout (8 × 2048 steps) against expert transitions sampled from mocap at the exact environment control dt, and its weights are pushed to every worker after each update. Episodes start from a random mocap frame (Reference State Initialization) with probability 0.5.

### Hyperparameters

**PPO**: lr 1e-4, n_steps 2048, batch 512, 5 epochs, target_kl 0.02, γ 0.99, GAE λ 0.95, clip 0.2, ent_coef 0.0, obs normalization (clip 10).

**AMP discriminator**: 90-D input (two 45-D heading-invariant frames), 2×512 MLP; LSGAN loss to +1/−1 targets; Adam lr 1e-4, weight decay 1e-4; 8 updates × batch 512 per rollout; gradient penalty 5.0 on real samples; score regularization 1e-4; grad-norm clip 1.0; 100k fake-transition replay buffer.

**Reward mixing**: 0.5 · task + 0.5 · style.

A second experiment (`models_exp2/`) uses a more conservative discriminator: disc lr 3e-5, 4 updates/rollout, gradient penalty 10.0, PPO lr 5e-5, RSI 0.3.

### Training data

~60 motion-capture clips (stand, walk, run, turns 45°–135°, backwards locomotion, gait transitions) retargeted to the Humanoid-v5 skeleton, stored as `qpos`/`qvel`/`fps` pickles. Clips originate from AMASS-format mocap (ACCAD subject sets, among others). **The original mocap datasets carry their own licenses (typically non-commercial for AMASS subsets) — verify before redistribution or commercial use.**

## Evaluation

Qualitative: see `replay.mp4` in the repo for a sample rollout. Run `humanoid_direction_evaluate.py` to render episodes and print per-episode rewards.

## Limitations

- Locomotion only (walk/run/turn/stand); no other skills are in the motion prior.
- Target speed fixed at 1.4 m/s; direction is randomized but speed is not commanded.
- Flat-ground MuJoCo simulation; no domain randomization — not directly transferable to a real robot.
- The policy is tied to this exact observation layout and `VecNormalize` statistics.

## Citation

```bibtex
@article{peng2021amp,
  title={AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control},
  author={Peng, Xue Bin and Ma, Ze and Abbeel, Pieter and Levine, Sergey and Kanazawa, Angjoo},
  journal={ACM Transactions on Graphics (TOG)},
  volume={40}, number={4}, pages={1--20}, year={2021}
}
```
