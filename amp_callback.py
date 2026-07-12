import os
from collections import deque

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class AMPDiscriminatorCallback(BaseCallback):
    """Trains the AMP discriminator and syncs weights into the envs.

    FIXES vs original:
    - Least-squares (MSE) loss to +1/-1 targets, per the AMP paper
      (was smooth_l1, which goes linear and weakens gradients exactly when
      the discriminator is wrong).
    - Robust trigger (`num_timesteps - last >= train_freq` instead of a
      modulo that silently never fires if train_freq isn't a multiple of
      n_envs).
    - Pushes updated weights to all envs via env_method("load_disc_state")
      -> required for SubprocVecEnv workers.
    - Sensible default budget: updates_per_call=8 x batch 512 per rollout
      (was 1 x 256 per 16384 steps at lr 3e-5 - the discriminator barely
      moved over an entire run).
    """

    def __init__(
        self,
        motion_lib,
        discriminator,
        optimizer,
        batch_size=512,
        updates_per_call=8,
        train_freq=16384,
        save_freq=500_000,
        save_path="./models/checkpoints",
        device="cpu",
        amp_mean=None,
        amp_std=None,
        fake_replay_size=100_000,
        gradient_penalty_weight=5.0,
        score_reg_weight=1e-4,
        max_grad_norm=1.0,
        verbose=1,
    ):
        super().__init__(verbose)
        self.motion_lib = motion_lib
        self.disc = discriminator
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.updates_per_call = updates_per_call
        self.train_freq = train_freq
        self.save_freq = save_freq
        self.save_path = save_path
        self.device = device
        self.amp_mean = amp_mean
        self.amp_std = amp_std
        self.fake_replay = deque(maxlen=fake_replay_size)
        self.gradient_penalty_weight = gradient_penalty_weight
        self.score_reg_weight = score_reg_weight
        self.max_grad_norm = max_grad_norm
        self.last_train_step = 0
        self.last_save_step = 0
        os.makedirs(self.save_path, exist_ok=True)

    # ------------------------------------------------------------------ #

    def _on_training_start(self):
        # Make sure every worker starts from the same weights as the master.
        self.push_disc_weights()

    def _on_step(self):
        if self.num_timesteps - self.last_train_step >= self.train_freq:
            self.train_discriminator()
            self.last_train_step = self.num_timesteps

        if self.num_timesteps - self.last_save_step >= self.save_freq:
            self.save_discriminator()
            self.last_save_step = self.num_timesteps
        return True

    def normalize_amp(self, x_np):
        if self.amp_mean is not None and self.amp_std is not None:
            return (x_np - self.amp_mean) / self.amp_std
        return x_np

    def push_disc_weights(self):
        state = {k: v.detach().cpu() for k, v in self.disc.state_dict().items()}
        self.training_env.env_method("load_disc_state", state)

    # ------------------------------------------------------------------ #

    def train_discriminator(self):
        fake_lists = self.training_env.env_method("pop_fake_transitions")
        fake_transitions = [t for lst in fake_lists for t in lst]
        if fake_transitions:
            self.fake_replay.extend(fake_transitions)

        if len(self.fake_replay) < self.batch_size:
            return

        self.disc.train()
        last_loss = None

        for _ in range(self.updates_per_call):
            idx = np.random.randint(0, len(self.fake_replay), size=self.batch_size)
            fake_np = self.normalize_amp(
                np.array([self.fake_replay[i] for i in idx], dtype=np.float32)
            )
            real_np = self.normalize_amp(
                np.array(
                    [self.motion_lib.sample_amp_transition()
                     for _ in range(self.batch_size)],
                    dtype=np.float32,
                )
            )

            real_x = torch.as_tensor(real_np, device=self.device)
            fake_x = torch.as_tensor(fake_np, device=self.device)

            real_scores = self.disc(real_x)
            fake_scores = self.disc(fake_x)

            # Least-squares GAN loss to targets +1 / -1 (AMP paper).
            real_loss = torch.square(real_scores - 1.0).mean()
            fake_loss = torch.square(fake_scores + 1.0).mean()
            gp_loss = self.gradient_penalty(real_x)
            score_reg = self.score_reg_weight * (
                real_scores.pow(2).mean() + fake_scores.pow(2).mean()
            )

            loss = (
                0.5 * (real_loss + fake_loss)
                + self.gradient_penalty_weight * gp_loss
                + score_reg
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.disc.parameters(), self.max_grad_norm)
            self.optimizer.step()
            last_loss = loss.item()

        self.disc.eval()
        self.push_disc_weights()

        # ---- diagnostics on the last minibatch ---- #
        with torch.no_grad():
            real_scores = self.disc(real_x)
            fake_scores = self.disc(fake_x)
            real_score = real_scores.mean().item()
            fake_score = fake_scores.mean().item()
            real_reward = self.disc.amp_reward(real_x).mean().item()
            fake_rewards = self.disc.amp_reward(fake_x)
            fake_reward = fake_rewards.mean().item()
            disc_acc = 0.5 * (
                (real_scores > 0).float().mean()
                + (fake_scores < 0).float().mean()
            ).item()

        if self.verbose:
            print(
                f"AMP Disc | loss={last_loss:.4f} acc={disc_acc:.2f} "
                f"real_score={real_score:.3f} fake_score={fake_score:.3f} "
                f"real_reward={real_reward:.3f} fake_reward={fake_reward:.3f}"
            )

        self.logger.record("amp/loss", last_loss)
        self.logger.record("amp/disc_acc", disc_acc)
        self.logger.record("amp/real_score", real_score)
        self.logger.record("amp/fake_score", fake_score)
        self.logger.record("amp/real_reward", real_reward)
        self.logger.record("amp/fake_reward", fake_reward)

    def gradient_penalty(self, real):
        real = real.clone().detach().requires_grad_(True)
        scores = self.disc(real)
        gradients = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=real,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients.pow(2).sum(dim=1).mean()

    def save_discriminator(self):
        path = os.path.join(
            self.save_path, f"amp_discriminator_{self.num_timesteps}.pt"
        )
        torch.save(self.disc.state_dict(), path)
        print("Saved discriminator checkpoint:", path)
