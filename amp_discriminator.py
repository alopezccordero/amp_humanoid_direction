import torch
import torch.nn as nn


class AMPDiscriminator(nn.Module):
    """Least-squares AMP discriminator (predicts +1 real / -1 fake).

    FIX vs original: `amp_reward` previously used exp(-0.05*(d-1)^2), which is
    almost flat: a confidently-fake score of -1 still earned reward 0.82 vs 1.0
    for perfectly-real. The style signal was ~constant, so PPO had nothing to
    optimize. This restores the paper reward

        r = clamp(1 - 0.25 * (d - 1)^2, 0, 1)

    which gives r=1 at d=+1 (real) and r=0 at d<=-1 (fake) - a full-range,
    informative reward.
    """

    def __init__(self, input_dim=90, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected AMP input dim {self.input_dim}, got {x.shape[-1]}"
            )

        return self.net(x).view(-1, 1)

    def predict_score(self, x):
        return self.forward(x)

    def amp_reward(self, x):
        d = self.forward(x)
        r = 1.0 - 0.25 * torch.square(d - 1.0)
        return torch.clamp(r, min=0.0, max=1.0)
