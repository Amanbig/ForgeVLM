import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatchingHead(nn.Module):
    """Continuous Visual Generation Head using Flow Matching (Transfusion / DiT style).

    Given conditioning text embeddings from the multimodal transformer and a noisy
    visual latent x_t at time t in [0, 1], predicts the velocity field v_t = dx_t / dt.
    """

    def __init__(self, latent_dim: int, cond_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Args:

            x_t: noisy latents (B, N, latent_dim)
            t: timesteps in [0, 1], shape (B, 1) or (B,)
            cond: conditioning vector from LM (B, cond_dim) or (B, N, cond_dim)
        Returns:
            predicted velocity v_t: (B, N, latent_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_emb = self.time_mlp(t)  # (B, hidden_dim)

        if cond.dim() == 2:
            c_emb = self.cond_proj(cond)  # (B, hidden_dim)
            # Expand time and conditioning along sequence length N
            n = x_t.size(1)
            t_emb = t_emb.unsqueeze(1).expand(-1, n, -1)
            c_emb = c_emb.unsqueeze(1).expand(-1, n, -1)
        else:
            c_emb = self.cond_proj(cond)
            t_emb = t_emb.unsqueeze(1).expand(-1, x_t.size(1), -1)

        feat = torch.cat((x_t, t_emb, c_emb), dim=-1)
        return self.net(feat)

    def compute_loss(self, x_1: torch.Tensor, cond: torch.Tensor):
        """Train step for Flow Matching:

        x_0 ~ N(0, I)
        x_t = (1 - t) * x_0 + t * x_1
        target_velocity = x_1 - x_0
        """
        b = x_1.size(0)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(b, 1, 1, device=x_1.device)

        x_t = (1.0 - t) * x_0 + t * x_1
        target_velocity = x_1 - x_0

        pred_velocity = self.forward(x_t, t.squeeze(-1), cond)
        return F.mse_loss(pred_velocity, target_velocity)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_latents: int, latent_dim: int, steps: int = 10):
        """Euler ODE solver from t=0 (noise) to t=1 (generated image latent)."""
        b = cond.size(0)
        x_t = torch.randn(b, num_latents, latent_dim, device=cond.device)
        dt = 1.0 / steps

        for step in range(steps):
            t = torch.full((b, 1), step * dt, device=cond.device)
            v = self.forward(x_t, t, cond)
            x_t = x_t + v * dt

        return x_t
