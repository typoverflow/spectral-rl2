import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.module.net.mlp import MLP


class Phi(nn.Module):
    def __init__(
        self,
        feature_dim,
        obs_dim,
        action_dim,
        hidden_dims,
    ) -> None:
        super().__init__()
        self.mlp = MLP(obs_dim+action_dim, feature_dim, hidden_dims, activation=nn.ELU, norm_layer=nn.LayerNorm)

    def forward(self, obs, action, dt=None):
        out = self.mlp(torch.concat([obs, action], dim=-1))
        out = torch.nn.functional.normalize(out, dim=-1)
        return out


class Mu(nn.Module):
    def __init__(
        self,
        feature_dim,
        obs_dim,
        hidden_dims,
    ) -> None:
        super().__init__()
        self.mlp = MLP(obs_dim, feature_dim, hidden_dims, activation=nn.ELU, norm_layer=nn.LayerNorm)

    def forward(self, obs):
        out = self.mlp(obs)
        out = torch.nn.functional.normalize(out, dim=-1)
        return out


class Theta(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dims,
    ) -> None:
        super().__init__()
        self.mlp = MLP(feature_dim, 1, hidden_dims, activation=nn.ELU, norm_layer=nn.LayerNorm)

    def forward(self, feature):
        return self.mlp(feature)


class RFFLayer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        learnable: bool = True
    ):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.layer = nn.Linear(feature_dim, hidden_dim)
        else:
            self.register_buffer("noise", torch.randn([feature_dim, hidden_dim], requires_grad=False))

    def forward(self, x):
        if self.learnable:
            return torch.sin(self.layer(x))
        else:
            return torch.sin(self.la)


class RFFCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int
    ):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.LayerNorm(feature_dim),
            RFFLayer(feature_dim, hidden_dim, learnable=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.net2 = nn.Sequential(
            nn.LayerNorm(feature_dim),
            RFFLayer(feature_dim, hidden_dim, learnable=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        q1 = self.net1(x)
        q2 = self.net2(x)
        return torch.stack([q1, q2], dim=0)
