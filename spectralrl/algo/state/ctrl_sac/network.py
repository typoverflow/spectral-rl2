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
        self.mlp = MLP(obs_dim+action_dim, feature_dim, hidden_dims, activation=nn.ELU)

    def forward(self, obs, action, dt=None):
        out = self.mlp(torch.concat([obs, action], dim=-1))
        return out


class Mu(nn.Module):
    def __init__(
        self,
        feature_dim,
        obs_dim,
        hidden_dims,
    ) -> None:
        super().__init__()
        self.mlp = MLP(obs_dim, feature_dim, hidden_dims, activation=nn.ELU)

    def forward(self, obs):
        out = self.mlp(obs)
        out = F.tanh(out)
        return out


class Theta(nn.Module):
    def __init__(
        self,
        feature_dim
    ) -> None:
        super().__init__()
        self.mlp = nn.Linear(feature_dim, 1)

    def forward(self, feature):
        return self.mlp(feature)
