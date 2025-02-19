import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.module.net.mlp import MLP


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)
    
    
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


class RFFCritic(nn.Module):
    def __init__(self,
        feature_dim: int,
        hidden_dim: int
    ):
        super().__init__()
        self.ln = nn.LayerNorm(feature_dim)
        self.l1 = nn.Linear(feature_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.l4 = nn.Linear(feature_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_feature=False):
        q1_feature, q2_feature = self.forward_feature(x)
        q1 = self.l3(q1_feature)
        q2 = self.l6(q2_feature)
        return torch.stack([q1, q2], dim=0)

    def forward_feature(self, x):
        x = self.ln(x)
        q1 = torch.sin(self.ln1(self.l1(x)))
        q1 = torch.nn.functional.elu(self.ln2(self.l2(q1)))

        q2 = torch.sin(self.ln4(self.l4(x)))
        q2 = torch.nn.functional.elu(self.ln5(self.l5(q2)))
        return q1, q2
