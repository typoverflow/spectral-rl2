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
        self.mlp = MLP(
            obs_dim+action_dim,
            feature_dim,
            hidden_dims,
            activation=nn.ELU,
            norm_layer=nn.LayerNorm
        )

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
        self.mlp = MLP(
            obs_dim,
            feature_dim,
            hidden_dims,
            activation=nn.ELU,
            norm_layer=nn.LayerNorm
        )

    def forward(self, obs):
        out = self.mlp(obs)
        out = torch.nn.functional.normalize(out, dim=-1)
        return out


class FactorizedInfoNCE(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        feature_dim,
        phi_hidden_dims,
        mu_hidden_dims,
        reward_hidden_dim,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.phi_net = MLP(
            obs_dim+action_dim,
            feature_dim,
            phi_hidden_dims,
            activation=nn.ELU,
            norm_layer=nn.LayerNorm
        )
        self.mu_net = MLP(
            obs_dim,
            feature_dim,
            mu_hidden_dims,
            activation=nn.ELU,
            norm_layer=nn.LayerNorm
        )
        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim))
        self.reward_net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            RFFLayer(feature_dim, reward_hidden_dim, learnable=True),
            nn.Linear(reward_hidden_dim, reward_hidden_dim),
            nn.LayerNorm(reward_hidden_dim),
            nn.ELU(),
            nn.Linear(reward_hidden_dim, 1)
        )

    def forward_phi(self, obs, action):
        out = self.phi_net(torch.concat([obs, action], dim=-1))
        # out = torch.nn.functional.normalize(out, dim=-1)
        return out

    def forward_mu(self, obs):
        out = self.mu_net(obs)
        out = torch.nn.functional.tanh(out)
        return out

    def compute_reward(self, z_phi):
        return self.reward_net(z_phi)

    def compute_logits(self, z_phi, z_mu):
        Wz = torch.matmul(self.W, z_mu.T)  # (z_dim,B)
        logits = torch.matmul(z_phi, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits








class RFFReward(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            RFFLayer(feature_dim, hidden_dim, learnable=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feature):
        return self.net(feature)


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
            self.layer = nn.Linear(feature_dim, hidden_dim//2)
        else:
            self.register_buffer("noise", torch.randn([feature_dim, hidden_dim], requires_grad=False))

    def forward(self, x):
        if self.learnable:
            x = self.layer(x)
            return torch.concat([torch.sin(x), torch.cos(x)], dim=-1)
        else:
            return torch.sin(x @ self.noise)


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
