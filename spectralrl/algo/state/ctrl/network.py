import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.algo.state.diffsr.nn_idql import MLPResNet
from spectralrl.module.net.mlp import MLP
from spectralrl.utils.utils import make_target, sync_target


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

        self.s_net = MLP(
            obs_dim,
            0,
            [512, 512, feature_dim],
            activation=[nn.ELU, nn.ELU, nn.Tanh],
            norm_layer=nn.LayerNorm
        )
        self.sa_net = MLP(
            feature_dim+action_dim,
            feature_dim,
            [512, ],
            activation=nn.ELU,
            norm_layer=nn.LayerNorm
        )
        # self.phi_net = MLP(
        #     obs_dim+action_dim,
        #     feature_dim,
        #     phi_hidden_dims,
        #     activation=nn.ELU,
        #     norm_layer=nn.LayerNorm
        # )
        # self.mu_net = MLP(
        #     obs_dim,
        #     feature_dim,
        #     mu_hidden_dims,
        #     activation=nn.ELU,
        #     norm_layer=nn.LayerNorm
        # )
        # self.phi_net = MLPResNet(
        #     num_blocks=len(phi_hidden_dims),
        #     input_dim=obs_dim+action_dim,
        #     out_dim=feature_dim,
        #     dropout_rate=None,
        #     use_layer_norm=True,
        #     hidden_dim=phi_hidden_dims[-1],
        #     activations=nn.Mish()
        # )
        # self.mu_net = MLPResNet(
        #     num_blocks=len(mu_hidden_dims),
        #     input_dim=obs_dim,
        #     out_dim=feature_dim,
        #     dropout_rate=None,
        #     use_layer_norm=True,
        #     hidden_dim=mu_hidden_dims[-1],
        #     activations=nn.Mish()
        # )
        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim))
        self.rff_layer = nn.Sequential(
            nn.LayerNorm(feature_dim),
            RFFLayer(feature_dim, reward_hidden_dim, learnable=True),
        )
        # self.rff_layer = nn.Identity()
        self.reward_net = nn.Sequential(
            nn.Linear(reward_hidden_dim, reward_hidden_dim),
            nn.LayerNorm(reward_hidden_dim),
            nn.ELU(),
            nn.Linear(reward_hidden_dim, 1)
        )
        # self.reward_net = nn.Linear(feature_dim, 1)
        # self.reward_net = nn.Sequential(
        #     nn.LayerNorm(feature_dim),
        #     RFFLayer(feature_dim, reward_hidden_dim, learnable=True),
        #     nn.Linear(reward_hidden_dim, reward_hidden_dim),
        #     nn.LayerNorm(reward_hidden_dim),
        #     nn.ELU(),
        #     nn.Linear(reward_hidden_dim, 1)
        # )

    def forward_phi(self, obs, action):
        z_s = self.s_net(obs)
        z_sa = self.sa_net(torch.concat([z_s, action], dim=-1))
        return z_sa
        # out = self.phi_net(torch.concat([obs, action], dim=-1))
        # out = torch.nn.functional.tanh(out)
        # return out

    def forward_mu(self, obs):
        z_s = self.s_net(obs)
        return z_s
        # out = self.mu_net(obs)
        # out = torch.nn.functional.tanh(out)
        # return out

    def compute_feature(self, obs, action):
        z_phi = self.forward_phi(obs, action)
        return self.rff_layer(z_phi)

    def compute_reward(self, z_phi):
        return self.reward_net(self.rff_layer(z_phi))

    def compute_logits(self, z_phi, z_mu):
        Wz = torch.matmul(self.W, z_mu.T)  # (z_dim,B)
        # Wz = z_mu.T
        logits = torch.matmul(z_phi, Wz)  # (B,B)
        # logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


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
            # nn.Linear(feature_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.ELU(),
            RFFLayer(feature_dim, hidden_dim, learnable=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.net2 = nn.Sequential(
            nn.LayerNorm(feature_dim),
            # nn.Linear(feature_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.ELU(),
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
