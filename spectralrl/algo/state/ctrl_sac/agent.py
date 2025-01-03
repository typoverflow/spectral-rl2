from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.algo.state import BaseStateAlgorithm
from spectralrl.algo.state.ctrl_sac.network import Mu, Phi, Theta
from spectralrl.module.actor import SquashedGaussianActor
from spectralrl.module.critic import EnsembleQ
from spectralrl.utils.utils import convert_to_tensor, make_target, sync_target


class Ctrl_SAC(BaseStateAlgorithm):
    def __init__(
        self,
        obs_dim,
        action_dim,
        cfg,
        device
    ) -> None:
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discount = cfg.discount
        self.tau = cfg.tau
        self.target_update_freq = cfg.target_update_freq
        self.auto_entropy = cfg.auto_entropy
        self.feature_dim = cfg.feature_dim
        self.feature_update_ratio = cfg.feature_update_ratio
        self.reward_coef = cfg.reward_coef

        # feature networks
        self.phi = Phi(
            feature_dim=cfg.feature_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=cfg.phi_hidden_dims,
        ).to(device)
        self.mu = Mu(
            feature_dim=cfg.feature_dim,
            obs_dim=obs_dim,
            hidden_dims=cfg.mu_hidden_dims,
        ).to(device)
        self.theta = Theta(
            feature_dim=cfg.feature_dim
        ).to(device)

        self.phi_target = make_target(self.phi)
        self.feature_optim = torch.optim.Adam(
            [*self.phi.parameters(), *self.mu.parameters(), *self.theta.parameters()],
            lr=cfg.feature_lr
        )

        # rl networks
        self.actor = SquashedGaussianActor(
            input_dim=self.obs_dim,
            output_dim=self.action_dim,
            hidden_dims=cfg.actor_hidden_dims,
            activation=nn.ELU
        ).to(self.device)
        self.critic = EnsembleQ(
            input_dim=self.feature_dim,
            hidden_dims=cfg.critic_hidden_dims,
            ensemble_size=2,
            activation=nn.ELU
        ).to(self.device)
        self.critic_target = make_target(self.critic)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        if self.auto_entropy:
            self.log_alpha = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=device), requires_grad=True)
            self.target_entropy = - self.action_dim
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        else:
            self.log_alpha = torch.tensor([torch.log(cfg.alpha)], dtype=torch.float32, device=device, requires_grad=False)

        self._step = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def train(self, training):
        self.actor.train(training)
        self.critic.train(training)
        self.phi.train(training)
        self.mu.train(training)
        self.theta.train(training)

    @torch.no_grad()
    def select_action(self, obs, step, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None, ...]
        action, *_ = self.actor.sample(obs, deterministic=deterministic)
        return action.squeeze(0).detach().cpu().numpy()

    def train_step(self, buffer, batch_size):
        tot_metrics = {}

        for _ in range(self.feature_update_ratio):
            batch = buffer.sample(batch_size)
            obs, action, next_obs, reward, terminal = [
                convert_to_tensor(b, self.device) for b in itemgetter("obs", "action", "next_obs", "reward", "terminal")(batch)
            ]
            feature_loss, feature_metrics = self.feature_step(obs, action, next_obs, reward)
            tot_metrics.update(feature_metrics)
            self.feature_optim.zero_grad()
            feature_loss.backward()
            self.feature_optim.step()
            sync_target(self.phi, self.phi_target, self.tau)

        critic_loss, critic_metrics = self.critic_step(obs, action, next_obs, reward, terminal)
        tot_metrics.update(critic_metrics)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss, actor_metrics = self.actor_step(obs)
        tot_metrics.update(actor_metrics)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self.auto_entropy:
            alpha_loss, alpha_metrics = self.alpha_step(obs)
            tot_metrics.update(alpha_metrics)
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        if self._step % self.target_update_freq == 0:
            sync_target(self.critic, self.critic_target, self.tau)

        self._step += 1
        return tot_metrics

    def feature_step(self, obs, action, next_obs, reward):
        z_phi = self.phi(obs, action)
        z_mu = self.mu(next_obs)
        label = torch.eye(obs.shape[0]).to(self.device)

        # we take NCE gamma = 1 here, while the paper uses 0.2
        contrastive = (z_phi.unsqueeze(1) * z_mu.unsqueeze(0)).sum(-1)
        model_loss = F.cross_entropy(contrastive, label)
        reward_loss = F.mse_loss(self.theta(z_phi), reward)
        feature_loss = model_loss + self.reward_coef * reward_loss
        return feature_loss, {
            "loss/feature_loss": feature_loss.item(),
            "loss/model_loss": model_loss.item(),
            "loss/reward_loss": reward_loss.item(),
        }

    def critic_step(self, obs, action, next_obs, reward, terminal):
        with torch.no_grad():
            next_action, next_logprob, *_ = self.actor.sample(next_obs)
            next_feature = self.get_feature(next_obs, next_action, use_target=True)
            q_target = self.critic_target(next_feature).min(0)[0] - self.alpha * next_logprob
            q_target = reward + self.discount * (1 - terminal) * q_target
        feature = self.get_feature(obs, action, use_target=True)
        q_pred = self.critic(feature)
        critic_loss = (q_target - q_pred).pow(2).sum(0).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss.item(),
        }

    def actor_step(self, obs):
        new_action, new_logprob, *_ = self.actor.sample(obs)
        new_feature = self.get_feature(obs, new_action, use_target=False)
        q_value = self.critic(new_feature)
        actor_loss = (self.alpha * new_logprob - q_value.min(0)[0]).mean()
        return actor_loss, {
            "misc/q_value_mean": q_value.mean().item(),
            "misc/q_value_std": q_value.std(0).mean().item(),
            "misc/q_value_min": q_value.min(0)[0].mean().item(),
            "loss/actor_loss": actor_loss.item()
        }

    def alpha_step(self, obs):
        with torch.no_grad():
            _, new_logprobs, _ = self.actor.sample(obs)
        alpha_loss = -(self.log_alpha * (new_logprobs + self.target_entropy)).mean()
        return alpha_loss, {"misc/alpha": self.alpha.item(), "loss/alpha_loss": alpha_loss.item()}

    def get_feature(self, obs, action, use_target=True):
        model = self.phi_target if use_target else self.phi
        return model(obs, action)
