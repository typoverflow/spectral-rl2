from operator import itemgetter

import torch
import torch.nn as nn

from spectralrl.algo.state.base import BaseStateAlgorithm
from spectralrl.module.actor import SquashedGaussianActor
from spectralrl.module.critic import EnsembleQ
from spectralrl.module.normalize import DummyNormalizer, RunningMeanStdNormalizer
from spectralrl.utils.utils import convert_to_tensor, make_target, sync_target


class SAC(BaseStateAlgorithm):
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
        self.actor_update_freq = cfg.actor_update_freq
        self.target_update_freq = cfg.target_update_freq
        self.auto_entropy = cfg.auto_entropy

        self.actor = SquashedGaussianActor(
            input_dim=self.obs_dim,
            output_dim=self.action_dim,
            hidden_dims=cfg.actor_hidden_dims,
            norm_layer=nn.LayerNorm
        ).to(self.device)
        self.critic = EnsembleQ(
            input_dim=self.obs_dim+self.action_dim,
            hidden_dims=cfg.critic_hidden_dims,
            ensemble_size=2,
            activation=nn.ELU,
            norm_layer=nn.LayerNorm
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

        self.normalize_obs = cfg.normalize_obs
        if self.normalize_obs:
            self.obs_rms = RunningMeanStdNormalizer(shape=(self.obs_dim,)).to(self.device)
        else:
            self.obs_rms = DummyNormalizer(shape=(self.obs_dim,)).to(self.device)

        self._step = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def train(self, training):
        self.actor.train(training)
        self.critic.train(training)

    @torch.no_grad()
    def select_action(self, obs, step, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None, ...]
        if self.normalize_obs:
            if step is not None:
                # meaning that we are training
                self.obs_rms.update(obs)
            obs = self.obs_rms.normalize(obs)
        action, *_ = self.actor.sample(obs, deterministic=deterministic)
        return action.squeeze(0).detach().cpu().numpy()

    def train_step(self, buffer, batch_size):
        tot_metrics = {}

        batch = buffer.sample(batch_size)
        obs, action, next_obs, reward, terminal = [
            convert_to_tensor(b, self.device) for b in itemgetter("obs", "action", "next_obs", "reward", "terminal")(batch)
        ]
        obs = self.obs_rms.normalize(obs)
        next_obs = self.obs_rms.normalize(next_obs)
        critic_loss, critic_metrics = self.critic_step(obs, action, next_obs, reward, terminal)
        tot_metrics.update(critic_metrics)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if self._step % self.actor_update_freq == 0:
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
            self.sync_target()

        self._step += 1
        return tot_metrics

    def critic_step(self, obs, action, next_obs, reward, terminal):
        with torch.no_grad():
            next_action, next_logprob, *_ = self.actor.sample(next_obs)
            q_target = self.critic_target(next_obs, next_action).min(0)[0] - self.alpha * next_logprob
            q_target = reward + self.discount * (1 - terminal) * q_target
        q_pred = self.critic(obs, action)
        critic_loss = (q_target - q_pred).pow(2).sum(0).mean()
        return critic_loss, {"loss/critic_loss": critic_loss.item()}

    def actor_step(self, obs):
        new_action, new_logprob, *_ = self.actor.sample(obs)
        q_value = self.critic(obs, new_action)
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

    def sync_target(self):
        sync_target(self.critic, self.critic_target, self.tau)
