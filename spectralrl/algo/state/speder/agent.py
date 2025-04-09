from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.algo.state.speder.network import FactorizedTransition, LinearCritic
from spectralrl.algo.state.td3.agent import TD3
from spectralrl.module.actor import SquashedDeterministicActor, SquashedGaussianActor
from spectralrl.module.normalize import DummyNormalizer, RunningMeanStdNormalizer
from spectralrl.utils.utils import convert_to_tensor, make_target, sync_target


class Speder_TD3(TD3):
    def __init__(
        self,
        obs_dim,
        action_dim,
        cfg,
        device
    ) -> None:
        super().__init__(obs_dim, action_dim, cfg, device)
        self.feature_dim = cfg.feature_dim
        self.feature_update_ratio = cfg.feature_update_ratio
        self.reward_coef = cfg.reward_coef
        self.back_critic_grad = cfg.back_critic_grad
        self.critic_coef = cfg.critic_coef
        self.feature_tau = cfg.feature_tau
        self.num_noises = cfg.num_noises

        # feature networks
        self.transition = FactorizedTransition(
            obs_dim=obs_dim,
            action_dim=action_dim,
            feature_dim=cfg.feature_dim,
            phi_hidden_dims=cfg.phi_hidden_dims,
            mu_hidden_dims=cfg.mu_hidden_dims,
            reward_hidden_dim=cfg.reward_hidden_dim,
            num_negatives=cfg.num_negatives,
            num_noises=cfg.num_noises,
            device=device
        ).to(device)
        self.transition_target = make_target(self.transition)
        self.feature_optim = torch.optim.Adam(
            [*self.transition.parameters()],
            lr=cfg.feature_lr
        )

        # rl networks
        self.actor = SquashedDeterministicActor(
            input_dim=self.obs_dim,
            output_dim=self.action_dim,
            hidden_dims=cfg.actor_hidden_dims,
            norm_layer=nn.LayerNorm,
        ).to(self.device)
        self.critic = LinearCritic(
            feature_dim=self.feature_dim,
            hidden_dim=cfg.critic_hidden_dim
        ).to(self.device)
        self.actor_target = make_target(self.actor)
        self.critic_target = make_target(self.critic)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.normalize_obs = cfg.normalize_obs
        if self.normalize_obs:
            self.obs_rms = RunningMeanStdNormalizer(shape=(obs_dim,)).to(self.device)
        else:
            self.obs_rms = DummyNormalizer(shape=(obs_dim,)).to(self.device)

    def train(self, training=True):
        super().train(training)
        self.transition.train(training)

    def train_step(self, buffer, batch_size):
        tot_metrics = {}

        for _ in range(self.feature_update_ratio):
            batch = buffer.sample(batch_size)
            obs, action, next_obs, reward, terminal = [
                convert_to_tensor(b, self.device)[:batch_size] for b in itemgetter("obs", "action", "next_obs", "reward", "terminal")(batch)
            ]
            obs = self.obs_rms.normalize(obs)
            next_obs = self.obs_rms.normalize(next_obs)

            feature_loss, feature_metrics = self.feature_step(obs, action, next_obs, reward)
            tot_metrics.update(feature_metrics)

            critic_loss, critic_metrics = self.critic_step(obs, action, next_obs, reward, terminal)
            tot_metrics.update(critic_metrics)

            self.feature_optim.zero_grad()
            self.critic_optim.zero_grad()
            (feature_loss + self.critic_coef * critic_loss).backward()
            self.critic_optim.step()
            self.feature_optim.step()

        if self._step % self.actor_update_freq == 0:
            actor_loss, actor_metrics = self.actor_step(obs)
            tot_metrics.update(actor_metrics)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        if self._step % self.target_update_freq == 0:
            self.sync_target()

        self._step += 1
        return tot_metrics

    def feature_step(self, obs, action, next_obs, reward):
        model_loss, reward_loss, feature_metrics = self.transition.compute_loss(obs, action, next_obs, reward)
        feature_loss = model_loss.mean() + self.reward_coef * reward_loss.mean()
        return feature_loss, feature_metrics

    def critic_step(self, obs, action, next_obs, reward, terminal):
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.target_policy_noise).clip(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target.sample(next_obs)[0] + noise).clip(-1.0, 1.0)
            next_feature = self.get_feature(next_obs, next_action, use_target=True)
            q_target = self.critic_target(next_feature).min(0)[0]
            q_target = reward + self.discount * (1 - terminal) * q_target
        if self.back_critic_grad:
            feature = self.get_feature(obs, action, use_target=False)
        else:
            feature = self.get_feature(obs, action, use_target=True).detach()
        q_pred = self.critic(feature)
        critic_loss = (q_target - q_pred).pow(2).sum(0).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss.item(),
        }

    def actor_step(self, obs):
        new_action, *_ = self.actor.sample(obs)
        if self.back_critic_grad:
            new_feature = self.get_feature(obs, new_action, use_target=False)
        else:
            new_feature = self.get_feature(obs, new_action, use_target=True)
        q_value = self.critic(new_feature)
        actor_loss =  - q_value.mean()
        return actor_loss, {
            "misc/q_value_mean": q_value.mean().item(),
            "misc/q_value_std": q_value.std(0).mean().item(),
            "misc/q_value_min": q_value.min(0)[0].mean().item(),
            "loss/actor_loss": actor_loss.item()
        }

    def get_feature(self, obs, action, use_target=True):
        model = self.transition_target if use_target else self.transition
        return model.compute_feature(obs, action)

    def sync_target(self):
        super().sync_target()
        sync_target(self.transition, self.transition_target, self.feature_tau)
