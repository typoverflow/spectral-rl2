from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.algo.state.diffsr.ddpm import DDPM
from spectralrl.algo.state.diffsr.network import RFFCritic
from spectralrl.algo.state.td3.agent import TD3
from spectralrl.module.actor import SquashedDeterministicActor, SquashedGaussianActor
from spectralrl.utils.utils import convert_to_tensor, make_target, sync_target


class DiffSR_TD3(TD3):
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
        self.back_critic_grad = cfg.back_critic_grad
        self.critic_coef = cfg.critic_coef
        self.reward_coef = cfg.reward_coef

        # diffusion
        self.diffusion = DDPM(
            cfg,
            obs_dim,
            action_dim,
            device=self.device
        ).to(self.device)
        self.diffusion_target = make_target(self.diffusion)
        self.diffusion_optim = torch.optim.Adam(self.diffusion.parameters(), lr=cfg.diffusion_lr)

        # rl network
        self.actor = SquashedDeterministicActor(
            input_dim=self.obs_dim,
            output_dim=self.action_dim,
            hidden_dims=cfg.actor_hidden_dims,
            norm_layer=nn.LayerNorm
        ).to(self.device)
        self.critic = RFFCritic(
            feature_dim=self.feature_dim,
            hidden_dim=cfg.critic_hidden_dim
        ).to(self.device)
        self.actor_target = make_target(self.actor)
        self.critic_target = make_target(self.critic)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def train(self, training):
        super().train(training)
        self.diffusion.train(training)

    def train_step(self, buffer, batch_size):
        tot_metrics = {}

        for _ in range(self.feature_update_ratio):
            batch = buffer.sample(batch_size)
            obs, action, next_obs, reward, terminal = [
                convert_to_tensor(b, self.device) for b in itemgetter("obs", "action", "next_obs", "reward", "terminal")(batch)
            ]
            diffusion_loss, reward_loss, diffusion_metrics = self.diffusion_step(obs, action, next_obs, reward)
            tot_metrics.update(diffusion_metrics)

            critic_loss, critic_metrics = self.critic_step(obs, action, next_obs, reward, terminal)
            tot_metrics.update(critic_metrics)

            self.diffusion_optim.zero_grad()
            self.critic_optim.zero_grad()
            (diffusion_loss + self.reward_coef * reward_loss + self.critic_coef * critic_loss).backward()
            self.critic_optim.step()
            self.diffusion_optim.step()

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

    def diffusion_step(self, obs, action, next_obs, reward):
        diffusion_loss, reward_loss, stats = self.diffusion.compute_loss(
            next_obs,
            obs,
            action,
            reward if self.reward_coef > 0 else None
        )
        metrics = {
            "loss/diffusion_loss": diffusion_loss.item(),
            "loss/reward_loss": reward_loss.item()
        }
        metrics.update(stats)
        return diffusion_loss, reward_loss, metrics

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
            feature = self.get_feature(obs, action, use_target=True)
        q_pred = self.critic(feature)
        critic_loss = (q_target - q_pred).pow(2).sum(0).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss.item(),
        }

    def actor_step(self, obs):
        new_action, *_ = self.actor.sample(obs)
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
        model = self.diffusion_target if use_target else self.diffusion
        return model.score.forward_psi(obs, action)

    def sync_target(self):
        super().sync_target()
        sync_target(self.diffusion, self.diffusion_target, self.tau)
