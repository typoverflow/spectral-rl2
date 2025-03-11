import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.algo.state.td3.agent import TD3
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
        self.temperature = cfg.temperature
        self.back_critic_grad = cfg.back_critic_grad
        self.critic_coef = cfg.critic_coef

        # feature networks
        self.infonce = FactorizedInfoNCE(
            obs_dim=obs_dim,
            action_dim=action_dim,
            feature_dim=cfg.feature_dim,
            phi_hidden_dims=cfg.phi_hidden_dims,
            mu_hidden_dims=cfg.mu_hidden_dims,
            reward_hidden_dim=cfg.reward_hidden_dim,
        ).to(device)
        self.infonce_target = make_target(self.infonce)
        self.feature_optim = torch.optim.Adam(
            [*self.infonce.parameters()],
            lr=cfg.feature_lr
        )
        # self.phi = Phi(
        #     feature_dim=cfg.feature_dim,
        #     obs_dim=obs_dim,
        #     action_dim=action_dim,
        #     hidden_dims=cfg.phi_hidden_dims,
        # ).to(device)
        # self.mu = Mu(
        #     feature_dim=cfg.feature_dim,
        #     obs_dim=obs_dim,
        #     hidden_dims=cfg.mu_hidden_dims,
        # ).to(device)
        # self.reward = RFFReward(
        #     feature_dim=cfg.feature_dim,
        #     hidden_dim=cfg.reward_hidden_dim,
        # ).to(device)

        # self.phi_target = make_target(self.phi)
        # self.feature_optim = torch.optim.Adam(
        #     [*self.phi.parameters(), *self.mu.parameters(), *self.reward.parameters()],
        #     lr=cfg.feature_lr
        # )

        # rl networks
        self.actor = SquashedDeterministicActor(
            input_dim=self.obs_dim,
            output_dim=self.action_dim,
            hidden_dims=cfg.actor_hidden_dims,
            norm_layer=nn.LayerNorm,
        ).to(self.device)
        self.critic = RFFCritic(
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
        self.infonce.train(training)

    def train_step(self, buffer, batch_size):
        tot_metrics = {}

        for _ in range(self.feature_update_ratio):
            batch = buffer.sample(batch_size)
            obs, action, next_obs, reward, terminal = [
                convert_to_tensor(b, self.device) for b in itemgetter("obs", "action", "next_obs", "reward", "terminal")(batch)
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
        z_phi = self.infonce.forward_phi(obs, action)
        z_mu = self.infonce.forward_mu(next_obs)
        logits = self.infonce.compute_logits(z_phi, z_mu)
        labels = torch.arange(logits.shape[0]).to(self.device)
        model_loss = F.cross_entropy(logits, labels)
        reward_loss = F.mse_loss(self.infonce.compute_reward(z_phi), reward)
        feature_loss = model_loss + self.reward_coef * reward_loss

        pos_logits_sum = logits[torch.arange(logits.shape[0]), labels].sum()
        neg_logits_sum = logits.sum() - pos_logits_sum
        pos_logits_mean = pos_logits_sum / logits.shape[0]
        neg_logits_mean = neg_logits_sum / (logits.shape[0]**2 - logits.shape[0])
        return feature_loss, {
            "loss/feature_loss": feature_loss.item(),
            "loss/model_loss": model_loss.item(),
            "loss/reward_loss": reward_loss.item(),
            "misc/phi_norm": z_phi.abs().mean().item(),
            "misc/mu_norm": z_mu.abs().mean().item(),
            "misc/pos_logits": pos_logits_mean.item(),
            "misc/neg_logits": neg_logits_mean.item(),
        }

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
        model = self.infonce_target if use_target else self.infonce
        return model.forward_phi(obs, action)

    def sync_target(self):
        super().sync_target()
        sync_target(self.infonce, self.infonce_target, self.tau)
