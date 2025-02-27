import numpy as np
import torch
import torch.nn as nn


class DummyNormalizer(nn.Module):
    def __init__(self, epsilon=1e-8, shape=(), dtype=torch.float32):
        super().__init__()

    def update(self, x):
        return

    def normalize(self, x):
        return x


class RunningMeanStdNormalizer(nn.Module):
    def __init__(self, epsilon=1e-8, shape=(), dtype=torch.float32):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=dtype))
        self.register_buffer("mean_square", torch.zeros(shape, dtype=dtype))
        self.register_buffer("count", torch.tensor(epsilon, dtype=dtype))
        self.epsilon = 1e-8

    def update(self, x):
        assert x.shape[0] == 1

        self.count += 1
        delta1 = x.squeeze() - self.mean
        self.mean += delta1 / self.count
        delta2 = x.squeeze().pow(2) - self.mean_square
        self.mean_square += delta2 / self.count

    def normalize(self, x):
        var = self.mean_square - self.mean.pow(2)
        std = torch.sqrt(var + self.epsilon)
        std = torch.clip(std, min=1e-4)
        return (x - self.mean) / std


# class RunningMeanStdNormalizer(nn.Module):
#     """Tracks the mean, variance and count of values."""

#     # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
#     def __init__(self, epsilon=1e-8, shape=(), dtype=torch.float32):
#         """Tracks the mean, variance and count of values."""
#         super().__init__()
#         self.register_buffer("mean", torch.zeros(shape, dtype=dtype))
#         self.register_buffer("var", torch.ones(shape, dtype=dtype))
#         self.register_buffer("count", torch.tensor(epsilon, dtype=dtype))

#     def update(self, x):
#         """Updates the mean, var and count from a batch of samples."""
#         batch_mean = torch.mean(x, axis=0)
#         batch_var = torch.var(x, axis=0)
#         batch_count = x.shape[0]
#         self.update_from_moments(batch_mean, batch_var, batch_count)

#     def update_from_moments(self, batch_mean, batch_var, batch_count):
#         """Updates from batch mean, variance and count moments."""
#         self.mean, self.var, self.count = update_mean_var_count_from_moments(
#             self.mean, self.var, self.count, batch_mean, batch_var, batch_count
#         )

#     def normalize(self, x):
#         return (x - self.mean) / np.sqrt(
#             self.var + 1e-8
#         )


# def update_mean_var_count_from_moments(
#     mean, var, count, batch_mean, batch_var, batch_count
# ):
#     """Updates the mean, var and count using the previous mean, var, count and batch values."""
#     delta = batch_mean - mean
#     tot_count = count + batch_count

#     new_mean = mean + delta * batch_count / tot_count
#     m_a = var * count
#     m_b = batch_var * batch_count
#     M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
#     new_var = M2 / tot_count
#     new_count = tot_count

#     return new_mean, new_var, new_count
