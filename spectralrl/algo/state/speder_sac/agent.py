from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.algo.state.ctrl_sac.agent import Ctrl_SAC
from spectralrl.utils.utils import convert_to_tensor, make_target, sync_target


class Speder_SAC(Ctrl_SAC):
    def __init__(
        self,
        obs_dim,
        action_dim,
        cfg,
        device
    ) -> None:
        super().__init__(obs_dim, action_dim, cfg, device)
