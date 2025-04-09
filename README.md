# Spectral RL: Spectral Representation for Reinforcement Learning

<p align="center">
  <a href="https://haotiansun14.github.io/rl-rep-page/">[Website]</a>
</p>

Spectral-RL is a reinforcement learning (RL) library dedicated to exploring representation learning in RL and Causal Inference. Specifically, we focus on algorithms that leverage spectral representation, including:

[1]: [Latent Variable Representation (LVRep)](https://arxiv.org/abs/2212.08765);

[2]: [Contrastive Representation (CTRL)](https://arxiv.org/abs/2207.07150);

[3]: [Multi-step Latent Variable Representation (μLVRep)](https://arxiv.org/abs/2311.12244);

[4]: [Spectral Decomposition Representation (Speder)](https://arxiv.org/abs/2208.09515);

[5]: [Diffusion Spectral Representation (Diff-SR)](https://arxiv.org/abs/2406.16121).

We provide implementations for these algorithms based on popular model-free RL algorithms and also evaluate them on a variety of benchmarks including MuJoCo, DMControl and MetaWorld.

## Installation
Spectral-RL is currently hosted on PyPI. It requires Python >= 3.10.

To install the latest version, you can simply clone this repository and install it:
```bash
git clone https://github.com/typoverflow/spectral-rl.git
cd spectral-rl
pip install -e .
```

## Usage
In `examples/`, we provide example scripts to run the algorithms on different environments.
```bash
examples
├── config            # config files
├── main_state_dmc.py # entry file for proprioceptive DMControl
├── main_state.py     # entry file for Gym-MuJoCo
└── main_visual.py    # entry file for MetaWorld and visual DMContrl
```

The basic command to run the algorithms is as follows:
```bash
python3 examples/main_state_dmc.py --algo <algorithm_name> --task <task_name>
```

## References
If you find our work useful, please consider citing our papers:

[1] Ren, Tongzheng, Chenjun Xiao, Tianjun Zhang, Na Li, Zhaoran Wang, Dale Schuurmans, and Bo Dai. "Latent Variable Representation for Reinforcement Learning." In The Eleventh International Conference on Learning Representations.

[2] Zhang, Tianjun, Tongzheng Ren, Mengjiao Yang, Joseph Gonzalez, Dale Schuurmans, and Bo Dai. "Making linear mdps practical via contrastive representation learning." In International Conference on Machine Learning, pp. 26447-26466. PMLR, 2022.

[3] Zhang, Hongming, Tongzheng Ren, Chenjun Xiao, Dale Schuurmans, and Bo Dai. "Provable Representation with Efficient Planning for Partially Observable Reinforcement Learning." In International Conference on Machine Learning, pp. 59759-59782. PMLR, 2024.

[4] Ren, Tongzheng, Tianjun Zhang, Lisa Lee, Joseph E. Gonzalez, Dale Schuurmans, and Bo Dai. "Spectral Decomposition Representation for Reinforcement Learning." In The Eleventh International Conference on Learning Representations.

[5] Shribak, Dmitry, Chen-Xiao Gao, Yitong Li, Chenjun Xiao, and Bo Dai. "Diffusion Spectral Representation for Reinforcement Learning." In The Thirty-eighth Annual Conference on Neural Information Processing Systems.
