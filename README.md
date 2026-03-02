# PPO BipedalWalker — Implemented from Scratch with PyTorch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-BipedalWalker--v3-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

> A clean, modular implementation of **Proximal Policy Optimization (PPO)** applied to the continuous-control benchmark `BipedalWalker-v3` — built entirely from scratch, without any RL framework.

---

## Overview

This project implements every component of the PPO algorithm from the ground up using PyTorch, with a focus on correctness and reproducibility:

- **Clipped surrogate objective** for stable, bounded policy updates
- **Generalized Advantage Estimation (GAE)** for low-variance advantage computation
- **Squashed Gaussian policy** with tanh squashing and proper log-prob correction (à la SAC)
- **Value function clipping** to stabilize critic training
- **Running observation normalization** (mean/variance)
- **TensorBoard** integration for real-time training monitoring
- **MPS (Apple Silicon) and CUDA** hardware acceleration support

---

## Architecture

The agent uses a **shared-trunk Actor-Critic** network with orthogonal weight initialization:

```
Observation (24-dim)
        |
+-------v------------------+
|     Shared MLP Trunk     |
|  Linear(24->64) + Tanh   |
|  Linear(64->64) + Tanh   |
+-------+----------+-------+
        |          |
  +-----v---+  +---v------+
  |  Actor  |  |  Critic  |
  |  mu-head|  |  V-head  |
  | (4-dim) |  | (scalar) |
  +---------+  +----------+
        |
  a = tanh(mu + sigma*eps)   <- Squashed Gaussian
```

Weight initialization follows the scheme from the PPO paper:
- Trunk layers: orthogonal, gain = sqrt(2)
- Policy head (mu): orthogonal, gain = 0.01
- Value head (V): orthogonal, gain = 1.0

---

## Key Implementation Details

| Component | Detail |
|---|---|
| **Policy** | Squashed Gaussian (tanh) with log-prob correction |
| **Advantage** | GAE (lambda=0.95, gamma=0.99) |
| **Policy loss** | Clipped surrogate objective (eps = 0.2) |
| **Value loss** | Clipped MSE to prevent large value updates |
| **Optimizer** | Adam with gradient clipping (max_norm = 0.5) |
| **KL early stop** | Optional per-update KL divergence threshold |
| **Obs. normalization** | Running mean/variance normalization |
| **Hardware** | MPS (Apple Silicon) + CUDA |
| **Monitoring** | TensorBoard: loss, entropy, KL divergence, explained variance |

---

## Project Structure

```
ppo-bipedalwalker/
├── scripts/
│   ├── train.py          # Launch training run
│   └── eval.py           # Deterministic evaluation + video recording
├── src/ppo/
│   ├── model.py          # ActorCritic network + Gaussian utilities
│   ├── buffer.py         # Rollout buffer with GAE computation
│   ├── update.py         # PPO update step (clipped objective + value clip)
│   ├── train_core.py     # Main training loop
│   ├── normalize.py      # Running observation normalizer
│   └── utils.py          # Explained variance and misc helpers
├── videos_eval/          # Recorded evaluation episodes (.mp4)
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone & setup

```bash
git clone https://github.com/01Giuliano01/ppo-bipedalwalker.git
cd ppo-bipedalwalker
python -m venv .envRL
source .envRL/bin/activate  # Windows: .envRL\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the agent

```bash
python -m scripts.train
```

Monitor training in real time:

```bash
tensorboard --logdir runs/
```

### 3. Evaluate a checkpoint

```bash
python -m scripts.eval --ckpt checkpoints/PPO_squashed_BipedalWalker-v3_seed0_1625811779_update100.pt
```

Evaluation episodes are recorded as `.mp4` files in `videos_eval/`.

---

## Requirements

```
Python 3.8+
torch
gymnasium[box2d]
tensorboard
numpy
```

---

## References

- Schulman et al. (2017) — [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Haarnoja et al. (2018) — [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) (squashed Gaussian policy)
- Huang et al. (2022) — [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
