<div align="center">

<h1>Spiking World Model with Multi-Compartment Neurons for Model-based Reinforcement Learning</h1>

[Yinqian Sun](https://scholar.google.com/citations?user=QtGgt2wAAAAJ&hl=zh-CN)<sup>1,4</sup>,
[Feifei Zhao](https://scholar.google.com/citations?hl=zh-CN&user=_-3Hn-EAAAAJ)<sup>1,3,4</sup>,
Mingyang Lyu<sup>1,2</sup>,
[Yi Zeng](https://scholar.google.com/citations?user=Rl-YqPEAAAAJ&hl=zh-CN)<sup>1,2,3,4</sup>,


<sup>1</sup>Brain-inspired Cognitive AI Lab, Institute of Automation, Chinese Academy of Sciences, 
<sup>2</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences
<sup>3</sup>State Key Laboratory of Brain Cognition and Brain-inspired Intelligence Technology
<sup>4</sup>Long-term AI

[![arXiv](https://img.shields.io/badge/arXiv-2503.00713-red)](https://arxiv.org/abs/2503.00713)
[![DOI](https://img.shields.io/badge/doi-PNAS-green)](https://www.pnas.org/doi/abs/10.1073/pnas.2513319122)

</div>



## 💡 Introduction
**Spiking-WM** is a brain-inspired **spiking world model** for model-based reinforcement learning that introduces **multi-compartment neurons (MCNs)** to equip SNNs with long-term temporal memory. Inspired by nonlinear dendritic integration in biological neurons, Spiking-WM integrates a spiking state-space model, a spiking encoder, and a spiking policy network to enable end-to-end planning and decision-making. Experiments on the DeepMind Control Suite demonstrate that Spiking-WM outperforms existing SNN-based approaches and achieves performance comparable to GRU-based ANN world models, while evaluations on long-sequence speech benchmarks (SHD, TIMIT, and LibriSpeech 100h) further confirm its superior capability for modeling long-range temporal dependencies.

<div style="text-align: center;">
  <img src="assets/overview.jpg" alt="Spiking-WM" width="888"/>
</div>

## 📢 News
[2025-12-12] Spiking-WM has been published online in Proceedings of the National Academy of Sciences ([PNAS](www.pnas.org/doi/10.1073/pnas.2513319122)).

## Table of Contents
- [💡 Introduction](#-introduction)
- [📢 News](#-news)
- [Table of Contents](#table-of-contents)
- [🛠️ Installation](#️-installation)
- [🚀 Training](#-training)
- [📜 Citing](#-citing)
- [🙏 Acknowledgement](#-acknowledgement)
<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>

## 🛠️ Installation

The project uses [uv](https://docs.astral.sh/uv/) — no Docker, no Conda, no sudo required.
A single script sets up Python 3.10, PyTorch 2.4.1 + CUDA 12.1, MuJoCo EGL rendering,
and all dependencies including a patched build of `loris`.

### Prerequisites

- NVIDIA driver ≥ 525 (CUDA 12.1 minor-version compatible)
- `build-essential` installed on the host (for loris C++ compilation)
- `uv` binary — see [installation](https://docs.astral.sh/uv/getting-started/installation/) or copy from another machine

### Setup

```bash
git clone <repo-url> && cd Spiking-WM-cluster
bash setup.sh
source .venv/bin/activate

# Verify
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

`setup.sh` runs three steps: `uv python install 3.10` → `uv sync` → patched loris build.
Full details in [DEPLOY_NOTE.md](DEPLOY_NOTE.md).

### Known dependency issues (handled automatically by setup.sh)

| Problem | Root cause | Fix |
|---|---|---|
| `loris==0.5.3` install fails | `setup.py` uses `__builtins__.__NUMPY_SETUP__` — a dict under modern setuptools, not a module | `install_loris.sh` patches the line before building |
| `loris` C++ compilation error | Uses `PyArray_DESCR->fields`, removed in numpy 2.0 | `numpy==1.26.4` pinned first in `pyproject.toml` |
| `undefined symbol: iJIT_NotifyEvent` on `import torch` | pip `mkl==2026.x` missing a VTune symbol that torch 2.4.1 expects | `mkl-service` excluded from deps |

<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>


## 🚀 Training

```bash
source .venv/bin/activate

# Smoke test — fast debug run (~15-30 min)
bash scripts/train.sh

# Full training run
MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python dreamer.py --configs dmc_vision --task dmc_walker_walk --seed 0 --logdir ./logs/walker
```

### Background training (survives SSH disconnect)

```bash
nohup bash scripts/train.sh > logs/train.log 2>&1 &
tail -f logs/train.log
```

### Config system

Configs are split into individual YAML files under `configs/`. Multiple configs are merged in order,
with each file overriding only the keys it specifies:

```
configs/defaults.yaml       ← full set of default hyperparameters
configs/dmc_vision.yaml     ← task-suite overrides (encoder/decoder keys, envs, …)
configs/walker_bs32_500k.yaml  ← experiment-level overrides (batch_size, steps, …)
```

Pass them with `--configs` (applied left to right after `defaults`):

```bash
python dreamer.py --configs dmc_vision walker_bs32_500k \
  --task dmc_walker_walk --seed 0 --logdir /workspace/logs/walker_bs32_500k
```

To create a new experiment config, add a file to `configs/` with only the keys you want to change:

```yaml
# configs/my_experiment.yaml
spike_times: 8
batch_size: 64
steps: 1000000
```

Available built-in configs: `dmc_vision`, `dmc_proprio`, `crafter`, `atari100k`, `minecraft`,
`memorymaze`, `debug`.

<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>





## 📜 Citing

If you find Spiking-WM is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry:

```
@article{sun2025spiking,
author = {Yinqian Sun  and Feifei Zhao  and Mingyang Lyu  and Yi Zeng },
title = {Spiking world model with multicompartment neurons for model-based reinforcement learning},
journal = {Proceedings of the National Academy of Sciences},
volume = {122},
number = {50},
pages = {e2513319122},
year = {2025},
doi = {10.1073/pnas.2513319122},
URL = {https://www.pnas.org/doi/abs/10.1073/pnas.2513319122},
eprint = {https://www.pnas.org/doi/pdf/10.1073/pnas.2513319122},
}
```
The model of this research is one of the core and part of [BrainCog Embot](https://www.brain-cog.network/embot).BrainCog Embot is an Embodied AI platform under the Brain-inspired Cognitive Intelligence Engine (BrainCog) framework, which is an open-source Brain-inspired AI platform based on Spiking Neural Network. 
```
@article{zeng2023braincog,
title={BrainCog: A spiking neural network based, brain-inspired cognitive intelligence engine for brain-inspired AI and brain simulation},
author={Zeng, Yi and Zhao, Dongcheng and Zhao, Feifei and Shen, Guobin and Dong, Yiting and Lu, Enmeng and Zhang, Qian and Sun, Yinqian and Liang, Qian and Zhao, Yuxuan and others},
journal={Patterns},
volume={4},
number={8},
year={2023},
publisher={Elsevier}
}
```

<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>



## 🙏 Acknowledgement
Our work is primarily based on the following codebases:[dreamerv3-torch](https://github.com/NM512/dreamerv3-torch). We are sincerely grateful for their work.

