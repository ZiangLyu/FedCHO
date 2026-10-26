# 【Information Sciences 2027】FedCHO: Closed-Loop Hierarchical Orchestration for Blockchain-Based Byzantine-Robust Federated Learning

[![Paper](https://img.shields.io/badge/Paper-Information%20Sciences-b31b1b.svg)](https://doi.org/10.1016/j.ins.2026.124194)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ins.2026.124194-2f5bd3.svg)](https://doi.org/10.1016/j.ins.2026.124194)
[![Code](https://img.shields.io/badge/Code-PyTorch-ee4c2c.svg)](https://github.com/ZiangLyu/FedCHO)
[![Tasks](https://img.shields.io/badge/Tasks-Byzantine%20Robustness%20%7C%20Non--IID%20FL-6f42c1.svg)](#-core-task)

[[📖 Paper](https://doi.org/10.1016/j.ins.2026.124194)] [[💻 Code](https://github.com/ZiangLyu/FedCHO)]

**Fanfan Shen**<sup>1,†</sup>, **Ziang Liu**<sup>1</sup>, **Chao Xu**<sup>1</sup>, **Zhiquan Liu**<sup>2</sup>, **Jun Feng**<sup>3</sup>, **Yanxiang He**<sup>4</sup>

<sup>1</sup>Nanjing Audit University &nbsp; <sup>2</sup>Jinan University &nbsp; <sup>3</sup>Huazhong University of Science and Technology &nbsp; <sup>4</sup>Wuhan University &nbsp; <sup>†</sup>Corresponding author



## 👀 About FedCHO

Blockchain-based federated learning (BFL) must simultaneously handle **Byzantine attacks**, **resource heterogeneity**, and **client drift under Non-IID data**. Existing systems usually treat filtering, role allocation, and aggregation as independent modules. This open-loop design overlooks their hierarchical coupling: contaminated trust signals distort role allocation, poor role allocation biases the aggregation state, and a drifted global model corrupts the next round of trust estimation.

**FedCHO** formulates BFL governance as **Closed-Loop Hierarchical Orchestration**. It organizes trust filtering, resource orchestration, and adaptive aggregation into three coupled layers whose outputs feed the next communication round:

1. **Curriculum-based Dynamic Filtering (CDF)** separates benign Non-IID updates from Byzantine updates using layer-wise similarity, curriculum-guided thresholds, and asymmetric reputation.
2. **Score-based Dynamic Role Allocation (SDRA)** assigns Worker, Validator, and Miner roles from device capability, trust, participation, and network state under controlled exploration.
3. **Decentralized Reinforcement Aggregation Policy (DRAP)** uses a DDPG agent and heuristic fusion to continuously optimize aggregation weights and mitigate client drift.


## 🎯 Core Task

**Input:**

- A set of heterogeneous edge devices with private, potentially Non-IID local datasets
- Dynamic computation power, bandwidth, stake, participation, and online status
- Byzantine workers performing data- or model-poisoning attacks

**Output:**

- A Byzantine-robust global model without sharing raw local data
- A validated blockchain record of accepted model updates
- Resource-aware Worker, Validator, and Miner assignments for the next round
- Adaptive aggregation weights learned from the evolving global system state

**Four stages in each communication round:**

1. **Local training:** Workers initialize from the global model and perform clipped local optimization.
2. **Trust filtering:** Validators verify signatures and use CDF to filter suspicious updates.
3. **Role orchestration:** A PoS-elected leader applies SDRA to produce the next round's role sets.
4. **Adaptive aggregation:** DRAP weights the accepted updates, produces the new global model, and records the winning block.


## 📂 Project Structure

```text
FedCHO/
├── main.py                         # End-to-end BFL simulation and experiment entry point
├── Device.py                       # Worker/Validator/Miner logic, CDF, aggregation, attacks
├── miner_server.py                 # SDRA, system state, resource scoring, client selection
├── Block.py                        # Block structure, hashing, signatures, and transactions
├── Blockchain.py                   # Local blockchain state and chain operations
├── DDPG/
│   └── DDPG.py                     # Actor, critic, replay buffer, and DDPG updates for DRAP
├── DatasetManager/
│   └── DatasetManager.py           # IID/Dirichlet partitioning and poisoning utilities
├── DatasetLoad.py                  # MNIST loading helpers and Gaussian-noise transform
├── Models.py                       # CNN, MLP, and residual model definitions
├── models/
│   ├── CifarNet.py                 # CIFAR CNN
│   └── ResNet18.py                 # CIFAR-100 ResNet-18 helper
├── data/                            # Local datasets; 
└── README.md
```

## 📐 Set Up

The experiments in the paper use Python 3.10 and PyTorch.

```bash
git clone https://github.com/ZiangLyu/FedCHO.git
cd FedCHO

conda create -n fedcho python=3.10 -y
conda activate fedcho

python -m pip install --upgrade pip
pip install torch torchvision numpy scipy matplotlib pycryptodome click

---

## 🚀 Training and Simulation

### Quick Start

Run a short intelligent-orchestration experiment:

```bash
python main.py \
  --max_num_comm 10 \
  --num_devices 20 \
  --hard_assign 12,5,3 \
  --num_malicious 3 \
  --batchsize 64 \
  --default_local_epochs 5 \
  --learning_rate 0.01 \
  --experiment_mode intelligent \
  --pow_difficulty 0
```


### Intelligent vs. Random Role Allocation

```bash
# FedCHO: state-aware SDRA
python main.py --experiment_mode intelligent --hard_assign 12,5,3

# Random-allocation baseline
python main.py --experiment_mode random --hard_assign 12,5,3
```

### Simulating Byzantine Workers

```bash
# 5 malicious devices among 20 total devices
python main.py \
  --num_devices 20 \
  --num_malicious 5 \
  --noise_variance 1 \
  --hard_assign 12,5,3 \
  --experiment_mode intelligent
```

### Useful Arguments

| Argument | Default | Description |
| :--- | :---: | :--- |
| `--gpu` | `0` | GPU identifier recorded by the experiment configuration |
| `--batchsize` | `10` | Local mini-batch size |
| `--learning_rate` | `0.01` | Local optimizer learning rate |
| `--optimizer` | `Adam` | Local optimizer |
| `--max_num_comm` | `100` | Number of communication rounds |
| `--num_devices` | `20` | Total simulated devices |
| `--num_malicious` | `0` | Number of malicious devices |
| `--noise_variance` | `1` | Gaussian poisoning strength |
| `--default_local_epochs` | `5` | Local epochs per worker per round |
| `--hard_assign` | `*,*,*` | Required Workers, Validators, and Miners, e.g. `12,5,3` |
| `--network_stability` | `1.0` | Probability that a device is online |
| `--pow_difficulty` | `0` | `0` for PoS; positive values enable PoW difficulty |
| `--validator_threshold` | `1.0` | Base validation threshold |
| `--experiment_mode` | `intelligent` | `intelligent` for SDRA or `random` for the baseline |
| `--save_network_snapshots` | `0` | Enable serialized network snapshots |
| `--save_freq` | `5` | Snapshot frequency in communication rounds |

Use `python main.py --help` for the complete argument list.


## 📁 Outputs

Each run produces timestamped and cumulative diagnostics:

```text
logs/<timestamp>/
├── args_used.txt
├── malicious_wokrer_identifying_log.db
├── correctly_kicked_workers.txt
├── mistakenly_kicked_workers.txt
├── kicked_lazy_workers.txt
└── comm_<round>/
    └── accuracy_comm_<round>.txt

snapshots/<timestamp>/snapshot_r_<round>   # when snapshot saving is enabled
Output_accuracy_change_m5_dir0.05.txt      # cumulative accuracy summary
Output_change_m5_dir0.05.txt               # communication, RL, blockchain, and device statistics
```


## 📌 Citation

If you find this project useful, please cite:

```bibtex
@article{shen2027fedcho,
  title   = {FedCHO: Closed-loop hierarchical orchestration for blockchain-based byzantine-robust federated learning},
  author  = {Shen, Fanfan and Liu, Ziang and Xu, Chao and Liu, Zhiquan and Feng, Jun and He, Yanxiang},
  journal = {Information Sciences},
  volume  = {762},
  pages   = {124194},
  year    = {2027},
  doi     = {10.1016/j.ins.2026.124194}
}
```

## 🙏 Acknowledgements

This work was supported by the Basic Science (Natural Science) Research Project of Colleges and Universities in Jiangsu Province, the National Natural Science Foundation of China, the Postgraduate Research & Practice Innovation Program of Jiangsu Province, and the Qinglan Project of Jiangsu Universities.
