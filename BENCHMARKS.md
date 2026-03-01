# AumOS Federated Learning — Academic Benchmarks

Gap #148: Academic Benchmarks Reference

This document maps key federated learning papers to their AumOS simulation
reproductions and provides hyperparameter reference tables.

## Reproduced Papers

### 1. FedAvg — McMahan et al. (2017)

**Citation**: McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A.
(2017). Communication-Efficient Learning of Deep Networks from Decentralized Data.
*AISTATS 2017*. arXiv:1602.05629.

**Key results (Table 1 — MNIST 2-layer MLP)**:
| Setting | Rounds to 97% accuracy |
|---------|------------------------|
| SGD (centralized) | 1,194 |
| FedAvg (C=0.1, E=1, B=∞) | 1,181 |
| FedAvg (C=0.1, E=5, B=50) | 24 |

**AumOS reproduction**:
```bash
python -m benchmarks.fl.reproduce_papers --paper fedavg
```

**Reference config** (`BENCHMARK_CONFIGS["fedavg"]`):
- `num_clients`: 100
- `fraction_fit`: 0.1 (C=0.1)
- `num_rounds`: 20
- `strategy`: fedavg

---

### 2. FedProx — Li et al. (2020)

**Citation**: Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Smola, A., & Smith, V.
(2020). Federated Optimization in Heterogeneous Networks.
*MLSys 2020*. arXiv:1812.06127.

**Key claim**: FedProx with μ=0.01 achieves 22% faster convergence vs FedAvg
on heterogeneous (non-IID) data by penalising client drift via proximal term.

**AumOS reproduction**:
```bash
python -m benchmarks.fl.reproduce_papers --paper fedprox
```

**Reference config** (`BENCHMARK_CONFIGS["fedprox"]`):
- `num_clients`: 100
- `fraction_fit`: 0.1
- `num_rounds`: 20
- `strategy`: fedprox
- `fedprox_mu`: 0.01

---

### 3. SCAFFOLD — Karimireddy et al. (2020)

**Citation**: Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S. U., &
Suresh, A. T. (2020). SCAFFOLD: Stochastic Controlled Averaging for Federated
Learning. *ICML 2020*. arXiv:1910.06378.

**Key claim**: SCAFFOLD uses control variates to correct client drift, achieving
linear speedup in the number of clients on non-IID data.

**AumOS reproduction**:
```bash
python -m benchmarks.fl.reproduce_papers --paper scaffold
```

**Reference config** (`BENCHMARK_CONFIGS["scaffold"]`):
- `num_clients`: 100
- `fraction_fit`: 0.1
- `num_rounds`: 20
- `strategy`: scaffold (falls back to FedAvg in Flower simulation)

---

### 4. DP-FedAvg — Dwork & Roth (2014) + McMahan et al. (2018)

**Citations**:
- Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy.
  *Foundations and Trends in TCS*.
- McMahan, B., et al. (2018). Learning Differentially Private Recurrent Language
  Models. *ICLR 2018*. arXiv:1710.06963.

**Key claim**: DP-SGD with ε=1.0 introduces modest accuracy loss (~2-5%) while
providing strong formal privacy guarantees.

**AumOS reproduction**:
```bash
python -m benchmarks.fl.reproduce_papers --paper dp_fedavg
```

**Reference config** (`BENCHMARK_CONFIGS["dp_fedavg"]`):
- `num_clients`: 50
- `fraction_fit`: 0.2
- `num_rounds`: 20
- `strategy`: fedavg
- `dp_epsilon`: 1.0

---

### 5. FedAsync — Xie et al. (2019)

**Citation**: Xie, C., Koyejo, O., & Gupta, I. (2019). Asynchronous Federated
Optimization. arXiv:1903.03934.

**Key claim**: FedAsync with staleness-aware weighting α(τ) = 1/(1+τ) maintains
accuracy within 2% of synchronous FedAvg while eliminating straggler wait time.

**AumOS implementation**: `adapters/async_aggregator.py` — `FedAsyncAggregator`

---

### 6. FedDF — Lin et al. (2020)

**Citation**: Lin, T., Kong, L., Stich, S. U., & Jaggi, M. (2020). Ensemble
Distillation for Robust Model Fusion in Federated Learning.
*NeurIPS 2020*. arXiv:2006.07242.

**Key claim**: FedDF aggregates logits rather than weights, enabling heterogeneous
model architectures to collaborate in federated training.

**AumOS implementation**: `adapters/strategies/fed_df.py` — `FedDFStrategy`

---

## Data Partitioning

See `benchmarks/fl/data_partitioning.py` for:

| Method | Description | Reference |
|--------|-------------|-----------|
| `iid_partition` | Uniform random split | McMahan et al. 2017 |
| `dirichlet_partition` | Dirichlet(α) non-IID | Li et al. 2020 |
| `pathological_partition` | k classes per client | McMahan et al. 2017 |

### Heterogeneity Guide

| alpha (Dirichlet) | Heterogeneity | Use case |
|-------------------|---------------|----------|
| 0.1 | Very high | Worst-case non-IID |
| 0.5 | High | Realistic enterprise FL |
| 1.0 | Moderate | Mixed environments |
| 100.0 | Low (≈IID) | Homogeneous data silos |

---

## Running All Benchmarks

```bash
# Install dependencies
pip install aumos-federated-learning[simulation]

# Run all benchmarks and save JSON results
python -m benchmarks.fl.reproduce_papers --all --output results/benchmarks.json
```

## Expected Hardware

Benchmarks run in simulation mode (no GPU required). Reference times on
a 4-core laptop:

| Benchmark | Rounds | Expected time |
|-----------|--------|---------------|
| fedavg    | 20     | ~30s          |
| fedprox   | 20     | ~35s          |
| scaffold  | 20     | ~30s          |
| dp_fedavg | 20     | ~25s          |
