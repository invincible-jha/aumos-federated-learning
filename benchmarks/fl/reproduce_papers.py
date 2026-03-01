"""Academic FL benchmark reproduction scripts.

Gap #148: No Academic Benchmarks.

Reproduces key results from seminal federated learning papers using
AumOS simulation infrastructure. Each benchmark runs the FL simulation
with reference hyperparameters and reports convergence metrics.

Papers reproduced:
    [1] McMahan et al. (2017). Communication-Efficient Learning of Deep Networks
        from Decentralized Data. arXiv:1602.05629. (FedAvg)
    [2] Li et al. (2020). Federated Optimization in Heterogeneous Networks.
        arXiv:1812.06127. (FedProx)
    [3] Karimireddy et al. (2020). SCAFFOLD: Stochastic Controlled Averaging for
        Federated Learning. arXiv:1910.06378. (SCAFFOLD)
    [4] Dwork & Roth (2014). The Algorithmic Foundations of Differential Privacy.
        (DP baseline)

Usage:
    python -m benchmarks.fl.reproduce_papers --paper fedavg
    python -m benchmarks.fl.reproduce_papers --paper fedprox
    python -m benchmarks.fl.reproduce_papers --all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Add src to path for direct execution
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run.

    Attributes:
        paper: Short identifier for the paper (e.g. "fedavg").
        strategy: FL strategy used.
        num_clients: Number of simulated participants.
        num_rounds: Training rounds completed.
        fraction_fit: Client sampling fraction.
        final_accuracy: Final centralized accuracy.
        convergence_round: Round at which 90% of final accuracy was reached.
        wall_time_seconds: Total wall-clock time for the simulation.
        per_round_losses: List of distributed loss values per round.
    """

    paper: str
    strategy: str
    num_clients: int
    num_rounds: int
    fraction_fit: float
    final_accuracy: float
    convergence_round: int | None
    wall_time_seconds: float
    per_round_losses: list[float]


# Reference hyperparameters from the original papers
BENCHMARK_CONFIGS: dict[str, dict[str, Any]] = {
    "fedavg": {
        # McMahan et al. 2017 — Table 1 (MNIST 2-layer MLP, 100 clients, C=0.1)
        "strategy": "fedavg",
        "num_clients": 100,
        "num_rounds": 20,
        "fraction_fit": 0.1,
        "dp_epsilon": None,
        "fedprox_mu": 0.0,
        "description": "FedAvg (McMahan et al. 2017) — 100 clients, 10% sampled, 20 rounds",
    },
    "fedprox": {
        # Li et al. 2020 — μ=0.01, heterogeneous data (simulated via diverse random seeds)
        "strategy": "fedprox",
        "num_clients": 100,
        "num_rounds": 20,
        "fraction_fit": 0.1,
        "dp_epsilon": None,
        "fedprox_mu": 0.01,
        "description": "FedProx (Li et al. 2020) — μ=0.01, 100 clients, heterogeneous",
    },
    "scaffold": {
        # Karimireddy et al. 2020 — SCAFFOLD reduces client drift on non-IID data
        "strategy": "scaffold",
        "num_clients": 100,
        "num_rounds": 20,
        "fraction_fit": 0.1,
        "dp_epsilon": None,
        "fedprox_mu": 0.0,
        "description": "SCAFFOLD (Karimireddy et al. 2020) — 100 clients, 20 rounds",
    },
    "dp_fedavg": {
        # Dwork & Roth (2014) DP baseline with ε=1.0
        "strategy": "fedavg",
        "num_clients": 50,
        "num_rounds": 20,
        "fraction_fit": 0.2,
        "dp_epsilon": 1.0,
        "fedprox_mu": 0.0,
        "description": "DP-FedAvg (Dwork & Roth 2014) — ε=1.0, 50 clients, 20 rounds",
    },
}


async def run_benchmark(paper: str) -> BenchmarkResult:
    """Run a single paper benchmark and return results.

    Args:
        paper: Key into BENCHMARK_CONFIGS (e.g. "fedavg").

    Returns:
        BenchmarkResult with convergence metrics.

    Raises:
        KeyError: If paper name is not in BENCHMARK_CONFIGS.
        ImportError: If flwr is not installed.
    """
    config = BENCHMARK_CONFIGS[paper]

    from aumos_federated_learning.adapters.simulation_runner import SimulationRunner  # noqa: PLC0415

    runner = SimulationRunner(
        strategy=config["strategy"],
        num_clients=config["num_clients"],
        num_rounds=config["num_rounds"],
        fraction_fit=config["fraction_fit"],
        dp_epsilon=config["dp_epsilon"],
        fedprox_mu=config["fedprox_mu"],
    )

    print(f"\n{'=' * 60}")
    print(f"Benchmark: {paper}")
    print(f"Description: {config['description']}")
    print(f"{'=' * 60}")

    start = time.perf_counter()
    simulation_id = uuid.uuid4()
    result = await runner.run(simulation_id)
    elapsed = time.perf_counter() - start

    losses = [m.distributed_loss for m in result.per_round_metrics]

    # Find convergence round: first round reaching 90% of final accuracy
    convergence_round: int | None = None
    target_accuracy = result.final_accuracy * 0.9
    for metric in result.per_round_metrics:
        if metric.centralized_accuracy >= target_accuracy:
            convergence_round = metric.round_number
            break

    bench_result = BenchmarkResult(
        paper=paper,
        strategy=config["strategy"],
        num_clients=config["num_clients"],
        num_rounds=config["num_rounds"],
        fraction_fit=config["fraction_fit"],
        final_accuracy=result.final_accuracy,
        convergence_round=convergence_round,
        wall_time_seconds=elapsed,
        per_round_losses=losses,
    )

    print(f"  Final accuracy:     {result.final_accuracy:.4f}")
    print(f"  Convergence round:  {convergence_round or 'not reached'}")
    print(f"  Wall time:          {elapsed:.1f}s")
    print(f"  Simulation ID:      {result.simulation_id}")

    return bench_result


async def run_all_benchmarks() -> list[BenchmarkResult]:
    """Run all configured benchmarks sequentially.

    Returns:
        List of BenchmarkResult objects, one per paper.
    """
    results: list[BenchmarkResult] = []
    for paper in BENCHMARK_CONFIGS:
        result = await run_benchmark(paper)
        results.append(result)
    return results


def save_results(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save benchmark results to a JSON file.

    Args:
        results: List of benchmark results.
        output_path: File path for JSON output.
    """
    output = {
        "benchmark_suite": "AumOS Federated Learning Academic Benchmarks",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [asdict(r) for r in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")


def main() -> None:
    """Entry point for benchmark reproduction CLI."""
    parser = argparse.ArgumentParser(
        description="Reproduce academic FL benchmark results using AumOS simulation"
    )
    parser.add_argument(
        "--paper",
        choices=list(BENCHMARK_CONFIGS.keys()),
        help="Run a single benchmark",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.json"),
        help="Output JSON path (default: benchmark_results.json)",
    )
    args = parser.parse_args()

    if not args.paper and not args.all:
        parser.print_help()
        sys.exit(1)

    if args.all:
        results = asyncio.run(run_all_benchmarks())
    else:
        results = [asyncio.run(run_benchmark(args.paper))]

    save_results(results, args.output)


if __name__ == "__main__":
    main()
