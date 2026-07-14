from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="veriweave",
        description="Run VeriWeave-VITA-BPA with coalitional verification, policy attention, and decision-drift certification.",
    )
    parser.add_argument("--max-tasks", type=int, help="Evaluate only the first N tasks (0 means all).")
    parser.add_argument("--top-k", type=int, help="Number of primary evidence items to retrieve.")
    parser.add_argument("--model", help="Ollama model identifier.")
    parser.add_argument("--benchmark", help="Path to the JSONL benchmark.")
    parser.add_argument("--methods", help="Comma-separated method names.")
    parser.add_argument("--vita-candidates", type=int, help="Maximum omitted clauses tested per closure round.")
    parser.add_argument("--coalition-size", type=int, help="Maximum omitted-evidence coalition size.")
    parser.add_argument("--vita-rounds", type=int, help="Maximum bounded closure rounds.")
    parser.add_argument("--boltzmann-candidates", type=int, help="Maximum clauses in the Boltzmann attention universe.")
    parser.add_argument("--temperature", type=float, help="Base Boltzmann policy-attention temperature.")
    parser.add_argument("--attention-mass", type=float, help="Target normalized marginal clause-attention mass captured by selected clauses.")
    parser.add_argument("--anneal-rate", type=float, help="Per-round temperature multiplier.")
    parser.add_argument("--offline", action="store_true", help="Disable Ollama and run the deterministic software smoke test.")
    args = parser.parse_args()

    if args.max_tasks is not None:
        os.environ["MAX_TASKS"] = str(args.max_tasks)
    if args.top_k is not None:
        os.environ["TOP_K"] = str(args.top_k)
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model
    if args.benchmark:
        os.environ["BENCHMARK_FILE"] = args.benchmark
    if args.methods:
        os.environ["METHODS"] = args.methods
    if args.vita_candidates is not None:
        os.environ["VITA_MAX_CANDIDATES"] = str(args.vita_candidates)
    if args.coalition_size is not None:
        os.environ["VITA_MAX_COALITION_SIZE"] = str(args.coalition_size)
    if args.vita_rounds is not None:
        os.environ["VITA_MAX_ROUNDS"] = str(args.vita_rounds)
    if args.boltzmann_candidates is not None:
        os.environ["BOLTZMANN_MAX_CANDIDATES"] = str(args.boltzmann_candidates)
    if args.temperature is not None:
        os.environ["BOLTZMANN_TEMPERATURE"] = str(args.temperature)
    if args.attention_mass is not None:
        os.environ["BOLTZMANN_ATTENTION_MASS"] = str(args.attention_mass)
    if args.anneal_rate is not None:
        os.environ["BOLTZMANN_ANNEAL_RATE"] = str(args.anneal_rate)
    if args.offline:
        os.environ["OLLAMA_ENABLED"] = "false"

    from .main import main as run

    run()


if __name__ == "__main__":
    main()
