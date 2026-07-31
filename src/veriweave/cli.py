from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="veriweave",
        description="Run VeriWeave-VITA-PRO with provenance-robust evidence selection and counterfactual certification.",
    )
    parser.add_argument("--max-tasks", type=int, help="Evaluate only the first N tasks (0 means all).")
    parser.add_argument("--top-k", type=int, help="Number of primary evidence items to retrieve.")
    parser.add_argument("--model", help="Ollama model identifier.")
    parser.add_argument("--benchmark", help="Path to the JSONL benchmark.")
    parser.add_argument("--methods", help="Comma-separated method names.")
    parser.add_argument("--vita-candidates", type=int, help="Maximum omitted clauses tested per closure round.")
    parser.add_argument("--coalition-size", type=int, help="Maximum omitted-evidence coalition size.")
    parser.add_argument("--vita-rounds", type=int, help="Maximum bounded closure rounds.")
    parser.add_argument("--boltzmann-candidates", type=int, help="Maximum clauses in the BPA comparison universe.")
    parser.add_argument("--temperature", type=float, help="Base BPA temperature for the BPA ablation.")
    parser.add_argument("--attention-mass", type=float, help="Target BPA marginal attention mass.")
    parser.add_argument("--anneal-rate", type=float, help="BPA per-round temperature multiplier.")
    parser.add_argument("--retries", type=int, help="Maximum Ollama retries after a failed request.")
    parser.add_argument("--non-strict", action="store_true", help="Permit fallback outputs after exhausted retries. Not valid for paper runs.")
    parser.add_argument("--offline", action="store_true", help="Disable Ollama and run the deterministic software smoke test.")
    args = parser.parse_args()

    mappings = {
        "MAX_TASKS": args.max_tasks,
        "TOP_K": args.top_k,
        "OLLAMA_MODEL": args.model,
        "BENCHMARK_FILE": args.benchmark,
        "METHODS": args.methods,
        "VITA_MAX_CANDIDATES": args.vita_candidates,
        "VITA_MAX_COALITION_SIZE": args.coalition_size,
        "VITA_MAX_ROUNDS": args.vita_rounds,
        "BOLTZMANN_MAX_CANDIDATES": args.boltzmann_candidates,
        "BOLTZMANN_TEMPERATURE": args.temperature,
        "BOLTZMANN_ATTENTION_MASS": args.attention_mass,
        "BOLTZMANN_ANNEAL_RATE": args.anneal_rate,
        "OLLAMA_MAX_RETRIES": args.retries,
    }
    for key, value in mappings.items():
        if value is not None:
            os.environ[key] = str(value)
    if args.non_strict:
        os.environ["STRICT_MODEL_RUN"] = "false"
    if args.offline:
        os.environ["OLLAMA_ENABLED"] = "false"
        os.environ["STRICT_MODEL_RUN"] = "false"

    from .main import main as run

    run()


if __name__ == "__main__":
    main()
