#!/usr/bin/env python3
"""List available models and preview model matrix experiments.

Usage:
    python3 list_models.py                                    # list available models
    python3 list_models.py --matrix config/three_agent_mesh.json  # preview homogeneous matrix
    python3 list_models.py --matrix config/three_agent_mesh.json --mode asymmetric
    python3 list_models.py --matrix config/three_agent_mesh.json --mode mixed
    python3 list_models.py --matrix config/three_agent_mesh.json --models gemma2:2b,llama3:latest
"""

import argparse
import json
import sys
from pathlib import Path

from sentinel.models import discover_models, generate_model_matrix, print_model_matrix_preview
from sentinel.ollama import OllamaClient


def main():
    parser = argparse.ArgumentParser(description="List models and preview model matrices")
    parser.add_argument("--matrix", help="Experiment config to generate matrix from")
    parser.add_argument("--mode", default="homogeneous",
                        choices=["homogeneous", "asymmetric", "mixed"],
                        help="Matrix mode (default: homogeneous)")
    parser.add_argument("--models", help="Comma-separated model names (default: all available)")
    args = parser.parse_args()

    client = OllamaClient()
    if not client.is_available():
        print("Error: Ollama is not running.", file=sys.stderr)
        sys.exit(1)

    available = discover_models(client)

    if not args.matrix:
        # Just list models
        print(f"\nAvailable Models ({len(available)}):")
        print(f"{'Name':25s} {'Family':10s} {'Params':10s} {'Quant':8s} {'Size':>6s}  {'Digest'}")
        print("-" * 80)
        for m in available:
            print(f"{m.name:25s} {m.family:10s} {m.parameter_size:10s} "
                  f"{m.quantization:8s} {m.size_gb:5.1f}G  {m.short_digest}")
        return

    # Generate model matrix
    config_path = Path(args.matrix)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        config = json.load(f)

    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
        # Validate
        available_names = {m.name for m in available}
        for name in model_names:
            if name not in available_names:
                print(f"Model not found: {name}", file=sys.stderr)
                print(f"Available: {', '.join(available_names)}", file=sys.stderr)
                sys.exit(1)
    else:
        model_names = [m.name for m in available]

    configs = generate_model_matrix(config, model_names, args.mode)
    print_model_matrix_preview(configs)

    if args.mode == "mixed" and len(model_names) > 2:
        total = len(model_names) ** len(config["agents"])
        print(f"\n  Warning: mixed mode with {len(model_names)} models × "
              f"{len(config['agents'])} agents = {total} experiments")


if __name__ == "__main__":
    main()
