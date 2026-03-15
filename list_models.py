#!/usr/bin/env python3

# Copyright 2026 Jason Gagne
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""List available models with system compatibility assessment.

Shows which models can safely run on this hardware, estimated VRAM,
and headroom. Prevents selecting models that will crash (e.g., NvMap
allocation failures on Jetson).

Usage:
    python3 list_models.py                                    # list models + system info
    python3 list_models.py --matrix config/three_agent_mesh.json  # preview model matrix
    python3 list_models.py --matrix config/three_agent_mesh.json --mode asymmetric
    python3 list_models.py --compatible-only                  # only show models that fit
"""

import argparse
import json
import sys
from pathlib import Path

from sentinel.models import (
    discover_models, get_system_specs, get_compatible_models,
    filter_compatible, generate_model_matrix, print_model_matrix_preview,
)
from sentinel.ollama import OllamaClient


def print_system_info(specs):
    """Print system hardware summary."""
    print(f"\nSystem: {specs.gpu_name}")
    print(f"  Platform:        {specs.platform}")
    print(f"  Total RAM:       {specs.total_ram_gb} GB")
    print(f"  Available RAM:   {specs.available_ram_gb} GB")
    if specs.is_unified_memory:
        print(f"  Memory:          unified (GPU shares system RAM)")
    if specs.is_jetson and specs.max_single_alloc_bytes > 0:
        print(f"  NvMap limit:     {specs.max_single_alloc_bytes / (1024**3):.1f} GB single alloc")
    print(f"  Max model size:  {specs.max_model_vram_gb} GB (with {0.5:.0f} GB headroom)")


def print_models_table(compatibilities):
    """Print models with compatibility status."""
    print(f"\n{'':4s} {'Model':25s} {'Family':10s} {'Params':10s} {'Quant':8s} {'Size':>6s} {'Margin':>7s}  {'Notes'}")
    print(f"{'':4s} {'':25s} {'':10s} {'':10s} {'':8s} {'':>6s} {'':>7s}  {'':20s}")
    for c in compatibilities:
        m = c.model
        notes = c.reason
        margin_str = f"{c.margin_gb:+.1f}G" if c.fits else f"{c.margin_gb:+.1f}G"
        print(f"{c.status_icon} {m.name:25s} {m.family:10s} {m.parameter_size:10s} "
              f"{m.quantization:8s} {m.size_gb:5.1f}G {margin_str:>7s}  {notes}")


def main():
    parser = argparse.ArgumentParser(description="List models with system compatibility")
    parser.add_argument("--matrix", help="Experiment config to generate matrix from")
    parser.add_argument("--mode", default="homogeneous",
                        choices=["homogeneous", "asymmetric", "mixed"],
                        help="Matrix mode (default: homogeneous)")
    parser.add_argument("--models", help="Comma-separated model names (default: all compatible)")
    parser.add_argument("--compatible-only", action="store_true",
                        help="Only show/use models that fit on this system")
    args = parser.parse_args()

    client = OllamaClient()
    if not client.is_available():
        print("Error: Ollama is not running.", file=sys.stderr)
        sys.exit(1)

    specs = get_system_specs()
    compatibilities = get_compatible_models(client, specs)

    if not args.matrix:
        print_system_info(specs)
        if args.compatible_only:
            compatibilities = [c for c in compatibilities if c.fits]
        print_models_table(compatibilities)

        compatible_count = sum(1 for c in compatibilities if c.fits)
        total = len(compatibilities)
        print(f"\n{compatible_count}/{total} models compatible with this system")
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
        # Validate availability
        available_names = {c.model.name for c in compatibilities}
        for name in model_names:
            if name not in available_names:
                print(f"Model not found: {name}", file=sys.stderr)
                print(f"Available: {', '.join(sorted(available_names))}", file=sys.stderr)
                sys.exit(1)
        # Warn about incompatible selections
        for c in compatibilities:
            if c.model.name in model_names and not c.fits:
                print(f"WARNING: {c.model.name} — {c.reason}", file=sys.stderr)
    else:
        # Default: only compatible models
        compatible = filter_compatible(compatibilities)
        model_names = [m.name for m in compatible]
        if not model_names:
            print("No compatible models found for this system.", file=sys.stderr)
            sys.exit(1)

    print_system_info(specs)
    configs = generate_model_matrix(config, model_names, args.mode)
    print_model_matrix_preview(configs)

    if args.mode == "mixed" and len(model_names) > 2:
        total = len(model_names) ** len(config["agents"])
        print(f"\n  Warning: mixed mode with {len(model_names)} models x "
              f"{len(config['agents'])} agents = {total} experiments")


if __name__ == "__main__":
    main()
