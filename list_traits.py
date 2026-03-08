#!/usr/bin/env python3
"""List all available persona trait dimensions and values.

Usage:
    python3 list_traits.py
    python3 list_traits.py --dimension role
"""

import argparse
from sentinel.persona import TRAIT_DIMENSIONS, DIMENSION_ORDER


def main():
    parser = argparse.ArgumentParser(description="List SENTINEL persona traits")
    parser.add_argument("--dimension", "-d", help="Show only this dimension")
    args = parser.parse_args()

    if args.dimension:
        if args.dimension not in TRAIT_DIMENSIONS:
            print(f"Unknown dimension: {args.dimension}")
            print(f"Available: {', '.join(DIMENSION_ORDER)}")
            return
        dims = {args.dimension: TRAIT_DIMENSIONS[args.dimension]}
    else:
        dims = {d: TRAIT_DIMENSIONS[d] for d in DIMENSION_ORDER}

    for dim_name, dim in dims.items():
        print(f"\n{dim_name}")
        print(f"  {dim['description']}")
        print()
        for key, desc in dim["values"].items():
            print(f"    {key:24s} {desc}")

    if not args.dimension:
        print(f"\n{len(DIMENSION_ORDER)} dimensions, "
              f"{sum(len(d['values']) for d in TRAIT_DIMENSIONS.values())} total trait values")


if __name__ == "__main__":
    main()
