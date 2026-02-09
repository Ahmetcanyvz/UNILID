#!/usr/bin/env python3
"""
Convert between UNILID model formats.

Pack (default):
    python convert.py results_100k
    python convert.py results_100k -o my_model.unilid

Unpack:
    python convert.py model.unilid --unpack -o tokenizers_dir/
"""
import argparse
from pathlib import Path

from unilid.model_io import save_unilid, unpack_unilid


def main():
    parser = argparse.ArgumentParser(
        description="Convert between UNILID model formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Pack tokenizers to .unilid
    python convert.py results_100k
    python convert.py results_100k -o my_model.unilid

    # Unpack .unilid to tokenizers directory
    python convert.py model.unilid --unpack
    python convert.py model.unilid --unpack -o tokenizers_dir/
        """,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input: model directory (for pack) or .unilid file (for unpack)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path",
    )
    parser.add_argument(
        "--unpack",
        action="store_true",
        help="Unpack .unilid to tokenizers directory",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} does not exist")
        return 1

    if args.unpack:
        # Unpack .unilid to directory
        if not args.input.suffix == ".unilid":
            print(f"Error: Expected .unilid file for --unpack, got {args.input}")
            return 1
        output = args.output or args.input.with_suffix("")
        unpack_unilid(args.input, output)
    else:
        # Pack directory to .unilid
        output = args.output or (args.input / "model.unilid")
        save_unilid(args.input, output)

    return 0


if __name__ == "__main__":
    exit(main())
