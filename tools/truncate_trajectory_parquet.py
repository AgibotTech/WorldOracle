#!/usr/bin/env python3
"""Keep the first N rows of a trajectory Parquet file.

Example:
    python tools/truncate_trajectory_parquet.py \
        /path/to/trajectory.parquet \
        --num-frames 657 \
        --output /path/to/trajectory_first_657.parquet
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input Parquet file.")
    parser.add_argument(
        "--num-frames",
        type=int,
        required=True,
        help="Number of rows to retain, starting at frame 0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <input_stem>_first_<N>.parquet).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_output(input_path: Path, num_frames: int) -> Path:
    return input_path.with_name(
        f"{input_path.stem}_first_{num_frames}{input_path.suffix}"
    )


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output(input_path, args.num_frames)
    )

    if args.num_frames <= 0:
        raise SystemExit("--num-frames must be positive")
    if not input_path.is_file():
        raise SystemExit(f"input file does not exist: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {output_path}; use --overwrite")

    table = pq.read_table(input_path)
    original_frames = table.num_rows
    if args.num_frames > original_frames:
        raise SystemExit(
            f"--num-frames={args.num_frames} exceeds input length "
            f"{original_frames}"
        )

    truncated = table.slice(0, args.num_frames)
    if "frame_index" in truncated.column_names:
        expected = list(range(args.num_frames))
        if truncated["frame_index"].to_pylist() != expected:
            raise SystemExit(
                "frame_index is not contiguous from 0; refusing to write"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(
            truncated,
            temporary_path,
            compression="zstd",
            store_schema=True,
        )
        written = pq.read_table(temporary_path)
        if written.num_rows != args.num_frames or written.schema != truncated.schema:
            raise RuntimeError("written Parquet verification failed")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    end_frame = args.num_frames - 1
    print(f"input:  {input_path} ({original_frames} rows)")
    print(f"range:  0:{end_frame} inclusive ({args.num_frames} rows)")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
