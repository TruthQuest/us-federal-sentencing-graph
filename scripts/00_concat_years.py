#!/usr/bin/env python3
"""
After 01_load_ussc_data.py has produced per-year Parquet files under
data/events/, this script concatenates them into one full-corpus
Parquet at data/sentencing_events.parquet, which is what scripts
03-09 all consume.

Runs in constant memory: iterates over per-year files, reads one at a
time, writes to a growing Parquet via pyarrow. Never holds the full
corpus in RAM.
"""
import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def concat_streaming(in_dir: Path, out_path: Path):
    files = sorted(in_dir.glob("fy*.parquet"))
    if not files:
        raise SystemExit(f"No fy*.parquet files under {in_dir}")

    print(f"Concatenating {len(files)} per-year Parquet files -> {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows = 0
    try:
        for f in files:
            table = pq.read_table(f)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            else:
                # Cast to first-file schema in case of minor type drift between
                # years (integer nullable vs int, etc). Fails loudly on real
                # incompatibility rather than silently reordering columns.
                table = table.cast(writer.schema, safe=False)
            writer.write_table(table)
            total_rows += table.num_rows
            print(f"  {f.name}: +{table.num_rows:,} rows  (total {total_rows:,})")
    finally:
        if writer:
            writer.close()

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nWrote {total_rows:,} rows to {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", default="data/events")
    ap.add_argument("--out", default="data/sentencing_events.parquet")
    args = ap.parse_args()
    concat_streaming(Path(args.in_dir), Path(args.out))
