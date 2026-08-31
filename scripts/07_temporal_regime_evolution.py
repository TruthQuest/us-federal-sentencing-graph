#!/usr/bin/env python3
"""
Temporal regime evolution across sentencing-policy epochs.

Federal sentencing changed structurally at least five times since 1991:

    Pre-Booker    : 1991-2004  (guidelines mandatory)
    Post-Booker   : 2005-2009  (guidelines advisory, Booker v. US)
    Post-FSA      : 2010-2013  (Fair Sentencing Act; crack-cocaine ratio)
    Post-A782     : 2014-2018  (Amendment 782; drug-quantity table)
    Post-FirstStep: 2019-present  (First Step Act; retroactivity + sentencing)

Analyses that pool across these average away real policy effects. This
script runs the stage-1 (adjusted co-occurrence) + step-4 (community
detection) pipeline SEPARATELY within each epoch, then tracks each
offense category's regime membership across epochs. An offense that
belongs to regime R_1 in the Pre-Booker epoch and regime R_2 in the
Post-FSA epoch has MIGRATED; migration events are the object of study
here.

To make migration comparable across epochs (community IDs are not
naturally aligned between independent Louvain runs), we align regimes
across adjacent epochs by maximum Jaccard overlap of offense-member sets.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EPOCHS = [
    ("Pre-Booker", 1991, 2004, "Sentencing Reform Act baseline"),
    ("Post-Booker", 2005, 2009, "US v. Booker (2005): guidelines advisory"),
    ("Post-FSA", 2010, 2013, "Fair Sentencing Act (2010): 18:1 crack/powder ratio"),
    ("Post-A782", 2014, 2018, "Amendment 782 (2014): drug quantity table"),
    ("Post-FirstStep", 2019, 2100, "First Step Act (2018): sentencing + retroactivity"),
]


def run_epoch_pipeline(events_path: Path, epoch_dir: Path, epoch_name: str,
                       start: int, end: int, n_perm: int, seed: int):
    """Slice events to epoch, write to temp file, run stage-1 co-occurrence
    and community detection, return the community assignment."""
    df = pd.read_parquet(events_path) if events_path.suffix == ".parquet" else pd.read_csv(events_path)
    sub = df[(df["fiscal_year"] >= start) & (df["fiscal_year"] <= end)].copy()
    if len(sub) < 500:
        print(f"  {epoch_name}: only {len(sub)} rows, skipping.")
        return None

    epoch_dir.mkdir(parents=True, exist_ok=True)
    events_out = epoch_dir / "events.parquet"
    sub.to_parquet(events_out, index=False)

    print(f"  {epoch_name}: {len(sub):,} events; running stage-1 ...")
    subprocess.run([
        sys.executable, "scripts/03_compute_adjusted_cooccurrence.py",
        "--in", str(events_out),
        "--out-dir", str(epoch_dir),
        "--n-perm", str(n_perm), "--seed", str(seed),
    ], check=True)

    # Loosen p-threshold proportionally to reduced n_perm within epochs:
    # the theoretical floor for a permutation p-value is 1/n_perm, so a
    # 0.001 threshold with 500 perms admits nothing structurally possible.
    epoch_p_threshold = max(0.01, 5.0 / n_perm)
    print(f"  {epoch_name}: running community detection (p<{epoch_p_threshold}) ...")
    subprocess.run([
        sys.executable, "scripts/04_community_detection.py",
        "--edges", str(epoch_dir / "adjusted_cooccurrence.csv"),
        "--out-dir", str(epoch_dir),
        "--p-threshold", str(epoch_p_threshold),
        "--n-boot", "50", "--seed", str(seed),
    ], check=True)

    comm_path = epoch_dir / "offense_communities.csv"
    if not comm_path.exists():
        return None
    return pd.read_csv(comm_path)


def align_communities(prev: pd.DataFrame, curr: pd.DataFrame) -> dict:
    """Map curr community IDs to prev community IDs by max Jaccard overlap."""
    if prev is None or curr is None:
        return {}
    prev_sets = {cid: set(g["offense_category"]) for cid, g in prev.groupby("community_id")}
    curr_sets = {cid: set(g["offense_category"]) for cid, g in curr.groupby("community_id")}
    mapping = {}
    for c_id, c_set in curr_sets.items():
        best_p, best_j = None, 0.0
        for p_id, p_set in prev_sets.items():
            j = len(c_set & p_set) / max(1, len(c_set | p_set))
            if j > best_j:
                best_p, best_j = p_id, j
        mapping[c_id] = (best_p, round(best_j, 3))
    return mapping


def detect_migrations(epoch_communities: dict) -> pd.DataFrame:
    """For each offense category, list its regime ID (aligned to Pre-Booker
    IDs where possible) across all epochs, and flag migration events."""
    all_offenses = set()
    for c in epoch_communities.values():
        if c is not None:
            all_offenses.update(c["offense_category"].tolist())

    aligned = {}
    prev = None
    prev_aligned_ids = None
    cumulative_mapping = {}
    for name, _, _, _ in EPOCHS:
        curr = epoch_communities.get(name)
        if curr is None:
            aligned[name] = None
            prev = None
            continue
        if prev is None:
            cumulative_mapping = {cid: cid for cid in curr["community_id"].unique()}
        else:
            step_map = align_communities(prev, curr)
            cumulative_mapping = {cid: (step_map.get(cid, (cid, 0))[0]) for cid in curr["community_id"].unique()}
        curr = curr.copy()
        curr["aligned_regime_id"] = curr["community_id"].map(cumulative_mapping)
        aligned[name] = curr
        prev = curr

    rows = []
    for offense in sorted(all_offenses):
        record = {"offense_category": offense}
        prev_regime = None
        migrations = 0
        for name, _, _, _ in EPOCHS:
            c = aligned.get(name)
            if c is None:
                record[name] = None
                continue
            match = c[c["offense_category"] == offense]
            if match.empty:
                record[name] = None
                continue
            r = int(match.iloc[0]["aligned_regime_id"]) if pd.notna(match.iloc[0]["aligned_regime_id"]) else None
            record[name] = r
            if prev_regime is not None and r is not None and r != prev_regime:
                migrations += 1
            if r is not None:
                prev_regime = r
        record["migration_count"] = migrations
        rows.append(record)
    if not rows:
        return pd.DataFrame(columns=["offense_category"] + [e[0] for e in EPOCHS] + ["migration_count"])
    return pd.DataFrame(rows).sort_values("migration_count", ascending=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default="data/sentencing_events.parquet")
    ap.add_argument("--out-dir", default="results/temporal")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    events_path = Path(args.events)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epoch_communities = {}
    for name, start, end, shock in EPOCHS:
        print(f"\n=== Epoch: {name} ({start}-{end}, {shock}) ===")
        epoch_dir = out_dir / name.lower().replace("-", "_")
        epoch_communities[name] = run_epoch_pipeline(
            events_path, epoch_dir, name, start, end, args.n_perm, args.seed,
        )

    migrations = detect_migrations(epoch_communities)
    migrations.to_csv(out_dir / "offense_regime_migrations.csv", index=False)
    print(f"\nWrote per-offense regime membership across epochs to "
          f"{out_dir / 'offense_regime_migrations.csv'}")
    top = migrations[migrations["migration_count"] > 0]
    if len(top):
        print(f"\nOffenses that migrated regime at least once ({len(top)}):")
        print(top.to_string(index=False))
