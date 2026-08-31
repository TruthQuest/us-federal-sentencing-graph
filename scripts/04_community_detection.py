#!/usr/bin/env python3
"""
Louvain community detection on the regression-adjusted co-occurrence graph
(results/adjusted_cooccurrence.csv from step 03), with a bootstrap
stability check analogous to the Cuba project's blind-Louvain validation
(Step 09: 100/100 stability, 97-98% analyst/algorithm agreement).

Here there is no hand-labeled "analyst regime" to compare against a priori
-- that was specific to the Cuba case, where the two-regime hypothesis
preceded the algorithmic check. The honest analog for a first pass on
USSC data is: how stable are the detected communities under resampling?
Low stability means the "regime" language is not yet warranted; it means
noise. Report stability as a number, do not narrate it as confirmation
either way.
"""
import argparse
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

try:
    import community as community_louvain  # python-louvain
except ImportError as e:
    raise SystemExit("pip install python-louvain") from e


def build_graph(edges: pd.DataFrame, p_threshold: float) -> nx.Graph:
    g = nx.Graph()
    filt = edges[edges["p_value"] < p_threshold]
    if "passes_effect_size_filter" in filt.columns:
        filt = filt[filt["passes_effect_size_filter"]]
    # Only positive-stacking edges enter the graph. Negative edges (offenses
    # that co-appear LESS than severity predicts) are structurally meaningful
    # but are the wrong input to community detection: Louvain groups things
    # that ARE close, not things that repel each other.
    filt = filt[filt["adjusted_cooccurrence_weight"] > 0]
    for _, row in filt.iterrows():
        w = float(row["adjusted_cooccurrence_weight"])
        if w <= 0:
            continue
        g.add_edge(row["offense_a"], row["offense_b"], weight=w)
    return g


def run_louvain(g: nx.Graph, seed: int) -> dict:
    if g.number_of_edges() == 0:
        return {}
    return community_louvain.best_partition(g, weight="weight", random_state=seed)


def bootstrap_stability(edges: pd.DataFrame, p_threshold: float, n_boot: int, seed: int):
    """Resample edges with replacement (weighted resample of the edge
    list, not of underlying cases -- underlying-case resampling would
    require re-running the full stage-1/stage-2 regression pipeline per
    bootstrap draw, which is the statistically correct but expensive
    version; this edge-level resample is a cheaper proxy for structural
    stability of the partition, not a full re-estimation)."""
    rng = np.random.default_rng(seed)
    base_g = build_graph(edges, p_threshold)
    base_partition = run_louvain(base_g, seed)
    if not base_partition:
        return 0.0, {}

    agreement_scores = []
    for b in range(n_boot):
        sample = edges.sample(n=len(edges), replace=True, random_state=rng.integers(0, 2**31 - 1))
        g = build_graph(sample, p_threshold)
        partition = run_louvain(g, seed + b + 1)
        if not partition:
            continue
        common_nodes = set(base_partition) & set(partition)
        if len(common_nodes) < 2:
            continue
        # pairwise co-membership agreement, restricted to nodes present in both
        pairs_agree, pairs_total = 0, 0
        nodes = sorted(common_nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b_node = nodes[i], nodes[j]
                base_same = base_partition[a] == base_partition[b_node]
                boot_same = partition[a] == partition[b_node]
                pairs_agree += int(base_same == boot_same)
                pairs_total += 1
        if pairs_total:
            agreement_scores.append(pairs_agree / pairs_total)

    stability = float(np.mean(agreement_scores)) if agreement_scores else 0.0
    return stability, base_partition


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edges", default="results/adjusted_cooccurrence.csv")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--p-threshold", type=float, default=0.001)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    edges = pd.read_csv(args.edges)
    stability, partition = bootstrap_stability(edges, args.p_threshold, args.n_boot, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comm_path = out_dir / "offense_communities.csv"
    if partition:
        comm_df = pd.DataFrame(
            [{"offense_category": k, "community_id": v} for k, v in partition.items()]
        ).sort_values(["community_id", "offense_category"])
        comm_df.to_csv(comm_path, index=False)
        print(f"Wrote {len(comm_df)} offense-to-community assignments.")
        print("\nCommunities found:")
        for cid, group in comm_df.groupby("community_id"):
            print(f"  Community {cid}: {', '.join(group['offense_category'])}")
    else:
        # Overwrite any stale file with an empty one so downstream scripts
        # (06, 07, 08, 09) either error loudly or handle empty input; a
        # stale communities file from a previous run silently poisons all
        # regime-dependent analyses.
        pd.DataFrame(columns=["offense_category", "community_id"]).to_csv(comm_path, index=False)
        print("No significant edges at this p-threshold; no communities to report. "
              "This is a valid negative result, not a pipeline failure -- report it as such. "
              f"Wrote empty {comm_path} to invalidate any stale downstream artifacts.")

    print(f"\nBootstrap co-membership stability ({args.n_boot} resamples, "
          f"edge-level, see docstring for what this does and does not validate): "
          f"{stability:.3f}")

    with open(out_dir / "community_detection_summary.md", "w") as f:
        f.write(f"# Community Detection Summary\n\n")
        f.write(f"- p-threshold for edge inclusion: {args.p_threshold}\n")
        f.write(f"- Bootstrap resamples: {args.n_boot}\n")
        f.write(f"- Co-membership stability: {stability:.3f}\n")
        f.write(f"- Communities detected: {len(set(partition.values())) if partition else 0}\n")
        f.write("\nStability is an edge-resampling proxy, not a full re-estimation of the "
                "underlying regressions per bootstrap draw. Treat as a lower bar to clear, "
                "not a substitute for out-of-sample or held-out-year validation.\n")
