#!/usr/bin/env python3
"""
Judge-level regime signatures.

For each judge with adequate case count and NON-SUPPRESSED identifier,
compute a regime-share vector: what fraction of their sentencing events
land in each detected regime. Then test:

  (1) SIGNATURE STABILITY. Is a judge's signature meaningfully different
      from the district baseline (does the judge apply regimes at rates
      distinguishable from the average judge in their district)?

  (2) CROSS-VENUE PERSISTENCE. For the subset of judges appearing in
      MORE THAN ONE district (senior-status designations, sitting by
      designation, etc.), does the signature persist across venues?
      This is the diagnostic that separates a genuine judge-level
      effect from a district effect: if the same judge exhibits the
      same regime pattern in two different districts, the pattern is a
      property of the judge, not the venue.

Interpretive care: judge-identifier suppression in USSC public files
means this analysis is only ever available for a SUBSET of judges, and
that subset is not random -- large-volume districts are overrepresented.
Report N of judges included as a hard number, do not extrapolate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def event_offenses(df):
    return df.apply(
        lambda r: [r["primary_offense"]] + [
            s for s in str(r.get("secondary_offenses") or "").split(";") if s
        ],
        axis=1,
    )


def assign_regime(offense_list, offense_to_regime):
    regimes = [offense_to_regime.get(o) for o in offense_list if o in offense_to_regime]
    if not regimes:
        return -1
    return int(pd.Series(regimes).mode().iloc[0])


def prep(df, communities):
    df = df.copy()
    df = df.dropna(subset=["judge_id"])
    o2r = dict(zip(communities["offense_category"], communities["community_id"]))
    df["offense_list"] = event_offenses(df)
    df["regime"] = df["offense_list"].apply(lambda lst: assign_regime(lst, o2r))
    df = df[df["regime"] >= 0].reset_index(drop=True)
    return df


def judge_signature_vector(sub: pd.DataFrame, regime_ids) -> np.ndarray:
    counts = sub["regime"].value_counts()
    total = counts.sum()
    if total == 0:
        return np.zeros(len(regime_ids))
    return np.array([counts.get(r, 0) / total for r in regime_ids])


def cosine(u, v):
    denom = (np.linalg.norm(u) * np.linalg.norm(v))
    return float(u @ v / denom) if denom > 0 else 0.0


def analyze_judges(df: pd.DataFrame, min_cases_per_venue: int) -> pd.DataFrame:
    regime_ids = sorted(df["regime"].unique())
    district_baselines = {
        d: judge_signature_vector(sub, regime_ids)
        for d, sub in df.groupby("district")
    }

    rows = []
    for judge_id, jdf in df.groupby("judge_id"):
        if len(jdf) < min_cases_per_venue:
            continue
        venues = jdf["district"].value_counts()
        adequate_venues = venues[venues >= min_cases_per_venue].index.tolist()

        overall_sig = judge_signature_vector(jdf, regime_ids)
        record = dict(
            judge_id=judge_id,
            n_cases=int(len(jdf)),
            n_venues_all=int(jdf["district"].nunique()),
            n_venues_adequate=int(len(adequate_venues)),
            regime_share=json.dumps({int(r): round(float(overall_sig[i]), 4)
                                      for i, r in enumerate(regime_ids)}),
            distance_from_primary_district_baseline=None,
            cross_venue_signature_stability=None,
        )

        if adequate_venues:
            primary = adequate_venues[0]
            base = district_baselines[primary]
            record["distance_from_primary_district_baseline"] = round(1 - cosine(overall_sig, base), 4)

        if len(adequate_venues) >= 2:
            sigs = [
                judge_signature_vector(jdf[jdf["district"] == v], regime_ids)
                for v in adequate_venues
            ]
            pairwise = [cosine(sigs[i], sigs[j])
                        for i in range(len(sigs)) for j in range(i + 1, len(sigs))]
            record["cross_venue_signature_stability"] = round(float(np.mean(pairwise)), 4)

        rows.append(record)

    result = pd.DataFrame(rows)
    if len(result) == 0:
        return pd.DataFrame(columns=[
            "judge_id", "n_cases", "n_venues_all", "n_venues_adequate",
            "regime_share", "distance_from_primary_district_baseline",
            "cross_venue_signature_stability",
        ])
    return result.sort_values(
        "distance_from_primary_district_baseline", ascending=False, na_position="last"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default="data/sentencing_events.parquet")
    ap.add_argument("--communities", default="results/offense_communities.csv")
    ap.add_argument("--out", default="results/judge_signatures.csv")
    ap.add_argument("--min-cases-per-venue", type=int, default=30)
    args = ap.parse_args()

    events_path = Path(args.events)
    df = pd.read_parquet(events_path) if events_path.suffix == ".parquet" else pd.read_csv(events_path)
    communities = pd.read_csv(args.communities)
    df = prep(df, communities)

    result = analyze_judges(df, args.min_cases_per_venue)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Wrote signatures for {len(result)} judges (>= {args.min_cases_per_venue} "
          f"cases in >= 1 venue) to {out_path}")
    if len(result) == 0:
        print("Zero judges qualified. Most common cause: judge_id is suppressed in "
               "public USSC files from FY2014 onward. This is a data-availability "
               "limit, not a pipeline failure.")
        exit(0)

    multi_venue = result.dropna(subset=["cross_venue_signature_stability"])
    if len(multi_venue):
        print(f"\n{len(multi_venue)} judges appear in >= 2 venues with adequate case counts.")
        print(f"Mean cross-venue signature stability: "
              f"{multi_venue['cross_venue_signature_stability'].mean():.3f}")
        print("Higher = signature persists across venue (judge-level effect); "
              "lower = signature is dominated by venue (district effect).")
        print("\nTop 5 most persistent (judge-effect candidates):")
        print(multi_venue.nlargest(5, "cross_venue_signature_stability")[
            ["judge_id", "n_cases", "n_venues_adequate", "cross_venue_signature_stability",
             "distance_from_primary_district_baseline"]
        ].to_string(index=False))
