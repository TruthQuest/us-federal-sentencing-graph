#!/usr/bin/env python3
"""
Generate synthetic sentencing data with four INDEPENDENTLY planted
signals, one per downstream analysis script, so each script can be
validated for detection sensitivity AND specificity (does it find what
was planted, does it fabricate what was not).

Planted signals:

  (A) STAGE-2 PAIR DISPARITY [detected by 03]
      ADMIN_JUST stacked with DRUG at elevated rate for Black/Hispanic
      defendants in districts D01-D05, holding severity constant.

  (B) COMMUNITY-LEVEL DISPARITY [detected by 06]
      Regime membership itself (drug-adjacent stacking cluster vs
      immigration cluster) predicted by race independent of severity,
      i.e. same-severity cases with equal probability of DRUG conviction
      still land in different regimes based on defendant race.

  (C) TEMPORAL REGIME MIGRATION [detected by 07]
      DRUG_POSS moves OUT of the drug-adjacent stacking cluster after
      the 2010 Fair Sentencing Act (year >= 2010) and eventually reduces
      further after Amendment 782 (year >= 2014); becomes structurally
      isolated in the post-2010 period.

  (D) JUDGE SIGNATURES [detected by 09]
      A subset of judges (IDs ending in odd digits) apply the stacking
      regime at 2x baseline rate; the pattern persists when they appear
      in multiple districts (a small fraction do), which is what
      distinguishes a judge signature from a district effect.

Output is entirely fabricated. Do not cite.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

OFFENSE_CATEGORIES = [
    "DRUG", "DRUG_POSS", "FIREARM", "IMMIG", "FRAUD", "LAUNDER", "RICO",
    "VIOLENT", "SEX", "ADMIN_JUST", "WEAPON", "TAX", "BRIBERY",
    "REGULATORY", "NATL_DEF", "OTHER",
]
RACE_CATEGORIES = ["White", "Black", "Hispanic", "Other"]
DISTRICTS = [f"D{str(i).zfill(2)}" for i in range(1, 21)]
CIRCUITS = {d: f"C{(int(d[1:]) - 1) // 3 + 1}" for d in DISTRICTS}

# A small pool of "traveling" judges to enable signal (D) detection:
# these judges' IDs appear in multiple districts over their careers.
TRAVELING_JUDGES = [f"TJ-{i}" for i in range(1, 11)]


def _offense_base_rates():
    weights = {
        "DRUG": 0.24, "DRUG_POSS": 0.03, "FIREARM": 0.13, "IMMIG": 0.22,
        "FRAUD": 0.12, "LAUNDER": 0.02, "RICO": 0.01, "VIOLENT": 0.05,
        "SEX": 0.04, "ADMIN_JUST": 0.03, "WEAPON": 0.01, "TAX": 0.02,
        "BRIBERY": 0.01, "REGULATORY": 0.01, "NATL_DEF": 0.005, "OTHER": 0.055,
    }
    probs = np.array([weights[c] for c in OFFENSE_CATEGORIES])
    return probs / probs.sum()


# Regime backbone: three fixed clusters of offenses that stack together
# above baseline rates, so community detection has real structure to find.
# Cluster 1 (Drug-Weapons): DRUG primaries commonly stack FIREARM, WEAPON
# Cluster 2 (Fraud-Fin): FRAUD primaries commonly stack LAUNDER, TAX
# Cluster 3 (Immigration-Docs): IMMIG primaries commonly stack ADMIN_JUST, OTHER
REGIME_BACKBONE = {
    "DRUG": [("FIREARM", 0.35), ("WEAPON", 0.15)],
    "FRAUD": [("LAUNDER", 0.30), ("TAX", 0.20)],
    "IMMIG": [("ADMIN_JUST", 0.25), ("OTHER", 0.15)],
}


def sample_bundle(rng, primary, fiscal_year, regime_backbone: bool, temporal_C: bool):
    """Baseline secondary-offense sampling PLUS optionally the regime
    backbone (population-level positive stacking, gated on regime_backbone)
    PLUS optionally the temporal-decay effect on DRUG_POSS (gated on
    temporal_C). Every planted signal is gated by an explicit flag so
    --null-only produces genuinely null-behaviour data."""
    secondaries = []
    if rng.random() < 0.35:
        candidates = [c for c in OFFENSE_CATEGORIES if c != primary]
        secondaries.append(rng.choice(candidates))
    if rng.random() < 0.10:
        secondaries.append(rng.choice([c for c in OFFENSE_CATEGORIES if c != primary]))

    if regime_backbone:
        for stack_offense, stack_rate in REGIME_BACKBONE.get(primary, []):
            if rng.random() < stack_rate:
                secondaries.append(stack_offense)

    if temporal_C and primary == "DRUG_POSS":
        if fiscal_year < 2010:
            if rng.random() < 0.55:
                secondaries.append(rng.choice(["ADMIN_JUST", "FIREARM", "DRUG"]))
        elif fiscal_year < 2014:
            if rng.random() < 0.20:
                secondaries.append(rng.choice(["ADMIN_JUST", "FIREARM", "DRUG"]))
        else:
            if rng.random() < 0.05:
                secondaries.append(rng.choice(["ADMIN_JUST", "FIREARM", "DRUG"]))
    return list(dict.fromkeys(secondaries))


def generate(n_rows: int, seed: int,
             disparity_A: bool, disparity_B: bool,
             temporal_C: bool, judge_D: bool, regime_backbone: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        fiscal_year = int(rng.integers(1991, 2025))
        # Signal (D) requires some judges to travel across districts.
        use_traveling_judge = judge_D and rng.random() < 0.05
        if use_traveling_judge:
            judge_id = str(rng.choice(TRAVELING_JUDGES))
            district = rng.choice(DISTRICTS)  # independent of judge => forces cross-venue appearances
        else:
            district = rng.choice(DISTRICTS)
            judge_suppressed = fiscal_year >= 2020 and rng.random() < 0.4
            judge_id = None if judge_suppressed else f"J{district}-{int(rng.integers(1, 9))}"
        circuit = CIRCUITS[district]
        race = rng.choice(RACE_CATEGORIES, p=[0.35, 0.30, 0.28, 0.07])
        sex = rng.choice(["M", "F"], p=[0.85, 0.15])
        citizenship = rng.choice(["US", "Non-US"], p=[0.78, 0.22])

        primary = rng.choice(OFFENSE_CATEGORIES, p=_offense_base_rates())
        final_offense_level = int(np.clip(rng.normal(24, 7), 4, 43))
        crim_hist_cat = int(rng.choice([1, 2, 3, 4, 5, 6],
                                        p=[0.45, 0.15, 0.13, 0.12, 0.08, 0.07]))
        mand_min = bool(rng.random() < (0.30 if primary in ("DRUG", "FIREARM") else 0.03))
        guideline_min = max(0, final_offense_level * 3 - crim_hist_cat * 2 + rng.normal(0, 4))
        guideline_max = guideline_min + rng.uniform(4, 14)
        sentence_months = float(np.clip(
            rng.normal((guideline_min + guideline_max) / 2, 6), 0, 470
        ))
        departure = rng.choice(
            ["none", "upward", "downward-substantial-assistance", "downward-other"],
            p=[0.55, 0.05, 0.20, 0.20],
        )

        secondaries = sample_bundle(rng, primary, fiscal_year, regime_backbone, temporal_C)

        # Signal (A): pair-level stacking disparity.
        if disparity_A and primary == "DRUG" and district in DISTRICTS[:5] \
                and race in ("Black", "Hispanic") and "ADMIN_JUST" not in secondaries:
            if rng.random() < 0.28:
                secondaries.append("ADMIN_JUST")

        # Signal (B): community-level disparity. Even holding primary and
        # severity constant, Black/Hispanic defendants land in the
        # drug-adjacent stacking cluster at elevated rate via FIREARM
        # or VIOLENT co-charges; White defendants of matched severity
        # land in an isolated pattern (no stack).
        if disparity_B and primary == "DRUG" and race in ("Black", "Hispanic"):
            if rng.random() < 0.22 and "FIREARM" not in secondaries:
                secondaries.append("FIREARM")
        if disparity_B and primary == "DRUG" and race == "White":
            secondaries = [s for s in secondaries if s not in ("FIREARM",)] \
                if rng.random() < 0.15 else secondaries

        # Signal (D): judge signatures. Traveling judges with odd trailing
        # digit apply stacking at 2x rate regardless of district.
        if judge_D and judge_id and judge_id.startswith("TJ-"):
            if int(judge_id.split("-")[-1]) % 2 == 1:
                if rng.random() < 0.45 and "ADMIN_JUST" not in secondaries:
                    secondaries.append("ADMIN_JUST")

        secondaries = list(dict.fromkeys(secondaries))
        rows.append(dict(
            case_id=f"SYN-{i:07d}",
            fiscal_year=fiscal_year, district=district, circuit=circuit,
            judge_id=judge_id,
            defendant_race=race, defendant_sex=sex, defendant_citizenship=citizenship,
            primary_offense=primary, secondary_offenses=";".join(secondaries),
            final_offense_level=final_offense_level,
            criminal_history_category=crim_hist_cat,
            guideline_min_months=round(guideline_min, 1),
            guideline_max_months=round(guideline_max, 1),
            mand_min_flag=mand_min,
            sentence_months=round(sentence_months, 1),
            departure_direction=departure,
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-rows", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic_sentencing.csv")
    ap.add_argument("--null-only", action="store_true",
                     help="Suppress ALL planted signals (negative control).")
    args = ap.parse_args()

    df = generate(args.n_rows, args.seed,
                  disparity_A=not args.null_only,
                  disparity_B=not args.null_only,
                  temporal_C=not args.null_only,
                  judge_D=not args.null_only,
                  regime_backbone=not args.null_only)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(f"Signals planted: {'NONE (negative control)' if args.null_only else 'A, B, C, D'}")
