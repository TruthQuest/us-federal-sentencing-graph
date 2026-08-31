#!/usr/bin/env python3
"""
Community-level disparity test. The novel move relative to standard USSC
disparity work: instead of testing "does race predict sentence length
conditional on guidelines" (mature literature), test "does race predict
which CHARGING REGIME the defendant lands in, conditional on severity
and district."

A charging regime here is an empirically-derived community from step 04
(scripts/04_community_detection.py output: results/offense_communities.csv),
i.e. a cluster of offense categories that co-occur at rates above what
guideline severity alone predicts. For each sentencing event we assign a
regime by looking at which community holds the plurality of that event's
offense codes. We then fit:

    P(regime = R_k | defendant) ~ race + sex + citizenship
                                  + final_offense_level + criminal_history_category
                                  + mand_min_flag + C(district) + C(fiscal_year_decade)

as a multinomial logit (or one-vs-rest if the multinomial doesn't
converge on real data). A statistically significant race coefficient
here, conditional on the severity block, is a claim about STRUCTURAL
selection into a regime -- structurally analogous to the Cuba
two-regimes finding but with the "regime" object empirically derived
rather than hand-labeled from state charging language.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

OFFENSE_CATEGORIES = [
    "DRUG", "DRUG_POSS", "FIREARM", "IMMIG", "FRAUD", "LAUNDER", "RICO",
    "VIOLENT", "SEX", "ADMIN_JUST", "WEAPON", "TAX", "BRIBERY",
    "REGULATORY", "NATL_DEF", "OTHER",
]
SEVERITY_COVARIATES = "final_offense_level + criminal_history_category + mand_min_flag"


def event_offenses(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda r: [r["primary_offense"]] + [
            s for s in str(r.get("secondary_offenses") or "").split(";") if s
        ],
        axis=1,
    )


def assign_regime(offense_list, offense_to_regime):
    regimes = [offense_to_regime.get(o) for o in offense_list if o in offense_to_regime]
    if not regimes:
        return None
    return int(pd.Series(regimes).mode().iloc[0])


def prep(df: pd.DataFrame, communities: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fiscal_year_decade"] = (df["fiscal_year"] // 10 * 10).astype(int)
    df["mand_min_flag"] = df["mand_min_flag"].astype(int)
    offense_to_regime = dict(zip(communities["offense_category"], communities["community_id"]))
    df["offense_list"] = event_offenses(df)
    df["regime"] = df["offense_list"].apply(lambda lst: assign_regime(lst, offense_to_regime))
    return df.dropna(subset=["regime"]).assign(regime=lambda d: d["regime"].astype(int))


def community_disparity(df: pd.DataFrame, ref_race: str,
                          condition_on_primary: bool) -> pd.DataFrame:
    """One-vs-rest logistic regressions, one per regime.

    Two specifications are supported:

    - condition_on_primary=False: regime membership ~ demographics +
      severity + venue. This shows how strongly race/sex/citizenship
      predict which regime a defendant ends up in overall.

    - condition_on_primary=True: regime membership ~ demographics +
      severity + venue + primary_offense. This CONTROLS for the fact
      that Black defendants face FIREARM (say) as primary at elevated
      rates; a race coefficient here is a claim about SECONDARY
      stacking patterns above and beyond primary-offense composition.

    Both are legitimate but answer different questions. The unconditional
    version can conflate "who ends up prosecuted for which primary
    offense" with "who gets stacked with what". The conditional version
    isolates the stacking-pattern disparity but risks overcontrolling if
    primary-offense selection itself is discriminatory. Report both."""
    df = df.dropna(subset=["defendant_race", "defendant_sex",
                             "defendant_citizenship", "district",
                             "final_offense_level", "criminal_history_category",
                             "primary_offense"]).copy()
    rows = []
    regime_ids = sorted(df["regime"].unique())
    race_categories = df["defendant_race"].unique().tolist()
    if ref_race not in race_categories:
        print(f"  ref_race {ref_race!r} not present in data; using {race_categories[0]!r}")
        ref_race = race_categories[0]
    df["defendant_race"] = pd.Categorical(
        df["defendant_race"],
        categories=[ref_race] + [r for r in race_categories if r != ref_race],
    )
    spec_label = "conditional_on_primary" if condition_on_primary else "unconditional"
    primary_term = " + C(primary_offense)" if condition_on_primary else ""
    for rid in regime_ids:
        df[f"in_regime_{rid}"] = (df["regime"] == rid).astype(int)
        formula = (
            f"in_regime_{rid} ~ C(defendant_race) + defendant_sex + defendant_citizenship + "
            f"{SEVERITY_COVARIATES} + C(district) + C(fiscal_year_decade){primary_term}"
        )
        try:
            model = smf.logit(formula, data=df).fit(disp=0, maxiter=200)
        except Exception as e:
            print(f"  regime {rid} model ({spec_label}) failed: {e}")
            continue
        for term in model.params.index:
            if "defendant_race" not in term and "defendant_sex" not in term \
                    and "defendant_citizenship" not in term:
                continue
            rows.append(dict(
                regime_id=rid,
                specification=spec_label,
                covariate_term=term,
                odds_ratio=round(float(np.exp(model.params[term])), 4),
                p_value=round(float(model.pvalues[term]), 6),
                n_cases=len(df),
                covariate_set=f"race({ref_race} ref) + sex + citizenship + "
                               f"{SEVERITY_COVARIATES} + district + fiscal_year_decade"
                               + (" + primary_offense" if condition_on_primary else ""),
                null_model="logistic regression: P(regime=k) ~ demographics | severity + venue"
                            + (" + primary offense" if condition_on_primary else ""),
            ))
    return pd.DataFrame(rows) if not rows else \
        pd.DataFrame(rows).sort_values(["regime_id", "specification", "p_value"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default="data/sentencing_events.parquet")
    ap.add_argument("--communities", default="results/offense_communities.csv")
    ap.add_argument("--out", default="results/community_disparity.csv")
    ap.add_argument("--ref-race", default="White")
    args = ap.parse_args()

    events_path = Path(args.events)
    df = pd.read_parquet(events_path) if events_path.suffix == ".parquet" else pd.read_csv(events_path)
    communities = pd.read_csv(args.communities)
    df = prep(df, communities)

    print(f"Assigned regimes to {len(df):,} sentencing events across "
          f"{df['regime'].nunique()} regimes.")
    if df["regime"].nunique() < 2:
        print("Only one regime detected -- community-level disparity is undefined "
               "(no alternative regime to compare against). This is a data-sparsity "
               "outcome, not a pipeline failure. Rerun with more years pooled.")
        pd.DataFrame(columns=["regime_id","specification","covariate_term","odds_ratio",
                                "p_value","n_cases","covariate_set","null_model"]).to_csv(
            Path(args.out), index=False)
        exit(0)

    print("Running BOTH unconditional and primary-conditional specifications ...")
    r1 = community_disparity(df, args.ref_race, condition_on_primary=False)
    r2 = community_disparity(df, args.ref_race, condition_on_primary=True)
    result = pd.concat([r1, r2], ignore_index=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Wrote {len(result)} community-disparity coefficients to {out_path}")
    sig = result[result["p_value"] < 0.001]
    if len(sig):
        print(f"\n{len(sig)} coefficients significant at p<0.001:")
        print(sig[["regime_id", "specification", "covariate_term", "odds_ratio", "p_value"]].to_string(index=False))
