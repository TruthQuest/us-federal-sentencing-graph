#!/usr/bin/env python3
"""
Counterfactual regime assignment.

Train a classifier to predict regime membership from LEGITIMATE severity
+ venue features alone: final offense level, criminal history category,
mandatory-minimum flag, district, fiscal-year decade. Deliberately omit
race/sex/citizenship from the predictor set.

Then apply the classifier back to every event and compare predicted vs
actual regime. An event whose predicted regime differs from its actual
regime is a DIVERGENCE. Test whether divergence rate is patterned by
race, controlling for district and severity.

The structural analog: in the Cuba pipeline this was the Louvain-vs-
analyst agreement check (100/100 stability, 97-98% agreement). Here
"analyst" is replaced by the actual world, and "algorithm" is the
severity-only prediction; disagreement between them is the finding.

Interpretive care: a systematic race pattern in divergence is a stronger
disparity claim than a stage-2 pair coefficient, because it says the
regime the defendant landed in cannot be reconstructed from legitimate
severity + venue features alone -- something else, correlated with race,
is driving assignment. It is NOT proof of prosecutorial intent; it is
evidence that the observed assignment is not reducible to severity.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


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


def prep(df: pd.DataFrame, communities: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop rows with missing severity BEFORE anything else; sklearn's
    # GradientBoostingClassifier does not accept NaN in features, and
    # ~3% of USSC cases have incomplete guideline computation.
    df = df.dropna(subset=["final_offense_level", "criminal_history_category",
                             "mand_min_flag", "district", "fiscal_year"]).copy()
    df["fiscal_year_decade"] = (df["fiscal_year"] // 10 * 10).astype(int)
    df["mand_min_flag"] = df["mand_min_flag"].astype(int)
    o2r = dict(zip(communities["offense_category"], communities["community_id"]))
    df["offense_list"] = event_offenses(df)
    df["actual_regime"] = df["offense_list"].apply(lambda lst: assign_regime(lst, o2r))
    df = df[df["actual_regime"] >= 0].reset_index(drop=True)
    return df


def counterfactual_predictions(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """5-fold cross-val predictions using severity + venue only. Yields
    an out-of-fold predicted regime per event so downstream divergence
    testing is not contaminated by in-sample fit."""
    feature_cols = ["final_offense_level", "criminal_history_category", "mand_min_flag"]
    X_num = df[feature_cols].to_numpy()
    X_district = pd.get_dummies(df["district"], prefix="dist").to_numpy()
    X_decade = pd.get_dummies(df["fiscal_year_decade"], prefix="dec").to_numpy()
    X = np.hstack([X_num, X_district, X_decade])
    y = df["actual_regime"].to_numpy()

    preds = np.full(len(df), -1, dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold_idx, (train, test) in enumerate(skf.split(X, y)):
        clf = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=seed + fold_idx)
        clf.fit(X[train], y[train])
        preds[test] = clf.predict(X[test])
    df = df.copy()
    df["predicted_regime"] = preds
    df["divergent"] = (df["predicted_regime"] != df["actual_regime"]).astype(int)
    return df


def divergence_disparity(df: pd.DataFrame, ref_race: str) -> pd.DataFrame:
    df = df.dropna(subset=["defendant_race", "defendant_sex",
                             "defendant_citizenship", "district",
                             "final_offense_level", "criminal_history_category"]).copy()
    race_categories = df["defendant_race"].unique().tolist()
    if ref_race not in race_categories:
        ref_race = race_categories[0]
    df["defendant_race"] = pd.Categorical(
        df["defendant_race"],
        categories=[ref_race] + [r for r in race_categories if r != ref_race],
    )
    formula = ("divergent ~ C(defendant_race) + defendant_sex + defendant_citizenship + "
               "final_offense_level + criminal_history_category + mand_min_flag + "
               "C(district) + C(fiscal_year_decade)")
    model = smf.logit(formula, data=df).fit(disp=0, maxiter=200)
    rows = []
    for term in model.params.index:
        if "defendant_race" not in term and "defendant_sex" not in term \
                and "defendant_citizenship" not in term:
            continue
        rows.append(dict(
            covariate_term=term,
            odds_ratio=round(float(np.exp(model.params[term])), 4),
            p_value=round(float(model.pvalues[term]), 6),
            n_cases=len(df),
            covariate_set=f"race({ref_race} ref) + sex + citizenship + severity + venue",
            null_model="logistic regression: P(divergent) ~ demographics | severity + venue",
        ))
    return pd.DataFrame(rows).sort_values("p_value")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default="data/sentencing_events.parquet")
    ap.add_argument("--communities", default="results/offense_communities.csv")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--ref-race", default="White")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    events_path = Path(args.events)
    df = pd.read_parquet(events_path) if events_path.suffix == ".parquet" else pd.read_csv(events_path)
    communities = pd.read_csv(args.communities)
    df = prep(df, communities)

    print(f"Predicting counterfactual regime from severity+venue only for {len(df):,} events ...")
    df = counterfactual_predictions(df, args.seed)
    div_rate = df["divergent"].mean()
    print(f"Overall divergence rate (predicted != actual): {div_rate:.3f}")
    print("Divergence rate by race:")
    print(df.groupby("defendant_race")["divergent"].mean().to_string())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df[["case_id", "actual_regime", "predicted_regime", "divergent"]].to_csv(
        out_dir / "counterfactual_predictions.csv", index=False,
    )

    disparity = divergence_disparity(df, args.ref_race)
    disparity.to_csv(out_dir / "divergence_disparity.csv", index=False)
    print(f"\nWrote divergence-disparity coefficients to {out_dir / 'divergence_disparity.csv'}")
    sig = disparity[disparity["p_value"] < 0.001]
    if len(sig):
        print(f"\n{len(sig)} coefficients significant at p<0.001:")
        print(sig.to_string(index=False))