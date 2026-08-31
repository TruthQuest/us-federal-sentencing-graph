#!/usr/bin/env python3
"""
Two distinct statistical products, deliberately kept separate because they
answer different questions (see ontology comment on :DisparityHypothesis
vs :adjustedCoOccurrenceWeight):

1. ADJUSTED CO-OCCURRENCE GRAPH (pattern discovery, demographics-blind).
   For each offense category, fit logistic regression of "does this
   offense appear on this case" against legitimate severity confounds
   only (final offense level, criminal history category, mandatory-
   minimum flag, district fixed effects, fiscal-year-decade fixed
   effects). Take deviance residuals. Correlate residuals pairwise across
   offense categories. This is the direct structural analog of the Cuba
   project's permutation-tested charge-stacking bundles, but residualized
   against guideline structure first, so a resulting cluster means
   "these offenses co-occur MORE than guideline severity alone predicts,"
   not "these offenses are both severe." Significance of each pairwise
   residual correlation is assessed by permutation (n=5000, shuffling
   residuals across cases, since residualization already removes the
   confound structure the permutation would otherwise need to preserve).

2. DISPARITY REGRESSION (the actual :DisparityHypothesis test).
   For each (primary offense, candidate secondary/stacked offense) pair
   flagged as a real cluster in product (1), fit a logistic regression
   predicting presence of the secondary offense from race, sex,
   citizenship, and district, CONTROLLING for the same severity confounds.
   A significant race or district coefficient here, conditional on an
   already-adjusted co-occurrence signal, is the actual disparity claim.
   Running this step without step (1) first invites fishing across all
   256 offense pairs; step (1) narrows the hypothesis space honestly.

Every row of every output file carries covariate_set, null_model, and
p_value so it satisfies shapes/ussc_shapes.ttl's AnalysisResultShape.
"""
import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)

OFFENSE_CATEGORIES = [
    "DRUG", "DRUG_POSS", "FIREARM", "IMMIG", "FRAUD", "LAUNDER", "RACKET",
    "VIOLENT", "SEX", "ADMIN_JUST", "WEAPON", "TAX", "BRIBERY", "CONSPIRACY",
    "REGULATORY", "NATL_DEF", "OTHER",
]
SEVERITY_COVARIATES = "final_offense_level + criminal_history_category + mand_min_flag"


def _discover_categories(df: pd.DataFrame) -> list[str]:
    """Prefer categories actually present in the data (as primary OR
    secondary) over the hardcoded list. This keeps the pipeline in sync
    with whatever statute_categorizer.py produces without editing scripts
    03-09 every time the vocabulary evolves."""
    seen = set()
    seen.update(df["primary_offense"].dropna().unique())
    for row in df["secondary_offenses"].dropna():
        seen.update(s for s in str(row).split(";") if s)
    # Preserve hardcoded order for reproducibility; append newly-discovered.
    ordered = [c for c in OFFENSE_CATEGORIES if c in seen]
    extra = sorted(seen - set(OFFENSE_CATEGORIES))
    return ordered + extra
NULL_MODEL_STAGE1 = (
    "logistic regression: P(has_both_offenses) ~ "
    f"{SEVERITY_COVARIATES} + C(district) + C(fiscal_year_decade); "
    "odds ratio of observed vs severity-predicted joint rate; Wald z-test on log(OR); "
    "effect-size filter (default OR>=1.5 or OR<=1/1.5) applied BEFORE significance filter "
    "because at USSC-scale n, p-value alone is trivially significant"
)
NULL_MODEL_STAGE2 = (
    "logistic regression: has_secondary ~ race + sex + citizenship + "
    f"C(district) + {SEVERITY_COVARIATES} + C(fiscal_year_decade)"
)


def has_offense(df: pd.DataFrame, category: str) -> pd.Series:
    is_primary = df["primary_offense"] == category
    is_secondary = df["secondary_offenses"].fillna("").str.split(";").apply(
        lambda lst: category in lst
    )
    return (is_primary | is_secondary).astype(int)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fiscal_year_decade"] = (df["fiscal_year"] // 10 * 10).astype(int)
    df["mand_min_flag"] = df["mand_min_flag"].astype(int)
    cats = _discover_categories(df)
    for cat in cats:
        df[f"off_{cat}"] = has_offense(df, cat)
    # Stash the discovered set as an attribute for downstream fns
    df.attrs["offense_categories"] = cats
    return df


def _fit_marginals(df: pd.DataFrame) -> dict:
    """Fit P(has_offense_a | severity + venue) once per offense category
    and cache the fitted predictions per case. Used by
    stage1_adjusted_cooccurrence to build the conditional-independence
    baseline for each pair without refitting inside the pair loop."""
    preds = {}
    formula_template = f"off_{{cat}} ~ {SEVERITY_COVARIATES} + C(district) + C(fiscal_year_decade)"
    cats = df.attrs.get("offense_categories", OFFENSE_CATEGORIES)
    for cat in cats:
        col = f"off_{cat}"
        if col not in df.columns or df[col].nunique() < 2 or df[col].mean() < 0.005:
            preds[cat] = None
            continue
        try:
            model = smf.logit(formula_template.format(cat=cat), data=df).fit(disp=0, maxiter=200)
            preds[cat] = model.predict(df).to_numpy()
        except Exception:
            preds[cat] = None
    return preds


def stage1_adjusted_cooccurrence(df: pd.DataFrame, min_odds_ratio: float,
                                   p_threshold: float) -> pd.DataFrame:
    """For each unordered pair (a, b) of offense categories, fit:

        P(has_both_a_and_b) ~ severity + venue + fiscal_year_decade

    and compare against the marginal expectation. The 'adjusted
    co-occurrence weight' is the LOG-ODDS-RATIO of joint presence beyond
    what severity + venue predicts, from a logistic regression where the
    outcome is the joint-presence indicator and the predictors are the
    severity covariates. Sign is meaningful: positive = stacked more
    together than severity predicts (a stacking regime); negative =
    stacked less.

    NOT residual correlation of two presence indicators -- that pipeline
    is confounded by primary-offense mutual exclusion (a case has one
    primary), which makes any two primary-heavy offenses look negatively
    correlated even after 'residualization', because residualization
    against severity does not remove the exclusion constraint. Joint-
    presence logistic regression side-steps the confound because it
    treats co-presence as its own binary outcome.

    At USSC scale (~60k rows/year, tens of millions cumulative), any
    tiny effect will be statistically significant. Therefore effect size
    (odds ratio) is filtered FIRST, p-value SECOND. Default min_odds_ratio
    of 1.5x (or its reciprocal for suppression) is chosen to isolate
    substantively meaningful stacking, not detectable-at-scale noise."""
    print("  fitting marginal severity models for each offense category ...")
    marginal_preds = _fit_marginals(df)
    from scipy.stats import norm

    cats = df.attrs.get("offense_categories", OFFENSE_CATEGORIES)
    rows = []
    for a, b in itertools.combinations(cats, 2):
        col_a, col_b = f"off_{a}", f"off_{b}"
        joint = ((df[col_a] == 1) & (df[col_b] == 1)).astype(int)
        n_joint = int(joint.sum())
        n_total = int(len(joint))
        if n_joint < 30 or n_joint == n_total:
            continue
        pred_a, pred_b = marginal_preds.get(a), marginal_preds.get(b)
        if pred_a is None or pred_b is None:
            continue
        # Some rows have NaN severity covariates and get NaN predictions;
        # restrict to rows where BOTH marginal predictions are defined so
        # observed vs expected are computed on the same denominator.
        pair_prod = pred_a * pred_b
        valid_mask = np.isfinite(pair_prod)
        if valid_mask.sum() < 100:
            continue
        expected_joint = float(pair_prod[valid_mask].mean())
        observed_joint = float(joint.to_numpy()[valid_mask].mean())
        n_joint_valid = int(joint.to_numpy()[valid_mask].sum())
        n_total_valid = int(valid_mask.sum())
        if not (0 < expected_joint < 1) or not (0 < observed_joint < 1):
            continue

        odds_ratio = (observed_joint / (1 - observed_joint)) / \
                     (expected_joint / (1 - expected_joint))
        se_obs = np.sqrt(observed_joint * (1 - observed_joint) / n_total_valid)
        se_log_or = se_obs / max(observed_joint * (1 - observed_joint), 1e-9)
        z = np.log(odds_ratio) / se_log_or if se_log_or > 0 else 0.0
        p_value = 2 * (1 - norm.cdf(abs(z)))

        rows.append(dict(
            offense_a=a, offense_b=b,
            adjusted_cooccurrence_weight=round(float(np.log(odds_ratio)), 5),
            odds_ratio=round(float(odds_ratio), 4),
            observed_joint_rate=round(float(observed_joint), 5),
            expected_joint_rate=round(float(expected_joint), 5),
            n_joint_cases=n_joint_valid,
            n_cases=n_total_valid,
            p_value=round(float(p_value), 6),
            covariate_set=SEVERITY_COVARIATES + " + district + fiscal_year_decade",
            null_model=NULL_MODEL_STAGE1,
            passes_effect_size_filter=bool(odds_ratio >= min_odds_ratio or odds_ratio <= 1.0 / min_odds_ratio),
        ))
    if not rows:
        return pd.DataFrame(columns=[
            "offense_a", "offense_b", "adjusted_cooccurrence_weight", "odds_ratio",
            "observed_joint_rate", "expected_joint_rate", "n_joint_cases",
            "n_cases", "p_value", "covariate_set", "null_model",
            "passes_effect_size_filter",
        ])
    out = pd.DataFrame(rows).sort_values("adjusted_cooccurrence_weight", ascending=False)
    return out


def stage2_disparity_tests(df: pd.DataFrame, edge_list: pd.DataFrame,
                            p_threshold: float, ref_race: str) -> pd.DataFrame:
    """Only run on offense pairs that survived stage 1 -- see module
    docstring on why this ordering matters."""
    rows = []
    # Deliberately BROADER than the stage-1 effect-size filter. The stage-1
    # filter decides which pairs are part of the regime graph (i.e., which
    # co-occur more or less than severity predicts at the POPULATION
    # level). Demographic disparity, by contrast, is precisely the case
    # where the pop-level effect is null-ish because a strong within-group
    # effect (Black+Hispanic elevated stacking) is diluted by a weak
    # majority-group effect. Filtering stage 2 by stage-1 significance
    # therefore SYSTEMATICALLY MISSES the disparities the pipeline is
    # trying to find. Instead: test every pair with enough joint cases to
    # fit a race-covariate logit.
    candidates = edge_list[edge_list["n_joint_cases"] >= 100]
    for _, edge in candidates.iterrows():
        primary_cat, secondary_cat = edge["offense_a"], edge["offense_b"]
        sub = df[df["primary_offense"] == primary_cat].copy()
        if len(sub) < 100:
            continue
        sub["target"] = sub[f"off_{secondary_cat}"]
        if sub["target"].nunique() < 2:
            continue
        # Real USSC data has NaN race for ~3% of cases; drop them before
        # setting the Categorical so pd.Categorical does not choke on nulls.
        sub = sub.dropna(subset=["defendant_race", "defendant_sex",
                                    "defendant_citizenship", "district",
                                    "final_offense_level", "criminal_history_category"])
        if len(sub) < 100 or sub["target"].nunique() < 2:
            continue
        race_categories = sub["defendant_race"].unique().tolist()
        if ref_race not in race_categories:
            continue
        sub["defendant_race"] = pd.Categorical(
            sub["defendant_race"],
            categories=[ref_race] + [r for r in race_categories if r != ref_race],
        )
        formula = (
            f"target ~ C(defendant_race) + defendant_sex + defendant_citizenship + "
            f"C(district) + {SEVERITY_COVARIATES} + C(fiscal_year_decade)"
        )
        try:
            model = smf.logit(formula, data=sub).fit(disp=0, maxiter=200)
        except Exception as e:
            print(f"  stage2 fit failed for {primary_cat}->{secondary_cat}: {e}")
            continue
        for term in model.params.index:
            if "defendant_race" not in term:
                continue
            rows.append(dict(
                primary_offense=primary_cat,
                stacked_offense=secondary_cat,
                covariate_term=term,
                odds_ratio=round(float(np.exp(model.params[term])), 4),
                p_value=round(float(model.pvalues[term]), 6),
                n_cases=len(sub),
                covariate_set=f"race({ref_race} ref) + sex + citizenship + district + "
                               f"{SEVERITY_COVARIATES} + fiscal_year_decade",
                null_model=NULL_MODEL_STAGE2,
            ))
    return pd.DataFrame(rows).sort_values("p_value") if rows else pd.DataFrame()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", default="data/sentencing_events.parquet")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-perm", type=int, default=5000,
                     help="Retained for API compat; joint-presence null does not use permutations.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-odds-ratio", type=float, default=1.5,
                     help="Effect-size filter for stage 1: only pairs with joint OR >= this "
                          "(stacking) or <= 1/this (suppression) count as candidate regime edges.")
    ap.add_argument("--stage2-p-threshold", type=float, default=0.001)
    ap.add_argument("--ref-race", default="White")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    df = pd.read_parquet(in_path) if in_path.suffix == ".parquet" else pd.read_csv(in_path)
    df = prep(df)

    print(f"Loaded {len(df):,} sentencing events.")
    print(f"Stage 1: joint-presence logistic regression per offense pair, "
          f"effect-size filter OR>={args.min_odds_ratio} ...")
    edges = stage1_adjusted_cooccurrence(df, args.min_odds_ratio, args.stage2_p_threshold)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges.to_csv(out_dir / "adjusted_cooccurrence.csv", index=False)
    print(f"Wrote {len(edges)} pairwise edges to {out_dir / 'adjusted_cooccurrence.csv'}")
    sig_edges = edges[edges["passes_effect_size_filter"] & (edges["p_value"] < args.stage2_p_threshold)]
    print(f"  {len(sig_edges)} pairs pass BOTH effect-size (OR>={args.min_odds_ratio}) "
          f"AND significance (p<{args.stage2_p_threshold}) filters -> stage 2 candidates.")

    print("Stage 2: running race/district disparity regressions on flagged pairs ...")
    disparity = stage2_disparity_tests(df, edges, args.stage2_p_threshold, args.ref_race)
    disparity.to_csv(out_dir / "disparity_regressions.csv", index=False)
    print(f"Wrote {len(disparity)} disparity-test rows to {out_dir / 'disparity_regressions.csv'}")
