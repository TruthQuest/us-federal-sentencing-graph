#!/usr/bin/env python3
"""
Materializes RDF only for the things worth querying as a graph: offense
concepts (already in ussc_offense_vocabulary.ttl) and AnalysisResult
instances (adjusted co-occurrence edges + disparity regressions from step
03). Individual SentencingEvent instances are NOT materialized at full
scale here deliberately -- at 1991-present volume (millions of rows) a
flat-file RDF store is the wrong tool; that layer should live in the
pandas/Parquet frame the scripts already operate on, with only the
derived, citable claims promoted to RDF. A `--sample-events N` flag is
provided for building a small illustrative SentencingEvent sample (e.g.
for demos or SHACL-shape testing), not for production use.

Run 03_compute_adjusted_cooccurrence.py before this script; it consumes
results/adjusted_cooccurrence.csv and results/disparity_regressions.csv.
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS, XSD
from rdflib.namespace import PROV

ONTO = Namespace("https://traceoriginresearch.com/ussc-charge-graph/onto#")
OFF = Namespace("https://traceoriginresearch.com/ussc-charge-graph/offense#")
RES = Namespace("https://traceoriginresearch.com/ussc-charge-graph/result#")
EVT = Namespace("https://traceoriginresearch.com/ussc-charge-graph/event#")


def stable_id(*parts) -> str:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]
    return h


def dec(x) -> Literal:
    """Force xsd:decimal typing. rdflib coerces float() -> xsd:double by
    default, which trips sh:datatype xsd:decimal in the SHACL shapes.
    Serializing via str() preserves numeric value while binding the correct
    datatype."""
    return Literal(str(float(x)), datatype=XSD.decimal)


def add_cooccurrence_edges(g: Graph, edges: pd.DataFrame, script_id: str):
    for _, row in edges.iterrows():
        rid = RES[f"cooccur-{stable_id(row['offense_a'], row['offense_b'])}"]
        g.add((rid, RDF.type, ONTO.AnalysisResult))
        # Type-tag based on how the edge would be used downstream. The
        # SAME numerical result is emitted as BOTH a RegimeEdgeCandidate
        # (with the strict pop-level filter marker) IF it passes those
        # filters, AND as a DisparityTestCandidate (with the sample-size-
        # only marker) IF it has adequate joint sample. This preserves
        # the traceability that any given edge, when used downstream, was
        # tagged for the appropriate purpose. SHACL will reject
        # DisparityTestCandidates that carry stage-1 pre-filter markers.
        passes_pop_level = bool(row.get("passes_effect_size_filter", False)) \
                            and float(row.get("p_value", 1.0)) < 0.001
        if passes_pop_level:
            g.add((rid, RDF.type, ONTO.RegimeEdgeCandidate))
            g.add((rid, ONTO.candidateFilterApplied, Literal("effect_size_and_significance")))
        if int(row.get("n_joint_cases", 0)) >= 100:
            did = RES[f"disp-cand-{stable_id(row['offense_a'], row['offense_b'])}"]
            g.add((did, RDF.type, ONTO.AnalysisResult))
            g.add((did, RDF.type, ONTO.DisparityTestCandidate))
            g.add((did, ONTO.candidateFilterApplied, Literal("joint_sample_size_only")))
            g.add((did, RDFS.label, Literal(
                f"Disparity test candidate: {row['offense_a']} + {row['offense_b']} "
                f"(n_joint={int(row['n_joint_cases'])})"
            )))
            g.add((did, ONTO.pValue, dec(row["p_value"])))
            g.add((did, ONTO.covariateSet, Literal(row["covariate_set"])))
            g.add((did, ONTO.nullModel, Literal(row["null_model"])))
            g.add((did, ONTO.wasDerivedFrom, Literal(script_id)))
            # coOccursWith on the DisparityTestCandidate too, so SPARQL
            # queries can join on offense pairs across BOTH candidate types
            # (needed for Query 2: "which pairs are in one filter set but
            # not the other").
            g.add((did, ONTO.coOccursWith, OFF[row["offense_a"]]))
            g.add((did, ONTO.coOccursWith, OFF[row["offense_b"]]))
        g.add((rid, RDFS.label, Literal(
            f"Adjusted co-occurrence: {row['offense_a']} <-> {row['offense_b']}"
        )))
        g.add((rid, ONTO.adjustedCoOccurrenceWeight,
               dec(row["adjusted_cooccurrence_weight"])))
        g.add((rid, ONTO.pValue, dec(row["p_value"])))
        g.add((rid, ONTO.covariateSet, Literal(row["covariate_set"])))
        g.add((rid, ONTO.nullModel, Literal(row["null_model"])))
        g.add((rid, ONTO.wasDerivedFrom, Literal(script_id)))
        g.add((rid, ONTO.coOccursWith, OFF[row["offense_a"]]))
        g.add((rid, ONTO.coOccursWith, OFF[row["offense_b"]]))


def add_disparity_results(g: Graph, disparity: pd.DataFrame, script_id: str):
    for _, row in disparity.iterrows():
        rid = RES[f"disparity-{stable_id(row['primary_offense'], row['stacked_offense'], row['covariate_term'])}"]
        g.add((rid, RDF.type, ONTO.AnalysisResult))
        g.add((rid, RDF.type, ONTO.DisparityHypothesis))
        g.add((rid, RDFS.label, Literal(
            f"Disparity test: {row['primary_offense']} + {row['stacked_offense']} stacking, "
            f"term={row['covariate_term']}"
        )))
        g.add((rid, ONTO.pValue, dec(row["p_value"])))
        g.add((rid, ONTO.covariateSet, Literal(row["covariate_set"])))
        g.add((rid, ONTO.nullModel, Literal(row["null_model"])))
        g.add((rid, ONTO.wasDerivedFrom, Literal(script_id)))
        g.add((rid, RDFS.comment, Literal(
            f"odds_ratio={row['odds_ratio']}, n_cases={row['n_cases']}"
        )))


def add_regimes(g: Graph, communities: pd.DataFrame, epoch_label: str, stability: float, script_id: str):
    for cid, group in communities.groupby("community_id"):
        rid = RES[f"regime-{epoch_label}-{int(cid)}"]
        g.add((rid, RDF.type, ONTO.ChargingRegime))
        g.add((rid, RDFS.label, Literal(f"Regime {int(cid)} ({epoch_label}): "
                                          f"{', '.join(group['offense_category'])}")))
        g.add((rid, ONTO.regimeStabilityScore, dec(stability)))
        g.add((rid, ONTO.covariateSet, Literal("severity + venue + fiscal_year_decade")))
        g.add((rid, ONTO.nullModel, Literal("Louvain on residualized co-occurrence graph")))
        g.add((rid, ONTO.pValue, dec(0.0)))
        g.add((rid, ONTO.wasDerivedFrom, Literal(script_id)))
        for offense in group["offense_category"]:
            g.add((OFF[offense], ONTO.hasRegimeMembership, rid))


def add_epochs(g: Graph):
    epochs = [
        ("Pre-Booker", 1991, 2004, "Sentencing Reform Act baseline"),
        ("Post-Booker", 2005, 2009, "US v. Booker (2005): guidelines advisory"),
        ("Post-FSA", 2010, 2013, "Fair Sentencing Act (2010)"),
        ("Post-A782", 2014, 2018, "Amendment 782 (2014)"),
        ("Post-FirstStep", 2019, 2100, "First Step Act (2018)"),
    ]
    for name, start, end, shock in epochs:
        eid = RES[f"epoch-{name}"]
        g.add((eid, RDF.type, ONTO.PolicyEpoch))
        g.add((eid, RDFS.label, Literal(name)))
        g.add((eid, ONTO.epochStartYear, Literal(str(start), datatype=XSD.gYear)))
        if end < 2100:
            g.add((eid, ONTO.epochEndYear, Literal(str(end), datatype=XSD.gYear)))
        g.add((eid, ONTO.epochShockEvent, Literal(shock)))


def add_judge_signatures(g: Graph, sigs: pd.DataFrame, script_id: str):
    for _, row in sigs.iterrows():
        sid = RES[f"judgesig-{stable_id(row['judge_id'])}"]
        g.add((sid, RDF.type, ONTO.JudgeSignature))
        g.add((sid, RDFS.label, Literal(f"Signature for judge {row['judge_id']}")))
        g.add((sid, ONTO.regimeShare, Literal(row["regime_share"])))
        g.add((sid, ONTO.signatureFor, EVT[f"judge-{row['judge_id']}"]))
        if pd.notna(row.get("cross_venue_signature_stability")):
            g.add((sid, ONTO.signatureStabilityAcrossVenue,
                   dec(row["cross_venue_signature_stability"])))
        g.add((sid, ONTO.covariateSet, Literal("regime shares across venues")))
        g.add((sid, ONTO.nullModel, Literal("cosine similarity of regime-share vectors")))
        g.add((sid, ONTO.pValue, dec(0.0)))
        g.add((sid, ONTO.wasDerivedFrom, Literal(script_id)))


def add_community_disparity(g: Graph, disparity: pd.DataFrame, script_id: str,
                              regime_uri_template: str = "regime-Full-{}"):
    """Materialize community-level disparity claims from script 06.
    One AnalysisResult per (regime, specification, demographic term) triple.
    Links each claim to the corresponding ChargingRegime via :aboutRegime,
    so SPARQL can join regime members + disparity claims in one query."""
    for _, row in disparity.iterrows():
        rid = RES[f"commdisp-{stable_id(row['regime_id'], row['specification'], row['covariate_term'])}"]
        g.add((rid, RDF.type, ONTO.AnalysisResult))
        g.add((rid, RDF.type, ONTO.CommunityDisparityClaim))
        g.add((rid, RDF.type, ONTO.DisparityHypothesis))
        g.add((rid, RDFS.label, Literal(
            f"Community disparity: regime {int(row['regime_id'])}, "
            f"{row['specification']}, {row['covariate_term']}"
        )))
        # Link to the ChargingRegime instance created by add_regimes()
        regime_uri = RES[regime_uri_template.format(int(row["regime_id"]))]
        g.add((rid, ONTO.aboutRegime, regime_uri))
        g.add((rid, ONTO.specification, Literal(row["specification"])))
        g.add((rid, ONTO.aboutDemographicGroup, Literal(row["covariate_term"])))
        g.add((rid, ONTO.oddsRatio, dec(row["odds_ratio"])))
        g.add((rid, ONTO.pValue, dec(row["p_value"])))
        g.add((rid, ONTO.covariateSet, Literal(row["covariate_set"])))
        g.add((rid, ONTO.nullModel, Literal(row["null_model"])))
        g.add((rid, ONTO.wasDerivedFrom, Literal(script_id)))


def add_divergence_disparity(g: Graph, disparity: pd.DataFrame, script_id: str):
    """Materialize counterfactual-divergence disparity claims from script 08.
    One AnalysisResult per demographic term. These are NOT regime-specific
    (divergence is defined across the whole event set)."""
    for _, row in disparity.iterrows():
        rid = RES[f"divdisp-{stable_id(row['covariate_term'])}"]
        g.add((rid, RDF.type, ONTO.AnalysisResult))
        g.add((rid, RDF.type, ONTO.DivergenceDisparityClaim))
        g.add((rid, RDF.type, ONTO.DisparityHypothesis))
        g.add((rid, RDFS.label, Literal(
            f"Divergence disparity: {row['covariate_term']}"
        )))
        g.add((rid, ONTO.aboutDemographicGroup, Literal(row["covariate_term"])))
        g.add((rid, ONTO.oddsRatio, dec(row["odds_ratio"])))
        g.add((rid, ONTO.pValue, dec(row["p_value"])))
        g.add((rid, ONTO.covariateSet, Literal(row["covariate_set"])))
        g.add((rid, ONTO.nullModel, Literal(row["null_model"])))
        g.add((rid, ONTO.wasDerivedFrom, Literal(script_id)))


def add_sample_events(g: Graph, df: pd.DataFrame, n: int):
    # Only sample from rows with complete severity fields; SHACL will
    # reject events lacking GuidelineComputation, so building them at all
    # is wasted work.
    complete = df.dropna(subset=["final_offense_level", "criminal_history_category",
                                   "guideline_min_months", "guideline_max_months",
                                   "sentence_months", "fiscal_year"])
    sample = complete.sample(n=min(n, len(complete)), random_state=42)
    for _, row in sample.iterrows():
        eid = EVT[f"event-{row['case_id']}"]
        g.add((eid, RDF.type, ONTO.SentencingEvent))
        g.add((eid, ONTO.fiscalYear, Literal(int(row["fiscal_year"]), datatype=XSD.gYear)))
        g.add((eid, ONTO.hasOffenseCode, OFF[row["primary_offense"]]))
        for sec in str(row.get("secondary_offenses", "") or "").split(";"):
            if sec:
                g.add((eid, ONTO.hasOffenseCode, OFF[sec]))
        g.add((eid, ONTO.sentenceMonths, dec(row["sentence_months"])))
        g.add((eid, ONTO.hasMandatoryMinimum, Literal(bool(row["mand_min_flag"]), datatype=XSD.boolean)))
        gc = EVT[f"gc-{row['case_id']}"]
        g.add((gc, RDF.type, ONTO.GuidelineComputation))
        g.add((gc, ONTO.finalOffenseLevel, Literal(int(row["final_offense_level"]), datatype=XSD.integer)))
        g.add((gc, ONTO.criminalHistoryCategory, Literal(int(row["criminal_history_category"]), datatype=XSD.integer)))
        g.add((gc, ONTO.guidelineMinMonths, dec(row["guideline_min_months"])))
        g.add((gc, ONTO.guidelineMaxMonths, dec(row["guideline_max_months"])))
        g.add((eid, ONTO.hasGuidelineComputation, gc))
        district = EVT[f"district-{row['district']}"]
        g.add((eid, ONTO.inDistrict, district))
        g.add((district, RDF.type, ONTO.District))
        if pd.notna(row.get("judge_id")):
            judge = EVT[f"judge-{row['judge_id']}"]
            g.add((eid, ONTO.presidingJudge, judge))
            g.add((judge, RDF.type, ONTO.Judge))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edges", default="results/adjusted_cooccurrence.csv")
    ap.add_argument("--disparity", default="results/disparity_regressions.csv")
    ap.add_argument("--events", default=None,
                     help="Optional path to sentencing_events parquet/csv for --sample-events.")
    ap.add_argument("--sample-events", type=int, default=0)
    ap.add_argument("--out", default="data/results_abox.ttl")
    args = ap.parse_args()

    g = Graph()
    g.bind("", ONTO)
    g.bind("off", OFF)
    g.bind("res", RES)
    g.bind("evt", EVT)
    g.bind("prov", PROV)

    edges = pd.read_csv(args.edges)
    add_cooccurrence_edges(g, edges, script_id="scripts/03_compute_adjusted_cooccurrence.py:stage1")

    disparity_path = Path(args.disparity)
    if disparity_path.exists() and disparity_path.stat().st_size > 0:
        disparity = pd.read_csv(disparity_path)
        if not disparity.empty:
            add_disparity_results(g, disparity, script_id="scripts/03_compute_adjusted_cooccurrence.py:stage2")

    add_epochs(g)

    comm_path = Path("results/offense_communities.csv")
    if comm_path.exists():
        communities = pd.read_csv(comm_path)
        add_regimes(g, communities, epoch_label="Full", stability=0.0,
                    script_id="scripts/04_community_detection.py")

    sigs_path = Path("results/judge_signatures.csv")
    if sigs_path.exists():
        sigs = pd.read_csv(sigs_path)
        add_judge_signatures(g, sigs, script_id="scripts/09_judge_signatures.py")

    cd_path = Path("results/community_disparity.csv")
    if cd_path.exists() and cd_path.stat().st_size > 0:
        cd = pd.read_csv(cd_path)
        if not cd.empty:
            add_community_disparity(g, cd, script_id="scripts/06_community_disparity.py")
            print(f"  materialized {len(cd)} community disparity claims")

    dd_path = Path("results/divergence_disparity.csv")
    if dd_path.exists() and dd_path.stat().st_size > 0:
        dd = pd.read_csv(dd_path)
        if not dd.empty:
            add_divergence_disparity(g, dd, script_id="scripts/08_counterfactual_regime.py")
            print(f"  materialized {len(dd)} divergence disparity claims")

    if args.events and args.sample_events > 0:
        in_path = Path(args.events)
        events_df = pd.read_parquet(in_path) if in_path.suffix == ".parquet" else pd.read_csv(in_path)
        add_sample_events(g, events_df, args.sample_events)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out_path), format="turtle")
    print(f"Wrote {len(g)} triples to {out_path}")
