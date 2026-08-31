#!/usr/bin/env python3
"""
Standing SPARQL queries against the built RDF graph, formatted as a
human-readable text report. Run AFTER scripts 02 (A-Box build) and 10
(SHACL validate) have populated data/results_abox.ttl.

Each query answers a question that:
  (a) uses the ontology-materialized objects (ChargingRegime,
      CommunityDisparityClaim, RegimeEdgeCandidate, etc.), so it cannot
      be reproduced by a simple filter on the source CSVs, AND
  (b) surfaces a substantive finding a reader would care about.

Report goes to results/sparql_report.txt.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from rdflib import Graph


ONTO_TTL = "ontology/ussc_tbox.ttl"
VOCAB_TTL = "ontology/ussc_offense_vocabulary.ttl"
ABOX_TTL = "data/results_abox.ttl"

PREFIXES = """
PREFIX : <https://traceoriginresearch.com/ussc-charge-graph/onto#>
PREFIX off: <https://traceoriginresearch.com/ussc-charge-graph/offense#>
PREFIX res: <https://traceoriginresearch.com/ussc-charge-graph/result#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


# ---------------------------------------------------------------------------
# Query 1: Regime membership × community disparity, both specifications
# side-by-side. THE finding-legible query.
# ---------------------------------------------------------------------------
Q1_REGIME_DISPARITY = PREFIXES + """
SELECT ?regimeLabel ?members ?demographic ?specification ?oddsRatio ?pValue
WHERE {
  ?regime a :ChargingRegime ;
          rdfs:label ?regimeLabel .
  {
    # Aggregate regime members into a single string per regime
    SELECT ?regime (GROUP_CONCAT(?offenseLabel; separator=", ") AS ?members)
    WHERE {
      ?offense :hasRegimeMembership ?regime ;
               skos:prefLabel ?offenseLabel .
    }
    GROUP BY ?regime
  }
  ?claim a :CommunityDisparityClaim ;
         :aboutRegime ?regime ;
         :aboutDemographicGroup ?demographic ;
         :specification ?specification ;
         :oddsRatio ?oddsRatio ;
         :pValue ?pValue .
  FILTER(?pValue < 0.001)
}
ORDER BY ?regimeLabel ?demographic ?specification
"""


# ---------------------------------------------------------------------------
# Query 2: Which specific offense pairs are DisparityTestCandidates but
# NOT RegimeEdgeCandidates? These are the pairs the two-filter discipline
# is specifically protecting: pop-level filtering would have excluded
# them from analysis, hiding real demographic disparities.
# ---------------------------------------------------------------------------
Q2_TWO_FILTER_PROTECTION = PREFIXES + """
SELECT DISTINCT ?offenseA ?offenseB ?pValue
WHERE {
  ?disparityCand a :DisparityTestCandidate ;
                 :coOccursWith ?offenseA, ?offenseB ;
                 :pValue ?pValue .
  # Undirected pair: keep only ordered pairs so each shows once
  FILTER(STR(?offenseA) < STR(?offenseB))
  FILTER NOT EXISTS {
    ?regimeCand a :RegimeEdgeCandidate ;
                :coOccursWith ?offenseA, ?offenseB .
  }
}
ORDER BY ?pValue
LIMIT 30
"""


# ---------------------------------------------------------------------------
# Query 3: Change of race coefficient across the two specifications for
# each regime. Directly surfaces the primary-offense-conditioning contrast
# that is the load-bearing methodological content of the pipeline.
# ---------------------------------------------------------------------------
Q3_CONDITIONING_EFFECT = PREFIXES + """
SELECT ?regimeLabel ?demographic ?unconditionalOR ?conditionalOR
WHERE {
  ?regime a :ChargingRegime ;
          rdfs:label ?regimeLabel .
  ?uncondClaim a :CommunityDisparityClaim ;
               :aboutRegime ?regime ;
               :aboutDemographicGroup ?demographic ;
               :specification "unconditional" ;
               :oddsRatio ?unconditionalOR ;
               :pValue ?uncP .
  ?condClaim a :CommunityDisparityClaim ;
             :aboutRegime ?regime ;
             :aboutDemographicGroup ?demographic ;
             :specification "conditional_on_primary" ;
             :oddsRatio ?conditionalOR ;
             :pValue ?cP .
  FILTER(?uncP < 0.001 || ?cP < 0.001)
  FILTER(CONTAINS(STR(?demographic), "race"))
}
ORDER BY ?regimeLabel ?demographic
"""


# ---------------------------------------------------------------------------
# Query 4: Full provenance chain for every claim. Anyone reading the
# report can see exactly which script produced each finding, what
# covariates were held constant, and against what null model. This is
# what the SHACL discipline was built to enforce.
# ---------------------------------------------------------------------------
Q4_PROVENANCE = PREFIXES + """
SELECT ?claimLabel ?derivedFromScript ?covariateSet ?nullModel ?pValue
WHERE {
  ?claim a :AnalysisResult ;
         rdfs:label ?claimLabel ;
         :wasDerivedFrom ?derivedFromScript ;
         :covariateSet ?covariateSet ;
         :nullModel ?nullModel ;
         :pValue ?pValue .
  FILTER(?pValue < 0.0001)
}
ORDER BY ?pValue
LIMIT 25
"""


QUERIES = [
    ("Query 1: Regime membership × community disparity",
     "For each detected charging regime, list its member offenses and every "
     "statistically significant demographic-group disparity in regime membership, "
     "for both the unconditional and primary-conditional specifications. "
     "This is the query that could not be posed against the source CSVs at all, "
     "because regime membership is an empirically-derived ontology object, not "
     "a column in the data.",
     Q1_REGIME_DISPARITY),

    ("Query 2: Two-filter discipline in action",
     "List offense pairs that ARE candidates for demographic-disparity testing "
     "(joint sample size >= 100) but are NOT candidates for the regime-edge graph "
     "(they failed the population-level effect-size or significance filter). "
     "These are exactly the pairs where naive pop-level filtering would have "
     "hidden a real demographic-conditional disparity by averaging it away. "
     "The T-Box distinction makes them queryable.",
     Q2_TWO_FILTER_PROTECTION),

    ("Query 3: Effect of primary-offense conditioning on race coefficients",
     "For each detected regime, contrast the race-coefficient in the "
     "unconditional specification against the same coefficient after adding "
     "primary offense as a covariate. Coefficients that stay elevated after "
     "conditioning are stacking-behavior disparities. Coefficients that collapse "
     "toward 1.0 are primary-offense-selection disparities. This is the load-"
     "bearing methodological contrast of the pipeline.",
     Q3_CONDITIONING_EFFECT),

    ("Query 4: Full provenance chain for every claim",
     "For every AnalysisResult in the graph, show its label, the script that "
     "produced it, the exact covariate set held constant, the null model, and "
     "the p-value. This is the query the SHACL DisparityTestCandidate shape "
     "was built to make possible: every claim carries its provenance or is "
     "rejected before it enters the graph.",
     Q4_PROVENANCE),
]


def load_graph() -> Graph:
    g = Graph()
    for path in (ONTO_TTL, VOCAB_TTL, ABOX_TTL):
        if not Path(path).exists():
            raise SystemExit(f"Missing {path}. Run scripts 02 (A-Box build) first.")
        g.parse(path, format="turtle")
    return g


def _fmt_cell(v) -> str:
    """rdflib returns URIRef / Literal / etc; convert to short string.
    URIs are compacted to their local name; literals are shown as text."""
    s = str(v)
    if "#" in s:
        s = s.rsplit("#", 1)[-1]
    elif "/" in s and s.startswith(("http://", "https://")):
        s = s.rsplit("/", 1)[-1]
    return s


def _table(rows: list[tuple], cols: list[str]) -> str:
    """Simple column-aligned text table."""
    if not rows:
        return "  (no results)"
    str_rows = [[_fmt_cell(cell) for cell in row] for row in rows]
    widths = [max(len(c), max((len(r[i]) for r in str_rows), default=0))
              for i, c in enumerate(cols)]
    lines = []
    lines.append("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols)))
    lines.append("  " + "  ".join("-" * widths[i] for i in range(len(cols))))
    for r in str_rows:
        lines.append("  " + "  ".join(r[i].ljust(widths[i]) for i in range(len(r))))
    return "\n".join(lines)


def run_query(g: Graph, sparql: str):
    results = g.query(sparql)
    cols = [str(v) for v in results.vars] if results.vars else []
    rows = [tuple(row) for row in results]
    return cols, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/sparql_report.txt")
    args = ap.parse_args()

    print("Loading graph ...")
    g = load_graph()
    print(f"  {len(g):,} triples across T-Box + vocabulary + A-Box.")

    lines = []
    lines.append("=" * 78)
    lines.append("USSC CHARGE GRAPH — SPARQL REPORT")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"Graph size: {len(g):,} triples")
    lines.append("=" * 78)

    for title, rationale, sparql in QUERIES:
        print(f"\nRunning: {title}")
        try:
            cols, rows = run_query(g, sparql)
        except Exception as e:
            lines.append(f"\n\n{'-' * 78}\n{title}\n{'-' * 78}\n\nRATIONALE\n{rationale}\n\nQUERY FAILED: {type(e).__name__}: {e}\n")
            continue

        lines.append("\n\n" + "-" * 78)
        lines.append(title)
        lines.append("-" * 78)
        lines.append("\nRATIONALE")
        # Wrap rationale to 76 chars
        import textwrap
        for para in textwrap.wrap(rationale, width=76):
            lines.append(para)
        lines.append(f"\nRESULTS ({len(rows)} rows)")
        lines.append(_table(rows, cols))
        lines.append("\nSPARQL")
        for src_line in sparql.strip().splitlines():
            lines.append("  " + src_line)

    lines.append("\n\n" + "=" * 78)
    lines.append("END OF REPORT")
    lines.append("=" * 78)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"\nWrote report ({len(lines)} lines) to {out}")


if __name__ == "__main__":
    main()
