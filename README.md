# us-federal-sentencing-graph

A formal ontology and reproducible analysis pipeline over twelve years
of U.S. federal sentencing records (FY2014-FY2025, 805,875 individual
sentencing events) that surfaces race-patterned structure in
prosecutorial charge stacking.

The finding: federal charge stacking clusters into two stable regimes
(a white-collar package of fraud + laundering + administration-of-
justice, and a gun/RICO package of firearms + racketeering). Which
regime a defendant lands in is race-patterned even after controlling
for offense severity, criminal history, mandatory-minimum
applicability, all 94 federal district fixed effects, AND primary
offense. Full writeup:
https://www.traceoriginresearch.com/what-the-charge-records-show/

## What is in this repository

```
us-federal-sentencing-graph/
├── ontology/
│   ├── ussc_tbox.ttl                    OWL 2 T-Box: ChargingRegime,
│   │                                     DisparityHypothesis,
│   │                                     RegimeEdgeCandidate,
│   │                                     DisparityTestCandidate,
│   │                                     CommunityDisparityClaim, etc.
│   └── ussc_offense_vocabulary.ttl      SKOS vocabulary of federal
│                                         offense categories
├── shapes/
│   └── ussc_shapes.ttl                  SHACL: enforces two-filter
│                                         discipline and required
│                                         provenance on every claim
├── scripts/
│   ├── statute_categorizer.py           NWSTAT (statute-citation) to
│   │                                     offense-category mapper
│   ├── 00_concat_years.py               Per-year Parquet concat
│   ├── 01_load_ussc_data.py             S3-streaming ETL
│   ├── 02_build_abox.py                 Materialize results as RDF
│   ├── 03_compute_adjusted_cooccurrence.py
│   │                                     Joint-presence OR vs
│   │                                     conditional-independence
│   ├── 04_community_detection.py        Louvain + bootstrap stability
│   ├── 05_generate_synthetic_data.py    Planted-signal test data
│   ├── 06_community_disparity.py        Dual-specification demographic
│   │                                     regressions on regime
│   │                                     membership
│   ├── 07_temporal_regime_evolution.py  Per-epoch pipeline + alignment
│   ├── 08_counterfactual_regime.py      Divergence disparity: actual
│   │                                     vs severity+venue-predicted
│   │                                     regime
│   ├── 09_judge_signatures.py           Judge regime-application
│   │                                     vectors (no output on FY14+
│   │                                     due to identifier suppression)
│   ├── 10_validate_shapes.py            SHACL validator
│   └── 11_sparql_report.py              Standing SPARQL queries;
│                                         human-readable text report
├── data/
│   └── (sentencing_events.parquet, results_abox.ttl)
├── results/
│   └── (CSVs per script, plus sparql_report.txt)
├── run_all.sh                           End-to-end orchestrator
├── LICENSE                              Proprietary; see LICENSE file
├── KNOWN_ISSUES.md
└── README.md                            (this file)
```

## Reproducing the analysis

Requires Python 3.11+, ~8 GB RAM, and ~15 GB disk for the full corpus.
Depends on: pandas, pyarrow, statsmodels, networkx, python-louvain,
scikit-learn, rdflib, pyshacl, boto3.

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run the full pipeline against USSC data (streams from public S3).
#    Substitute --synthetic for scripts/05 output to test locally first.
bash run_all.sh

# 3. Read the generated report
cat results/sparql_report.txt
```

The SHACL validator (`scripts/10_validate_shapes.py`) must exit clean
before the SPARQL report is generated. If it does not, the T-Box has
drifted from the ontology's discipline and the pipeline should not be
trusted until the shape violations are addressed.

## Data

Source: **U.S. Sentencing Commission, Individual Offender Datafiles**,
FY2014 through FY2025. Public domain at source. Available from
https://www.ussc.gov/research/datafiles/commission-datafiles.

The A-Box materialized into `data/results_abox.ttl` is a derivative
work (statistical summaries, community memberships, disparity
coefficients with provenance). Individual defendants are represented
only by USSC-anonymized identifiers, never by name. Judge
identifiers are suppressed by USSC in FY2014-forward public files;
the judge-signature analysis (`scripts/09`) accordingly produces no
output on this corpus.

## What the pipeline does NOT do

- **No causal inference.** The pipeline detects structural patterns in
  charging outcomes conditional on standard controls. It does not
  establish that any prosecutor, office, or policy caused those
  patterns.

- **No claim of intent.** A regime is a statistical object, not a
  legal one. Nothing in this repository supports a claim that any
  particular defendant was charged discriminatorily, and no output
  should be cited for that purpose without independent case-specific
  legal analysis.

- **No original-charge analysis.** USSC data reflects conviction, not
  original charge. Plea bargaining strips substantial charging-level
  signal before it reaches the sentencing record. Every "charging
  regime" finding is shorthand for "regime observable at the
  conviction layer" and understates the true charging-level structure.

- **No sentence-length regression.** This pipeline tests regime
  membership as the outcome. Sentence-length disparity is a separate
  and well-established literature; this work is orthogonal to it.

See `KNOWN_ISSUES.md` for methodological caveats not covered above.

## Citation

If you cite this repository in academic work, policy submissions, or
journalism:

> Brattin, E. (2026). us-federal-sentencing-graph: A formal ontology
> and analysis of race-patterned charge stacking in the federal
> criminal system, FY2014-2025. Trace Origin LLC.
> https://github.com/TruthQuest/us-federal-sentencing-graph

Please also separately cite the underlying data:

> U.S. Sentencing Commission (2014-2025). Individual Offender
> Datafiles, FY2014 through FY2025.
> https://www.ussc.gov/research/datafiles/commission-datafiles

## License

Proprietary. See `LICENSE` for the full terms. Non-commercial academic
citation, journalistic quotation, and inspection for reproducibility
are permitted with attribution. Commercial use, redistribution,
derivative works, and use as ML training data require prior written
permission.

## Contact

  Trace Origin LLC
  ebrattin@traceoriginresearch.com
  https://www.traceoriginresearch.com
