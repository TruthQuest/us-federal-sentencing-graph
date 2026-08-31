#!/usr/bin/env python3
"""SHACL-validate the built graph against ussc_shapes.ttl."""
import sys
from pathlib import Path
from pyshacl import validate
from rdflib import Graph

data_g = Graph()
data_g.parse("ontology/ussc_tbox.ttl", format="turtle")
data_g.parse("ontology/ussc_offense_vocabulary.ttl", format="turtle")
data_g.parse("data/results_abox.ttl", format="turtle")

shapes_g = Graph()
shapes_g.parse("shapes/ussc_shapes.ttl", format="turtle")

conforms, report_g, report_text = validate(
    data_g, shacl_graph=shapes_g,
    inference="rdfs", abort_on_first=False, meta_shacl=False,
    advanced=True, js=False, debug=False,
)
print(f"Conforms: {conforms}")
print(report_text)
sys.exit(0 if conforms else 1)
