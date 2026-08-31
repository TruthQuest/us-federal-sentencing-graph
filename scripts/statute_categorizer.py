"""
Statute-citation to offense-category mapper.

USSC NWSTAT1-18 fields hold up to 18 raw statutory citations per case,
concatenated as [title][section][subsection], e.g. '21841A1' = 21 USC
§841(a)(1), '81326B' = 8 USC §1326(b). This module maps each citation
to one of the offense categories in ontology/ussc_offense_vocabulary.ttl.

The mapping is based on the frequency-weighted top ~50 citations
observed in FY2014 and known USC organization. It is NOT the USSC's
own primary-statute -> offense-type crosswalk (which is undocumented
in the public codebook and varies by year); it is a defensible
heuristic that captures the top ~85% of citation volume correctly.

For citations not matched by any rule, category = OTHER. Every USSC
year should be spot-checked: the counts of OTHER-bucketed citations
per year should stay under ~15%, and if they rise, the mapper needs
new rules.
"""
import re
from typing import Optional


def parse_citation(cite: str) -> Optional[tuple[int, str]]:
    """'21841A1' -> (21, '841A1'). '81326B' -> (8, '1326B'). None if unparseable."""
    if not isinstance(cite, str) or not cite:
        return None
    # Title numbers in USSC NWSTAT are 1-2 digits without leading zero.
    # Order of alternatives matters: match TWO-digit titles first so that
    # "18922G1" parses as (18, "922G1") not (1, "8922G1"). For Title 8
    # citations like "81326A", we need to know Title 8 exists; the
    # explicit alternation forces the parser to try known titles.
    m = re.match(r"^(50|46|42|31|29|26|22|21|18|16|15|8)(\d+[A-Z0-9]*)$", cite.strip().upper())
    if not m:
        return None
    return int(m.group(1)), m.group(2)


# Section-prefix rules per USC title, evaluated in order per title.
# Each rule: (section_prefix, category). First match wins per citation.
# Sections must start with the digit(s) shown to match.
_RULES: dict[int, list[tuple[str, str]]] = {
    8: [
        ("1326", "IMMIG"),
        ("1324", "IMMIG"),
        ("1325", "IMMIG"),
        ("1327", "IMMIG"),
        ("1328", "IMMIG"),
    ],
    18: [
        # Firearms (Title 18 Chapter 44)
        ("922", "FIREARM"),
        ("923", "FIREARM"),
        ("924", "FIREARM"),
        ("925", "FIREARM"),
        # Weapons of mass destruction / non-firearm weapons
        ("2332a", "WEAPON"),
        ("2332b", "WEAPON"),
        # Fraud
        ("1028", "FRAUD"),     # identification fraud
        ("1029", "FRAUD"),     # access-device fraud
        ("1030", "FRAUD"),     # computer fraud
        ("1341", "FRAUD"),     # mail fraud
        ("1343", "FRAUD"),     # wire fraud
        ("1344", "FRAUD"),     # bank fraud
        ("1347", "FRAUD"),     # healthcare fraud
        ("1348", "FRAUD"),     # securities fraud
        ("1349", "FRAUD"),     # attempt & conspiracy to commit fraud (Title 18)
        ("471", "FRAUD"),      # counterfeit obligations
        ("472", "FRAUD"),      # counterfeit obligations, passing
        ("286", "FRAUD"),      # false claims conspiracy
        ("287", "FRAUD"),      # false claims
        ("641", "FRAUD"),      # theft of public property/money
        ("1014", "FRAUD"),     # false statement to lending institution
        ("1708", "FRAUD"),     # mail theft
        ("1709", "FRAUD"),     # mail theft by employee
        # Money laundering
        ("1956", "LAUNDER"),
        ("1957", "LAUNDER"),
        # RICO / racketeering / extortion
        ("1951", "RACKET"),
        ("1952", "RACKET"),
        ("1959", "RACKET"),
        ("1961", "RACKET"),
        ("1962", "RACKET"),
        # Violent / robbery / assault / murder
        ("2113", "VIOLENT"),   # bank robbery
        ("2114", "VIOLENT"),   # postal robbery
        ("2119", "VIOLENT"),   # carjacking
        ("1111", "VIOLENT"),   # murder (put before "111" so it matches first)
        ("1112", "VIOLENT"),
        ("111",  "VIOLENT"),   # assault on federal officer
        ("113",  "VIOLENT"),   # assault
        ("115",  "VIOLENT"),
        ("117",  "VIOLENT"),
        # Sex offenses / child pornography / SORNA
        ("2250", "SEX"),       # SORNA failure to register
        ("2251", "SEX"),
        ("2252", "SEX"),
        ("2253", "SEX"),
        ("2260", "SEX"),
        ("2421", "SEX"),
        ("2422", "SEX"),
        ("2423", "SEX"),
        # Immigration / passport / border documents (Title 18, not Title 8)
        ("1542", "IMMIG"),     # false statement in passport application
        ("1544", "IMMIG"),     # misuse of passport
        ("1546", "IMMIG"),     # fraud in immigration documents
        # Administration of justice (obstruction, perjury, contempt,
        # false statements, failure to appear, escape). The primary
        # federal stacking category.
        ("1001", "ADMIN_JUST"),
        ("1503", "ADMIN_JUST"),
        ("1505", "ADMIN_JUST"),
        ("1510", "ADMIN_JUST"),
        ("1512", "ADMIN_JUST"),
        ("1621", "ADMIN_JUST"),
        ("1622", "ADMIN_JUST"),
        ("1623", "ADMIN_JUST"),
        ("3146", "ADMIN_JUST"),
        ("751", "ADMIN_JUST"),   # escape from custody
        ("752", "ADMIN_JUST"),
        ("1791", "ADMIN_JUST"),  # contraband in prison
        # National security / terrorism
        ("2339", "NATL_DEF"),
        ("2381", "NATL_DEF"),
        ("2382", "NATL_DEF"),
        ("2384", "NATL_DEF"),
        # Bribery / corruption
        ("201",  "BRIBERY"),
        ("666",  "BRIBERY"),
        # Conspiracy (frequently a stacking charge, treated as its own
        # category rather than folded into the primary — this preserves
        # the analytic signal that 'primary + 18 USC 371' is a distinct
        # stacking pattern from primary alone)
        ("371",  "CONSPIRACY"),
    ],
    21: [
        # Drug trafficking / manufacture / distribution
        ("841", "DRUG"),
        ("843", "DRUG"),      # drug abuse prevention & control
        ("846", "DRUG"),      # drug conspiracy — kept in DRUG (not CONSPIRACY)
                                # because it IS the drug-conspiracy statute
                                # rather than a stacked general-conspiracy charge
        ("848", "DRUG"),      # continuing criminal enterprise
        ("851", "DRUG"),      # sentencing enhancement (prior drug conviction)
        ("856", "DRUG"),      # maintaining drug-involved premises
        ("860", "DRUG"),      # drug distribution near schools/playgrounds
        ("952", "DRUG"),      # import
        ("959", "DRUG"),
        ("960", "DRUG"),
        ("963", "DRUG"),
        # Simple possession
        ("844", "DRUG_POSS"),
    ],
    26: [
        # Tax
        ("7201", "TAX"),
        ("7202", "TAX"),
        ("7203", "TAX"),
        ("7206", "TAX"),
        ("7212", "TAX"),
        # NFA firearms (Title 26 Ch 53 — Machine guns, silencers, etc.)
        ("5861", "FIREARM"),
        ("5871", "FIREARM"),
    ],
    31: [
        # Bank Secrecy Act / money laundering (financial reporting)
        ("5322", "LAUNDER"),
        ("5324", "LAUNDER"),
        ("5331", "LAUNDER"),
    ],
    42: [
        # SORNA (sex offender registration) & SSA fraud
        ("14135", "SEX"),
        ("14072", "SEX"),
        ("408", "FRAUD"),      # SSA fraud
    ],
    46: [
        # Maritime drug trafficking
        ("70503", "DRUG"),
        ("70506", "DRUG"),
    ],
    50: [
        ("1705", "NATL_DEF"),   # IEEPA violations
    ],
}


# Statutes that are NOT substantive offenses but theory-of-liability tags
# (aiding-and-abetting, maritime jurisdiction, misprision, assimilative
# crimes). These NEVER stand alone as the offense being punished and, if
# treated as a stacked category, inflate co-occurrence counts falsely. The
# categorizer returns SKIP for these and the ETL layer drops them from the
# secondary_offenses list.
_SKIP_STATUTES = {
    (18, "2"),          # 18 USC §2, aiding and abetting
    (18, "3"),          # 18 USC §3, accessory after the fact
    (18, "4"),          # 18 USC §4, misprision of felony
    (18, "7"),          # 18 USC §7, maritime and territorial jurisdiction
    (18, "13"),         # 18 USC §13, assimilative crimes (state law adopted)
}


def categorize(cite: str) -> str:
    """Map one NWSTAT citation to an offense category label matching
    the SKOS vocabulary in ontology/ussc_offense_vocabulary.ttl."""
    parsed = parse_citation(cite)
    if parsed is None:
        return "OTHER"
    title, section = parsed
    if (title, section) in _SKIP_STATUTES:
        return "SKIP"
    rules = _RULES.get(title, [])
    for prefix, category in rules:
        if section.startswith(prefix):
            return category
    return "OTHER"


def categorize_all(cites: list[str]) -> list[str]:
    """Vectorized-style batch categorization. Preserves order, drops empty."""
    return [categorize(c) for c in cites if c]
