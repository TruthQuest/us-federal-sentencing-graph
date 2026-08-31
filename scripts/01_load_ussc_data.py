#!/usr/bin/env python3
"""
Streaming ETL for USSC individual-offender CSV datafiles.

Reads either from local disk or DIRECTLY FROM S3 (no local download of
the raw file needed) using chunked pandas reads with column selection.

Real USSC per-year files are ~2 GB uncompressed with 25,001 columns, of
which ~25 are analytic. Column selection at read time reduces the working
set from ~65 GB (33 years) to ~500 MB total in Parquet, so the whole
corpus fits comfortably in-memory on a laptop after ETL.

Usage:

    # Stream one file from S3, write per-year Parquet:
    python 01_load_ussc_data.py \\
        --s3 s3://us-federal-sentencing-graph/sentencing-data/opafy14nid.csv \\
        --fiscal-year 2014 \\
        --out data/events/fy2014.parquet

    # Stream ALL files matching a prefix from S3 (one Parquet per year):
    python 01_load_ussc_data.py \\
        --s3-prefix s3://us-federal-sentencing-graph/sentencing-data/ \\
        --out-dir data/events/

    # Local file (backwards compatible):
    python 01_load_ussc_data.py --local opafy14/opafy14nid.csv --fiscal-year 2014 \\
        --out data/events/fy2014.parquet

    # Synthetic (unchanged from earlier scaffold):
    python 01_load_ussc_data.py --synthetic data/synthetic_sentencing.csv \\
        --out data/sentencing_events.parquet
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from statute_categorizer import categorize as statute_to_category


# --- Real USSC column mapping (validated against FY2014 CSV) ---------------
# Column names verified against opafy14nid.csv header. Older/newer years
# may differ; the streaming loader tolerates missing columns (fills NaN)
# so schema-drift causes graceful degradation rather than crash.
USSC_COLUMNS = {
    "case_id": "USSCIDN",
    "primary_offense_code": "OFFTYPE2",   # USSC's own integer offense-type code
    "final_offense_level": "XFOLSOR",
    "criminal_history_category": "XCRHISSR",
    "guideline_min_months": "GLMIN",
    "guideline_max_months": "GLMAX",
    "sentence_months": "SENTTOT",
    "defendant_race": "NEWRACE",           # Preferred over MONRACE (newer coding)
    "defendant_sex": "MONSEX",
    "defendant_citizenship": "CITIZEN",
    "defendant_age": "AGE",
    "district": "DISTRICT",
    "circuit": "CIRCDIST",
    # judge intentionally absent from FY14+ public files (suppression);
    # loader will simply not populate judge_id, downstream handles NaN.
}
NWSTAT_COLS = [f"NWSTAT{i}" for i in range(1, 19)]      # 18 charge slots
MAND_COLS = [f"MAND{i}" for i in range(1, 7)]           # 6 mand-min flags

CANONICAL_COLUMNS = [
    "case_id", "fiscal_year", "district", "circuit", "judge_id",
    "defendant_race", "defendant_sex", "defendant_citizenship", "defendant_age",
    "primary_offense", "secondary_offenses",
    "final_offense_level", "criminal_history_category",
    "guideline_min_months", "guideline_max_months", "mand_min_flag",
    "sentence_months", "departure_direction",
    # Provenance
    "primary_offense_code_raw", "n_raw_charges",
]

# USSC categorical decoders (verified from FY2014 value distributions;
# stable across the modern USSC coding era but worth spot-checking older years).
# Each dict returns None for unknown codes so downstream can filter.
NEWRACE_DECODE = {1: "White", 2: "Black", 3: "Hispanic", 6: "Other"}
MONSEX_DECODE = {0: "M", 1: "F"}
CITIZEN_DECODE = {
    1: "US Citizen",
    2: "Legal Alien",
    3: "Illegal Alien",
    4: "Non-US Extra-Territorial",
    5: "Unknown",
}
# MAND1-6 coding: 4 = "no mandatory minimum applies" (75%+ of cases in
# FY2014). Any other value {1, 2, 3, 5, 6} indicates a specific mand-min
# statute was in play. This is the opposite of what a naive "any non-null"
# interpretation would give and is a documented USSC codebook subtlety.
MAND_NO_MIN_VALUE = 4


# --- Streaming reader ------------------------------------------------------

def _extract_fiscal_year_from_path(path: str) -> int | None:
    """opafy14nid.csv -> 2014, opafy99nid.csv -> 1999."""
    m = re.search(r"opafy(\d{2})", path.lower())
    if not m:
        return None
    yy = int(m.group(1))
    return 2000 + yy if yy < 90 else 1900 + yy


def _iter_source_chunks(source: str, chunksize: int, storage_options: dict | None):
    """Return an iterator of DataFrame chunks, one column-selected chunk at
    a time, from local file or S3 URL. Column selection happens INSIDE the
    read so we never materialize the full 25k-column row in memory.

    Handles three input forms transparently:
      - Local .csv
      - Local .zip containing a .csv
      - s3://.../.csv or s3://.../.zip

    For zips we open the archive as a stream (s3fs for S3 or plain open()
    for local), locate the single .csv inside, and hand zipfile's
    ZipExtFile to pandas. The ~2 GB uncompressed CSV never hits disk:
    zipfile decompresses on read, pandas chunksize means only chunksize
    rows are materialized at once."""
    all_wanted_cols = list(USSC_COLUMNS.values()) + NWSTAT_COLS + MAND_COLS
    wanted_set = set(all_wanted_cols)

    is_zip = source.lower().endswith(".zip")
    is_s3 = source.startswith("s3://")

    if not is_zip:
        return pd.read_csv(
            source,
            usecols=lambda c: c in wanted_set,
            dtype=str,
            chunksize=chunksize,
            low_memory=False,
            storage_options=storage_options,
            encoding="latin-1",   # USSC files aren't all UTF-8-clean;
                                    # latin-1 maps every byte, preserves ASCII
        )

    import zipfile
    if is_s3:
        import s3fs
        anon = bool(storage_options and storage_options.get("anon"))
        fs = s3fs.S3FileSystem(anon=anon)
        archive_stream = fs.open(source.replace("s3://", ""), "rb")
    else:
        archive_stream = open(source, "rb")

    zf = zipfile.ZipFile(archive_stream)
    csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if not csv_members:
        raise SystemExit(f"No .csv found inside {source}: {zf.namelist()}")
    if len(csv_members) > 1:
        opafy = [n for n in csv_members if "opafy" in n.lower()]
        member = opafy[0] if opafy else csv_members[0]
        print(f"  zip {source} has {len(csv_members)} CSVs; using {member!r}",
              file=sys.stderr)
    else:
        member = csv_members[0]

    inner_stream = zf.open(member, "r")
    inner_iter = pd.read_csv(
        inner_stream,
        usecols=lambda c: c in wanted_set,
        dtype=str,
        chunksize=chunksize,
        low_memory=False,
        encoding="latin-1",   # per-year USSC files aren't UTF-8-clean;
                                # latin-1 never fails, preserves ASCII fields
    )

    def _cleanup_gen():
        try:
            for chunk in inner_iter:
                yield chunk
        finally:
            inner_stream.close()
            zf.close()
            archive_stream.close()

    return _cleanup_gen()


def _process_chunk(chunk: pd.DataFrame, fiscal_year: int) -> pd.DataFrame:
    """Turn a raw USSC CSV chunk into canonical-column rows."""
    df = pd.DataFrame(index=chunk.index)

    for canon, source in USSC_COLUMNS.items():
        df[canon] = chunk.get(source, pd.NA)

    df["fiscal_year"] = fiscal_year
    df["judge_id"] = pd.NA
    df["departure_direction"] = pd.NA

    # Numeric coercion
    for c in ["final_offense_level", "criminal_history_category",
              "guideline_min_months", "guideline_max_months",
              "sentence_months", "defendant_age"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Decode categoricals from USSC integer codes to human-readable labels.
    # Race decoding uses NEWRACE (post-2000s consolidated coding); older
    # years may need MONRACE, which has 8+ categories with a different
    # crosswalk -- guard by fiscal year in a real per-year pipeline.
    race_int = pd.to_numeric(df["defendant_race"], errors="coerce")
    df["defendant_race"] = race_int.map(NEWRACE_DECODE)
    sex_int = pd.to_numeric(df["defendant_sex"], errors="coerce")
    df["defendant_sex"] = sex_int.map(MONSEX_DECODE)
    cit_int = pd.to_numeric(df["defendant_citizenship"], errors="coerce")
    df["defendant_citizenship"] = cit_int.map(CITIZEN_DECODE)

    # Retain the raw OFFTYPE2 code for provenance but do NOT use it as the
    # primary-offense category. OFFTYPE2 semantics drift across years
    # (FY2014 code 27 = Immigration reentry, not Administration of Justice
    # as in older codebooks) and USSC does not publish a stable crosswalk.
    # Instead, derive primary_offense from NWSTAT1 (first statutory
    # citation), categorized by the statute-based mapper, which IS stable
    # across years because it maps directly from USC title+section.
    df["primary_offense_code_raw"] = pd.to_numeric(
        chunk.get("OFFTYPE2", pd.Series(dtype=float)), errors="coerce"
    ).astype("Int64")

    # Vectorize NWSTAT categorization: convert to object array, categorize
    # in place, then derive primary and secondaries.
    nwstat_present = [c for c in NWSTAT_COLS if c in chunk.columns]
    if not nwstat_present:
        df["primary_offense"] = "OTHER"
        df["secondary_offenses"] = ""
        df["n_raw_charges"] = 0
    else:
        nw_arr = chunk[nwstat_present].to_numpy(dtype=object)
        cat_matrix = np.empty(nw_arr.shape, dtype=object)
        n_charges = np.zeros(nw_arr.shape[0], dtype=int)
        for i in range(nw_arr.shape[0]):
            n_present = 0
            for j in range(nw_arr.shape[1]):
                v = nw_arr[i, j]
                if isinstance(v, str) and v:
                    cat_matrix[i, j] = statute_to_category(v)
                    n_present += 1
                else:
                    cat_matrix[i, j] = None
            n_charges[i] = n_present

        # Primary = first non-null, non-SKIP categorized statute (NWSTAT1
        # usually). Theory-of-liability tags (18 USC §2 aiding-and-abetting,
        # §4 misprision, §13 assimilative crimes) return "SKIP" from the
        # categorizer and are dropped so they do not inflate stacking counts.
        primary = np.empty(cat_matrix.shape[0], dtype=object)
        secondaries = []
        for i in range(cat_matrix.shape[0]):
            row_cats = [c for c in cat_matrix[i] if c is not None and c != "SKIP"]
            primary[i] = row_cats[0] if row_cats else "OTHER"
            seen = {primary[i]}
            out = []
            for cat in row_cats[1:]:
                if cat not in seen:
                    seen.add(cat)
                    out.append(cat)
            secondaries.append(";".join(out))
        df["primary_offense"] = primary
        df["secondary_offenses"] = secondaries
        df["n_raw_charges"] = n_charges

    # mand_min_flag: True if ANY MAND* value is present AND not equal to 4
    # (the "no mandatory minimum applies" code). See MAND_NO_MIN_VALUE
    # constant for provenance.
    mand_present = [c for c in MAND_COLS if c in chunk.columns]
    if mand_present:
        mand_matrix = chunk[mand_present].apply(pd.to_numeric, errors="coerce")
        # True if any column has a non-null, non-4 value
        df["mand_min_flag"] = mand_matrix.apply(
            lambda row: any(
                (pd.notna(v) and v != MAND_NO_MIN_VALUE) for v in row
            ),
            axis=1,
        )
    else:
        df["mand_min_flag"] = False

    return df[CANONICAL_COLUMNS]


def load_from_source(source: str, fiscal_year: int, out_path: Path,
                       chunksize: int, storage_options: dict | None):
    """Stream one file (local or S3) into one Parquet."""
    print(f"Streaming {source} -> {out_path} (chunk={chunksize:,}) ...", file=sys.stderr)
    frames = []
    n = 0
    for chunk in _iter_source_chunks(source, chunksize, storage_options):
        processed = _process_chunk(chunk, fiscal_year)
        frames.append(processed)
        n += len(processed)
        print(f"  processed {n:,} rows ...", file=sys.stderr)
    if not frames:
        raise SystemExit(f"No data read from {source}.")
    df = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="snappy")
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {len(df):,} rows, {df.shape[1]} columns to {out_path} ({size_mb:.1f} MB)")
    return df


def _s3_storage_options(anon: bool) -> dict:
    """Return storage_options dict for pandas S3 reads. anon=True is only
    appropriate for public buckets; the USSC-mirror bucket you're pointing
    at (us-federal-sentencing-graph) requires AWS credentials, so the
    default is anon=False, which falls through to the standard boto3
    credential chain (env vars > ~/.aws/credentials > IAM role)."""
    return {"anon": True} if anon else {}


def load_from_s3_prefix(prefix: str, out_dir: Path, chunksize: int, anon: bool):
    """List every opafy*_csv file under the S3 prefix and stream each."""
    import s3fs
    fs = s3fs.S3FileSystem(anon=anon)
    prefix_path = prefix.replace("s3://", "")
    listing = fs.ls(prefix_path)

    csv_files = [f for f in listing
                  if (f.endswith(".csv") or f.endswith(".zip"))
                  and "opafy" in f.lower()]
    if not csv_files:
        raise SystemExit(f"No opafy*.{{csv,zip}} files found under {prefix}. "
                          f"Listed contents: {listing[:10]}")

    storage_options = _s3_storage_options(anon)
    failures = []
    for path in sorted(csv_files):
        fy = _extract_fiscal_year_from_path(path)
        if fy is None:
            print(f"  skipping {path}: cannot infer fiscal year", file=sys.stderr)
            continue
        out_path = out_dir / f"fy{fy}.parquet"
        if out_path.exists():
            print(f"  {out_path} already exists, skipping. Delete to re-process.",
                  file=sys.stderr)
            continue
        try:
            load_from_source(f"s3://{path}", fy, out_path, chunksize, storage_options)
        except Exception as e:
            # Log and continue rather than killing a multi-hour batch on
            # one bad file. Failed years are reported at the end so the
            # user can retry individually.
            print(f"  FAILED on {path}: {type(e).__name__}: {e}", file=sys.stderr)
            failures.append((path, str(e)))
            # Remove partial output if any so a retry starts clean
            if out_path.exists():
                out_path.unlink()

    if failures:
        print(f"\nCompleted with {len(failures)} failure(s):", file=sys.stderr)
        for path, err in failures:
            print(f"  {path}: {err[:120]}", file=sys.stderr)
        print("Retry individual failed files with --s3 <path> --fiscal-year <yy>",
              file=sys.stderr)


def load_synthetic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[CANONICAL_COLUMNS]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--s3", metavar="S3_URL",
                     help="Stream a single S3 CSV. Requires --fiscal-year.")
    src.add_argument("--s3-prefix", metavar="S3_PREFIX",
                     help="Stream ALL opafy*.csv files under this S3 prefix.")
    src.add_argument("--local", metavar="LOCAL_PATH",
                     help="Read a local CSV. Requires --fiscal-year.")
    src.add_argument("--synthetic", metavar="CSV_PATH",
                     help="Load the synthetic CSV instead of real USSC files.")
    ap.add_argument("--fiscal-year", type=int, default=None,
                     help="Required for --s3 and --local; ignored for --s3-prefix.")
    ap.add_argument("--out", default=None,
                     help="Output Parquet path (for --s3, --local, --synthetic).")
    ap.add_argument("--out-dir", default="data/events",
                     help="Output directory (for --s3-prefix).")
    ap.add_argument("--chunksize", type=int, default=10_000)
    ap.add_argument("--anon-s3", action="store_true", default=False,
                     help="Use anonymous S3 access (only for public buckets). "
                          "Default: use AWS credential chain "
                          "(env vars > ~/.aws/credentials > IAM role).")
    args = ap.parse_args()

    if args.synthetic:
        df = load_synthetic(Path(args.synthetic))
        out = Path(args.out or "data/sentencing_events.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"Wrote {len(df):,} synthetic rows to {out}")

    elif args.s3_prefix:
        load_from_s3_prefix(args.s3_prefix, Path(args.out_dir),
                             args.chunksize, args.anon_s3)

    else:
        source = args.s3 or args.local
        if args.fiscal_year is None:
            fy = _extract_fiscal_year_from_path(source)
            if fy is None:
                raise SystemExit("--fiscal-year required (could not infer from filename)")
        else:
            fy = args.fiscal_year
        out = Path(args.out or f"data/events/fy{fy}.parquet")
        storage_options = _s3_storage_options(args.anon_s3) if args.s3 else None
        load_from_source(source, fy, out, args.chunksize, storage_options)
