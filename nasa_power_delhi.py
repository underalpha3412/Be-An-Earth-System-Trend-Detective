#!/usr/bin/env python3
"""
Fetch monthly climate data from the official NASA POWER API
(https://power.larc.nasa.gov/api/temporal/monthly/point) for New Delhi,
transform it into a wide-format pandas DataFrame that mirrors the standard
NASA POWER tabular layout (PARAMETER, YEAR, JAN..DEC, ANN), print it fully,
and optionally save it to CSV.

Usage:
    python nasa_power_delhi.py            # fetch, transform, print
    python nasa_power_delhi.py --save     # also write nasa_delhi_climate_2015_2025.csv
    python nasa_power_delhi.py --save-out path/to/file.csv
"""

import argparse
import sys

import pandas as pd
import requests

API_BASE_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"

LATITUDE = 28.6209          # New Delhi
LONGITUDE = 77.0907         # New Delhi
START_YEAR = 2015
END_YEAR = 2025
COMMUNITY = "ag"            # Agroclimatology community
FORMAT = "JSON"

MONTH_NAMES = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

PARAMETERS = [
    "PRECTOTCORR",   # Precipitation (bias-corrected)
    "QV2M",          # Specific Humidity at 2 Meters
    "T2M_MAX",       # Temperature at 2 Meters Maximum
    "T2M_MIN",       # Temperature at 2 Meters Minimum
    "TSOIL1",        # Soil Temperature, Layer 1
    "WS10M",         # Wind Speed at 10 Meters
]

DEFAULT_CSV_NAME = "nasa_delhi_climate_2015_2025.csv"

# NASA's nodata sentinel (appears in header.fill_value)
FILL_VALUE = -999.0

REQUEST_TIMEOUT = 90  # seconds; POWER can be slow on cold cache


def build_url() -> str:
    """Construct the NASA POWER monthly/point request URL."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start": START_YEAR,
        "end": END_YEAR,
        "community": COMMUNITY,
        "parameters": ",".join(PARAMETERS),
        "format": FORMAT,
    }
    return API_BASE_URL, requests.Request("GET", API_BASE_URL, params=params).prepare().url


def fetch_monthly_data() -> dict:
    """
    Call the NASA POWER API and return the parsed JSON dict.

    Raises RuntimeError with a readable message on any failure.
    """
    base_url, url = build_url()
    print(f"Requesting NASA POWER data...\nURL: {url}\n")

    try:
        response = requests.get(base_url, params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start": START_YEAR,
            "end": END_YEAR,
            "community": COMMUNITY.upper(),
            "parameters": ",".join(PARAMETERS),
            "format": FORMAT,
        }, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Timed out after {REQUEST_TIMEOUT}s contacting NASA POWER. "
            "The service can be slow; try again shortly."
        ) from None
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not connect to NASA POWER ({exc.__class__.__name__}). "
            "Check your internet connection / proxy settings."
        ) from None
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"NASA POWER request failed: {exc}") from None

    if response.status_code == 404:
        raise RuntimeError(
            "NASA POWER returned 404 (Not Found). Common causes: an invalid "
            "parameter name, or a date range with no data for this point."
        )
    if response.status_code == 429:
        raise RuntimeError(
            "NASA POWER returned 429 (Too Many Requests). Service rate limit "
            "hit - wait a minute and retry."
        )
    if response.status_code == 503:
        raise RuntimeError(
            "NASA POWER returned 503 (Service Unavailable). The API may be "
            "temporarily down for maintenance; try again later."
        )
    if not response.ok:
        raise RuntimeError(
            f"NASA POWER request failed with HTTP {response.status_code}:\n"
            f"{response.text[:500]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "NASA POWER responded with non-JSON content; the payload may be a "
            "maintenance page. Response preview:\n" + response.text[:300]
        ) from exc

    return payload


def parse_header(payload: dict) -> str:
    """Extract title/units summary and surface POWER's warning messages."""
    header = payload.get("header") or {}
    title = header.get("title", "NASA POWER data")
    fill = header.get("fill_value")
    print(f"Service : {header.get('service', 'POWER')}")
    print(f"Title   : {title}")

    messages = payload.get(" messages") or payload.get("messages") or []
    for msg in messages:
        m = str(msg).strip()
        if m:
            print(f"Note    : {m}")

    if fill is not None:
        print(f"Fill    : missing values are flagged as {fill}\n")
    return title


def extract_units(payload: dict) -> dict:
    """Map each requested parameter to its unit string from the API metadata.

    Structure (verified live): payload['parameters'][PARAM]['units'].
    """
    info = payload.get("parameters") or {}
    return {p: (info.get(p) or {}).get("units", "") for p in PARAMETERS}


def transform_to_dataframe(payload: dict) -> pd.DataFrame:
    """
    Transform the nested NASA POWER JSON into the standard wide tabular layout:

        PARAMETER, YEAR, JAN, FEB, ..., DEC, ANN

    NASA encodes monthly values as keys 'YYYYMM' (MM = 1..12 for months, 13 for
    the annual mean/aggregate). Each (parameter, year) pair becomes exactly one
    row: months fill JAN..DEC and the 'YYYY13' value fills ANN. The unused
    'ANN' dict key under each parameter is API metadata (lat/lon/height), not
    data, and is skipped here.
    """
    parameters_block = payload.get("properties", {}).get("parameter")
    if not parameters_block:
        raise RuntimeError("API response contained no 'properties.parameter' data block.")

    missing_params = [p for p in PARAMETERS if p not in parameters_block]
    if missing_params:
        raise RuntimeError(
            "NASA POWER did not return the requested parameter(s): "
            + ", ".join(missing_params)
        )

    fill_value = (payload.get("header") or {}).get("fill_value", FILL_VALUE)

    def clean(v):
        # Flag nodata sentinels so downstream math is safe.
        if v is None or (isinstance(v, (int, float)) and v <= fill_value):
            return pd.NA
        return round(float(v), 2)

    rows: dict[tuple[str, int], dict] = {}
    for param in PARAMETERS:
        series = parameters_block[param]
        for key, value in series.items():
            # temporal/data keys are 'YYYYMM' with MM = 1..12 (months) or 13 (annual).
            if not (isinstance(key, str) and len(key) == 6 and key.isdigit()):
                continue  # skip e.g. the 'ANN' metadata dict (lat/lon/height)

            year, month = int(key[:4]), int(key[4:])
            if not (START_YEAR <= year <= END_YEAR and 1 <= month <= 13):
                continue

            row = rows.setdefault(
                (param, year),
                {"PARAMETER": param, "YEAR": year,
                 **{name: pd.NA for name in MONTH_NAMES}, "ANN": pd.NA},
            )
            if month == 13:
                row["ANN"] = clean(value)
            else:
                row[MONTH_NAMES[month - 1]] = clean(value)

    if not rows:
        raise RuntimeError(
            "No usable monthly data points found for "
            f"{START_YEAR}-{END_YEAR} at the given coordinates."
        )

    df = pd.DataFrame(list(rows.values()),
                      columns=["PARAMETER", "YEAR", *MONTH_NAMES, "ANN"])

    # Sort parameters in the requested order, years ascending within each.
    param_order = {p: i for i, p in enumerate(PARAMETERS)}
    df["_p"] = df["PARAMETER"].map(param_order)
    df = df.sort_values(["_p", "YEAR"]).drop(columns="_p").reset_index(drop=True)
    return df


def print_table(df: pd.DataFrame, units: dict | None = None) -> None:
    """Print the full wide table without truncation and with 2 decimals."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.float_format", "{:.2f}".format)

    print("NASA POWER Monthly Climate Data - New Delhi "
          f"(lat {LATITUDE}, lon {LONGITUDE}), {START_YEAR}-{END_YEAR}")
    if units:
        print("Units:")
        for p in PARAMETERS:
            print(f"  {p:<12} {units.get(p, '-')}")
    print("=" * 130)
    print(df.to_string(index=False, na_rep=""))
    print("=" * 130)


def save_to_csv(df: pd.DataFrame, path: str, units: dict | None = None) -> None:
    """Write the wide table, with a units comment line appended at the end."""
    df.to_csv(path, index=False)
    if units:
        parts = " ".join(f"{p}={units.get(p) or '-'}" for p in PARAMETERS)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"\n# Units: {parts}\n")
    print(f"Saved CSV -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", action="store_true",
                        help=f"save the table to {DEFAULT_CSV_NAME}")
    parser.add_argument("--save-out", metavar="PATH",
                        help="save the table to a custom CSV path")
    args = parser.parse_args()

    try:
        payload = fetch_monthly_data()
        parse_header(payload)  # also surfaces POWER's warning messages
        units = extract_units(payload)
        df = transform_to_dataframe(payload)
        print()

        print_table(df, units=units)

        if args.save_out:
            save_to_csv(df, args.save_out, units=units)
        elif args.save:
            save_to_csv(df, DEFAULT_CSV_NAME, units=units)
        else:
            print(f"\nTip: run with --save to also write {DEFAULT_CSV_NAME}")
        return 0
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
