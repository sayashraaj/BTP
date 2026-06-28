from __future__ import annotations

"""
Data pipeline: fetch, parse, validate, and export Indian Railways schedule data.

Priority order:
1. Local TrainScheduleDB.csv (real Indian Railways data, 237,449 rows)
2. data.gov.in open data CSV
3. Synthetic schedule generation (guaranteed fallback)
"""

import logging
import random
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SCHEDULE_CSV = PROCESSED_DIR / "schedule.csv"
REAL_DATA_PATH = RAW_DIR / "TrainScheduleDB.csv"

SCHEMA_COLUMNS = [
    "train_id",
    "train_name",
    "station_id",
    "station_name",
    "arrival_time_seconds",
    "departure_time_seconds",
    "sequence_number",
]


def _hms_to_seconds(time_str: str, day: int = 1) -> int | None:
    """Convert 'HH:MM' or 'HH:MM:SS' to seconds, offset by day."""
    if not time_str or str(time_str).strip() in ("", "None", "nan", "NaT", "--", "Source", "Destination"):
        return None
    parts = str(time_str).strip().split(":")
    try:
        h, m = int(parts[0]), int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
        return (day - 1) * 86400 + h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return None


def _parse_real_csv(path: Path) -> pd.DataFrame | None:
    """
    Parse TrainScheduleDB.csv into the standardised schema.

    The real data has: TrainId, TrainName, StationId, StationName,
    ScheduledArrival, ScheduledDeparture, Distance, Day.

    Time encoding: Day * 86400 + HH*3600 + MM*60 (multi-day journeys).
    'Source' and 'Destination' are markers for first/last stops.
    """
    logger.info("Parsing real Indian Railways data from %s", path)
    try:
        df = pd.read_csv(path, encoding="cp1252", dtype=str)
    except Exception:
        try:
            df = pd.read_csv(path, encoding="utf-8", dtype=str)
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)
            return None

    rows = []
    seq_counters = {}

    for _, row in df.iterrows():
        tid = str(row.get("TrainId", "")).strip()
        tname = str(row.get("TrainName", "")).strip()
        sid = str(row.get("StationId", "")).strip()
        sname = str(row.get("StationName", "")).strip()
        arr_str = str(row.get("ScheduledArrival", "")).strip()
        dep_str = str(row.get("ScheduledDeparture", "")).strip()
        day = int(row.get("Day", 1))

        if not tid or tid == "nan":
            continue

        arr_s = _hms_to_seconds(arr_str, day)
        dep_s = _hms_to_seconds(dep_str, day)

        if arr_s is None and dep_s is None:
            continue
        if arr_s is None:
            arr_s = dep_s
        if dep_s is None:
            dep_s = arr_s

        seq_counters[tid] = seq_counters.get(tid, 0) + 1

        rows.append({
            "train_id": tid,
            "train_name": tname,
            "station_id": sid,
            "station_name": sname,
            "arrival_time_seconds": arr_s,
            "departure_time_seconds": dep_s,
            "sequence_number": seq_counters[tid],
        })

    if len(rows) < 100:
        logger.warning("Only %d valid rows parsed; insufficient", len(rows))
        return None

    result = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    logger.info("Parsed %d rows from real data (%d trains, %d stations)",
                len(result), result["train_id"].nunique(),
                result["station_id"].nunique())
    return result


def _try_download_gov_data() -> Path | None:
    """Attempt to download schedule CSV from data.gov.in."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / "indian_railways_timetable.csv"
    if cache_path.exists() and cache_path.stat().st_size > 1000:
        logger.info("Cache hit: %s (%d bytes)", cache_path, cache_path.stat().st_size)
        return cache_path

    urls = [
        "https://data.gov.in/files/ogdpv2dms/s3fs-public/datafile/Indian_Railways_Train_Time_Table.csv",
    ]
    for url in urls:
        logger.info("Attempting download from %s", url)
        try:
            resp = requests.get(url, timeout=60, stream=True)
            if resp.status_code == 200 and len(resp.content) > 500:
                cache_path.write_bytes(resp.content)
                logger.info("Downloaded %d bytes to %s", len(resp.content), cache_path)
                return cache_path
        except requests.RequestException as e:
            logger.warning("Download failed: %s", e)
    return None


def _parse_gov_csv(path: Path) -> pd.DataFrame | None:
    """Parse the data.gov.in CSV into the standardised schema."""
    logger.info("Parsing %s", path)
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", dtype=str)
    except Exception as e:
        logger.warning("CSV parse failed: %s", e)
        return None

    col_map = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for key, candidates in {
        "train_id": ["train no", "train_no", "trainno", "train number"],
        "train_name": ["train name", "train_name", "trainname"],
        "station_name": ["station name", "station_name", "source station name"],
        "station_id": ["station code", "station_code", "source station code"],
        "arrival": ["arrival time", "arrival_time", "arrival"],
        "departure": ["departure time", "departure_time", "departure"],
        "sequence": ["seq", "sequence", "stop_seq", "stop sequence", "serial no", "sno", "islno"],
    }.items():
        for cand in candidates:
            if cand in lower_cols:
                col_map[key] = lower_cols[cand]
                break

    if not {"train_id", "train_name", "station_name"}.issubset(col_map.keys()):
        logger.warning("Missing required columns. Found: %s", list(col_map.keys()))
        return None

    rows = []
    for idx, row in df.iterrows():
        tid = str(row.get(col_map.get("train_id", ""), "")).strip()
        tname = str(row.get(col_map.get("train_name", ""), "")).strip()
        sname = str(row.get(col_map.get("station_name", ""), "")).strip()
        sid = str(row.get(col_map.get("station_id", ""), sname)).strip()
        arr_s = _hms_to_seconds(str(row.get(col_map.get("arrival", ""), "")))
        dep_s = _hms_to_seconds(str(row.get(col_map.get("departure", ""), "")))

        try:
            seq = int(float(str(row.get(col_map.get("sequence", ""), idx))))
        except (ValueError, TypeError):
            seq = int(idx)

        if not tid or not sname or tid == "nan":
            continue
        if arr_s is None and dep_s is None:
            continue
        if arr_s is None:
            arr_s = dep_s
        if dep_s is None:
            dep_s = arr_s

        rows.append({
            "train_id": tid, "train_name": tname, "station_id": sid if sid != "nan" else sname,
            "station_name": sname, "arrival_time_seconds": arr_s,
            "departure_time_seconds": dep_s, "sequence_number": seq,
        })

    if len(rows) < 100:
        return None

    result = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    logger.info("Parsed %d rows from government data", len(result))
    return result


# ---------------------------------------------------------------------------
# Synthetic schedule generator (guaranteed fallback)
# ---------------------------------------------------------------------------

_STATION_POOL = [
    ("NDLS", "NEW DELHI"), ("MAS", "CHENNAI CENTRAL"), ("BCT", "MUMBAI CENTRAL"),
    ("HWH", "HOWRAH JUNCTION"), ("BPL", "BHOPAL JUNCTION"), ("JHS", "JHANSI JUNCTION"),
    ("WL", "WARANGAL"), ("BZA", "VIJAYAWADA JUNCTION"), ("CNB", "KANPUR CENTRAL"),
    ("NGP", "NAGPUR JUNCTION"), ("GWL", "GWALIOR JUNCTION"), ("AGC", "AGRA CANTT."),
    ("TPJ", "TIRUCHCHIRAPPALLI JUNCTION"), ("NZM", "HAZRAT NIZAMUDDIN"),
    ("BSB", "VARANASI JUNCTION"), ("CDG", "CHANDIGARH"), ("LKO", "LUCKNOW"),
    ("JP", "JAIPUR JUNCTION"), ("ADI", "AHMEDABAD JUNCTION"), ("PUNE", "PUNE JUNCTION"),
    ("SBC", "KSR BENGALURU CITY JUNCTION"), ("HYB", "HYDERABAD DECCAN"),
    ("SC", "SECUNDERABAD JUNCTION"), ("RNC", "RANCHI JUNCTION"), ("PNBE", "PATNA JUNCTION"),
    ("GD", "GUDUR JUNCTION"), ("NLR", "NELLORE"), ("BBS", "BHUBANESWAR"),
    ("VSKP", "VISAKHAPATNAM JUNCTION"), ("RJT", "RAJKOT JUNCTION"),
    ("UDZ", "UDAIPUR CITY"), ("JU", "JODHPUR JUNCTION"), ("BKN", "BIKANER JUNCTION"),
    ("AII", "AJMER JUNCTION"), ("ERS", "ERNAKULAM JUNCTION"),
    ("TVC", "THIRUVANANTHAPURAM CENTRAL"), ("MAQ", "MANGALURU JUNCTION"),
    ("CBE", "COIMBATORE JUNCTION"), ("MDU", "MADURAI JUNCTION"), ("MS", "CHENNAI EGMORE"),
    ("CGL", "CHENGALPATTU JUNCTION"), ("VM", "VILLUPURAM JUNCTION"),
    ("TNJ", "THANJAVUR JUNCTION"), ("KMU", "KUMBAKONAM"),
    ("ET", "ITARSI JUNCTION"), ("BPQ", "BALHARSHAH JUNCTION"),
    ("CD", "CHANDRAPUR"), ("SEG", "SEWAGRAM JUNCTION"), ("KMT", "KHAMMAM"),
    ("OGL", "ONGOLE"), ("CLT", "CHIRALA"), ("TEL", "TENALI JUNCTION"),
    ("RDM", "RAMAGUNDAM"), ("SKZR", "SIRPUR KAGHAZNAGAR"),
    ("SA", "SALEM JUNCTION"), ("JTJ", "JOLARPETTAI JUNCTION"),
    ("KPD", "KATPADI JUNCTION"), ("RU", "RENIGUNTA JUNCTION"),
    ("DG", "DINDIGUL JUNCTION"), ("KRR", "KARUR JUNCTION"),
]


def _generate_synthetic_schedule(
    n_trains: int = 600, min_stops: int = 4, max_stops: int = 20, seed: int = 42
) -> pd.DataFrame:
    """Generate a realistic synthetic Indian Railways schedule."""
    logger.info("Generating synthetic schedule: %d trains, %d-%d stops each",
                n_trains, min_stops, max_stops)
    rng = random.Random(seed)
    rows = []
    train_counter = 10001

    for _ in range(n_trains):
        tid = str(train_counter)
        train_counter += 1
        n_stops = rng.randint(min_stops, max_stops)
        route_stations = rng.sample(_STATION_POOL, min(n_stops, len(_STATION_POOL)))

        base_hour = rng.randint(0, 23)
        base_minute = rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        current_time = base_hour * 3600 + base_minute * 60

        suffixes = ["EXPRESS", "SF EXPRESS", "SUPERFAST", "MAIL", "SPECIAL", "RAJDHANI", "SHATABDI"]
        tname = f"{route_stations[0][1].split()[0]} - {route_stations[-1][1].split()[0]} {rng.choice(suffixes)}"

        for seq, (sid, sname) in enumerate(route_stations, start=1):
            dwell = rng.randint(1, 5) * 60 if seq > 1 else 0
            arr_time = current_time
            dep_time = arr_time + dwell
            rows.append({
                "train_id": tid, "train_name": tname, "station_id": sid, "station_name": sname,
                "arrival_time_seconds": arr_time, "departure_time_seconds": dep_time,
                "sequence_number": seq,
            })
            current_time = dep_time + rng.randint(60, 300) * 60

    result = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    logger.info("Generated %d rows across %d trains", len(result), n_trains)
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_schedule(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Validate schedule DataFrame. Returns (clean_df, list_of_issues)."""
    issues = []

    null_station = df["station_id"].isna() | (df["station_id"] == "")
    if null_station.any():
        bad_rows = df[null_station].index.tolist()
        issues.append({
            "check": "null_station_id", "rows": bad_rows,
            "detail": f"{len(bad_rows)} rows with null station_id",
        })
        df = df[~null_station].copy()

    arr_gt_dep = df["arrival_time_seconds"] > df["departure_time_seconds"]
    if arr_gt_dep.any():
        bad_rows = df[arr_gt_dep].index.tolist()
        issues.append({
            "check": "arrival_after_departure", "rows": bad_rows[:20],
            "detail": f"{len(bad_rows)} rows where arrival > departure",
        })
        df.loc[arr_gt_dep, "departure_time_seconds"] = df.loc[arr_gt_dep, "arrival_time_seconds"]

    for tid, grp in df.groupby("train_id"):
        seqs = grp["sequence_number"].values
        if len(seqs) > 1:
            diffs = seqs[1:] - seqs[:-1]
            if (diffs <= 0).any():
                issues.append({
                    "check": "non_monotonic_sequence", "train_id": tid,
                    "sequences": seqs.tolist(),
                    "detail": f"Train {tid}: sequence not strictly increasing",
                })

    logger.info("Validation complete: %d issues found, %d clean rows", len(issues), len(df))
    return df, issues


# ---------------------------------------------------------------------------
# Station lookup
# ---------------------------------------------------------------------------

_STATION_CODE_TO_NAME: dict[str, str] = {}


def get_station_name(schedule: pd.DataFrame, code_or_name: str) -> str:
    """Resolve a station code or name to the canonical station name in the schedule."""
    global _STATION_CODE_TO_NAME
    if not _STATION_CODE_TO_NAME:
        for _, row in schedule.drop_duplicates("station_id").iterrows():
            _STATION_CODE_TO_NAME[str(row["station_id"]).strip()] = str(row["station_name"]).strip()

    if code_or_name in _STATION_CODE_TO_NAME:
        return _STATION_CODE_TO_NAME[code_or_name]
    if code_or_name in _STATION_CODE_TO_NAME.values():
        return code_or_name
    return code_or_name


def get_station_code(schedule: pd.DataFrame, name_or_code: str) -> str:
    """Resolve a station name to its code."""
    matches = schedule[schedule["station_name"] == name_or_code]
    if len(matches) > 0:
        return str(matches.iloc[0]["station_id"])
    matches = schedule[schedule["station_id"] == name_or_code]
    if len(matches) > 0:
        return str(matches.iloc[0]["station_id"])
    return name_or_code


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(force_synthetic: bool = False) -> pd.DataFrame:
    """Execute the full data pipeline. Returns the validated schedule DataFrame."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if SCHEDULE_CSV.exists() and not force_synthetic:
        logger.info("Loading cached processed schedule from %s", SCHEDULE_CSV)
        return pd.read_csv(SCHEDULE_CSV)

    df = None

    if not force_synthetic:
        # Priority 1: Local real data
        if REAL_DATA_PATH.exists():
            t0 = time.time()
            df = _parse_real_csv(REAL_DATA_PATH)
            if df is not None:
                logger.info("Real data parsed in %.1fs", time.time() - t0)

        # Priority 2: data.gov.in
        if df is None:
            t0 = time.time()
            raw_path = _try_download_gov_data()
            if raw_path:
                df = _parse_gov_csv(raw_path)
                logger.info("Government data stage took %.1fs", time.time() - t0)

    if df is None:
        logger.info("Falling back to synthetic schedule generation")
        t0 = time.time()
        df = _generate_synthetic_schedule()
        logger.info("Synthetic generation took %.1fs", time.time() - t0)

    t0 = time.time()
    df, issues = validate_schedule(df)
    logger.info("Validation took %.1fs", time.time() - t0)
    for issue in issues:
        logger.warning("Validation issue: %s", issue["detail"])

    df.to_csv(SCHEDULE_CSV, index=False)
    logger.info("Exported %d rows to %s", len(df), SCHEDULE_CSV)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    schedule = run_pipeline()
    print(f"Pipeline complete: {len(schedule)} rows, {schedule['train_id'].nunique()} trains, "
          f"{schedule['station_id'].nunique()} stations")
