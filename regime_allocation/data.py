"""Local provider snapshots, exchange closes, and point-in-time input alignment."""

from datetime import date, datetime, timedelta, timezone
import hashlib
import csv
from importlib.metadata import version
import json
import os

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import requests

from . import config as c


class DataUnavailableError(RuntimeError):
    """A required provider input or endpoint is unavailable; stop preparation."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _code_hash(path):
    """Git newline conversion must not change a source file's recorded identity."""
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def write_manifest(manifest):
    c.RETRIEVAL_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    c.RETRIEVAL_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _retrieve(source_id, url, parameters, manifest, *, fred=False, document=False):
    """Cache exact response bytes and their original retrieval time, without keys."""
    signature = json.dumps([url, parameters], sort_keys=True).encode()
    name = f"{source_id}_{hashlib.sha256(signature).hexdigest()[:16]}"
    snapshot = c.RAW_DATA_DIR / f"{name}.{'html' if document else 'json'}"
    metadata_path = c.RAW_DATA_DIR / f"{name}.metadata.json"
    if snapshot.exists() != metadata_path.exists():
        raise DataUnavailableError(f"Incomplete local snapshot: {name}; restore both files.")
    if snapshot.exists():
        content = snapshot.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("source_id") != source_id or metadata.get("url") != url
                or metadata.get("parameters") != parameters):
            raise DataUnavailableError(f"Snapshot request metadata mismatch: {snapshot.name}.")
        if hashlib.sha256(content).hexdigest() != metadata["sha256"]:
            raise DataUnavailableError(f"Snapshot hash mismatch: {snapshot.name}.")
        cached = True
    else:
        query = dict(parameters)
        if fred:
            query["api_key"] = c.fred_api_key()
        try:
            response = requests.get(
                url, params=query, headers={"User-Agent": c.HTTP_USER_AGENT},
                timeout=c.HTTP_TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            raise DataUnavailableError(f"{source_id}: provider access failed ({type(error).__name__}).") from None
        if response.status_code != 200:
            raise DataUnavailableError(f"{source_id}: provider returned HTTP {response.status_code}.")
        content = response.content
        # Never archive a response that echoes an authenticated query string.
        key = os.environ.get(c.FRED_API_KEY_ENV, "").encode()
        if key and key in content:
            raise DataUnavailableError(f"{source_id}: response unexpectedly contains a credential.")
        metadata = {
            "source_id": source_id, "url": url, "parameters": parameters,
            "retrieved_at_utc": _now(), "http_status": response.status_code,
            "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
            "local_snapshot": snapshot.relative_to(c.PROJECT_ROOT).as_posix(),
            "redistributed": False,
        }
        c.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(content)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        cached = False
    manifest["retrievals"].append({**metadata, "reused_local_snapshot": cached})
    if document:
        return content
    try:
        return json.loads(content)
    except (ValueError, UnicodeDecodeError):
        raise DataUnavailableError(f"{source_id}: response is not valid JSON.") from None


def session_schedule():
    """XNYS includes holidays, early closes, DST, and extraordinary closures."""
    last = c.research_calendar()["scheduled_target_end_date"].max()
    calendar = xcals.get_calendar(
        c.MARKET_CALENDAR,
        start=c.MACRO_OBSERVATION_START,
        end=last + pd.Timedelta(days=c.CALENDAR_BUFFER_DAYS),
    )
    return calendar.schedule[["close"]].rename(columns={"close": "market_close"})


def decision_cutoffs(dates, sessions):
    """Use Friday's actual close; on a closed Friday retain a 16:00 cutoff.

    A Friday holiday changes the price observation to the last prior session,
    while a macro release on Friday can still be available for that week's decision.
    """
    return pd.DatetimeIndex([
        sessions.at[day, "market_close"] if day in sessions.index else
        (pd.Timestamp(day) + pd.Timedelta(hours=c.REGULAR_CLOSE_HOUR))
        .tz_localize(c.MARKET_TIMEZONE).tz_convert("UTC")
        for day in pd.DatetimeIndex(dates).normalize()
    ])


def align_calendar(sessions, calendar=None):
    """Resolve research weeks to actual predictor, execution, and target closes."""
    calendar = c.research_calendar() if calendar is None else calendar.copy()
    closes = pd.DatetimeIndex(sessions["market_close"].array)

    def next_close(day):
        position = sessions.index.searchsorted(pd.Timestamp(day))
        if position == len(sessions):
            raise DataUnavailableError(f"Exchange calendar does not cover endpoint {day}.")
        return closes[position]

    cutoffs = decision_cutoffs(calendar["decision_date"], sessions)
    positions = closes.searchsorted(cutoffs, side="right") - 1
    if (positions < 0).any():
        raise DataUnavailableError("Exchange calendar lacks predictor history.")
    calendar["decision_cutoff"] = cutoffs
    calendar["predictor_close"] = closes[positions]
    calendar["execution_close"] = [next_close(day) for day in calendar["scheduled_execution_date"]]
    calendar["realization_close"] = [
        next_close(timestamp.tz_convert(c.MARKET_TIMEZONE).date()
                   + pd.Timedelta(weeks=c.FORECAST_HORIZON_WEEKS))
        for timestamp in calendar["execution_close"]
    ]
    calendar["next_execution_close"] = [
        next_close(day + pd.Timedelta(weeks=c.REFIT_EVERY_WEEKS))
        for day in calendar["scheduled_execution_date"]
    ]
    boundary = decision_cutoffs([c.FIRST_RETURN_PRICE_BOUNDARY], sessions)[0]
    boundary_position = closes.searchsorted(boundary, side="right") - 1
    if boundary_position < 0:
        raise DataUnavailableError("Exchange calendar lacks the first return boundary.")
    calendar["previous_predictor_close"] = [closes[boundary_position], *calendar["predictor_close"].iloc[:-1]]
    if not ((calendar["predictor_close"] <= calendar["decision_cutoff"])
            & (calendar["decision_cutoff"] < calendar["execution_close"])
            & (calendar["execution_close"] < calendar["realization_close"])).all():
        raise DataUnavailableError("Decision, execution, and realization timestamps are inconsistent.")
    return calendar


def adjusted_prices(raw):
    """Apply Yahoo's distribution/split adjustment factor to OHLC prices."""
    required = ("Open", "High", "Low", "Close", "Adj Close")
    if not set(required).issubset(raw.columns):
        raise DataUnavailableError("Yahoo response lacks unadjusted OHLC or adjusted closes.")
    values = raw[list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise DataUnavailableError("Yahoo response contains missing or nonpositive prices.")
    if not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise DataUnavailableError("Yahoo price timestamps are duplicated or unordered.")
    adjusted = raw.drop(columns="Adj Close").copy()
    ohlc = ["Open", "High", "Low", "Close"]
    adjusted[ohlc] = raw[ohlc].mul(raw["Adj Close"] / raw["Close"], axis=0)
    adjusted["Close"] = raw["Adj Close"]
    return adjusted


def _parse_market(payload, asset, sessions):
    chart = payload.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise DataUnavailableError(f"Yahoo {asset}: no chart history returned.")
    result = chart["result"][0]
    if result.get("meta", {}).get("symbol") != asset:
        raise DataUnavailableError(f"Yahoo {asset}: unexpected source identifier.")
    timestamps = result.get("timestamp", [])
    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(c.MARKET_TIMEZONE).tz_localize(None).normalize()
    if not len(dates) or not dates.isin(sessions.index).all():
        raise DataUnavailableError(f"Yahoo {asset}: missing timestamps or non-session bars.")
    indicators = result.get("indicators", {})
    quote = (indicators.get("quote") or [{}])[0]
    adjusted = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    if adjusted is None:
        raise DataUnavailableError(f"Yahoo {asset}: adjusted close history is unavailable.")
    raw = pd.DataFrame({
        "Open": quote.get("open"), "High": quote.get("high"),
        "Low": quote.get("low"), "Close": quote.get("close"),
        "Adj Close": adjusted,
    }, index=pd.DatetimeIndex(sessions.loc[dates, "market_close"].array, name="market_close"))
    return adjusted_prices(raw)["Close"].rename(asset)


def _fred(endpoint, series, parameters, manifest):
    return _retrieve(
        f"alfred_{series}_{endpoint.replace('/', '_')}", c.FRED_API_BASE + endpoint,
        {"series_id": series, "file_type": "json", **parameters}, manifest, fred=True,
    )


def _macro_history(series, last_decision, manifest):
    metadata = _fred("series", series, {}, manifest)
    release = _fred("series/release", series, {}, manifest)
    vintage_dates = _fred("series/vintagedates", series, {
        "realtime_start": str(c.MACRO_OBSERVATION_START), "realtime_end": str(last_decision),
        "limit": c.FRED_VINTAGE_DATE_LIMIT, "sort_order": "asc",
    }, manifest)
    dates = vintage_dates.get("vintage_dates", [])
    if not dates or vintage_dates.get("count") != len(dates):
        raise DataUnavailableError(f"ALFRED {series}: incomplete vintage-date coverage.")
    rows = []
    first_year = c.MACRO_OBSERVATION_START.year
    for year in range(first_year, last_decision.year + 1, c.FRED_VINTAGE_CHUNK_YEARS):
        start = max(c.MACRO_OBSERVATION_START, pd.Timestamp(year, 1, 1).date())
        end = min(last_decision, pd.Timestamp(year + c.FRED_VINTAGE_CHUNK_YEARS, 1, 1).date() - pd.Timedelta(days=1))
        offset = 0
        while True:
            payload = _fred("series/observations", series, {
                "observation_start": str(c.MACRO_OBSERVATION_START), "observation_end": str(last_decision),
                "realtime_start": str(start), "realtime_end": str(end),
                "output_type": 1, "units": "lin", "sort_order": "asc",
                "limit": c.FRED_PAGE_LIMIT, "offset": offset,
            }, manifest)
            page = payload.get("observations", [])
            if "count" not in payload or (not page and offset < payload["count"]):
                raise DataUnavailableError(f"ALFRED {series}: incomplete observation page.")
            rows.extend({"series": series, **row} for row in page)
            offset += len(page)
            if offset >= payload["count"]:
                break
    if not rows:
        raise DataUnavailableError(f"ALFRED {series}: no historical vintages returned.")
    manifest["macro_sources"][series] = {
        "series_url": f"https://fred.stlouisfed.org/series/{series}",
        "attribution": metadata.get("seriess", [{}])[0].get("title"),
        "release_ids": [row["id"] for row in release.get("releases", [])],
        "vintage_dates": {"count": len(dates), "first": min(dates), "last": max(dates)},
        "observation_rows": len(rows), "historical_intraday_release_times": "not supplied by ALFRED",
        "availability_rule": c.MACRO_AVAILABILITY_RULE,
        "redistribution": "observations withheld; publish retrieval metadata and attribution only",
    }
    return pd.DataFrame(rows)


def normalize_vintages(rows):
    """Keep revisions as separate events and bound date-only availability safely."""
    required = {"series", "date", "realtime_start", "realtime_end", "value"}
    if not required.issubset(rows.columns):
        raise DataUnavailableError("ALFRED response lacks observation/vintage fields.")
    rows = rows.drop_duplicates().copy()
    numeric = pd.to_numeric(rows["value"].replace(".", np.nan), errors="raise")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise DataUnavailableError("ALFRED response contains infinite values.")
    rows["value"] = numeric
    # Real-time query boundaries clip otherwise unchanged vintage intervals.
    # Coalesce contiguous equal values so January 1 is not invented as a release.
    merged = []
    for _, group in rows.groupby(["series", "date"], sort=True):
        intervals = []
        for row in group.sort_values("realtime_start").to_dict("records"):
            if intervals:
                previous = intervals[-1]
                same_value = (row["value"] == previous["value"]
                              or (pd.isna(row["value"]) and pd.isna(previous["value"])))
                adjacent = (previous["realtime_end"] == "9999-12-31"
                            or row["realtime_start"] <= (date.fromisoformat(previous["realtime_end"]) + timedelta(days=1)).isoformat())
                if same_value and adjacent:
                    previous["realtime_end"] = max(previous["realtime_end"], row["realtime_end"])
                    continue
                if row["realtime_start"] <= previous["realtime_end"]:
                    raise DataUnavailableError("ALFRED contains conflicting overlapping vintage intervals.")
            intervals.append(row)
        merged.extend(intervals)
    rows = pd.DataFrame(merged)
    rows["observation_date"] = pd.to_datetime(rows["date"], format="%Y-%m-%d")
    starts = pd.to_datetime(rows["realtime_start"], format="%Y-%m-%d")
    rows["available_at"] = (
        (starts + pd.DateOffset(days=c.MACRO_AVAILABILITY_DELAY_DAYS))
        .dt.tz_localize(c.MACRO_AVAILABILITY_TIMEZONE).dt.tz_convert("UTC")
    )
    # Inclusive real-time ends expire with the same conservative publication lag.
    rows["valid_until"] = pd.to_datetime([
        pd.NaT if end == "9999-12-31" else
        (pd.Timestamp(end) + pd.DateOffset(days=1 + c.MACRO_AVAILABILITY_DELAY_DAYS))
        .tz_localize(c.MACRO_AVAILABILITY_TIMEZONE).tz_convert("UTC")
        for end in rows["realtime_end"]
    ], utc=True)
    return rows.sort_values(["series", "available_at", "observation_date"])


def available_macro(vintages, cutoffs):
    """Reconstruct each observation's known vintage before choosing the latest date."""
    selected = []
    for series in c.MACRO_SERIES:
        history = vintages.loc[vintages["series"].eq(series)]
        for week, cutoff in cutoffs.items():
            known = history.loc[
                history["available_at"].le(cutoff)
                & (history["valid_until"].isna() | history["valid_until"].gt(cutoff))
            ]
            known = known.loc[known["observation_date"].le(cutoff.tz_convert(c.MARKET_TIMEZONE).tz_localize(None).normalize())]
            known = known.sort_values(["observation_date", "available_at"])
            known = known.drop_duplicates("observation_date", keep="last").dropna(subset=["value"])
            if known.empty:
                raise DataUnavailableError(f"ALFRED {series}: no released observation at cutoff {cutoff.isoformat()} (week {week}).")
            row = known.iloc[-1]
            selected.append({
                "research_week": week, "series": series, "value": float(row["value"]),
                "observation_date": row["observation_date"], "available_at": row["available_at"],
                "vintage_start": row["realtime_start"], "vintage_end": row["realtime_end"],
                "valid_until": row["valid_until"],
                "decision_cutoff": cutoff,
            })
    return pd.DataFrame(selected).set_index(["research_week", "series"]).sort_index()


def macro_cutoffs(calendar, sessions):
    """Include four preceding decision weeks to define the first macro change."""
    first = calendar["decision_date"].iloc[0]
    dates = pd.date_range(
        end=first - pd.Timedelta(weeks=1), periods=c.MACRO_CHANGE_WEEKS,
        freq=pd.Timedelta(days=c.DAYS_PER_WEEK),
    )
    prefix = pd.Series(
        decision_cutoffs(dates, sessions),
        index=range(1 - c.MACRO_CHANGE_WEEKS, 1), name="decision_cutoff",
    )
    return pd.concat([prefix, calendar["decision_cutoff"]])


def validate_price_coverage(prices, calendar, sessions):
    start = calendar["previous_predictor_close"].iloc[0]
    end = calendar["realization_close"].max()
    expected = pd.DatetimeIndex(sessions.loc[
        sessions["market_close"].between(start, end), "market_close"
    ].array, name="market_close")
    if not prices.index.is_unique:
        raise DataUnavailableError("Adjusted prices have duplicate session closes.")
    aligned = prices.reindex(index=expected, columns=c.ASSETS)
    missing = aligned.isna()
    if missing.any().any():
        details = {
            asset: [time.isoformat() for time in aligned.index[missing[asset]][:5]]
            for asset in c.ASSETS if missing[asset].any()
        }
        raise DataUnavailableError(f"Required Yahoo trading sessions are unavailable: {details}")
    if not np.isfinite(aligned.to_numpy()).all() or (aligned <= 0).any().any():
        raise DataUnavailableError("Adjusted prices must be finite and strictly positive.")
    return aligned


def _new_manifest():
    return {
        "schema_version": 1, "status": "preparing", "generated_at_utc": _now(),
        "retrievals": [], "macro_sources": {}, "coverage": {},
        "unavailable_required_inputs": [], "locked_test_evaluated": False,
        "market_source": {"provider": c.MARKET_DATA_SOURCE, "assets": list(c.ASSETS),
                          "retrieval_client": "requests; Yahoo chart JSON endpoint",
                          "snapshot_representation": "exact response body bytes after HTTP content decoding"},
        "publication": {
            "contents": "retrieval metadata, dates, hashes, methods, and attribution only",
            "raw_and_derived_observations": "private under ignored data/; no observations published",
            "yahoo": "personal research use; no permission to redistribute snapshots is assumed",
            "fred": "series owners retain rights; VIXCLS and T10Y2Y require attribution; PPIACO is public-domain with citation requested",
            "terms_urls": dict(c.SOURCE_TERMS),
            "terms_reviewed_on": str(c.SOURCE_TERMS_REVIEWED_ON),
            "terms_review_method": (
                "official Yahoo terms saved locally; official FRED API terms and ALFRED help "
                "verified through web access (documentation-host raw download timed out)"
            ),
        },
        "methods": {
            "exchange_calendar": c.MARKET_CALENDAR,
            "friday_holiday": "Friday 16:00 local cutoff; last prior exchange close for prices",
            "friday_early_close": "actual exchange close is the decision cutoff",
            "execution": c.EXECUTION_RULE,
            "target": "actual execution date plus four calendar weeks; next session if closed",
            "macro_availability": c.MACRO_AVAILABILITY_RULE,
            "macro_timing_precision": "date-only vintages; intraday release/ingestion times are unknown, not invented",
            "query_boundary_intervals": (
                "adjacent unchanged ALFRED intervals are coalesced before assigning availability; "
                "raw clipped intervals are preserved"
            ),
            "macro_changes": "difference or fractional percentage change of levels known at their respective weekly cutoffs",
            "adjustment": "Yahoo adjusted close; OHLC multiplied by adjusted-close/raw-close (the yfinance adjustment formula)",
            "market_vintages": "current provider history snapshotted exactly; historical market-data revision vintages are not supplied",
            "vintage_paging": f"{c.FRED_VINTAGE_CHUNK_YEARS}-year real-time intervals and observation-offset pagination",
        },
        "reconstruction": [
            "Use CPython 3.12.14 and install requirements.txt into the project .venv.",
            "Supply FRED_API_KEY through the process environment; no key appears in this manifest.",
            "Run python run_pipeline.py --stage data from the project root with the .venv interpreter.",
            "Restore the listed private snapshots and adjacent metadata files to data/raw/ "
            "to replay exact bytes; every cached response is hash-checked.",
            "Without local snapshots the command retrieves the recorded endpoints/parameters "
            "again; later provider revisions can change bytes and hashes.",
            "Neither original provider observations nor derived price/features/targets are distributed with this manifest.",
        ],
        "dependency_versions": {
            name: version(name) for name in (
                "pandas", "numpy", "yfinance", "fredapi", "requests", "exchange-calendars"
            )
        },
    }


def prepare_sources():
    """Verify access, terms, and complete endpoint coverage before feature work."""
    manifest = _new_manifest()
    try:
        for source_id in ("yahoo_terms", "yahoo_finance_terms"):
            _retrieve(source_id, c.SOURCE_TERMS[source_id], {}, manifest, document=True)
        sessions = session_schedule()
        calendar = align_calendar(sessions)
        start = pd.Timestamp(c.FIRST_RETURN_PRICE_BOUNDARY).tz_localize(c.MARKET_TIMEZONE)
        end = (calendar["realization_close"].max().tz_convert(c.MARKET_TIMEZONE).normalize()
               + pd.DateOffset(days=1))
        parameters = {
            **c.YAHOO_CHART_PARAMETERS,
            "period1": int(start.timestamp()), "period2": int(end.timestamp()),
        }
        histories = []
        for asset in c.ASSETS:
            payload = _retrieve(f"yahoo_{asset}", c.YAHOO_CHART_BASE + asset, parameters, manifest)
            histories.append(_parse_market(payload, asset, sessions))
        prices = validate_price_coverage(pd.concat(histories, axis=1).sort_index(), calendar, sessions)
        last_decision = calendar["decision_date"].iloc[-1].date()
        vintages = normalize_vintages(pd.concat([
            _macro_history(series, last_decision, manifest) for series in c.MACRO_SERIES
        ], ignore_index=True))
        availability = available_macro(vintages, macro_cutoffs(calendar, sessions))
        manifest["coverage"] = {
            "research_weeks": len(calendar), "first_week_start": str(c.RESEARCH_START),
            "splits": {name: last - first + 1 for name, (first, last) in c.SPLIT_WEEKS.items()},
            "first_price_boundary_utc": prices.index.min().isoformat(),
            "last_required_realization_utc": prices.index.max().isoformat(),
            "required_daily_sessions_per_asset": len(prices),
            "assets": {asset: {"missing_sessions": 0, "adjusted_prices": True} for asset in c.ASSETS},
            "macro_cutoffs_per_series_including_history": len(availability) // len(c.MACRO_SERIES),
            "macro_missing_cutoffs": 0,
            "friday_holiday_decisions": int((
                calendar["predictor_close"].dt.tz_convert(c.MARKET_TIMEZONE)
                .dt.tz_localize(None).dt.normalize() != calendar["decision_date"]
            ).sum()),
            "holiday_shifted_executions": int((
                calendar["execution_close"].dt.tz_convert(c.MARKET_TIMEZONE)
                .dt.tz_localize(None).dt.normalize() != calendar["scheduled_execution_date"]
            ).sum()),
        }
        manifest["status"] = "sources_ready"
        write_manifest(manifest)
        return prices, calendar, availability, manifest
    except (DataUnavailableError, ValueError, KeyError) as error:
        manifest["status"] = "failed"
        manifest["unavailable_required_inputs"] = [str(error)]
        write_manifest(manifest)
        raise DataUnavailableError(str(error)) from None


def save_prepared(tables, manifest):
    """Write observations locally; publish only hashes and coverage metadata."""
    artifacts = []
    for name, frame in tables.items():
        path = c.PREPARED_PATHS[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, float_format="%.17g", lineterminator="\n")
        artifacts.append({
            "name": name, "local_path": path.relative_to(c.PROJECT_ROOT).as_posix(),
            "rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "redistributed": False,
        })
    manifest["local_artifacts"] = artifacts
    validation_targets = tables["targets"].xs(c.FORECAST_ASSETS[0], level="asset")
    eligible = validation_targets["selection_eligible"]
    manifest["coverage"]["selection_eligible_validation_origins"] = int(eligible.sum())
    manifest["coverage"]["boundary_crossing_validation_origins"] = validation_targets.index[
        validation_targets["split"].eq("validation") & ~eligible
    ].tolist()
    manifest["code_hash_format"] = "sha256 of UTF-8 source text normalized to LF line endings"
    manifest["code_hashes"] = {
        name: _code_hash(c.PROJECT_ROOT / name)
        for name in ("regime_allocation/config.py", "regime_allocation/data.py",
                     "regime_allocation/features.py", "run_pipeline.py", "requirements.txt")
    }
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = _now()
    write_manifest(manifest)


def prepare_data():
    """Run source validation before constructing any dependent feature tables."""
    from .features import build_dataset

    prices, calendar, availability, manifest = prepare_sources()
    try:
        tables = build_dataset(prices, calendar, availability)
        save_prepared(tables, manifest)
    except (DataUnavailableError, ValueError, KeyError) as error:
        manifest["status"] = "failed"
        manifest["unavailable_required_inputs"] = [str(error)]
        write_manifest(manifest)
        raise DataUnavailableError(str(error)) from None
    return manifest


def load_prepared(period="validation"):
    """Read verified snapshots; selection never receives locked-test outcomes."""
    manifest = json.loads(c.RETRIEVAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["unavailable_required_inputs"]:
        raise DataUnavailableError("Data preparation is incomplete.")
    for entry in [*manifest["retrievals"], *manifest["local_artifacts"]]:
        path = c.PROJECT_ROOT / entry.get("local_snapshot", entry.get("local_path"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise DataUnavailableError(f"Recorded input hash mismatch: {path.name}.")
    indexes = {
        "calendar": ["research_week"], "adjusted_prices": ["market_close"],
        "weekly_returns": ["research_week"], "macro_predictors": ["research_week"],
        "macro_availability": ["research_week", "series"],
        "features": ["research_week", "asset"], "targets": ["research_week", "asset"],
    }
    utc_columns = {
        "market_close", "decision_cutoff", "predictor_close", "execution_close",
        "realization_close", "next_execution_close", "previous_predictor_close",
        "available_at", "valid_until",
    }
    date_columns = {"week_start", "decision_date", "scheduled_execution_date",
                    "scheduled_target_end_date", "observation_date"}
    strings = utc_columns | date_columns | {
        "research_week", "asset", "series", "split", "selection_eligible", "vintage_start", "vintage_end",
    }
    tables = {}
    for name, path in c.PREPARED_PATHS.items():
        with path.open(encoding="utf-8", newline="") as stream:
            columns = next(csv.reader(stream))
        numeric = {column: (lambda value: float(value) if value else np.nan)
                   for column in columns if column not in strings}
        frame = pd.read_csv(path, converters=numeric)
        for column in frame.columns.intersection(list(utc_columns)):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        for column in frame.columns.intersection(list(date_columns)):
            frame[column] = pd.to_datetime(frame[column])
        tables[name] = frame.set_index(indexes[name]).sort_index()
    cutoff = tables["calendar"].at[c.SPLIT_WEEKS["locked_test"][0], "decision_cutoff"]
    if period == "validation":
        end = c.SPLIT_WEEKS["validation"][1]
        for name, frame in tables.items():
            if name == "adjusted_prices":
                tables[name] = frame.loc[frame.index < cutoff]
            else:
                weeks = frame.index.get_level_values("research_week")
                tables[name] = frame.loc[weeks <= end].copy()
        future = tables["targets"]["realization_close"] >= cutoff
        tables["targets"].loc[future, "target_4w_excess_return"] = np.nan
    elif period != "locked_test":
        raise ValueError(f"Unknown evaluation period: {period}")
    tables["locked_test_cutoff"] = cutoff
    tables["retrieval_manifest"] = manifest
    return tables
