import calendar
import csv
import hashlib
import json
import os
import re
import time
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import requests
from tqdm.auto import tqdm

@dataclass
class PipelineConfig:
    history_start_date: str = '2026-05-01'
    history_end_date: str = '2026-05-10'
    valid_length_days: int = 1
    test_length_days: int = 1
    setting: str = 'offline'
    dataset_time_granularity: str = 'day'
    output_dir: str = 'generated_datasets'
    gdelt_raw_dir: str = '.cache/gdelt'
    cache_dir: str = '.cache/construction'
    use_gdelt_raw_cache: bool = True
    use_gdelt_manifest_cache: bool = True
    gdelt_source_policy: str = 'auto'
    gdelt_manifest_cache_file: str = 'gdelt_file_manifest.json'
    gdelt_download_max_retries: int = 3
    gdelt_download_retry_backoff_seconds: float = 5.0
    use_wikidata_cache: bool = True
    use_existing_entity_cache_only: bool = False
    repeat_lookback_days: int = 1
    leak_sr_max_objects: int = 3
    leak_s_max_objects: int = 5
    wikidata_match_threshold: float = 0.6
    wikidata_batch_size: int = 25
    wikidata_delay_seconds: float = 0.2
    wikidata_request_timeout: int = 60
    wikidata_max_retries: int = 5
    wikidata_retry_backoff_seconds: float = 2.0
    wikidata_min_request_interval_seconds: float = 2.0
    wikidata_continue_on_batch_error: bool = True
    openai_model: str = os.environ.get('LLM_MODEL', '')
    openai_base_url: str = os.environ.get('LLM_BASE_URL', '')
    openai_api_key_env: str = 'LLM_API_KEY'
    openai_request_timeout_seconds: int = 60
    openai_max_retries: int = 0
    openai_web_search_tool_type: str = 'web_search'
    openai_external_web_access: bool = True
    openai_enable_search: bool = False
    openai_forced_search: bool = False
    openai_prompt_version: str = 'pipeline-v1'
    enable_llm_entity_resolution: bool = True
    llm_entity_resolution_workers: int = 1
    llm_entity_resolution_request_interval_seconds: float = 0.5
    enable_llm_entity_relation_validation: bool = True
    enable_llm_interval_refinement: bool = False
    enable_llm_online_interval_verification: bool = False
    semantic_domain: str = 'politics'
    domain_classification_method: str = 'llm'
    domain_endpoint_policy: str = 'either'
    include_point_time_facts: bool = True
    include_interval_facts: bool = True
    enable_wikidata_temporal_expansion: bool = True
    wikidata_expansion_max_depth: int = 1
    max_new_entities_per_depth: int = 5000
    enable_llm_resolution: Optional[bool] = None
    enable_online_interval_verification: Optional[bool] = None
    user_agent: str = os.environ.get('BENCHMARK_USER_AGENT', 'TemporalBenchmark/1.0')

def ensure_output_dirs(config: PipelineConfig) -> None:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.gdelt_raw_dir).mkdir(parents=True, exist_ok=True)
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

def config_to_jsonable(config: PipelineConfig) -> Dict[str, Any]:
    private_runtime_fields = {'output_dir', 'cache_dir', 'gdelt_raw_dir', 'openai_base_url', 'openai_api_key_env', 'user_agent'}
    return {key: value for key, value in asdict(config).items() if key not in private_runtime_fields}

def cache_file_path(config: PipelineConfig, filename: str) -> Path:
    return Path(config.cache_dir) / filename
_JSONL_CACHE_LOCKS: Dict[str, threading.RLock] = {}
_JSONL_CACHE_LOCKS_GUARD = threading.Lock()

def jsonl_cache_lock(path: Path) -> threading.RLock:
    key = str(Path(path).resolve())
    with _JSONL_CACHE_LOCKS_GUARD:
        lock = _JSONL_CACHE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _JSONL_CACHE_LOCKS[key] = lock
        return lock

def load_jsonl_cache(path: Path, key_field: str) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    with jsonl_cache_lock(path):
        cache = {}
        if not path.exists():
            return cache
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                key = row.get(key_field)
                if key is not None and key != '':
                    cache[str(key)] = row
        return cache
WIKIDATA_API_URL = 'https://www.wikidata.org/w/api.php'
WIKIDATA_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_WIKIDATA_REQUEST_LOCK = threading.Lock()
_WIKIDATA_LAST_REQUEST_AT = [0.0]

def wait_for_wikidata_request_slot(config: PipelineConfig) -> None:
    interval = float(getattr(config, 'wikidata_min_request_interval_seconds', 0.0) or 0.0)
    if interval <= 0:
        return
    with _WIKIDATA_REQUEST_LOCK:
        elapsed = time.monotonic() - _WIKIDATA_LAST_REQUEST_AT[0]
        if elapsed < interval:
            time.sleep(interval - elapsed)
        _WIKIDATA_LAST_REQUEST_AT[0] = time.monotonic()

def append_jsonl_record(path: Path, row: Dict[str, Any], sort_keys: bool=True) -> None:
    path = Path(path)
    with jsonl_cache_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=sort_keys) + '\n')

def record_wikidata_failed_batch(config: PipelineConfig, kind: str, qids: Sequence[str], error: Exception, attempts: int) -> None:
    row = {'kind': kind, 'qids': list(qids), 'error_type': type(error).__name__, 'error_message': str(error), 'attempts': attempts}
    append_jsonl_record(cache_file_path(config, 'wikidata_failed_batches.jsonl'), row)

def wikidata_get_json(params: Dict[str, Any], config: PipelineConfig, kind: str, qids: Sequence[str]) -> Dict[str, Any]:
    max_attempts = max(1, int(getattr(config, 'wikidata_max_retries', 5) or 1))
    backoff = float(getattr(config, 'wikidata_retry_backoff_seconds', 2.0) or 0.0)
    timeout = getattr(config, 'wikidata_request_timeout', 60)
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            wait_for_wikidata_request_slot(config)
            response = requests.get(WIKIDATA_API_URL, params=params, timeout=timeout, headers={'User-Agent': config.user_agent})
            status_code = getattr(response, 'status_code', None)
            if status_code in WIKIDATA_RETRYABLE_STATUS_CODES:
                raise requests.exceptions.HTTPError(f'Retryable Wikidata HTTP status {status_code}', response=response)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError, requests.exceptions.RequestException) as exc:
            last_error = exc
            if attempt >= max_attempts:
                raise
            sleep_seconds = backoff * 2 ** (attempt - 1)
            print(f'[Wikidata] {kind} batch retry {attempt}/{max_attempts} for {len(qids)} id(s): {type(exc).__name__}: {exc}', flush=True)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    if last_error is not None:
        raise last_error
    return {}

def llm_entity_resolution_enabled(config: PipelineConfig) -> bool:
    if config.enable_llm_resolution is not None:
        return bool(config.enable_llm_resolution)
    return bool(config.enable_llm_entity_resolution)

def llm_entity_relation_validation_enabled(config: PipelineConfig) -> bool:
    return bool(getattr(config, 'enable_llm_entity_relation_validation', False))

def llm_interval_refinement_enabled(config: PipelineConfig) -> bool:
    if config.enable_llm_resolution is not None:
        return bool(config.enable_llm_resolution)
    return bool(config.enable_llm_interval_refinement)

def llm_online_interval_verification_enabled(config: PipelineConfig) -> bool:
    if config.enable_online_interval_verification is not None:
        return bool(config.enable_online_interval_verification)
    return bool(config.enable_llm_online_interval_verification)
LOOKUP_BASE = 'https://www.gdeltproject.org/data/lookups/'
GDELT_EVENTS_BASE = 'http://data.gdeltproject.org/events/'
VALID_TIME_GRANULARITIES = {'day', 'month', 'year'}
TIME_GRANULARITY_RANK = {'year': 1, 'month': 2, 'day': 3}

def parse_date(value: str) -> date:
    value = str(value).strip()
    if re.fullmatch('\\d{8}', value):
        return datetime.strptime(value, '%Y%m%d').date()
    return date.fromisoformat(value)

def validate_time_granularity(granularity: str) -> str:
    granularity = str(granularity or 'day').strip().lower()
    if granularity not in VALID_TIME_GRANULARITIES:
        raise ValueError(f'dataset_time_granularity must be one of {sorted(VALID_TIME_GRANULARITIES)}, got {granularity!r}')
    return granularity

def validate_pipeline_setting(config: PipelineConfig) -> str:
    setting = str(config.setting or 'offline').strip().lower()
    if setting in {'transductive', 'inductive'}:
        raise ValueError("transductive/inductive are experiment-time masks and are no longer valid pipeline setting values; use setting='offline' or setting='online_testing'.")
    if setting not in {'offline', 'online_testing'}:
        raise ValueError("setting must be 'offline' or 'online_testing'")
    return setting

def infer_time_value_precision(value: str) -> str:
    value = str(value or '').strip()
    if re.fullmatch('\\d{8}', value):
        return 'day'
    if re.fullmatch('\\d{4}', value):
        return 'year'
    if re.fullmatch('\\d{4}-\\d{2}', value):
        return 'month'
    match = re.fullmatch('(\\d{4})-(\\d{2})-(\\d{2})', value)
    if not match:
        return 'unknown'
    (year, month, day) = match.groups()
    if month == '00' and day == '00':
        return 'year'
    if month != '00' and day == '00':
        try:
            date(int(year), int(month), 1)
        except ValueError:
            return 'unknown'
        return 'month'
    if month != '00' and day != '00':
        try:
            date(int(year), int(month), int(day))
        except ValueError:
            return 'unknown'
        return 'day'
    return 'unknown'

def normalize_time_value(value: str, granularity: str) -> str:
    granularity = validate_time_granularity(granularity)
    value = str(value or '').strip()
    if not value:
        return ''
    if re.fullmatch('\\d{8}', value):
        value = datetime.strptime(value, '%Y%m%d').date().isoformat()
    precision = infer_time_value_precision(value)
    if granularity == 'day':
        if precision != 'day':
            raise ValueError(f'{value!r} is not day-precision')
        return parse_date(value).isoformat()
    if granularity == 'month':
        if precision == 'day':
            return parse_date(value).strftime('%Y-%m')
        if precision == 'month':
            return value[:7]
        raise ValueError(f'{value!r} is not at least month-precision')
    if precision in {'day', 'month', 'year'}:
        return value[:4]
    raise ValueError(f'{value!r} is not at least year-precision')

def effective_time_precision(value: str, precision: str) -> str:
    precision = str(precision or '').strip().lower()
    inferred = infer_time_value_precision(value)
    if precision in {'day', 'day_refined', 'day_verified'}:
        return 'day' if inferred == 'day' else 'unknown'
    if precision == 'month':
        return 'month' if inferred in {'day', 'month'} else 'unknown'
    if precision == 'year':
        return 'year' if inferred in {'day', 'month', 'year'} else 'unknown'
    if precision == 'coarse' and inferred in {'month', 'year'}:
        return inferred
    return inferred

def time_precision_satisfies(value: str, precision: str, granularity: str) -> bool:
    granularity = validate_time_granularity(granularity)
    if not str(value or '').strip():
        return False
    effective = effective_time_precision(value, precision)
    return effective in TIME_GRANULARITY_RANK and TIME_GRANULARITY_RANK[effective] >= TIME_GRANULARITY_RANK[granularity]

def time_bucket_bounds(value: str, granularity: str) -> Tuple[str, str]:
    granularity = validate_time_granularity(granularity)
    normalized = normalize_time_value(value, granularity)
    if granularity == 'day':
        return (normalized, normalized)
    if granularity == 'month':
        (year, month) = map(int, normalized.split('-'))
        last_day = calendar.monthrange(year, month)[1]
        return (f'{year:04d}-{month:02d}-01', f'{year:04d}-{month:02d}-{last_day:02d}')
    year = int(normalized)
    return (f'{year:04d}-01-01', f'{year:04d}-12-31')

def compare_time_values(left: str, right: str, granularity: str) -> int:
    left_start = time_bucket_bounds(left, granularity)[0]
    right_start = time_bucket_bounds(right, granularity)[0]
    return (left_start > right_start) - (left_start < right_start)

def time_values_overlap(left_start: str, left_end: str, right_start: str, right_end: str, granularity: str) -> bool:
    granularity = validate_time_granularity(granularity)
    l_start = '0001-01-01' if not left_start else time_bucket_bounds(left_start, granularity)[0]
    l_end = '9999-12-31' if not left_end else time_bucket_bounds(left_end, granularity)[1]
    r_start = time_bucket_bounds(right_start, granularity)[0]
    r_end = time_bucket_bounds(right_end, granularity)[1]
    return l_start <= r_end and l_end >= r_start

def time_value_in_window(value: str, window_start: str, window_end: str, granularity: str) -> bool:
    return time_values_overlap(value, value, window_start, window_end, granularity)

def shift_time_value(value: str, steps: int, granularity: str) -> str:
    granularity = validate_time_granularity(granularity)
    normalized = normalize_time_value(value, granularity)
    if granularity == 'day':
        return (parse_date(normalized) + timedelta(days=steps)).isoformat()
    if granularity == 'month':
        (year, month) = map(int, normalized.split('-'))
        month_index = year * 12 + month - 1 + steps
        (new_year, month_zero) = divmod(month_index, 12)
        return f'{new_year:04d}-{month_zero + 1:02d}'
    return f'{int(normalized) + steps:04d}'

def date_is_granularity_boundary(value: str, granularity: str, boundary: str) -> bool:
    granularity = validate_time_granularity(granularity)
    current = parse_date(value)
    if granularity == 'day':
        return True
    if granularity == 'month':
        if boundary == 'start':
            return current.day == 1
        return current.day == calendar.monthrange(current.year, current.month)[1]
    if boundary == 'start':
        return current.month == 1 and current.day == 1
    return current.month == 12 and current.day == 31

def validate_time_window_boundary_alignment(window_start: str, window_end: str, granularity: str, label: str) -> None:
    granularity = validate_time_granularity(granularity)
    if parse_date(window_start) > parse_date(window_end):
        raise ValueError(f'{label} window start must be <= end')
    if not date_is_granularity_boundary(window_start, granularity, 'start') or not date_is_granularity_boundary(window_end, granularity, 'end'):
        raise ValueError(f'{label} window must align to {granularity} boundary: {window_start}..{window_end}')

def split_history_dates(history_dates: Sequence[str], valid_length_days: int) -> Tuple[List[str], List[str]]:
    dates = list(history_dates)
    if valid_length_days <= 0:
        return (dates, [])
    if valid_length_days >= len(dates):
        return ([], dates)
    return (dates[:-valid_length_days], dates[-valid_length_days:])

def validate_pipeline_time_window_alignment(config: PipelineConfig, history_dates: Sequence[str], test_dates: Sequence[str]) -> None:
    granularity = validate_time_granularity(config.dataset_time_granularity)
    (train_dates, valid_dates) = split_history_dates(history_dates, config.valid_length_days)
    if granularity == 'year' and valid_dates:
        validate_time_window_boundary_alignment(history_dates[0], history_dates[-1], 'year', 'history')
        validate_time_window_boundary_alignment(list(test_dates)[0], list(test_dates)[-1], 'year', 'test')
        validate_time_window_boundary_alignment(valid_dates[0], valid_dates[-1], 'month', 'valid')
        if valid_dates[0][:7] != valid_dates[-1][:7]:
            raise ValueError(f'valid window must be exactly one natural month for year-granularity raw validation holdout: {valid_dates[0]}..{valid_dates[-1]}')
        if train_dates:
            validate_time_window_boundary_alignment(train_dates[0], train_dates[-1], 'month', 'train raw')
        return
    for (label, dates) in (('train', train_dates), ('valid', valid_dates), ('test', list(test_dates))):
        if dates:
            validate_time_window_boundary_alignment(dates[0], dates[-1], granularity, label)

def normalize_time_window_from_dates(dates: Sequence[str], granularity: str, default: Optional[Tuple[str, str]]=None) -> Tuple[str, str]:
    dates = list(dates)
    if not dates:
        if default is None:
            raise ValueError('Cannot build a time window from an empty date sequence without a default')
        return default
    return (normalize_time_value(dates[0], granularity), normalize_time_value(dates[-1], granularity))

def event_or_date_window(events: Sequence[Dict[str, Any]], fallback_dates: Sequence[str], granularity: str, default: Optional[Tuple[str, str]]=None) -> Tuple[str, str]:
    event_dates = sorted({row.get('date') for row in events if row.get('date')})
    if event_dates:
        return (event_dates[0], event_dates[-1])
    return normalize_time_window_from_dates(fallback_dates, granularity, default)

def infer_row_time_granularity(row: Dict[str, Any], default: str='day') -> str:
    explicit = row.get('time_granularity')
    if explicit:
        return validate_time_granularity(explicit)
    for key in ('t', 't_start', 't_end', 'date'):
        inferred = infer_time_value_precision(row.get(key, ''))
        if inferred in VALID_TIME_GRANULARITIES:
            return inferred
    return validate_time_granularity(default)

def date_to_gdelt(value: str) -> str:
    return parse_date(value).strftime('%Y%m%d')

def iso_date(value: str) -> str:
    return parse_date(value).isoformat()

def date_range_inclusive(start_date: str, end_date: str) -> List[str]:
    start = parse_date(start_date)
    end = parse_date(end_date)
    if start > end:
        raise ValueError('start_date must be <= end_date')
    days = (end - start).days
    return [(start + timedelta(days=i)).isoformat() for i in range(days + 1)]

def empty_gdelt_manifest() -> Dict[str, Dict[str, str]]:
    return {'daily': {}, 'monthly': {}, 'yearly': {}}

def parse_gdelt_manifest_html(html: str) -> Dict[str, Dict[str, str]]:
    manifest = empty_gdelt_manifest()
    files = re.findall('HREF="([^"]+\\.zip)"', html)
    for filename in files:
        if re.fullmatch('\\d{8}\\.export\\.CSV\\.zip', filename):
            manifest['daily'][filename[:8]] = filename
        elif re.fullmatch('\\d{6}\\.zip', filename):
            manifest['monthly'][filename[:6]] = filename
        elif re.fullmatch('\\d{4}\\.zip', filename):
            manifest['yearly'][filename[:4]] = filename
    return manifest

def load_gdelt_file_manifest(config: PipelineConfig) -> Dict[str, Dict[str, str]]:
    cache_path = cache_file_path(config, config.gdelt_manifest_cache_file)
    if config.use_gdelt_manifest_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding='utf-8'))
            if all((key in cached for key in ('daily', 'monthly', 'yearly'))):
                return {key: dict(cached.get(key, {})) for key in ('daily', 'monthly', 'yearly')}
        except Exception:
            pass
    response = requests.get(GDELT_EVENTS_BASE, timeout=30, headers={'User-Agent': config.user_agent})
    response.raise_for_status()
    manifest = parse_gdelt_manifest_html(response.text)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    return manifest

def discover_latest_gdelt_date(config: PipelineConfig) -> str:
    manifest = load_gdelt_file_manifest(config)
    dates = sorted(manifest.get('daily', {}))
    if not dates:
        raise RuntimeError('Could not find daily GDELT export files on the events index page')
    return datetime.strptime(dates[-1], '%Y%m%d').date().isoformat()

def load_official_mappings(config: PipelineConfig) -> Tuple[Dict[str, str], set, Dict[str, str]]:
    headers = {'User-Agent': config.user_agent}
    lookup_cache_dir = Path(config.cache_dir) / 'gdelt_lookup_cache'
    lookup_cache_dir.mkdir(parents=True, exist_ok=True)
    max_retries = max(1, int(getattr(config, 'gdelt_download_max_retries', 3) or 1))
    backoff = float(getattr(config, 'gdelt_download_retry_backoff_seconds', 5.0) or 0.0)

    def parse_lookup_text(text: str) -> Dict[str, str]:
        mapping = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t', 1)
            if len(parts) < 2:
                parts = re.split('\\s+', line, maxsplit=1)
            if len(parts) >= 2:
                mapping[parts[0].strip()] = parts[1].strip()
        return mapping

    def load_lookup(filename: str) -> Dict[str, str]:
        cache_path = lookup_cache_dir / filename
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return parse_lookup_text(cache_path.read_text(encoding='utf-8', errors='replace'))
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(LOOKUP_BASE + filename, timeout=30, headers=headers)
                response.raise_for_status()
                text = response.text
                cache_path.write_text(text, encoding='utf-8')
                return parse_lookup_text(text)
            except Exception as exc:
                last_error = exc
                if attempt < max_retries and backoff:
                    time.sleep(backoff * attempt)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return parse_lookup_text(cache_path.read_text(encoding='utf-8', errors='replace'))
        raise RuntimeError(f'Failed to load GDELT lookup {filename}') from last_error
    country_map = load_lookup('CAMEO.country.txt')
    group_map = load_lookup('CAMEO.knowngroup.txt')
    event_map = load_lookup('CAMEO.eventcodes.txt')
    entity_name_map = dict(country_map)
    entity_name_map.update(group_map)
    return (entity_name_map, set(entity_name_map), event_map)

def select_gdelt_file_for_date(date_str: str, manifest: Dict[str, Dict[str, str]], config: Optional[PipelineConfig]=None) -> Optional[str]:
    policy = getattr(config, 'gdelt_source_policy', 'auto') if config is not None else 'auto'
    current = parse_date(date_str)
    ymd = current.strftime('%Y%m%d')
    ym = current.strftime('%Y%m')
    year = current.strftime('%Y')
    if policy == 'daily_only':
        return manifest.get('daily', {}).get(ymd)
    if policy == 'auto':
        return manifest.get('daily', {}).get(ymd) or manifest.get('monthly', {}).get(ym) or manifest.get('yearly', {}).get(year)
    if policy == 'monthly_yearly':
        return manifest.get('monthly', {}).get(ym) or manifest.get('yearly', {}).get(year)
    raise ValueError(f'Unsupported gdelt_source_policy: {policy}')

def build_gdelt_download_plans(dates: Sequence[str], manifest: Dict[str, Dict[str, str]], config: Optional[PipelineConfig]=None) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[str]] = {}
    missing = []
    for current_date in sorted(set((iso_date(value) for value in dates))):
        filename = select_gdelt_file_for_date(current_date, manifest, config)
        if not filename:
            missing.append(current_date)
            continue
        grouped.setdefault(filename, []).append(current_date)
    for current_date in missing:
        print(f'[GDELT] No available source file for {current_date}; skipped before download', flush=True)
    return [{'filename': filename, 'dates': grouped[filename]} for filename in grouped]

def extracted_gdelt_csv_path(filename: str, config: PipelineConfig) -> Optional[Path]:
    extract_dir = Path(config.gdelt_raw_dir) / filename.replace('.zip', '')
    if not extract_dir.exists():
        return None
    raw_files = sorted([p for p in extract_dir.rglob('*') if p.is_file() and p.suffix.lower() in {'.csv', '.txt'}])
    if raw_files:
        return raw_files[0]
    legacy = Path(config.gdelt_raw_dir) / filename.replace('.zip', '')
    if legacy.exists() and legacy.is_file():
        return legacy
    return None

def download_gdelt_source_file(filename: str, config: PipelineConfig) -> Optional[Path]:
    raw_dir = Path(config.gdelt_raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cached_csv = extracted_gdelt_csv_path(filename, config)
    if config.use_gdelt_raw_cache and cached_csv and cached_csv.exists():
        print(f'[GDELT] Raw cache hit: {filename} -> {cached_csv}', flush=True)
        return cached_csv
    zip_path = raw_dir / filename
    extract_dir = raw_dir / filename.replace('.zip', '')
    url = f'{GDELT_EVENTS_BASE}{filename}'
    max_retries = max(1, int(getattr(config, 'gdelt_download_max_retries', 3) or 1))
    backoff = float(getattr(config, 'gdelt_download_retry_backoff_seconds', 5.0) or 0.0)
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=120, headers={'User-Agent': config.user_agent})
            response.raise_for_status()
            zip_path.write_bytes(response.content)
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            zip_path.unlink(missing_ok=True)
            return extracted_gdelt_csv_path(filename, config)
        except Exception as exc:
            last_exc = exc
            zip_path.unlink(missing_ok=True)
            if attempt < max_retries:
                print(f'[GDELT] Download retry {attempt}/{max_retries} for {filename}: {type(exc).__name__}: {exc}', flush=True)
                if backoff:
                    time.sleep(backoff * attempt)
            else:
                print(f'GDELT download failed for {filename}: {last_exc}', flush=True)
    return None

def download_raw_gdelt_csv(date_str: str, config: PipelineConfig) -> Optional[Path]:
    gdelt_date = date_to_gdelt(date_str)
    legacy_csv = Path(config.gdelt_raw_dir) / f'{gdelt_date}.export.CSV'
    if config.use_gdelt_raw_cache and legacy_csv.exists():
        print(f'[GDELT] Raw cache hit: {legacy_csv}', flush=True)
        return legacy_csv
    manifest = load_gdelt_file_manifest(config)
    filename = select_gdelt_file_for_date(iso_date(date_str), manifest, config)
    if not filename:
        print(f'[GDELT] No available source file for {iso_date(date_str)}; skipped before download', flush=True)
        return None
    return download_gdelt_source_file(filename, config)

def load_conceptual_entity_blacklist(path: str | None=None) -> set:
    blacklist_path = Path(path) if path else Path(__file__).with_name('concept_entity_blacklist.txt')
    if not blacklist_path.exists():
        raise FileNotFoundError(f'Concept entity blacklist file not found: {blacklist_path}')
    text = blacklist_path.read_text(encoding='utf-8')
    values = {value.strip() for value in re.findall('"([^"]+)"', text) if value.strip()}
    text_without_quoted_values = re.sub('"[^"]+"\\s*,?', '', text)
    for line in text_without_quoted_values.splitlines():
        line = line.split('#', 1)[0].strip().strip(',')
        if line:
            values.add(line)
    if not values:
        raise ValueError(f'Concept entity blacklist file is empty: {blacklist_path}')
    return values
GENERIC_ENTITY_TERMS = {'ARMED POLICE', 'ARMED GROUP', 'ARMED MEN', 'AID GROUP', 'AMERICAN TROOPS', 'FOREIGN MINISTER', 'LAWMAKER', 'SENATOR', 'WEBSITE', 'NEWS AGENCY', 'PUBLICATION', 'GAZETTE', 'LAW ENFORCEMENT AUTHORITIES', 'MILITARY JUNTA', 'MILITARY TRIBUNAL', 'MILITARY WING', 'INSPECTOR GENERAL', 'PUBLIC SECTOR', 'BORDER SECURITY', 'CABINET MEETING', 'A CABINET MEETING', 'MALE', 'RETIRED', 'GARRISON', 'MIDWIFE', 'PIRATE', 'PEDOPHILE', 'SOVEREIGN', 'RESERVIST', 'TURNCOAT', 'UNDERSECRETARY'}
GENERIC_ENTITY_SUFFIXES = (' GROUP', ' MEN', ' TROOPS', ' AUTHORITIES', ' MINISTRY', ' COURT', ' AGENCY', ' WEBSITE', ' SECTOR', ' TRIBUNAL', ' COMMISSION', ' COALITION', ' INDUSTRY')
ARTICLE_COUNTRY_ALIASES = {('A US', 'USA'): 'USA', ('A U.S.', 'USA'): 'USA', ('AN US', 'USA'): 'USA', ('AUSSIE', 'AUS'): 'AUS'}

def normalize_entity_text(name: str) -> str:
    return re.sub('\\s+', ' ', str(name or '').strip()).upper()
_NORMALIZED_CONCEPTUAL_BLACKLIST_CACHE: Dict[int, set] = {}

def normalized_conceptual_blacklist(conceptual_entities: set) -> set:
    key = id(conceptual_entities)
    cached = _NORMALIZED_CONCEPTUAL_BLACKLIST_CACHE.get(key)
    if cached is None:
        cached = {normalize_entity_text(item) for item in conceptual_entities}
        _NORMALIZED_CONCEPTUAL_BLACKLIST_CACHE[key] = cached
    return cached

def normalize_relation_label(label: str) -> str:
    label = str(label or '').strip()
    label = re.sub('matyerial', 'material', label, flags=re.IGNORECASE)
    label = re.sub('\\s*,?\\s*not\\s+specified\\s+below\\s*', '', label, flags=re.IGNORECASE)
    label = re.sub('\\s*,\\s*', ', ', label)
    label = re.sub('(?:,\\s*)+', ', ', label)
    label = re.sub('\\s+', ' ', label).strip(' ,')
    return label

def looks_like_generic_entity(name: str) -> bool:
    raw = str(name or '').strip()
    norm = normalize_entity_text(raw)
    if not norm:
        return True
    if norm in GENERIC_ENTITY_TERMS:
        return True
    if any((norm.endswith(suffix) for suffix in GENERIC_ENTITY_SUFFIXES)):
        return True
    if re.fullmatch('A\\s+[A-Z]{2,4}', norm):
        return True
    if raw == raw.lower() and len(norm.split()) <= 3:
        return True
    return False

def normalize_gdelt_actor(row: Sequence[str], side: str, entity_name_map: Dict[str, str], conceptual_entities: set) -> Optional[str]:
    if side == 'actor1':
        (code_idx, name_idx, country_idx, type_start) = (5, 6, 7, 12)
    elif side == 'actor2':
        (code_idx, name_idx, country_idx, type_start) = (15, 16, 17, 22)
    else:
        raise ValueError(f'Unsupported actor side: {side}')
    code = row[code_idx].strip() if len(row) > code_idx else ''
    name = row[name_idx].strip() if len(row) > name_idx else ''
    country = row[country_idx].strip() if len(row) > country_idx else ''
    actor_types = {row[i].strip() for i in range(type_start, min(type_start + 3, len(row))) if row[i].strip()}
    norm_name = normalize_entity_text(name)
    alias_country = ARTICLE_COUNTRY_ALIASES.get((norm_name, country))
    if alias_country and alias_country in entity_name_map:
        return entity_name_map[alias_country]
    if norm_name in conceptual_entities or norm_name in GENERIC_ENTITY_TERMS:
        return None
    if code in conceptual_entities or code in {'UAF'}:
        return None
    if looks_like_generic_entity(name):
        if country and country in entity_name_map and code.startswith(country) and (not actor_types):
            return entity_name_map[country]
        return None
    if not is_specific_entity(name, conceptual_entities):
        return None
    return name

def normalize_gdelt_actor_cached(row: Sequence[str], side: str, entity_name_map: Dict[str, str], conceptual_entities: set, actor_cache: Dict[Tuple[Any, ...], Optional[str]]) -> Optional[str]:
    if side == 'actor1':
        (code_idx, name_idx, country_idx, type_start) = (5, 6, 7, 12)
    elif side == 'actor2':
        (code_idx, name_idx, country_idx, type_start) = (15, 16, 17, 22)
    else:
        raise ValueError(f'Unsupported actor side: {side}')
    key = (side, row[code_idx].strip() if len(row) > code_idx else '', row[name_idx].strip() if len(row) > name_idx else '', row[country_idx].strip() if len(row) > country_idx else '', tuple((row[i].strip() for i in range(type_start, min(type_start + 3, len(row))) if row[i].strip())))
    if key not in actor_cache:
        actor_cache[key] = normalize_gdelt_actor(row, side, entity_name_map, conceptual_entities)
    return actor_cache[key]

def is_specific_entity(name: str, conceptual_entities: set) -> bool:
    raw = str(name or '').strip()
    norm = normalize_entity_text(raw)
    if not norm:
        return False
    normalized_blacklist = normalized_conceptual_blacklist(conceptual_entities)
    if norm in normalized_blacklist:
        return False
    if raw.endswith('ER'):
        return False
    if looks_like_generic_entity(raw):
        return False
    tokens = [token for token in norm.split() if token]
    if tokens and all((token in normalized_blacklist for token in tokens)):
        return False
    return True

def is_gdelt_reduced_file(path: Path) -> bool:
    text = str(path).upper()
    return 'MASTERREDUCED' in text or Path(path).suffix.lower() == '.txt'

def normalize_gdelt_reduced_actor(code: str, entity_name_map: Dict[str, str], conceptual_entities: set) -> Optional[str]:
    code = str(code or '').strip()
    if not code:
        return None
    norm_code = normalize_entity_text(code)
    normalized_blacklist = normalized_conceptual_blacklist(conceptual_entities)
    if norm_code in normalized_blacklist or norm_code in GENERIC_ENTITY_TERMS:
        return None
    if code in entity_name_map:
        candidate = entity_name_map[code]
    elif len(code) > 3 and code[:3] in entity_name_map:
        candidate = entity_name_map[code[:3]]
    else:
        return None
    if not candidate or normalize_entity_text(candidate) in normalized_blacklist:
        return None
    return candidate

def parse_reduced_gdelt_events_for_dates(target_dates: Sequence[str], csv_path: Path, event_map: Dict[str, str], conceptual_entities: set, entity_name_map: Optional[Dict[str, str]]=None) -> List[Dict[str, str]]:
    entity_name_map = entity_name_map or {}
    gdelt_to_iso = {date_to_gdelt(value): iso_date(value) for value in target_dates}
    min_target_date = min(gdelt_to_iso) if gdelt_to_iso else ''
    max_target_date = max(gdelt_to_iso) if gdelt_to_iso else ''
    seen = set()
    events = []
    total_bytes = csv_path.stat().st_size if Path(csv_path).exists() else None
    desc = f'GDELT reduced rows {Path(csv_path).name}'
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f, tqdm(total=total_bytes, desc=desc, unit='B', unit_scale=True) as progress:
        for line in f:
            progress.update(len(line.encode('utf-8', errors='replace')))
            row = line.rstrip('\n').split('\t')
            if not row or row[0] == 'Date' or len(row) < 4:
                continue
            row_date = row[0].strip()
            if max_target_date and row_date > max_target_date:
                break
            if min_target_date and row_date < min_target_date:
                continue
            if row_date not in gdelt_to_iso:
                continue
            source_name = normalize_gdelt_reduced_actor(row[1].strip(), entity_name_map, conceptual_entities)
            target_name = normalize_gdelt_reduced_actor(row[2].strip(), entity_name_map, conceptual_entities)
            event_code = row[3].strip()
            if event_code not in event_map:
                continue
            if not source_name or not target_name or source_name == target_name:
                continue
            relation_label = normalize_relation_label(event_map[event_code])
            if not relation_label:
                continue
            item = (source_name, relation_label, target_name, gdelt_to_iso[row_date])
            if item in seen:
                continue
            seen.add(item)
            events.append({'source_name': item[0], 'relation_label': item[1], 'target_name': item[2], 'date': item[3]})
    return events

def parse_events_from_raw_csv_for_dates(target_dates: Sequence[str], csv_path: Path, event_map: Dict[str, str], conceptual_entities: set, entity_name_map: Optional[Dict[str, str]]=None) -> List[Dict[str, str]]:
    if is_gdelt_reduced_file(csv_path):
        return parse_reduced_gdelt_events_for_dates(target_dates, csv_path, event_map, conceptual_entities, entity_name_map)
    entity_name_map = entity_name_map or {}
    gdelt_to_iso = {date_to_gdelt(value): iso_date(value) for value in target_dates}
    seen = set()
    events = []
    actor_cache: Dict[Tuple[Any, ...], Optional[str]] = {}
    total_bytes = csv_path.stat().st_size if Path(csv_path).exists() else None
    desc = f'GDELT rows {Path(csv_path).name}'
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f, tqdm(total=total_bytes, desc=desc, unit='B', unit_scale=True) as progress:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row:
                progress.update(sum((len(value) for value in row)) + max(0, len(row) - 1) + 1)
            if len(row) < 27:
                continue
            row_date = row[1].strip()
            if row_date not in gdelt_to_iso:
                continue
            event_code = row[26].strip()
            if event_code not in event_map:
                continue
            source_name = normalize_gdelt_actor_cached(row, 'actor1', entity_name_map, conceptual_entities, actor_cache)
            target_name = normalize_gdelt_actor_cached(row, 'actor2', entity_name_map, conceptual_entities, actor_cache)
            if not source_name or not target_name or source_name == target_name:
                continue
            relation_label = normalize_relation_label(event_map[event_code])
            if not relation_label:
                continue
            item = (source_name, relation_label, target_name, gdelt_to_iso[row_date])
            if item in seen:
                continue
            seen.add(item)
            events.append({'source_name': item[0], 'relation_label': item[1], 'target_name': item[2], 'date': item[3]})
    return events

def parse_events_from_raw_csv(date_str: str, csv_path: Path, event_map: Dict[str, str], conceptual_entities: set, entity_name_map: Optional[Dict[str, str]]=None) -> List[Dict[str, str]]:
    return parse_events_from_raw_csv_for_dates([date_str], csv_path, event_map, conceptual_entities, entity_name_map)

def collect_gdelt_events(dates: Sequence[str], config: PipelineConfig, event_map: Dict[str, str], conceptual_entities: set, entity_name_map: Optional[Dict[str, str]]=None) -> List[Dict[str, str]]:
    all_events = []
    print(f'[GDELT] Collecting events for {len(dates)} day(s)', flush=True)
    manifest = load_gdelt_file_manifest(config)
    plans = build_gdelt_download_plans(dates, manifest, config)
    print(f'[GDELT] Planned {len(plans)} source file(s) for {len(dates)} day(s)', flush=True)
    for plan in tqdm(plans, desc='GDELT source files', unit='file'):
        csv_path = download_gdelt_source_file(plan['filename'], config)
        if not csv_path:
            print(f"[GDELT] {plan['filename']}: skipped, raw source unavailable", flush=True)
            continue
        source_events = parse_events_from_raw_csv_for_dates(plan['dates'], csv_path, event_map, conceptual_entities, entity_name_map)
        print(f"[GDELT] {plan['filename']}: {len(source_events)} filtered event facts for {len(plan['dates'])} day(s)", flush=True)
        all_events.extend(source_events)
    sorted_events = sorted(all_events, key=lambda row: (row['date'], row['source_name'], row['relation_label'], row['target_name']))
    print(f'[GDELT] Finished: {len(sorted_events)} total filtered event facts', flush=True)
    return sorted_events

def dedupe_events(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for row in sorted(events, key=lambda x: (x['date'], x['source_name'], x['relation_label'], x['target_name'])):
        key = (row['source_name'], row['relation_label'], row['target_name'], row['date'])
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(row))
    return output

def split_history_events(events: Sequence[Dict[str, Any]], valid_length_days: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    events = dedupe_events(events)
    if valid_length_days <= 0:
        return (events, [])
    dates = sorted({row['date'] for row in events})
    if not dates:
        return ([], [])
    valid_dates = set(dates[-valid_length_days:])
    train = [row for row in events if row['date'] not in valid_dates]
    valid = [row for row in events if row['date'] in valid_dates]
    return (train, valid)

def split_history_events_by_date_windows(events: Sequence[Dict[str, Any]], train_dates: Sequence[str], valid_dates: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    events = dedupe_events(events)
    train_set = set(train_dates)
    valid_set = set(valid_dates)
    train = [row for row in events if row.get('date') in train_set]
    valid = [row for row in events if row.get('date') in valid_set]
    return (train, valid)

def build_test_window_dates(config: PipelineConfig, latest_gdelt_date: Optional[str]=None) -> List[str]:
    if config.test_length_days <= 0:
        raise ValueError('test_length_days must be positive')
    setting = validate_pipeline_setting(config)
    if setting == 'online_testing':
        if latest_gdelt_date is None:
            latest_gdelt_date = discover_latest_gdelt_date(config)
        end = parse_date(latest_gdelt_date)
        start = end - timedelta(days=config.test_length_days - 1)
    else:
        start = parse_date(config.history_end_date) + timedelta(days=1)
        end = start + timedelta(days=config.test_length_days - 1)
    return date_range_inclusive(start.isoformat(), end.isoformat())

def normalize_event_times(events: Sequence[Dict[str, Any]], config: PipelineConfig) -> List[Dict[str, Any]]:
    granularity = validate_time_granularity(config.dataset_time_granularity)
    normalized = []
    for row in events:
        updated = dict(row)
        updated['date'] = normalize_time_value(row['date'], granularity)
        updated['time_granularity'] = granularity
        normalized.append(updated)
    return dedupe_events(normalized)

def debias_test_facts(train_events: Sequence[Dict[str, Any]], test_events: Sequence[Dict[str, Any]], config: PipelineConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_sr_to_objects = defaultdict(set)
    train_s_to_objects = defaultdict(set)
    train_so = set()
    for row in train_events:
        (s, r, o) = (row['source_name'], row['relation_label'], row['target_name'])
        train_sr_to_objects[s, r].add(o)
        train_s_to_objects[s].add(o)
        train_so.add((s, o))
    by_date_triplets = defaultdict(set)
    kept = []
    filtered = []
    for row in dedupe_events(test_events):
        triplet = (row['source_name'], row['relation_label'], row['target_name'])
        reason = None
        for lookback in range(1, config.repeat_lookback_days + 1):
            previous = shift_time_value(row['date'], -lookback, config.dataset_time_granularity)
            if triplet in by_date_triplets.get(previous, set()):
                reason = 'repeat_within_lookback'
                break
        (s, r, o) = triplet
        if reason is None:
            objects = train_sr_to_objects.get((s, r), set())
            if 0 < len(objects) <= config.leak_sr_max_objects and o in objects:
                reason = 'leak_sr_low_diversity'
        if reason is None:
            objects = train_s_to_objects.get(s, set())
            if 0 < len(objects) <= config.leak_s_max_objects and (s, o) in train_so:
                reason = 'leak_s_low_diversity'
        by_date_triplets[row['date']].add(triplet)
        if reason is None:
            kept.append(row)
        else:
            rejected = dict(row)
            rejected['reason'] = reason
            filtered.append(rejected)
    return (kept, filtered)

def debias_canonical_test_facts(train_events: Sequence[Dict[str, Any]], test_events: Sequence[Dict[str, Any]], config: PipelineConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    (kept, filtered) = debias_test_facts(train_events, test_events, config)
    for row in filtered:
        row['filter_stage'] = 'canonical_post_filter'
    return (kept, filtered)
TIME_DESCRIPTION_PATTERNS = ['\\(\\s*\\d{4}\\s*[-–—]\\s*\\d{4}\\s*\\)', '\\(\\s*\\d{4}\\s*\\)', 'from\\s+\\d{4}\\s+to\\s+\\d{4}', 'since\\s+\\d{4}\\s+to\\s+\\d{4}', '\\d{4}\\s*[-–—]\\s*\\d{4}', 'born\\s+\\d{4}', '\\b\\d{4}\\s*-\\s*\\d{4}\\b', '\\b\\d{4}\\s*–\\s*\\d{4}\\b', '\\b\\d{4}\\s*—\\s*\\d{4}\\b']

def clean_description(description: str) -> str:
    cleaned = str(description or '')
    for pattern in TIME_DESCRIPTION_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub('\\s+', ' ', cleaned)
    cleaned = re.sub('\\s*,\\s*,', ',', cleaned)
    cleaned = re.sub('^\\s*[,-]\\s*', '', cleaned)
    cleaned = re.sub('\\s*[,-]\\s*$', '', cleaned)
    return cleaned.strip().replace('()', '')

def wikidata_search_entity(entity_name: str, config: PipelineConfig, limit: int=5) -> Dict[str, Any]:
    params = {'action': 'wbsearchentities', 'search': entity_name, 'language': 'en', 'format': 'json', 'limit': limit}
    best = {'original_text': entity_name, 'qid': '', 'wikidata_label': '', 'short_description': '', 'match_score': 0.0}
    try:
        data = wikidata_get_json(params, config, 'entity_search', [entity_name])
    except Exception as exc:
        record_wikidata_failed_batch(config, 'entity_search', [entity_name], exc, getattr(config, 'wikidata_max_retries', 5))
        if getattr(config, 'wikidata_continue_on_batch_error', True):
            return best
        raise
    for result in data.get('search', []):
        label = result.get('label', '')
        score = SequenceMatcher(None, entity_name.lower(), label.lower()).ratio() if label else 0.0
        reverse_score = SequenceMatcher(None, label.lower(), entity_name.lower()).ratio() if label else 0.0
        final_score = min(score, reverse_score)
        if final_score > best['match_score']:
            best = {'original_text': entity_name, 'qid': result.get('id', ''), 'wikidata_label': label, 'short_description': clean_description(result.get('description', '')), 'match_score': float(final_score)}
    return best

def fetch_wikidata_labels(entity_ids: Sequence[str], config: PipelineConfig) -> Dict[str, Dict[str, str]]:
    output = {}
    valid_ids = [qid for qid in entity_ids if qid and re.fullmatch('[QP]\\d+', qid)]
    print(f'[Wikidata] Fetching labels/descriptions for {len(valid_ids)} id(s)', flush=True)
    for start in tqdm(range(0, len(valid_ids), config.wikidata_batch_size), desc='Wikidata label batches', unit='batch'):
        batch = valid_ids[start:start + config.wikidata_batch_size]
        params = {'action': 'wbgetentities', 'ids': '|'.join(batch), 'format': 'json', 'languages': 'en', 'props': 'labels|descriptions'}
        try:
            data = wikidata_get_json(params, config, 'labels', batch)
        except Exception as exc:
            record_wikidata_failed_batch(config, 'labels', batch, exc, getattr(config, 'wikidata_max_retries', 5))
            if getattr(config, 'wikidata_continue_on_batch_error', True):
                continue
            raise
        for (qid, entity) in data.get('entities', {}).items():
            label = entity.get('labels', {}).get('en', {}).get('value', '')
            description = entity.get('descriptions', {}).get('en', {}).get('value', '')
            output[qid] = {'label': label, 'description': clean_description(description), 'original_description': description}
        if config.wikidata_delay_seconds:
            time.sleep(config.wikidata_delay_seconds)
    print(f'[Wikidata] Labels/descriptions fetched: {len(output)}', flush=True)
    return output

def is_high_confidence_wikidata_match(row: Dict[str, Any], threshold: float) -> bool:
    try:
        score = float(row.get('match_score', 0.0) or 0.0)
    except Exception:
        score = 0.0
    return bool(row.get('qid') and row.get('wikidata_label') and (score >= threshold))

def build_entity_context_examples(events: Sequence[Dict[str, Any]], entity_name: str, max_examples: int=8) -> List[Dict[str, str]]:
    examples = []
    seen = set()
    for row in events or []:
        source = str(row.get('source_name', ''))
        target = str(row.get('target_name', ''))
        if entity_name not in {source, target}:
            continue
        item = {'source_name': source, 'relation_label': str(row.get('relation_label', '')), 'target_name': target, 'date': str(row.get('date', row.get('t', '')))}
        key = (item['source_name'], item['relation_label'], item['target_name'])
        if key in seen:
            continue
        seen.add(key)
        examples.append(item)
        if len(examples) >= max_examples:
            break
    return examples

def build_entity_context_index(events: Sequence[Dict[str, Any]], max_examples: int=8) -> Dict[str, List[Dict[str, str]]]:
    index = defaultdict(list)
    seen_by_entity = defaultdict(set)
    for row in events or []:
        source = str(row.get('source_name', ''))
        target = str(row.get('target_name', ''))
        item = {'source_name': source, 'relation_label': str(row.get('relation_label', '')), 'target_name': target, 'date': str(row.get('date', row.get('t', '')))}
        key = (item['source_name'], item['relation_label'], item['target_name'])
        for entity_name in (source, target):
            if not entity_name or len(index[entity_name]) >= max_examples or key in seen_by_entity[entity_name]:
                continue
            seen_by_entity[entity_name].add(key)
            index[entity_name].append(item)
    return dict(index)

def llm_entity_name_cache_key(entity_name: str, context_examples: Sequence[Dict[str, str]], config: PipelineConfig) -> str:
    payload = {'prompt_version': config.openai_prompt_version, 'model': config.openai_model, 'base_url': config.openai_base_url, 'entity_name': entity_name, 'context_examples': list(context_examples)}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()

def infer_wikidata_candidate_names_with_llm(entity_name: str, context_examples: Sequence[Dict[str, str]], config: PipelineConfig) -> Dict[str, Any]:
    cache_path = cache_file_path(config, 'llm_entity_name_resolution_cache.jsonl')
    cache_key = llm_entity_name_cache_key(entity_name, context_examples, config)
    if config.use_wikidata_cache:
        cached = load_jsonl_cache(cache_path, 'cache_key').get(cache_key)
        if cached and (not cached.get('error')) and (not cached.get('skipped')):
            row = dict(cached)
            row['cache_hit'] = True
            return row
    example_lines = []
    for (idx, row) in enumerate(context_examples, 1):
        example_lines.append(f"{idx}. {row.get('source_name', '')} -- {row.get('relation_label', '')} --> {row.get('target_name', '')}, date={row.get('date', '')}")
    prompt = f'Given a noisy GDELT entity name and several observed interaction facts, infer the standard English name or likely Wikidata search names for that entity. Do not return a Wikidata QID. Return compact JSON only with canonical_name, candidate_names, entity_type, confidence as a number from 0 to 1, and rationale.\nGDELT entity name: {entity_name}\nObserved interaction facts:\n' + ('\n'.join(example_lines) if example_lines else 'No examples available.')
    row = openai_web_search_json(prompt, config, 'llm_entity_name_resolution_cache.jsonl')
    row['cache_key'] = cache_key
    row['entity_name'] = entity_name
    row['context_examples'] = list(context_examples)
    if config.use_wikidata_cache:
        upsert_jsonl_cache(cache_path, [row], 'cache_key')
    return row

def llm_candidate_names(llm_row: Dict[str, Any], original_name: str, max_names: int=5) -> List[str]:
    parsed = parse_llm_json_response(llm_row)
    candidates = []
    canonical = str(parsed.get('canonical_name', '')).strip() if parsed else ''
    if canonical:
        candidates.append(canonical)
    raw_names = parsed.get('candidate_names', []) if parsed else []
    if isinstance(raw_names, str):
        raw_names = [raw_names]
    if isinstance(raw_names, list):
        candidates.extend((str(name).strip() for name in raw_names))
    output = []
    seen = {original_name.strip().lower()}
    for name in candidates:
        key = name.lower()
        if name and key not in seen:
            output.append(name)
            seen.add(key)
        if len(output) >= max_names:
            break
    return output

def accepted_entity_mapping(original_name: str, row: Dict[str, Any], match_source: str, **extra: Any) -> Dict[str, Any]:
    output = dict(row)
    output['original_text'] = original_name
    output['status'] = 'accepted'
    output['match_source'] = match_source
    output.update(extra)
    return output

def rejected_entity_mapping(original_name: str, reason: str, first_pass: Optional[Dict[str, Any]]=None, **extra: Any) -> Dict[str, Any]:
    output = {'original_text': original_name, 'status': 'rejected', 'qid': '', 'wikidata_label': '', 'short_description': '', 'match_score': 0.0, 'reject_reason': reason}
    if first_pass is not None:
        output['first_pass'] = first_pass
    output.update(extra)
    return output
ENTITY_MAPPING_CACHE_FIELDS = ('cache_key', 'match_score', 'qid', 'short_description', 'wikidata_label', 'llm_entity_name_resolution_cache_key', 'llm_entity_relation_validation_cache_key', 'note')

def safe_match_score(row: Dict[str, Any]) -> float:
    try:
        return float(row.get('match_score', 0.0) or 0.0)
    except Exception:
        return 0.0

def collect_mapping_candidates(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = []
    for candidate in (row, row.get('first_pass') if isinstance(row.get('first_pass'), dict) else None):
        if isinstance(candidate, dict) and candidate.get('qid') and candidate.get('wikidata_label'):
            candidates.append(candidate)
    for field in ('second_pass_attempts', 'failed_candidates'):
        attempts = row.get(field)
        if isinstance(attempts, list):
            for candidate in attempts:
                if isinstance(candidate, dict) and candidate.get('qid') and candidate.get('wikidata_label'):
                    candidates.append(candidate)
    return candidates

def llm_cache_key_from_mapping(row: Dict[str, Any], embedded_field: str, canonical_field: str) -> str:
    if row.get(canonical_field):
        return str(row.get(canonical_field) or '')
    embedded = row.get(embedded_field)
    if isinstance(embedded, dict):
        return str(embedded.get('cache_key') or '')
    return ''

def canonical_entity_mapping_cache_row(cache_key: str, row: Dict[str, Any]) -> Dict[str, Any]:
    candidates = collect_mapping_candidates(row)
    best = max(candidates, key=safe_match_score) if candidates else None
    source = best or row
    relation_failed = row.get('reject_reason') == 'llm_relation_validation_failed' or (source.get('llm_relation_validated') and (not source.get('llm_relation_valid')))
    qid = str(source.get('qid') or '') if source else ''
    label = str(source.get('wikidata_label') or '') if source else ''
    description = str(source.get('short_description') or source.get('description') or '') if source else ''
    score = safe_match_score(source) if qid and label else 0.0
    note = ''
    if relation_failed:
        score = 0.0
        note = 'llm_relation_validation_failed'
    elif not qid or not label:
        score = 0.0
        note = 'no_wikidata_match'
    name_key = llm_cache_key_from_mapping(row, 'llm_name_resolution', 'llm_entity_name_resolution_cache_key')
    relation_key = llm_cache_key_from_mapping(source, 'llm_relation_validation', 'llm_entity_relation_validation_cache_key')
    if not relation_key:
        relation_key = llm_cache_key_from_mapping(row, 'llm_relation_validation', 'llm_entity_relation_validation_cache_key')
    return {'cache_key': str(cache_key), 'match_score': float(score), 'qid': qid, 'short_description': description, 'wikidata_label': label, 'llm_entity_name_resolution_cache_key': name_key, 'llm_entity_relation_validation_cache_key': relation_key, 'note': note}

def write_jsonl_cache(path: Path, rows: Sequence[Dict[str, Any]], key_field: str='cache_key') -> None:
    path = Path(path)
    with jsonl_cache_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        latest = {}
        for row in rows:
            key = row.get(key_field)
            if key is not None and key != '':
                latest[str(key)] = row
        temp_path = path.with_name(f'{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            for key in sorted(latest):
                f.write(json.dumps(latest[key], ensure_ascii=False, sort_keys=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)

def upsert_jsonl_cache(path: Path, rows: Sequence[Dict[str, Any]], key_field: str='cache_key') -> None:
    path = Path(path)
    with jsonl_cache_lock(path):
        existing = load_jsonl_cache(path, key_field) if path.exists() else {}
        for row in rows:
            key = row.get(key_field)
            if key is not None and key != '':
                existing[str(key)] = dict(row)
        write_jsonl_cache(path, list(existing.values()), key_field)

def write_entity_mapping_cache(path: Path, cache_rows: Dict[str, Dict[str, Any]]) -> None:
    canonical_rows = [canonical_entity_mapping_cache_row(key, row) for (key, row) in cache_rows.items()]
    write_jsonl_cache(path, canonical_rows, 'cache_key')

def upsert_entity_mapping_cache_row(path: Path, entity_cache: Dict[str, Dict[str, Any]], cache_key: str, row: Dict[str, Any]) -> None:
    canonical = canonical_entity_mapping_cache_row(cache_key, row)
    entity_cache[cache_key] = canonical
    upsert_jsonl_cache(path, [canonical], 'cache_key')

def mapping_row_from_entity_cache(row: Dict[str, Any], original_name: str) -> Dict[str, Any]:
    output = dict(row)
    output['original_text'] = original_name
    if output.get('qid') and output.get('wikidata_label') and (safe_match_score(output) > 0):
        output['status'] = 'accepted'
        output['match_source'] = 'cache_best_match'
    else:
        output['status'] = 'rejected'
        output['reject_reason'] = output.get('note') or 'no_wikidata_match'
    if output.get('llm_entity_relation_validation_cache_key'):
        output['llm_relation_validated'] = True
        output['llm_relation_valid'] = output.get('note') != 'llm_relation_validation_failed' and safe_match_score(output) > 0
    return output

def cache_candidate_score(row: Dict[str, Any]) -> float:
    return safe_match_score(row)

def best_mapping_candidate_from_cache(row: Dict[str, Any], original_name: str) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    if set(ENTITY_MAPPING_CACHE_FIELDS).issubset(set(row.keys())):
        return mapping_row_from_entity_cache(row, original_name)
    candidates = collect_mapping_candidates(row)
    if not candidates:
        return None
    best = max(candidates, key=cache_candidate_score)
    output = dict(best)
    output['original_text'] = original_name
    output['status'] = 'accepted'
    output['match_source'] = 'cache_best_match'
    if 'short_description' not in output and row.get('short_description'):
        output['short_description'] = row.get('short_description')
    output['cache_source_status'] = row.get('status', '')
    if row.get('llm_entity_name_resolution_cache_key'):
        output['llm_entity_name_resolution_cache_key'] = row.get('llm_entity_name_resolution_cache_key')
    if row.get('llm_entity_relation_validation_cache_key'):
        output['llm_entity_relation_validation_cache_key'] = row.get('llm_entity_relation_validation_cache_key')
    return output

def terminal_mapping_from_cache(row: Dict[str, Any], original_name: str, threshold: float) -> Optional[Dict[str, Any]]:
    cached = best_mapping_candidate_from_cache(row, original_name)
    if cached is None:
        return None
    output = dict(cached)
    output['original_text'] = original_name
    output['cache_hit'] = True
    if output.get('qid') and output.get('wikidata_label') and (safe_match_score(output) >= threshold):
        output['status'] = 'accepted'
        output.setdefault('match_source', 'cache_best_match')
    else:
        output['status'] = 'rejected'
        if not output.get('reject_reason'):
            if output.get('note'):
                output['reject_reason'] = output.get('note')
            elif output.get('qid') and output.get('wikidata_label'):
                output['reject_reason'] = 'wikidata_match_below_threshold'
            else:
                output['reject_reason'] = 'no_wikidata_match'
    return output

def entity_relation_validation_cache_key(original_name: str, candidate: Dict[str, Any], context_examples: Sequence[Dict[str, str]], config: PipelineConfig) -> str:
    payload = {'prompt_version': config.openai_prompt_version, 'model': config.openai_model, 'base_url': config.openai_base_url, 'original_name': original_name, 'qid': candidate.get('qid', ''), 'wikidata_label': candidate.get('wikidata_label', ''), 'short_description': candidate.get('short_description', ''), 'context_examples': list(context_examples)}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()

def validate_entity_relation_with_llm(original_name: str, candidate: Dict[str, Any], context_examples: Sequence[Dict[str, str]], config: PipelineConfig) -> Dict[str, Any]:
    cache_path = cache_file_path(config, 'llm_entity_relation_validation_cache.jsonl')
    cache_key = entity_relation_validation_cache_key(original_name, candidate, context_examples, config)
    if config.use_wikidata_cache:
        cached = load_jsonl_cache(cache_path, 'cache_key').get(cache_key)
        if cached and (not cached.get('error')) and (not cached.get('skipped')):
            row = dict(cached)
            row['cache_hit'] = True
            return row
    example_lines = []
    for (idx, row) in enumerate(context_examples, 1):
        role = 'source' if row.get('source_name') == original_name else 'target'
        example_lines.append(f"{idx}. role={role}; {row.get('source_name', '')} -- {row.get('relation_label', '')} --> {row.get('target_name', '')}; date={row.get('date', '')}")
    prompt = f"Decide whether the matched Wikidata entity is a concrete instance that can plausibly participate in the observed GDELT relations. Reject conceptual names, creative works, domains, numbers, and other non-actors when the observed relation would be implausible. Return compact JSON only with is_valid as true or false and reason. Do not include a confidence or score field.\nGDELT entity name: {original_name}\nMatched Wikidata QID: {candidate.get('qid', '')}\nMatched Wikidata label: {candidate.get('wikidata_label', '')}\nMatched Wikidata description: {candidate.get('short_description', '')}\nObserved GDELT interaction facts:\n" + ('\n'.join(example_lines) if example_lines else 'No examples available.')
    row = openai_web_search_json(prompt, config, 'llm_entity_relation_validation_cache.jsonl')
    row['cache_key'] = cache_key
    row['entity_name'] = original_name
    row['candidate_qid'] = candidate.get('qid', '')
    row['candidate_label'] = candidate.get('wikidata_label', '')
    row['candidate_description'] = candidate.get('short_description', '')
    row['context_examples'] = list(context_examples)
    if config.use_wikidata_cache:
        upsert_jsonl_cache(cache_path, [row], 'cache_key')
    return row

def parse_llm_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {'true', 'yes', 'y', '1', 'valid', 'reasonable', 'plausible'}

def apply_entity_relation_validation(candidate: Dict[str, Any], original_name: str, context_examples: Sequence[Dict[str, str]], config: PipelineConfig) -> Dict[str, Any]:
    updated = dict(candidate)
    if not llm_entity_relation_validation_enabled(config):
        return updated
    if updated.get('llm_relation_validated'):
        return updated
    validation = validate_entity_relation_with_llm(original_name, updated, context_examples, config)
    parsed = parse_llm_json_response(validation)
    is_valid = parse_llm_boolean(parsed.get('is_valid')) if parsed else False
    updated['llm_relation_validation'] = validation
    updated['llm_relation_validated'] = True
    updated['llm_relation_valid'] = bool(is_valid)
    updated['llm_relation_entity_type'] = str(parsed.get('entity_type', '')) if parsed else ''
    updated['llm_relation_invalid_reason'] = str(parsed.get('invalid_reason', '')) if parsed else str(validation.get('reason', ''))
    return updated

def relation_validation_passed(candidate: Dict[str, Any], config: PipelineConfig) -> bool:
    if not llm_entity_relation_validation_enabled(config):
        return True
    if candidate.get('llm_relation_validated'):
        return bool(candidate.get('llm_relation_valid'))
    return bool(candidate.get('llm_relation_validation_status') == 'legacy_unvalidated')

def try_accept_entity_candidate(original_name: str, candidate: Dict[str, Any], match_source: str, context_examples: Sequence[Dict[str, str]], config: PipelineConfig, **extra: Any) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    candidate = dict(candidate or {})
    if not is_high_confidence_wikidata_match(candidate, config.wikidata_match_threshold):
        return (None, candidate)
    candidate = apply_entity_relation_validation(candidate, original_name, context_examples, config)
    if not relation_validation_passed(candidate, config):
        return (None, candidate)
    return (accepted_entity_mapping(original_name, candidate, match_source, **extra), candidate)

def match_entities_to_wikidata(entity_names: Sequence[str], config: PipelineConfig, events: Optional[Sequence[Dict[str, Any]]]=None) -> List[Dict[str, Any]]:
    unique_names = sorted(set(entity_names))
    cache_path = cache_file_path(config, 'wikidata_entity_mapping_cache.jsonl')
    entity_cache = load_jsonl_cache(cache_path, 'cache_key') if config.use_wikidata_cache else {}
    stats = defaultdict(int)
    results_by_name: Dict[str, Dict[str, Any]] = {}
    pending_names: List[str] = []
    print(f'[EntityResolver] Resolving {len(unique_names)} entity name(s)', flush=True)
    for name in tqdm(unique_names, desc='Entity resolver', unit='entity'):
        cached = entity_cache.get(name)
        cached_terminal = terminal_mapping_from_cache(cached, name, config.wikidata_match_threshold) if cached else None
        if cached_terminal is not None:
            results_by_name[name] = cached_terminal
            if cached_terminal.get('status') == 'accepted':
                stats['cache_best_accepted'] += 1
            else:
                stats['cache_best_rejected'] += 1
            continue
        if config.use_existing_entity_cache_only:
            results_by_name[name] = rejected_entity_mapping(name, 'missing_from_existing_entity_cache')
            stats['cache_only_rejected'] += 1
            continue
        pending_names.append(name)
    context_index = build_entity_context_index(events or []) if pending_names else {}
    if pending_names:
        stats['context_index_built'] += 1
    rate_lock = threading.Lock()
    last_worker_start = [0.0]

    def wait_for_new_entity_worker_slot() -> None:
        interval = float(getattr(config, 'llm_entity_resolution_request_interval_seconds', 0.0) or 0.0)
        if interval <= 0:
            return
        with rate_lock:
            elapsed = time.monotonic() - last_worker_start[0]
            if elapsed < interval:
                time.sleep(interval - elapsed)
            last_worker_start[0] = time.monotonic()

    def get_context(entity_name: str) -> List[Dict[str, str]]:
        return list(context_index.get(entity_name, []))

    def resolve_pending_entity(name: str) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
        wait_for_new_entity_worker_slot()
        local_stats = defaultdict(int)
        cached = entity_cache.get(name)
        context = get_context(name)
        first = None
        failed_candidates = []
        if cached and cached.get('qid'):
            first = mapping_row_from_entity_cache(cached, name) if set(ENTITY_MAPPING_CACHE_FIELDS).issubset(set(cached.keys())) else {k: v for (k, v) in cached.items() if k != 'cache_key'}
            first.setdefault('original_text', name)
        else:
            try:
                first = wikidata_search_entity(name, config)
            except Exception as exc:
                first = {'original_text': name, 'qid': '', 'wikidata_label': '', 'short_description': '', 'match_score': 0.0, 'error': str(exc)}
            if config.wikidata_delay_seconds:
                time.sleep(config.wikidata_delay_seconds)
        (row, checked_first) = try_accept_entity_candidate(name, first, 'wikidata_direct', context, config)
        if row is not None:
            local_stats['accepted_first_pass'] += 1
        else:
            if is_high_confidence_wikidata_match(checked_first, config.wikidata_match_threshold):
                failed_candidates.append(checked_first)
            if llm_entity_resolution_enabled(config):
                llm_row = infer_wikidata_candidate_names_with_llm(name, context, config)
                attempts = []
                for candidate in llm_candidate_names(llm_row, name):
                    try:
                        attempt = wikidata_search_entity(candidate, config)
                    except Exception as exc:
                        attempt = {'original_text': candidate, 'qid': '', 'wikidata_label': '', 'short_description': '', 'match_score': 0.0, 'error': str(exc)}
                    if config.wikidata_delay_seconds:
                        time.sleep(config.wikidata_delay_seconds)
                    (accepted, checked_attempt) = try_accept_entity_candidate(name, attempt, 'llm_name_then_wikidata', context, config, first_pass=first, llm_name_resolution=llm_row, llm_candidate_name=candidate)
                    attempts.append(checked_attempt)
                    if accepted is not None:
                        accepted['second_pass_attempts'] = attempts
                        row = accepted
                        local_stats['accepted_after_llm_name'] += 1
                        break
                if row is None:
                    failed_candidates.extend([attempt for attempt in attempts if is_high_confidence_wikidata_match(attempt, config.wikidata_match_threshold)])
                    reason = 'llm_relation_validation_failed' if failed_candidates else 'wikidata_match_below_threshold'
                    row = rejected_entity_mapping(name, reason, first_pass=first, llm_name_resolution=llm_row, second_pass_attempts=attempts, failed_candidates=failed_candidates)
                    local_stats['rejected'] += 1
            else:
                reason = 'llm_relation_validation_failed' if failed_candidates else 'wikidata_match_below_threshold'
                row = rejected_entity_mapping(name, reason, first_pass=first, failed_candidates=failed_candidates)
                local_stats['rejected'] += 1
        return (name, row, dict(local_stats))
    workers = max(1, int(getattr(config, 'llm_entity_resolution_workers', 1) or 1))
    use_parallel = llm_entity_resolution_enabled(config) and workers > 1 and (len(pending_names) > 1)
    if pending_names:
        print(f'[EntityResolver] Cache resolved {len(results_by_name)}; resolving {len(pending_names)} cache miss(es)' + (f' with {workers} worker(s)' if use_parallel else ' serially'), flush=True)
    if use_parallel:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(resolve_pending_entity, name) for name in pending_names]
            for future in tqdm(as_completed(futures), total=len(futures), desc='Entity resolver new', unit='entity'):
                (name, row, local_stats) = future.result()
                results_by_name[name] = row
                for (key, value) in local_stats.items():
                    stats[key] += value
                if config.use_wikidata_cache:
                    upsert_entity_mapping_cache_row(cache_path, entity_cache, name, row)
    else:
        for name in tqdm(pending_names, desc='Entity resolver new', unit='entity'):
            (name, row, local_stats) = resolve_pending_entity(name)
            results_by_name[name] = row
            for (key, value) in local_stats.items():
                stats[key] += value
            if config.use_wikidata_cache:
                upsert_entity_mapping_cache_row(cache_path, entity_cache, name, row)
    print('[EntityResolver] ' + ', '.join((f'{key}={stats[key]}' for key in sorted(stats))), flush=True)
    return [results_by_name[name] for name in unique_names]

def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    append_jsonl_record(path, row, sort_keys=True)

def extract_response_text(response: Any) -> str:
    try:
        dumped = response.model_dump() if hasattr(response, 'model_dump') else response
    except Exception:
        dumped = response
    if isinstance(dumped, dict):
        choices = dumped.get('choices')
        if isinstance(choices, list) and choices:
            message = choices[0].get('message', {}) if isinstance(choices[0], dict) else {}
            if isinstance(message, dict) and message.get('content'):
                return str(message['content'])
        output = dumped.get('output')
        texts = []
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict) and item.get('type') == 'message':
                    for content in item.get('content', []) or []:
                        if isinstance(content, dict) and content.get('type') in {'output_text', 'text'}:
                            value = content.get('text') or content.get('value')
                            if value:
                                texts.append(str(value))
        if texts:
            return '\n'.join(texts)
        for key in ('output_text', 'text', 'message'):
            if dumped.get(key):
                return str(dumped[key])
        return json.dumps(dumped, ensure_ascii=False)
    try:
        text = getattr(response, 'output_text')
        if text:
            return str(text)
    except Exception:
        pass
    return str(response)

def extract_response_reasoning(response: Any) -> str:
    try:
        dumped = response.model_dump() if hasattr(response, 'model_dump') else response
    except Exception:
        dumped = response
    if not isinstance(dumped, dict):
        return ''
    choices = dumped.get('choices')
    if isinstance(choices, list) and choices:
        message = choices[0].get('message', {}) if isinstance(choices[0], dict) else {}
        if isinstance(message, dict) and message.get('reasoning_content'):
            return str(message['reasoning_content'])
    return ''

def build_openai_chat_kwargs(prompt: str, config: PipelineConfig) -> Dict[str, Any]:
    extra_body = {}
    if getattr(config, 'openai_enable_search', False):
        extra_body['enable_search'] = True
    if getattr(config, 'openai_forced_search', False):
        extra_body.setdefault('search_options', {})['forced_search'] = True
    kwargs = {'model': config.openai_model, 'messages': [{'role': 'user', 'content': prompt}]}
    if extra_body:
        kwargs['extra_body'] = extra_body
    return kwargs

def build_openai_responses_kwargs(prompt: str, config: PipelineConfig) -> Dict[str, Any]:
    tool = {'type': config.openai_web_search_tool_type}
    if config.openai_external_web_access:
        tool['external_web_access'] = True
    return {'model': config.openai_model, 'tools': [tool], 'input': prompt}

def build_openai_client(config: PipelineConfig, OpenAIClass=None):
    api_key = os.environ.get(config.openai_api_key_env)
    if not api_key:
        raise RuntimeError(f'{config.openai_api_key_env} not set')
    if OpenAIClass is None:
        from openai import OpenAI as OpenAIClass
    kwargs = {'api_key': api_key}
    if config.openai_base_url:
        kwargs['base_url'] = config.openai_base_url
    timeout = getattr(config, 'openai_request_timeout_seconds', None)
    if timeout is not None:
        kwargs['timeout'] = timeout
    max_retries = getattr(config, 'openai_max_retries', None)
    if max_retries is not None:
        kwargs['max_retries'] = int(max_retries)
    return OpenAIClass(**kwargs)

def validate_llm_configuration(config: PipelineConfig) -> None:
    needs_entity_resolution = llm_entity_resolution_enabled(config) and (not config.use_existing_entity_cache_only)
    needs_relation_validation = llm_entity_relation_validation_enabled(config) and (not config.use_existing_entity_cache_only)
    if not needs_entity_resolution and (not needs_relation_validation) and (not llm_interval_refinement_enabled(config)) and (not llm_online_interval_verification_enabled(config)):
        return
    if not os.environ.get(config.openai_api_key_env):
        raise RuntimeError(f'{config.openai_api_key_env} not set. Set os.environ[{config.openai_api_key_env!r}] in the notebook kernel or disable LLM/web verification explicitly.')

def parse_llm_json_response(response_row: Dict[str, Any]) -> Dict[str, Any]:
    if not response_row or response_row.get('skipped'):
        return {}
    return extract_first_json_object(response_row.get('response_text', ''))

def parse_confidence_score(value: Any, default: float=0.0) -> float:
    if value is None or value == '':
        return default
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(value).strip().lower()
        aliases = {'very high': 0.95, 'certain': 0.95, 'high': 0.85, 'likely': 0.8, 'probable': 0.8, 'medium': 0.5, 'moderate': 0.5, 'low': 0.2, 'uncertain': 0.2, 'unknown': default}
        if text in aliases:
            return aliases[text]
        percent = re.fullmatch('([0-9]+(?:\\.[0-9]+)?)\\s*%', text)
        if percent:
            score = float(percent.group(1)) / 100.0
        else:
            match = re.search('[0-9]+(?:\\.[0-9]+)?', text)
            if not match:
                return default
            score = float(match.group(0))
    if score > 1.0 and score <= 100.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))

def openai_web_search_json(prompt: str, config: PipelineConfig, cache_file: str) -> Dict[str, Any]:
    base_row = {'prompt_version': config.openai_prompt_version, 'model': config.openai_model, 'base_url': config.openai_base_url, 'api_mode': 'chat_completions', 'web_search_tool_type': config.openai_web_search_tool_type, 'search_verification': 'unverified_forced_search_summary' if getattr(config, 'openai_enable_search', False) or getattr(config, 'openai_forced_search', False) else 'none', 'prompt': prompt}
    if not os.environ.get(config.openai_api_key_env):
        row = dict(base_row)
        row.update({'skipped': True, 'reason': f'{config.openai_api_key_env} not set'})
        return row
    try:
        client = build_openai_client(config)
    except Exception as exc:
        row = dict(base_row)
        row.update({'skipped': True, 'reason': f'OpenAI-compatible client unavailable: {exc}'})
        return row
    kwargs = build_openai_chat_kwargs(prompt, config)
    try:
        response = client.chat.completions.create(**kwargs)
        response_text = extract_response_text(response)
        reasoning_text = extract_response_reasoning(response)
        try:
            dumped = response.model_dump() if hasattr(response, 'model_dump') else {}
        except Exception:
            dumped = {}
        row = dict(base_row)
        row.update({'request_extra_body': kwargs.get('extra_body', {}), 'response_text': response_text})
        if reasoning_text:
            row['reasoning_text'] = reasoning_text
        if isinstance(dumped, dict):
            choices = dumped.get('choices')
            if isinstance(choices, list) and choices:
                message = choices[0].get('message', {}) if isinstance(choices[0], dict) else {}
                if isinstance(message, dict):
                    row['tool_calls_present'] = bool(message.get('tool_calls'))
            if dumped.get('error'):
                row['error'] = dumped['error']
        if not response_text:
            row['warning'] = 'chat.completions response has no message content'
    except Exception as exc:
        row = dict(base_row)
        row.update({'error': {'type': type(exc).__name__, 'message': str(exc)}})
    return row

def apply_llm_entity_resolution(row: Dict[str, Any], llm_row: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    updated = dict(row)
    updated['llm_resolution'] = llm_row
    parsed = parse_llm_json_response(llm_row)
    confidence = parse_confidence_score(parsed.get('confidence')) if parsed else 0.0
    qid = str(parsed.get('qid', '')).strip() if parsed else ''
    label = str(parsed.get('label', '')).strip() if parsed else ''
    if qid and label and (confidence >= threshold):
        updated['qid'] = qid
        updated['wikidata_label'] = label
        updated['match_score'] = confidence
        updated['llm_confidence'] = confidence
        updated['llm_verified'] = True
        updated['match_source'] = 'llm_json'
        if parsed.get('evidence_urls'):
            updated['llm_evidence_urls'] = parsed.get('evidence_urls')
        if parsed.get('rationale'):
            updated['llm_rationale'] = parsed.get('rationale')
    return updated

def resolve_low_confidence_matches(matches: List[Dict[str, Any]], config: PipelineConfig) -> List[Dict[str, Any]]:
    if not llm_entity_resolution_enabled(config):
        print('[LLM] Entity resolution disabled', flush=True)
        return matches
    resolved = []
    needs_count = sum((1 for row in matches if not row.get('qid') or float(row.get('match_score', 0.0)) < config.wikidata_match_threshold))
    print(f'[LLM] Resolving {needs_count}/{len(matches)} low-confidence entity match(es)', flush=True)
    for row in tqdm(matches, desc='LLM entity resolution', unit='entity'):
        row = dict(row)
        needs_resolution = not row.get('qid') or float(row.get('match_score', 0.0)) < config.wikidata_match_threshold
        if needs_resolution:
            prompt = f"Find the best Wikidata entity for this GDELT entity name. Use forced web/search summary if available. Return compact JSON only with qid, label, confidence as a number from 0 to 1, evidence_urls, and rationale. Entity name: {row.get('original_text', '')}"
            llm_row = openai_web_search_json(prompt, config, 'llm_resolution_cache.jsonl')
            row = apply_llm_entity_resolution(row, llm_row, config.wikidata_match_threshold)
        resolved.append(row)
    print('[LLM] Entity resolution pass finished', flush=True)
    return resolved

def normalize_wikidata_time(value: str) -> Tuple[str, str]:
    value = str(value or '').strip().lstrip('+')
    if 'T' in value:
        value = value.split('T', 1)[0]
    if not value:
        return ('', 'missing')
    if re.fullmatch('\\d{4}-\\d{2}-\\d{2}', value):
        if '-00' in value:
            return (value, 'coarse')
        return (value, 'day')
    if re.fullmatch('\\d{4}-\\d{2}', value):
        return (value + '-00', 'coarse')
    if re.fullmatch('\\d{4}', value):
        return (value + '-00-00', 'coarse')
    return (value, 'unknown')

def refine_coarse_date(entity_label: str, relation_label: str, tail_label: str, date_value: str, role: str, config: PipelineConfig) -> Dict[str, Any]:
    prompt = f'Find the exact YYYY-MM-DD date for a coarse Wikidata temporal qualifier. Return compact JSON with date, confidence as a number from 0 to 1, evidence_urls, and rationale. Head: {entity_label}; relation: {relation_label}; tail: {tail_label}; {role}: {date_value}'
    return openai_web_search_json(prompt, config, 'web_verification_cache.jsonl')

def apply_date_refinement(row: Dict[str, Any], role: str, refinement: Dict[str, Any], min_confidence: float=0.75) -> Dict[str, Any]:
    updated = dict(row)
    parsed = parse_llm_json_response(refinement)
    date_value = str(parsed.get('date', ''))
    confidence = parse_confidence_score(parsed.get('confidence'))
    if re.fullmatch('\\d{4}-\\d{2}-\\d{2}', date_value) and confidence >= min_confidence:
        if role == 'start':
            updated['t_start'] = date_value
            updated['start_precision'] = 'day_refined'
        elif role == 'end':
            updated['t_end'] = date_value
            updated['end_precision'] = 'day_refined'
        else:
            raise ValueError(f'Unsupported refinement role: {role}')
    else:
        updated[f'{role}_precision'] = updated.get(f'{role}_precision', 'coarse') + '_unresolved'
    updated[f'{role}_refinement'] = refinement
    return updated

def verify_online_interval_current(interval: Dict[str, Any], config: PipelineConfig) -> Dict[str, Any]:
    prompt = f'Verify whether this Wikidata interval relation is still current today. If it ended, return compact JSON with ended=true, end_date=YYYY-MM-DD, confidence as a number from 0 to 1, evidence_urls, and rationale. Interval: {interval}'
    return openai_web_search_json(prompt, config, 'web_verification_cache.jsonl')

def extract_first_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    match = re.search('\\{.*\\}', text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}

def fetch_wikidata_entities(entity_ids: Sequence[str], config: PipelineConfig) -> Dict[str, Any]:
    result = {}
    ids = [qid for qid in entity_ids if qid and re.fullmatch('Q\\d+', qid)]
    print(f'[Wikidata] Fetching full entity payloads for {len(ids)} id(s)', flush=True)
    for start in tqdm(range(0, len(ids), config.wikidata_batch_size), desc='Wikidata entity batches', unit='batch'):
        batch = ids[start:start + config.wikidata_batch_size]
        params = {'action': 'wbgetentities', 'ids': '|'.join(batch), 'format': 'json', 'languages': 'en'}
        try:
            data = wikidata_get_json(params, config, 'entities', batch)
        except Exception as exc:
            record_wikidata_failed_batch(config, 'entities', batch, exc, getattr(config, 'wikidata_max_retries', 5))
            if getattr(config, 'wikidata_continue_on_batch_error', True):
                continue
            raise
        result.update(data.get('entities', {}))
        if config.wikidata_delay_seconds:
            time.sleep(config.wikidata_delay_seconds)
    print(f'[Wikidata] Entity payloads fetched: {len(result)}', flush=True)
    return result

def extract_snak_value_id(mainsnak: Dict[str, Any]) -> Tuple[str, str]:
    datavalue = mainsnak.get('datavalue') or {}
    value_type = datavalue.get('type', '')
    value = datavalue.get('value', '')
    if value_type == 'wikibase-entityid':
        return (value.get('id', ''), 'entity')
    if value_type == 'time':
        (normalized, _) = normalize_wikidata_time(value.get('time', ''))
        return (normalized, 'time')
    if value_type == 'string':
        return (str(value), 'string')
    if value_type == 'monolingualtext':
        return (str(value.get('text', '')), 'text')
    return (str(value), value_type or 'other')

def extract_time_qualifiers(qualifiers: Dict[str, Any], prop: str) -> List[Tuple[str, str]]:
    values = []
    for snak in qualifiers.get(prop, []):
        datavalue = snak.get('datavalue') or {}
        if datavalue.get('type') == 'time':
            values.append(normalize_wikidata_time(datavalue.get('value', {}).get('time', '')))
    return values

def extract_temporal_claims(entity_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    claims = []
    print(f'[Wikidata] Extracting temporal claims from {len(entity_payload)} entity payload(s)', flush=True)
    for (head_qid, entity) in tqdm(entity_payload.items(), desc='Wikidata temporal claims', unit='entity'):
        for (prop_id, prop_claims) in entity.get('claims', {}).items():
            for claim in prop_claims:
                qualifiers = claim.get('qualifiers', {})
                if 'P580' not in qualifiers and 'P582' not in qualifiers:
                    continue
                (tail_value, value_type) = extract_snak_value_id(claim.get('mainsnak', {}))
                if value_type != 'entity' or not tail_value.startswith('Q'):
                    continue
                start_values = extract_time_qualifiers(qualifiers, 'P580') or [('', 'missing')]
                end_values = extract_time_qualifiers(qualifiers, 'P582') or [('', 'missing')]
                for (start_value, start_precision) in start_values:
                    for (end_value, end_precision) in end_values:
                        claims.append({'head_qid': head_qid, 'property_id': prop_id, 'tail_qid': tail_value, 't_start': start_value, 't_end': end_value, 'start_precision': start_precision, 'end_precision': end_precision})
    print(f'[Wikidata] Temporal claims extracted: {len(claims)}', flush=True)
    return claims

def extract_point_time_claims(entity_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    points = []
    print(f'[Wikidata] Extracting point-in-time claims from {len(entity_payload)} entity payload(s)', flush=True)
    for (head_qid, entity) in tqdm(entity_payload.items(), desc='Wikidata point-time claims', unit='entity'):
        for (prop_id, prop_claims) in entity.get('claims', {}).items():
            for claim in prop_claims:
                qualifiers = claim.get('qualifiers', {})
                if 'P585' not in qualifiers:
                    continue
                (tail_value, value_type) = extract_snak_value_id(claim.get('mainsnak', {}))
                if value_type != 'entity' or not tail_value.startswith('Q'):
                    continue
                for (time_value, time_precision) in extract_time_qualifiers(qualifiers, 'P585'):
                    points.append({'head_qid': head_qid, 'property_id': prop_id, 'tail_qid': tail_value, 't': time_value, 'time_precision': time_precision})
    print(f'[Wikidata] Point-in-time claims extracted: {len(points)}', flush=True)
    return points

def intervals_overlap(interval_start: str, interval_end: str, window_start: str, window_end: str, granularity: str='day') -> bool:
    return time_values_overlap(interval_start, interval_end, window_start, window_end, granularity)

def assign_interval_splits(intervals: Sequence[Dict[str, Any]], windows: Dict[str, Tuple[str, str]], granularity: str='day') -> Dict[str, List[Dict[str, Any]]]:
    granularity = validate_time_granularity(granularity)
    assigned = {name: [] for name in windows}
    for row in intervals:
        if row.get('t_start') and row.get('t_end') and (compare_time_values(row['t_start'], row['t_end'], granularity) > 0):
            continue
        for (split, (start, end)) in windows.items():
            if intervals_overlap(row.get('t_start', ''), row.get('t_end', ''), start, end, granularity):
                assigned[split].append(dict(row))
    return assigned

def assign_point_splits(points: Sequence[Dict[str, Any]], windows: Dict[str, Tuple[str, str]], granularity: str='day') -> Dict[str, List[Dict[str, Any]]]:
    granularity = validate_time_granularity(granularity)
    assigned = {name: [] for name in windows}
    for row in points:
        timestamp = row.get('t', '')
        if not timestamp:
            continue
        for (split, (start, end)) in windows.items():
            if time_value_in_window(timestamp, start, end, granularity):
                assigned[split].append(dict(row))
    return assigned

def is_day_date(value: str) -> bool:
    return bool(re.fullmatch('\\d{4}-\\d{2}-\\d{2}', str(value or ''))) and '-00' not in str(value or '')

def interval_has_required_time_precision(row: Dict[str, Any], granularity: str) -> bool:
    granularity = validate_time_granularity(granularity)
    for (value_key, precision_key) in (('t_start', 'start_precision'), ('t_end', 'end_precision')):
        value = row.get(value_key, '')
        if not value:
            continue
        if not time_precision_satisfies(value, row.get(precision_key, ''), granularity):
            return False
    return True

def interval_has_required_day_precision(row: Dict[str, Any]) -> bool:
    return interval_has_required_time_precision(row, 'day')

def add_tail_mapping_once(extended_mappings: List[Dict[str, Any]], known_mapping_qids: set, tail_mapping: Dict[str, Any]) -> None:
    tail_qid = tail_mapping.get('qid')
    if tail_qid and tail_qid not in known_mapping_qids:
        extended_mappings.append(dict(tail_mapping))
        known_mapping_qids.add(tail_qid)

def temporal_tail_mapping_from_row(row: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    tail_qid = row.get('tail_qid', '')
    if not tail_qid:
        return None
    tail_label = row.get('target_name') or tail_qid
    tail_description = row.get('target_description') or row.get('tail_description') or row.get('short_description') or ''
    return {'original_text': tail_label, 'qid': tail_qid, 'wikidata_label': tail_label, 'short_description': tail_description, 'match_score': 1.0, 'source': source}

def tail_mappings_from_temporal_rows(rows: Sequence[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    by_qid = {}
    for row in rows:
        mapping = temporal_tail_mapping_from_row(row, source)
        if mapping:
            by_qid[mapping['qid']] = mapping
    return [by_qid[qid] for qid in sorted(by_qid)]

def build_interval_rows_from_claims(claims: Sequence[Dict[str, Any]], qid_to_name: Dict[str, str], labels: Dict[str, Dict[str, str]], config: PipelineConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_rows = []
    tail_mappings = []
    print(f'[Wikidata] Processing {len(claims)} temporal claim(s)', flush=True)
    for claim in tqdm(claims, desc='Wikidata interval facts', unit='claim'):
        head_qid = claim['head_qid']
        tail_qid = claim['tail_qid']
        if head_qid not in qid_to_name:
            continue
        head_label = labels.get(head_qid, {}).get('label') or qid_to_name[head_qid]
        tail_label = labels.get(tail_qid, {}).get('label') or tail_qid
        relation_label = normalize_relation_label(labels.get(claim['property_id'], {}).get('label') or claim['property_id'])
        tail_mapping = {'original_text': tail_label, 'qid': tail_qid, 'wikidata_label': tail_label, 'short_description': labels.get(tail_qid, {}).get('description', ''), 'match_score': 1.0, 'source': 'wikidata_interval_tail'}
        if not is_acceptable_wikidata_entity(tail_mapping, config.wikidata_match_threshold):
            continue
        base_rows.append({'source_name': head_label, 'source_description': labels.get(head_qid, {}).get('description', ''), 'relation_label': relation_label, 'target_name': tail_label, 'target_description': labels.get(tail_qid, {}).get('description', ''), 't_start': claim.get('t_start', ''), 't_end': claim.get('t_end', ''), 'head_qid': head_qid, 'tail_qid': tail_qid, 'property_id': claim['property_id'], 'start_precision': claim.get('start_precision', ''), 'end_precision': claim.get('end_precision', '')})
        tail_mappings.append(tail_mapping)
    return (base_rows, tail_mappings)

def build_base_intervals_for_missing_qids(missing_qids: Sequence[str], qid_to_name: Dict[str, str], config: PipelineConfig, cache_path: Optional[Path]=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not missing_qids:
        return ([], [])
    if config.use_existing_entity_cache_only:
        print(f'[Wikidata] Cache-only mode: skipping {len(missing_qids)} missing interval entity/entities', flush=True)
        return ([], [])
    all_base_rows = []
    all_tail_mappings = []
    qids = sorted(set(missing_qids))
    for start in tqdm(range(0, len(qids), config.wikidata_batch_size), desc='Wikidata interval source batches', unit='batch'):
        batch_qids = qids[start:start + config.wikidata_batch_size]
        entity_payload = fetch_wikidata_entities(batch_qids, config)
        claims = extract_temporal_claims(entity_payload)
        label_ids = set()
        for claim in claims:
            label_ids.add(claim['head_qid'])
            label_ids.add(claim['tail_qid'])
            label_ids.add(claim['property_id'])
        labels = fetch_wikidata_labels(sorted(label_ids), config) if label_ids else {}
        (batch_base_rows, batch_tail_mappings) = build_interval_rows_from_claims(claims, qid_to_name, labels, config)
        if cache_path is not None:
            cache_base_intervals(cache_path, batch_base_rows, batch_tail_mappings, batch_qids)
        all_base_rows.extend(batch_base_rows)
        all_tail_mappings.extend(batch_tail_mappings)
    return (all_base_rows, all_tail_mappings)

def cache_base_intervals(cache_path: Path, base_rows: Sequence[Dict[str, Any]], tail_mappings: Sequence[Dict[str, Any]], requested_qids: Optional[Sequence[str]]=None) -> None:
    by_head = defaultdict(list)
    for row in base_rows:
        head_qid = row.get('head_qid')
        if head_qid:
            by_head[head_qid].append(dict(row))
    rows = [{'cache_key': qid, 'qid': qid, 'intervals': facts} for (qid, facts) in by_head.items()]
    for qid in sorted(set(requested_qids or [])):
        if qid and qid not in by_head:
            rows.append({'cache_key': qid, 'qid': qid, 'intervals': [], 'cache_status': 'no_temporal_claims'})
    if rows:
        upsert_jsonl_cache(cache_path, rows, 'cache_key')

def finalize_interval_row(base_row: Dict[str, Any], config: PipelineConfig, dataset_end: str) -> Optional[Dict[str, Any]]:
    granularity = validate_time_granularity(config.dataset_time_granularity)
    row = dict(base_row)
    for (role, value_key, precision_key) in (('start', 't_start', 'start_precision'), ('end', 't_end', 'end_precision')):
        value = row.get(value_key, '')
        if not value:
            continue
        if not time_precision_satisfies(value, row.get(precision_key, ''), granularity):
            if row.get(precision_key) != 'coarse' or not llm_interval_refinement_enabled(config):
                return None
            row = apply_date_refinement(row, role, refine_coarse_date(row['source_name'], row['relation_label'], row['target_name'], value, role, config))
    if not interval_has_required_time_precision(row, granularity):
        return None
    for value_key in ('t_start', 't_end'):
        if row.get(value_key):
            row[value_key] = normalize_time_value(row[value_key], granularity)
    row['time_granularity'] = granularity
    if validate_pipeline_setting(config) == 'online_testing' and llm_online_interval_verification_enabled(config) and (not row['t_end'] or compare_time_values(row['t_end'], dataset_end, granularity) >= 0):
        verification = verify_online_interval_current(row, config)
        row['online_verification'] = verification
        parsed = extract_first_json_object(verification.get('response_text', ''))
        end_date = str(parsed.get('end_date', ''))
        if parsed.get('ended') is True and time_precision_satisfies(end_date, 'day', granularity):
            row['t_end'] = normalize_time_value(end_date, granularity)
            row['end_precision'] = 'day_verified'
    if row.get('t_start') and row.get('t_end') and (compare_time_values(row['t_start'], row['t_end'], granularity) > 0):
        return None
    return row

def build_point_rows_from_claims(point_claims: Sequence[Dict[str, Any]], qid_to_name: Dict[str, str], labels: Dict[str, Dict[str, str]], config: PipelineConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_rows = []
    tail_mappings = []
    print(f'[Wikidata] Processing {len(point_claims)} point-in-time claim(s)', flush=True)
    for claim in tqdm(point_claims, desc='Wikidata point facts', unit='claim'):
        head_qid = claim['head_qid']
        tail_qid = claim['tail_qid']
        if head_qid not in qid_to_name:
            continue
        head_label = labels.get(head_qid, {}).get('label') or qid_to_name[head_qid]
        tail_label = labels.get(tail_qid, {}).get('label') or tail_qid
        relation_label = normalize_relation_label(labels.get(claim['property_id'], {}).get('label') or claim['property_id'])
        tail_mapping = {'original_text': tail_label, 'qid': tail_qid, 'wikidata_label': tail_label, 'short_description': labels.get(tail_qid, {}).get('description', ''), 'match_score': 1.0, 'source': 'wikidata_point_tail'}
        if not is_acceptable_wikidata_entity(tail_mapping, config.wikidata_match_threshold):
            continue
        base_rows.append({'source_name': head_label, 'source_description': labels.get(head_qid, {}).get('description', ''), 'relation_label': relation_label, 'target_name': tail_label, 'target_description': labels.get(tail_qid, {}).get('description', ''), 't': claim.get('t', ''), 'head_qid': head_qid, 'tail_qid': tail_qid, 'property_id': claim['property_id'], 'time_precision': claim.get('time_precision', '')})
        tail_mappings.append(tail_mapping)
    return (base_rows, tail_mappings)

def build_base_points_for_missing_qids(missing_qids: Sequence[str], qid_to_name: Dict[str, str], config: PipelineConfig, cache_path: Optional[Path]=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not missing_qids:
        return ([], [])
    if config.use_existing_entity_cache_only:
        print(f'[Wikidata] Cache-only mode: skipping {len(missing_qids)} missing point entity/entities', flush=True)
        return ([], [])
    all_base_rows = []
    all_tail_mappings = []
    qids = sorted(set(missing_qids))
    for start in tqdm(range(0, len(qids), config.wikidata_batch_size), desc='Wikidata point source batches', unit='batch'):
        batch_qids = qids[start:start + config.wikidata_batch_size]
        entity_payload = fetch_wikidata_entities(batch_qids, config)
        point_claims = extract_point_time_claims(entity_payload)
        label_ids = set()
        for claim in point_claims:
            label_ids.add(claim['head_qid'])
            label_ids.add(claim['tail_qid'])
            label_ids.add(claim['property_id'])
        labels = fetch_wikidata_labels(sorted(label_ids), config) if label_ids else {}
        (batch_base_rows, batch_tail_mappings) = build_point_rows_from_claims(point_claims, qid_to_name, labels, config)
        if cache_path is not None:
            cache_base_points(cache_path, batch_base_rows, batch_tail_mappings, batch_qids)
        all_base_rows.extend(batch_base_rows)
        all_tail_mappings.extend(batch_tail_mappings)
    return (all_base_rows, all_tail_mappings)

def cache_base_points(cache_path: Path, base_rows: Sequence[Dict[str, Any]], tail_mappings: Sequence[Dict[str, Any]], requested_qids: Optional[Sequence[str]]=None) -> None:
    by_head = defaultdict(list)
    for row in base_rows:
        head_qid = row.get('head_qid')
        if head_qid:
            by_head[head_qid].append(dict(row))
    rows = [{'cache_key': qid, 'qid': qid, 'points': facts} for (qid, facts) in by_head.items()]
    for qid in sorted(set(requested_qids or [])):
        if qid and qid not in by_head:
            rows.append({'cache_key': qid, 'qid': qid, 'points': [], 'cache_status': 'no_point_claims'})
    if rows:
        upsert_jsonl_cache(cache_path, rows, 'cache_key')

def finalize_point_row(base_row: Dict[str, Any], config: PipelineConfig) -> Optional[Dict[str, Any]]:
    granularity = validate_time_granularity(config.dataset_time_granularity)
    row = dict(base_row)
    if not row.get('t') or not time_precision_satisfies(row.get('t', ''), row.get('time_precision', ''), granularity):
        return None
    row['t'] = normalize_time_value(row['t'], granularity)
    row['time_granularity'] = granularity
    return row

def build_points_from_wikidata_mappings(mappings: Sequence[Dict[str, Any]], windows: Dict[str, Tuple[str, str]], config: PipelineConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not config.include_point_time_facts:
        return ([], list(mappings))
    reliable = []
    for row in mappings:
        if not row.get('qid') or float(row.get('match_score', 0.0)) < config.wikidata_match_threshold:
            continue
        reliable.append(row)
    qid_to_name = {row['qid']: row.get('wikidata_label') or row.get('original_text') for row in reliable}
    print(f'[Wikidata] Building points from {len(reliable)} reliable mapped entity/entities', flush=True)
    if not qid_to_name:
        return ([], list(mappings))
    cache_path = cache_file_path(config, 'wikidata_point_cache.jsonl')
    point_cache = load_jsonl_cache(cache_path, 'cache_key') if config.use_wikidata_cache else {}
    cached_base_rows = []
    cached_tail_mappings = []
    missing_qids = []
    for qid in sorted(qid_to_name):
        cached = point_cache.get(qid)
        if cached is not None:
            cached_points = [dict(row) for row in cached.get('points', [])]
            cached_base_rows.extend(cached_points)
            cached_tail_mappings.extend(tail_mappings_from_temporal_rows(cached_points, 'wikidata_point_tail_cache'))
        else:
            missing_qids.append(qid)
    print(f'[Wikidata] Point cache hits={len(qid_to_name) - len(missing_qids)}, misses={len(missing_qids)}', flush=True)
    if config.use_existing_entity_cache_only and missing_qids:
        print(f'[Wikidata] Cache-only mode: skipping {len(missing_qids)} point cache miss(es)', flush=True)
        missing_qids = []
    (new_base_rows, new_tail_mappings) = build_base_points_for_missing_qids(missing_qids, qid_to_name, config, cache_path if config.use_wikidata_cache else None)
    extended_mappings = list(mappings)
    known_mapping_qids = {row.get('qid') for row in mappings if row.get('qid')}
    for tail_mapping in cached_tail_mappings + new_tail_mappings:
        add_tail_mapping_once(extended_mappings, known_mapping_qids, tail_mapping)
    points = []
    base_rows = cached_base_rows + new_base_rows
    print(f'[Wikidata] Finalizing {len(base_rows)} base point fact(s)', flush=True)
    for base_row in tqdm(base_rows, desc='Wikidata point facts', unit='claim'):
        row = finalize_point_row(base_row, config)
        if row is not None and any((time_value_in_window(row['t'], start, end, config.dataset_time_granularity) for (start, end) in windows.values())):
            points.append(row)
    return (points, extended_mappings)

def build_intervals_from_wikidata_mappings(mappings: Sequence[Dict[str, Any]], windows: Dict[str, Tuple[str, str]], config: PipelineConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    reliable = []
    for row in mappings:
        if not row.get('qid') or float(row.get('match_score', 0.0)) < config.wikidata_match_threshold:
            continue
        reliable.append(row)
    qid_to_name = {row['qid']: row.get('wikidata_label') or row.get('original_text') for row in reliable}
    print(f'[Wikidata] Building intervals from {len(reliable)} reliable mapped entity/entities', flush=True)
    if not qid_to_name:
        return ([], list(mappings))
    cache_path = cache_file_path(config, 'wikidata_interval_cache.jsonl')
    interval_cache = load_jsonl_cache(cache_path, 'cache_key') if config.use_wikidata_cache else {}
    cached_base_rows = []
    cached_tail_mappings = []
    missing_qids = []
    for qid in sorted(qid_to_name):
        cached = interval_cache.get(qid)
        if cached is not None:
            cached_intervals = [dict(row) for row in cached.get('intervals', [])]
            cached_base_rows.extend(cached_intervals)
            cached_tail_mappings.extend(tail_mappings_from_temporal_rows(cached_intervals, 'wikidata_interval_tail_cache'))
        else:
            missing_qids.append(qid)
    print(f'[Wikidata] Interval cache hits={len(qid_to_name) - len(missing_qids)}, misses={len(missing_qids)}', flush=True)
    if config.use_existing_entity_cache_only and missing_qids:
        print(f'[Wikidata] Cache-only mode: skipping {len(missing_qids)} interval cache miss(es)', flush=True)
        missing_qids = []
    (new_base_rows, new_tail_mappings) = build_base_intervals_for_missing_qids(missing_qids, qid_to_name, config, cache_path if config.use_wikidata_cache else None)
    extended_mappings = list(mappings)
    known_mapping_qids = {row.get('qid') for row in mappings if row.get('qid')}
    for tail_mapping in cached_tail_mappings + new_tail_mappings:
        add_tail_mapping_once(extended_mappings, known_mapping_qids, tail_mapping)
    dataset_end = max((end for (_, end) in windows.values()))
    intervals = []
    base_rows = cached_base_rows + new_base_rows
    print(f'[Wikidata] Finalizing {len(base_rows)} base interval fact(s)', flush=True)
    for base_row in tqdm(base_rows, desc='Wikidata interval facts', unit='claim'):
        row = finalize_interval_row(base_row, config, dataset_end)
        if row is not None:
            intervals.append(row)
    return (intervals, extended_mappings)

def dedupe_temporal_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for row in rows:
        key = tuple((row.get(k, '') for k in keys))
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(row))
    return output

def build_temporal_knowledge_from_wikidata_mappings(mappings: Sequence[Dict[str, Any]], windows: Dict[str, Tuple[str, str]], config: PipelineConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    max_depth = max(1, int(config.wikidata_expansion_max_depth or 1)) if config.enable_wikidata_temporal_expansion else 1
    all_mappings = list(mappings)
    known_qids = {row.get('qid') for row in all_mappings if row.get('qid')}
    frontier = list(mappings)
    all_intervals = []
    all_points = []
    visited_frontier_qids = set()
    for depth in range(1, max_depth + 1):
        frontier = [row for row in frontier if row.get('qid') and row.get('qid') not in visited_frontier_qids]
        if not frontier:
            break
        for row in frontier:
            visited_frontier_qids.add(row.get('qid'))
        print(f'[Wikidata] Temporal expansion depth {depth}/{max_depth}: frontier={len(frontier)}', flush=True)
        (depth_intervals, interval_mappings) = build_intervals_from_wikidata_mappings(frontier, windows, config) if config.include_interval_facts else ([], list(frontier))
        (depth_points, point_mappings) = build_points_from_wikidata_mappings(frontier, windows, config) if config.include_point_time_facts else ([], list(frontier))
        for row in depth_intervals:
            row['expansion_depth'] = depth
        for row in depth_points:
            row['expansion_depth'] = depth
        all_intervals.extend(depth_intervals)
        all_points.extend(depth_points)
        new_frontier = []
        for candidate in interval_mappings + point_mappings:
            qid = candidate.get('qid')
            if not qid or qid in known_qids or qid in visited_frontier_qids:
                continue
            all_mappings.append(dict(candidate))
            known_qids.add(qid)
            new_frontier.append(dict(candidate))
            if len(new_frontier) >= config.max_new_entities_per_depth:
                break
        frontier = new_frontier
    all_intervals = dedupe_temporal_rows(all_intervals, ('source_name', 'relation_label', 'target_name', 't_start', 't_end'))
    all_points = dedupe_temporal_rows(all_points, ('source_name', 'relation_label', 'target_name', 't'))
    return (all_intervals, all_points, all_mappings)
SEMANTIC_DOMAIN_PATTERNS = {'finance': ('\\bcompany\\b', '\\bcorporation\\b', '\\bbank\\b', '\\bfinancial\\b', '\\bfinance\\b', '\\beconomist\\b', '\\beconomy\\b', '\\bbusiness\\b', '\\btrade\\b', '\\bindustry\\b', '\\bmanufacturer\\b', '\\benterprise\\b', '\\bmarket\\b', '\\binvestment\\b', '\\bcurrency\\b', '\\bstock exchange\\b', '\\bcentral bank\\b', '\\bminister of finance\\b'), 'media_entertainment': ('\\bsport\\b', '\\bfootball\\b', '\\bsoccer\\b', '\\bbasketball\\b', '\\bclub\\b', '\\bteam\\b', '\\bathlete\\b', '\\bcoach\\b', '\\bmusic\\b', '\\bmusician\\b', '\\bsinger\\b', '\\bband\\b', '\\balbum\\b', '\\bsong\\b', '\\bfilm\\b', '\\bmovie\\b', '\\bactor\\b', '\\bactress\\b', '\\bartist\\b', '\\bwriter\\b', '\\bmuseum\\b', '\\bcultural\\b', '\\bmedia\\b', '\\bnewspaper\\b', '\\bjournalist\\b', '\\bnews\\b', '\\bbroadcaster\\b', '\\btelevision\\b', '\\bradio\\b', '\\bmagazine\\b', '\\bpress\\b', '\\binformation\\b'), 'politics': ('\\bcountry\\b', '\\bsovereign state\\b', '\\brepublic\\b', '\\bkingdom\\b', '\\bfederation\\b', '\\bnation\\b', '\\bterritory\\b', '\\bstate in\\b', '\\bpolitician\\b', '\\bdiplomat\\b', '\\bpresident\\b', '\\bprime minister\\b', '\\bminister\\b', '\\bambassador\\b', '\\bgovernor\\b', '\\bmonarch\\b', '\\bofficial\\b', '\\bgovernment\\b', '\\bpolitical party\\b', '\\binternational organi[sz]ation\\b', '\\bministry\\b', '\\bdepartment\\b', '\\bparliament\\b', '\\bcongress\\b', '\\bsenate\\b', '\\badministrative\\b')}

def classify_semantic_domain_rule(mapping: Dict[str, Any]) -> str:
    text = ' '.join((str(mapping.get(key, '')) for key in ('original_text', 'wikidata_label', 'short_description', 'description'))).lower()
    for domain in ('finance', 'media_entertainment', 'politics'):
        if any((re.search(pattern, text) for pattern in SEMANTIC_DOMAIN_PATTERNS[domain])):
            return domain
    return 'other'

def semantic_domain_cache_key(mapping: Dict[str, Any], config: PipelineConfig) -> str:
    payload = {'prompt_version': config.openai_prompt_version, 'model': config.openai_model, 'base_url': config.openai_base_url, 'qid': mapping.get('qid', ''), 'wikidata_label': mapping.get('wikidata_label', ''), 'description': mapping.get('short_description') or mapping.get('description') or ''}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
VALID_SEMANTIC_DOMAINS = {'politics', 'finance', 'media_entertainment', 'other'}

def load_semantic_domain_cache(config: PipelineConfig) -> Dict[str, Dict[str, Any]]:
    if not config.use_wikidata_cache:
        return {}
    return load_jsonl_cache(cache_file_path(config, 'entity_domain_cache.jsonl'), 'cache_key')

def classify_semantic_domain_with_llm(mapping: Dict[str, Any], config: PipelineConfig, domain_cache: Optional[Dict[str, Dict[str, Any]]]=None) -> str:
    cache_path = cache_file_path(config, 'entity_domain_cache.jsonl')
    cache_key = semantic_domain_cache_key(mapping, config)
    if config.use_wikidata_cache:
        if domain_cache is None:
            domain_cache = load_semantic_domain_cache(config)
        cached = domain_cache.get(cache_key)
        if cached and cached.get('domain') in VALID_SEMANTIC_DOMAINS:
            return cached['domain']
    prompt = f"Classify this Wikidata-aligned entity into exactly one domain: politics, finance, media_entertainment, or other. Politics includes countries, political people, governments, political organizations, international organizations, and administrative agencies. Finance includes companies, banks, financial organizations, trade organizations, and finance/economic people. Media_entertainment includes sports, music, media, broadcasters, television, journalists, and public information entities. Return compact JSON only with domain, is_domain_core_entity, and reason.\nEntity name: {mapping.get('original_text', '')}\nWikidata label: {mapping.get('wikidata_label', '')}\nDescription: {mapping.get('short_description') or mapping.get('description') or ''}"
    row = openai_web_search_json(prompt, config, 'entity_domain_cache.jsonl')
    parsed = parse_llm_json_response(row)
    domain = str(parsed.get('domain', '')).strip().lower() if parsed else ''
    if domain not in VALID_SEMANTIC_DOMAINS:
        domain = classify_semantic_domain_rule(mapping)
    cache_row = dict(row)
    cache_row.update({'cache_key': cache_key, 'qid': mapping.get('qid', ''), 'wikidata_label': mapping.get('wikidata_label', ''), 'description': mapping.get('short_description') or mapping.get('description') or '', 'domain': domain, 'is_domain_core_entity': parsed.get('is_domain_core_entity', '') if parsed else '', 'classification_method': 'llm'})
    if config.use_wikidata_cache:
        append_jsonl(cache_path, cache_row)
        if domain_cache is not None:
            domain_cache[cache_key] = cache_row
    return domain

def classify_semantic_domain(mapping: Dict[str, Any], config: PipelineConfig, domain_cache: Optional[Dict[str, Dict[str, Any]]]=None) -> str:
    method = getattr(config, 'domain_classification_method', 'rule') or 'rule'
    rule_domain = classify_semantic_domain_rule(mapping)
    if method == 'rule':
        return rule_domain
    if method == 'rule_then_llm' and rule_domain != 'other':
        return rule_domain
    if not os.environ.get(config.openai_api_key_env):
        return rule_domain
    return classify_semantic_domain_with_llm(mapping, config, domain_cache)

def semantic_domain_cache_hit_for_config(mapping: Dict[str, Any], config: PipelineConfig, domain_cache: Optional[Dict[str, Dict[str, Any]]]) -> bool:
    if not domain_cache:
        return False
    cached = domain_cache.get(semantic_domain_cache_key(mapping, config))
    return bool(cached and cached.get('domain') in VALID_SEMANTIC_DOMAINS)

def build_entity_domain_map(mappings: Sequence[Dict[str, Any]], config: PipelineConfig) -> Dict[str, str]:
    domain_map = {}
    method = getattr(config, 'domain_classification_method', 'rule') or 'rule'
    domain_cache = load_semantic_domain_cache(config) if method != 'rule' else None
    accepted_rows = [row for row in mappings if is_acceptable_wikidata_entity(row, config.wikidata_match_threshold)]
    cached_count = sum((1 for row in accepted_rows if semantic_domain_cache_hit_for_config(row, config, domain_cache)))
    print(f'[SemanticDomain] Building domain map for {len(accepted_rows)} accepted entity mapping(s); cached={cached_count}; method={method}', flush=True)
    domain_counts: Dict[str, int] = {}
    for row in tqdm(accepted_rows, desc='Semantic domain map', unit='entity'):
        domain = classify_semantic_domain(row, config, domain_cache)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        for key in ('original_text', 'wikidata_label', 'source_name', 'target_name'):
            name = str(row.get(key, '')).strip()
            if name:
                domain_map[name] = domain
    print(f'[SemanticDomain] Domain map complete: {domain_counts}', flush=True)
    return domain_map

def semantic_domain_entity_rows(mappings: Sequence[Dict[str, Any]], config: PipelineConfig) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    method = getattr(config, 'domain_classification_method', 'rule') or 'rule'
    domain_cache = load_semantic_domain_cache(config) if method != 'rule' else None
    candidate_rows = []
    for row in mappings:
        name = str(row.get('wikidata_label') or row.get('original_text') or '').strip()
        qid = str(row.get('qid') or '')
        key = (qid, name)
        if not name or key in seen or (not is_acceptable_wikidata_entity(row, config.wikidata_match_threshold)):
            continue
        seen.add(key)
        candidate_rows.append((row, name, qid))
    cached_count = sum((1 for (row, _, _) in candidate_rows if semantic_domain_cache_hit_for_config(row, config, domain_cache)))
    print(f'[SemanticDomain] Building semantic_domain_entities rows for {len(candidate_rows)} entity/entities; cached={cached_count}; method={method}', flush=True)
    for (row, name, qid) in tqdm(candidate_rows, desc='Semantic domain entity rows', unit='entity'):
        rows.append({'entity_name': name, 'original_text': str(row.get('original_text') or ''), 'qid': qid, 'wikidata_label': str(row.get('wikidata_label') or ''), 'description': str(row.get('short_description') or row.get('description') or ''), 'semantic_domain': classify_semantic_domain(row, config, domain_cache), 'classification_method': getattr(config, 'domain_classification_method', 'rule')})
    return rows

def event_matches_semantic_domain(source_domain: str, target_domain: str, selected_domain: str, endpoint_policy: str='either') -> Tuple[bool, str]:
    selected_domain = selected_domain or 'all'
    if selected_domain == 'all':
        return (True, '')
    domains = {source_domain or 'other', target_domain or 'other'}
    if selected_domain == 'politics' and ('finance' in domains or 'media_entertainment' in domains):
        return (False, 'semantic_domain_excluded_counterpart')
    if endpoint_policy == 'both':
        matched = source_domain == selected_domain and target_domain == selected_domain
    elif endpoint_policy == 'source':
        matched = source_domain == selected_domain
    elif endpoint_policy == 'target':
        matched = target_domain == selected_domain
    else:
        matched = source_domain == selected_domain or target_domain == selected_domain
    return (matched, '' if matched else 'semantic_domain_mismatch')

def politics_counterpart_excluded(source_domain: str, target_domain: str, selected_domain: str) -> bool:
    domains = {source_domain or 'other', target_domain or 'other'}
    return selected_domain == 'politics' and ('finance' in domains or 'media_entertainment' in domains)

def event_domain_pair(row: Dict[str, Any], entity_domain_map: Dict[str, str]) -> Tuple[str, str]:
    return (entity_domain_map.get(row.get('source_name', ''), 'other'), entity_domain_map.get(row.get('target_name', ''), 'other'))

def event_is_domain_seed(source_domain: str, target_domain: str, selected_domain: str) -> bool:
    return source_domain == selected_domain or target_domain == selected_domain

def filter_events_by_either_semantic_domain(events: Sequence[Dict[str, Any]], entity_domain_map: Dict[str, str], selected_domain: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    seed_entities = set()
    for row in events:
        (source_domain, target_domain) = event_domain_pair(row, entity_domain_map)
        if politics_counterpart_excluded(source_domain, target_domain, selected_domain):
            continue
        if event_is_domain_seed(source_domain, target_domain, selected_domain):
            seed_entities.add(row.get('source_name', ''))
            seed_entities.add(row.get('target_name', ''))
    kept = []
    filtered = []
    for row in events:
        (source_domain, target_domain) = event_domain_pair(row, entity_domain_map)
        if politics_counterpart_excluded(source_domain, target_domain, selected_domain):
            reason = 'semantic_domain_excluded_counterpart'
        elif row.get('source_name', '') in seed_entities and row.get('target_name', '') in seed_entities:
            kept_row = dict(row)
            kept_row.update({'source_domain': source_domain, 'target_domain': target_domain, 'semantic_domain': selected_domain, 'domain_filter_stage': 'seed' if event_is_domain_seed(source_domain, target_domain, selected_domain) else 'induced'})
            kept.append(kept_row)
            continue
        else:
            reason = 'semantic_domain_seed_mismatch'
        rejected = dict(row)
        rejected.update({'reason': reason, 'source_domain': source_domain, 'target_domain': target_domain, 'semantic_domain': selected_domain})
        filtered.append(rejected)
    return (kept, filtered)

def filter_events_by_semantic_domain(events: Sequence[Dict[str, Any]], entity_domain_map: Dict[str, str], config: PipelineConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected_domain = getattr(config, 'semantic_domain', 'all') or 'all'
    if selected_domain == 'all':
        return ([dict(row) for row in events], [])
    endpoint_policy = getattr(config, 'domain_endpoint_policy', 'either')
    if endpoint_policy == 'either':
        return filter_events_by_either_semantic_domain(events, entity_domain_map, selected_domain)
    kept = []
    filtered = []
    for row in events:
        (source_domain, target_domain) = event_domain_pair(row, entity_domain_map)
        (matched, reason) = event_matches_semantic_domain(source_domain, target_domain, selected_domain, endpoint_policy)
        if matched:
            kept.append(dict(row))
        else:
            rejected = dict(row)
            rejected.update({'reason': reason, 'source_domain': source_domain, 'target_domain': target_domain, 'semantic_domain': selected_domain})
            filtered.append(rejected)
    return (kept, filtered)

def train_entity_set(train_events: Sequence[Dict[str, Any]]) -> set:
    entities = set()
    for row in train_events:
        entities.add(row['source_name'])
        entities.add(row['target_name'])
    return entities

def classify_test_entity_overlap(train_entities: set, row: Dict[str, Any]) -> str:
    return 'transductive' if row.get('source_name') in train_entities and row.get('target_name') in train_entities else 'inductive'

def annotate_test_entity_overlap_types(train_events: Sequence[Dict[str, Any]], test_events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    train_entities = train_entity_set(train_events)
    annotated = []
    for row in test_events:
        item = dict(row)
        item['test_entity_overlap_type'] = classify_test_entity_overlap(train_entities, item)
        annotated.append(item)
    return annotated

def test_entity_overlap_availability_counts(test_events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(test_events)
    transductive_count = sum((1 for row in test_events if row.get('test_entity_overlap_type') == 'transductive'))
    inductive_count = sum((1 for row in test_events if row.get('test_entity_overlap_type') == 'inductive'))
    return {'transductive_event_fact_count': transductive_count, 'inductive_event_fact_count': inductive_count, 'transductive_ratio': transductive_count / total if total else 0.0, 'inductive_ratio': inductive_count / total if total else 0.0}

def canonicalize_events_with_mappings(events: Sequence[Dict[str, Any]], mappings: Sequence[Dict[str, Any]], config: PipelineConfig) -> List[Dict[str, Any]]:
    original_to_canonical = {}
    for row in mappings:
        score = float(row.get('match_score', 0.0) or 0.0)
        if row.get('qid') and score >= config.wikidata_match_threshold and row.get('wikidata_label'):
            original_to_canonical[row['original_text']] = row['wikidata_label']
    canonical = []
    for row in events:
        new_row = dict(row)
        new_row['source_name'] = original_to_canonical.get(row['source_name'], row['source_name'])
        new_row['target_name'] = original_to_canonical.get(row['target_name'], row['target_name'])
        canonical.append(new_row)
    return dedupe_events(canonical)

def is_acceptable_wikidata_entity(mapping: Dict[str, Any], threshold: float=0.8) -> bool:
    label = str(mapping.get('wikidata_label') or mapping.get('original_text') or '').strip()
    try:
        score = float(mapping.get('match_score', 0.0) or 0.0)
    except Exception:
        score = 0.0
    llm_verified = bool(mapping.get('llm_verified')) and parse_confidence_score(mapping.get('llm_confidence')) >= threshold
    if mapping.get('status') == 'rejected':
        return False
    if not mapping.get('qid'):
        return False
    if score < threshold and (not llm_verified):
        return False
    if not label or label.startswith('.') or re.fullmatch('\\d+', label):
        return False
    if mapping.get('llm_relation_validated') and (not mapping.get('llm_relation_valid')):
        return False
    if looks_like_generic_entity(label):
        return False
    return True

def accepted_entity_names_from_mappings(mappings: Sequence[Dict[str, Any]], threshold: float=0.8) -> set:
    accepted = set()
    for row in mappings:
        if is_acceptable_wikidata_entity(row, threshold):
            accepted.add(str(row.get('original_text', '')).strip())
            if row.get('wikidata_label'):
                accepted.add(str(row['wikidata_label']).strip())
    accepted.discard('')
    return accepted

def looks_like_geo_containment(source_name: str, target_name: str) -> bool:
    s = str(source_name or '').strip().lower()
    o = str(target_name or '').strip().lower()
    if not s or not o or s == o:
        return False
    return s in o or o in s

def filter_events_by_entity_acceptance_and_self_loops(events: Sequence[Dict[str, Any]], mappings: Sequence[Dict[str, Any]], threshold: float=0.8) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted = accepted_entity_names_from_mappings(mappings, threshold)
    kept = []
    filtered = []
    for row in events:
        s = row.get('source_name', '')
        o = row.get('target_name', '')
        reason = None
        if s == o:
            reason = 'self_loop'
        elif looks_like_geo_containment(s, o):
            reason = 'geo_containment'
        elif s not in accepted or o not in accepted:
            reason = 'unaccepted_entity'
        if reason:
            rejected = dict(row)
            rejected['reason'] = reason
            filtered.append(rejected)
        else:
            kept.append(dict(row))
    return (kept, filtered)

def event_hash(row: Dict[str, Any]) -> str:
    return hashlib.sha256(f"event|{row['s_id']}|{row['r_id']}|{row['o_id']}|{row['t']}".encode('utf-8')).hexdigest()

def interval_hash(row: Dict[str, Any]) -> str:
    return hashlib.sha256(f"interval|{row['s_id']}|{row['r_id']}|{row['o_id']}|{row['t_start']}|{row['t_end']}".encode('utf-8')).hexdigest()

def point_hash(row: Dict[str, Any]) -> str:
    return hashlib.sha256(f"point|{row['s_id']}|{row['r_id']}|{row['o_id']}|{row['t']}".encode('utf-8')).hexdigest()

def split_hash(hashes: Iterable[str]) -> str:
    payload = '\n'.join(sorted(hashes))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def mapping_index_by_entity_name(mappings: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index = {}
    for row in mappings or []:
        for key in ('wikidata_label', 'original_text', 'source_name', 'target_name'):
            name = str(row.get(key, '')).strip()
            if name and name not in index:
                index[name] = row
    return index

def entity_metadata_row(entity_id: int, entity_name: str, mapping_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    mapping = mapping_index.get(entity_name, {})
    description = str(mapping.get('short_description') or mapping.get('description') or '')
    return {'entity_id': entity_id, 'entity_name': entity_name, 'name': entity_name, 'label': str(mapping.get('wikidata_label') or entity_name), 'wikidata_label': str(mapping.get('wikidata_label') or ''), 'qid': str(mapping.get('qid') or ''), 'description': description, 'short_description': description, 'original_text': str(mapping.get('original_text') or ''), 'match_score': mapping.get('match_score', ''), 'match_source': str(mapping.get('match_source') or mapping.get('source') or ''), 'status': str(mapping.get('status') or '')}

def build_id_tables(events: Sequence[Dict[str, Any]], intervals: Sequence[Dict[str, Any]], mappings: Optional[Sequence[Dict[str, Any]]]=None, points: Optional[Sequence[Dict[str, Any]]]=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    entity_names = set()
    relation_labels = set()
    for row in events:
        entity_names.add(row['source_name'])
        entity_names.add(row['target_name'])
        relation_labels.add(normalize_relation_label(row['relation_label']))
    for row in intervals:
        if 'source_name' in row and 'target_name' in row:
            entity_names.add(row['source_name'])
            entity_names.add(row['target_name'])
        if 'relation_label' in row:
            relation_labels.add(normalize_relation_label(row['relation_label']))
    for row in points or []:
        if 'source_name' in row and 'target_name' in row:
            entity_names.add(row['source_name'])
            entity_names.add(row['target_name'])
        if 'relation_label' in row:
            relation_labels.add(normalize_relation_label(row['relation_label']))
    entity_to_id = {name: i for (i, name) in enumerate(sorted(entity_names))}
    relation_to_id = {name: i for (i, name) in enumerate(sorted(relation_labels))}
    mapping_index = mapping_index_by_entity_name(mappings or [])
    entities = [entity_metadata_row(i, name, mapping_index) for (name, i) in sorted(entity_to_id.items(), key=lambda item: item[1])]
    relations = [{'relation_id': i, 'relation_label': name} for (name, i) in sorted(relation_to_id.items(), key=lambda item: item[1])]
    mapped = []
    for row in events:
        relation_label = normalize_relation_label(row['relation_label'])
        mapped_row = {'s_id': entity_to_id[row['source_name']], 'r_id': relation_to_id[relation_label], 'o_id': entity_to_id[row['target_name']], 't': row['date'], 'source_name': row['source_name'], 'relation_label': relation_label, 'target_name': row['target_name']}
        for extra_key in ('time_granularity', 'domain_filter_stage', 'source_domain', 'target_domain'):
            if extra_key in row:
                mapped_row[extra_key] = row[extra_key]
        mapped_row['hash'] = event_hash(mapped_row)
        mapped.append(mapped_row)
    return (entities, relations, mapped)

def map_interval_ids(intervals: Sequence[Dict[str, Any]], entity_to_id: Dict[str, int], relation_to_id: Dict[str, int]) -> List[Dict[str, Any]]:
    mapped = []
    for row in intervals:
        relation_label = normalize_relation_label(row.get('relation_label', ''))
        if row.get('source_name') not in entity_to_id or row.get('target_name') not in entity_to_id or relation_label not in relation_to_id:
            continue
        mapped_row = dict(row)
        mapped_row['relation_label'] = relation_label
        mapped_row['s_id'] = entity_to_id[row['source_name']]
        mapped_row['r_id'] = relation_to_id[relation_label]
        mapped_row['o_id'] = entity_to_id[row['target_name']]
        mapped_row['hash'] = interval_hash(mapped_row)
        mapped.append(mapped_row)
    return mapped

def map_point_ids(points: Sequence[Dict[str, Any]], entity_to_id: Dict[str, int], relation_to_id: Dict[str, int]) -> List[Dict[str, Any]]:
    mapped = []
    for row in points:
        relation_label = normalize_relation_label(row.get('relation_label', ''))
        if row.get('source_name') not in entity_to_id or row.get('target_name') not in entity_to_id or relation_label not in relation_to_id:
            continue
        mapped_row = dict(row)
        mapped_row['relation_label'] = relation_label
        mapped_row['s_id'] = entity_to_id[row['source_name']]
        mapped_row['r_id'] = relation_to_id[relation_label]
        mapped_row['o_id'] = entity_to_id[row['target_name']]
        mapped_row['hash'] = point_hash(mapped_row)
        mapped.append(mapped_row)
    return mapped
RUNTIME_EXPORT_FIELDS = {'cache_hit'}

def strip_runtime_export_fields(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{key: value for (key, value) in dict(row).items() if key not in RUNTIME_EXPORT_FIELDS} for row in rows]

def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')

def relation_entity_sets(events: Sequence[Dict[str, Any]], intervals: Sequence[Dict[str, Any]], points: Optional[Sequence[Dict[str, Any]]]=None) -> Tuple[set, set]:
    entities = set()
    relations = set()
    for row in events:
        entities.add(row.get('s_id'))
        entities.add(row.get('o_id'))
        relations.add(row.get('r_id'))
    for row in intervals:
        entities.add(row.get('s_id'))
        entities.add(row.get('o_id'))
        relations.add(row.get('r_id'))
    for row in points or []:
        entities.add(row.get('s_id'))
        entities.add(row.get('o_id'))
        relations.add(row.get('r_id'))
    entities.discard(None)
    relations.discard(None)
    return (entities, relations)

def count_intervals_ending_in_window(intervals: Sequence[Dict[str, Any]], window: Tuple[str, str], granularity: str='day') -> int:
    (start, end) = window
    return sum((1 for row in intervals if row.get('t_end') and time_value_in_window(row['t_end'], start, end, granularity)))

def build_dataset_stats(splits: Dict[str, Dict[str, List[Dict[str, Any]]]], windows: Dict[str, Tuple[str, str]], config: Optional[PipelineConfig]=None) -> Dict[str, Any]:
    granularity = validate_time_granularity(config.dataset_time_granularity if config is not None else 'day')
    stats = {}
    for (split, payload) in splits.items():
        events = payload.get('events', [])
        intervals = payload.get('intervals', [])
        points = payload.get('points', [])
        (entities, relations) = relation_entity_sets(events, intervals, points)
        stats[split] = {'event_fact_count': len(events), 'interval_fact_count': len(intervals), 'point_fact_count': len(points), 'entity_count': len(entities), 'relation_count': len(relations), 'intervals_ending_in_window': count_intervals_ending_in_window(intervals, windows[split], granularity), 'time_granularity': granularity}
        if split == 'test':
            stats[split]['test_entity_overlap_availability'] = test_entity_overlap_availability_counts(events)
    return stats

def assert_dataset_integrity(splits: Dict[str, Dict[str, List[Dict[str, Any]]]], entity_ids: set, relation_ids: set, windows: Optional[Dict[str, Tuple[str, str]]]=None) -> None:
    for (split, payload) in splits.items():
        for row in payload.get('events', []):
            assert row['s_id'] in entity_ids and row['o_id'] in entity_ids, f'Unknown event entity in {split}'
            assert row['r_id'] in relation_ids, f'Unknown event relation in {split}'
        for row in payload.get('intervals', []):
            assert row['s_id'] in entity_ids and row['o_id'] in entity_ids, f'Unknown interval entity in {split}'
            assert row['r_id'] in relation_ids, f'Unknown interval relation in {split}'
            if row.get('t_start') and row.get('t_end'):
                assert compare_time_values(row['t_start'], row['t_end'], infer_row_time_granularity(row)) <= 0, f'Invalid interval in {split}: {row}'
        for row in payload.get('points', []):
            assert row['s_id'] in entity_ids and row['o_id'] in entity_ids, f'Unknown point entity in {split}'
            assert row['r_id'] in relation_ids, f'Unknown point relation in {split}'
            if windows is not None:
                assert time_value_in_window(row['t'], windows[split][0], windows[split][1], infer_row_time_granularity(row)), f'Point outside split window in {split}: {row}'

def dataset_knowledge_hash(splits: Dict[str, Dict[str, List[Dict[str, Any]]]], config: Optional[PipelineConfig]=None) -> str:
    hashes = []
    for split_name in sorted(splits):
        payload = splits[split_name]
        hashes.extend((row['hash'] for row in payload.get('events', [])))
        hashes.extend((row['hash'] for row in payload.get('intervals', [])))
        hashes.extend((row['hash'] for row in payload.get('points', [])))
    payload = {'knowledge_hashes': sorted(hashes)}
    if config is not None:
        payload['config'] = config_to_jsonable(config)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

def export_dataset(config: PipelineConfig, splits: Dict[str, Dict[str, List[Dict[str, Any]]]], entities: List[Dict[str, Any]], relations: List[Dict[str, Any]], mappings: List[Dict[str, Any]], filtered_test: List[Dict[str, Any]], windows: Dict[str, Tuple[str, str]], filtered_entity_facts: Optional[List[Dict[str, Any]]]=None, filtered_domain_facts: Optional[List[Dict[str, Any]]]=None, semantic_domain_entities: Optional[List[Dict[str, Any]]]=None) -> None:
    out = Path(config.output_dir) / dataset_knowledge_hash(splits, config)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'entities.csv', entities)
    write_csv(out / 'relations.csv', relations)
    write_csv(out / 'entity_wikidata_mapping.csv', strip_runtime_export_fields(mappings))
    write_csv(out / 'filtered_test_facts.csv', filtered_test)
    write_csv(out / 'filtered_entity_facts.csv', filtered_entity_facts or [])
    write_csv(out / 'filtered_semantic_domain_facts.csv', filtered_domain_facts or [])
    write_csv(out / 'semantic_domain_entities.csv', semantic_domain_entities or [])
    for (split, payload) in splits.items():
        write_csv(out / f'facts_{split}.csv', payload.get('events', []))
        write_csv(out / f'intervals_{split}.csv', payload.get('intervals', []))
        write_csv(out / f'points_{split}.csv', payload.get('points', []))
    split_hashes = {}
    for (split, payload) in splits.items():
        hashes = [row['hash'] for row in payload.get('events', [])] + [row['hash'] for row in payload.get('intervals', [])] + [row['hash'] for row in payload.get('points', [])]
        split_hashes[split] = split_hash(hashes)
    write_json(out / 'dataset_hashes.json', split_hashes)
    write_json(out / 'dataset_stats.json', build_dataset_stats(splits, windows, config))
    write_json(out / 'dataset_config.json', config_to_jsonable(config))

def read_csv_dict_rows(path: Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, 'r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))

def pipeline_config_from_json(config_data: Dict[str, Any], **overrides: Any) -> PipelineConfig:
    fields = set(PipelineConfig.__dataclass_fields__)
    payload = {key: value for (key, value) in dict(config_data or {}).items() if key in fields}
    payload.update({key: value for (key, value) in overrides.items() if key in fields})
    return PipelineConfig(**payload)

def load_pipeline_config_from_dataset_dir(dataset_dir: Path, **overrides: Any) -> PipelineConfig:
    dataset_dir = Path(dataset_dir)
    config_path = dataset_dir / 'dataset_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'Missing dataset_config.json in {dataset_dir}')
    config_data = json.loads(config_path.read_text(encoding='utf-8'))
    return pipeline_config_from_json(config_data, **overrides)

def train_entity_names_from_dataset_dir(dataset_dir: Path) -> set:
    train_rows = read_csv_dict_rows(Path(dataset_dir) / 'facts_train.csv')
    entities = set()
    for row in train_rows:
        for key in ('source_name', 'target_name'):
            value = str(row.get(key, '')).strip()
            if value:
                entities.add(value)
    return entities

def numeric_id(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return -1

def base_entity_rows_by_name(dataset_dir: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_csv_dict_rows(Path(dataset_dir) / 'entities.csv')
    by_name = {}
    for row in rows:
        name = str(row.get('entity_name') or row.get('name') or row.get('label') or '').strip()
        if name:
            by_name[name] = dict(row)
    return by_name

def base_relation_rows_by_label(dataset_dir: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_csv_dict_rows(Path(dataset_dir) / 'relations.csv')
    by_label = {}
    for row in rows:
        label = normalize_relation_label(str(row.get('relation_label') or ''))
        if label:
            item = dict(row)
            item['relation_label'] = label
            by_label[label] = item
    return by_label

def filter_online_intervals_for_window(intervals: Sequence[Dict[str, Any]], window: Tuple[str, str], granularity: str='day') -> List[Dict[str, Any]]:
    kept = []
    (window_start, window_end) = window
    for row in intervals:
        if row.get('t_start') and row.get('t_end') and (compare_time_values(row['t_start'], row['t_end'], granularity) > 0):
            continue
        if intervals_overlap(row.get('t_start', ''), row.get('t_end', ''), window_start, window_end, granularity):
            kept.append(dict(row))
    return kept

def online_interval_counts(intervals: Sequence[Dict[str, Any]], window: Tuple[str, str], granularity: str='day') -> Dict[str, int]:
    open_count = 0
    overlap_count = 0
    for row in intervals:
        if not row.get('t_end'):
            open_count += 1
        if intervals_overlap(row.get('t_start', ''), row.get('t_end', ''), window[0], window[1], granularity):
            overlap_count += 1
    return {'open_interval_fact_count': open_count, 'window_overlapping_interval_fact_count': overlap_count}

def online_entity_metadata_row(entity_id: int, entity_name: str, mapping_index: Dict[str, Dict[str, Any]], base_rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if entity_name in mapping_index:
        return entity_metadata_row(entity_id, entity_name, mapping_index)
    base = dict(base_rows.get(entity_name, {}))
    if base:
        base['entity_id'] = entity_id
        base.setdefault('entity_name', entity_name)
        base.setdefault('name', entity_name)
        base.setdefault('label', entity_name)
        return base
    return entity_metadata_row(entity_id, entity_name, mapping_index)

def build_online_id_maps(events: Sequence[Dict[str, Any]], intervals: Sequence[Dict[str, Any]], base_dataset_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    base_entities = base_entity_rows_by_name(base_dataset_dir)
    base_relations = base_relation_rows_by_label(base_dataset_dir)
    entity_names = set()
    relation_labels = set()
    for row in events:
        entity_names.add(row['source_name'])
        entity_names.add(row['target_name'])
        relation_labels.add(normalize_relation_label(row['relation_label']))
    for row in intervals:
        if row.get('source_name') and row.get('target_name'):
            entity_names.add(row['source_name'])
            entity_names.add(row['target_name'])
        if row.get('relation_label'):
            relation_labels.add(normalize_relation_label(row['relation_label']))
    entity_to_id = {}
    used_entity_ids = set()
    for name in sorted(entity_names):
        base_id = numeric_id(base_entities.get(name, {}).get('entity_id'))
        if base_id >= 0:
            entity_to_id[name] = base_id
            used_entity_ids.add(base_id)
    next_entity_id = max(used_entity_ids | {numeric_id(row.get('entity_id')) for row in base_entities.values()} | {-1}) + 1
    for name in sorted(entity_names):
        if name not in entity_to_id:
            while next_entity_id in used_entity_ids:
                next_entity_id += 1
            entity_to_id[name] = next_entity_id
            used_entity_ids.add(next_entity_id)
            next_entity_id += 1
    relation_to_id = {}
    used_relation_ids = set()
    for label in sorted(relation_labels):
        base_id = numeric_id(base_relations.get(label, {}).get('relation_id'))
        if base_id >= 0:
            relation_to_id[label] = base_id
            used_relation_ids.add(base_id)
    next_relation_id = max(used_relation_ids | {numeric_id(row.get('relation_id')) for row in base_relations.values()} | {-1}) + 1
    for label in sorted(relation_labels):
        if label not in relation_to_id:
            while next_relation_id in used_relation_ids:
                next_relation_id += 1
            relation_to_id[label] = next_relation_id
            used_relation_ids.add(next_relation_id)
            next_relation_id += 1
    return (entity_to_id, relation_to_id)

def build_online_test_tables(events: Sequence[Dict[str, Any]], intervals: Sequence[Dict[str, Any]], base_dataset_dir: Path, mappings: Sequence[Dict[str, Any]], config: PipelineConfig, online_window: Optional[Tuple[str, str]]=None) -> Dict[str, Any]:
    base_dataset_dir = Path(base_dataset_dir)
    granularity = validate_time_granularity(config.dataset_time_granularity)
    if online_window is None:
        event_dates = sorted({row.get('date') for row in events if row.get('date')})
        if event_dates:
            online_window = (event_dates[0], event_dates[-1])
        else:
            online_window = (config.history_start_date, config.history_start_date)
    train_entities = train_entity_names_from_dataset_dir(base_dataset_dir)
    annotated_events = []
    for row in events:
        item = dict(row)
        item['test_entity_overlap_type'] = classify_test_entity_overlap(train_entities, item)
        annotated_events.append(item)
    online_intervals = filter_online_intervals_for_window(intervals, online_window, granularity)
    (entity_to_id, relation_to_id) = build_online_id_maps(annotated_events, online_intervals, base_dataset_dir)
    mapping_index = mapping_index_by_entity_name(mappings or [])
    base_entities = base_entity_rows_by_name(base_dataset_dir)
    entities = [online_entity_metadata_row(entity_id, name, mapping_index, base_entities) for (name, entity_id) in sorted(entity_to_id.items(), key=lambda item: item[1])]
    relations = [{'relation_id': relation_id, 'relation_label': label} for (label, relation_id) in sorted(relation_to_id.items(), key=lambda item: item[1])]
    mapped_events = []
    for row in annotated_events:
        relation_label = normalize_relation_label(row['relation_label'])
        mapped_row = dict(row)
        mapped_row['relation_label'] = relation_label
        mapped_row['s_id'] = entity_to_id[row['source_name']]
        mapped_row['r_id'] = relation_to_id[relation_label]
        mapped_row['o_id'] = entity_to_id[row['target_name']]
        mapped_row['t'] = row.get('date', row.get('t', ''))
        mapped_row['hash'] = event_hash(mapped_row)
        mapped_events.append(mapped_row)
    mapped_intervals = map_interval_ids(online_intervals, entity_to_id, relation_to_id)
    split_payload = {'events': mapped_events, 'intervals': mapped_intervals, 'points': []}
    stats = build_dataset_stats({'test': split_payload}, {'test': online_window}, config)['test']
    stats.update(online_interval_counts(mapped_intervals, online_window, granularity))
    stats['online_window_start'] = online_window[0]
    stats['online_window_end'] = online_window[1]
    return {'entities': entities, 'relations': relations, 'events': mapped_events, 'intervals': mapped_intervals, 'points': [], 'stats': stats, 'window': online_window, 'entity_to_id': entity_to_id, 'relation_to_id': relation_to_id}

def filter_mappings_for_entity_names(mappings: Sequence[Dict[str, Any]], entity_names: set) -> List[Dict[str, Any]]:
    filtered = []
    seen = set()
    for row in mappings:
        names = {str(row.get('original_text', '')).strip(), str(row.get('wikidata_label', '')).strip()}
        if names & entity_names:
            key = row.get('qid') or row.get('original_text') or row.get('wikidata_label')
            if key and key not in seen:
                filtered.append(dict(row))
                seen.add(key)
    return filtered

def dataset_fact_rows_for_debias(dataset_dir: Path, filename: str='facts_train.csv') -> List[Dict[str, Any]]:
    rows = read_csv_dict_rows(Path(dataset_dir) / filename)
    facts = []
    for row in rows:
        source = str(row.get('source_name') or row.get('s') or '').strip()
        relation = str(row.get('relation_label') or row.get('r') or '').strip()
        target = str(row.get('target_name') or row.get('o') or '').strip()
        timestamp = str(row.get('date') or row.get('t') or '').strip()
        if source and relation and target and timestamp:
            facts.append({'source_name': source, 'relation_label': relation, 'target_name': target, 'date': timestamp})
    return facts

def export_online_test_artifact(base_dataset_dir: Path, config: PipelineConfig, tables: Dict[str, Any], mappings: Sequence[Dict[str, Any]], semantic_domain_entities: Sequence[Dict[str, Any]], filtered_entity_facts: Sequence[Dict[str, Any]], filtered_domain_facts: Sequence[Dict[str, Any]], latest_gdelt_date: str, filtered_test_facts: Optional[Sequence[Dict[str, Any]]]=None) -> Path:
    out = Path(base_dataset_dir) / 'online_test'
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'facts_online_test.csv', tables['events'])
    write_csv(out / 'intervals_online_test.csv', tables['intervals'])
    write_csv(out / 'points_online_test.csv', [])
    write_csv(out / 'entities_online.csv', tables['entities'])
    write_csv(out / 'relations_online.csv', tables['relations'])
    write_csv(out / 'entity_wikidata_mapping_online.csv', strip_runtime_export_fields(mappings))
    write_csv(out / 'semantic_domain_entities_online.csv', semantic_domain_entities)
    write_csv(out / 'filtered_entity_facts_online.csv', filtered_entity_facts)
    write_csv(out / 'filtered_semantic_domain_facts_online.csv', filtered_domain_facts)
    write_csv(out / 'filtered_test_facts_online.csv', list(filtered_test_facts or []))
    split_hashes = {'online_test': split_hash([row['hash'] for row in tables['events']] + [row['hash'] for row in tables['intervals']])}
    write_json(out / 'online_test_hashes.json', split_hashes)
    stats = dict(tables['stats'])
    stats['gdelt_latest_date'] = latest_gdelt_date
    stats['semantic_domain'] = config.semantic_domain
    stats['dataset_time_granularity'] = config.dataset_time_granularity
    write_json(out / 'online_test_stats.json', stats)
    write_json(out / 'online_test_config.json', config_to_jsonable(config))
    return out

def build_online_test_artifact(base_dataset_dir: Path, config: Optional[PipelineConfig]=None, latest_gdelt_date: Optional[str]=None) -> Dict[str, Any]:
    base_dataset_dir = Path(base_dataset_dir)
    if config is None:
        config = load_pipeline_config_from_dataset_dir(base_dataset_dir)
    config = pipeline_config_from_json(config_to_jsonable(config), setting='online_testing', dataset_time_granularity='day', test_length_days=30, include_point_time_facts=False, include_interval_facts=True, enable_llm_online_interval_verification=False, enable_online_interval_verification=False, llm_entity_resolution_workers=1, llm_entity_resolution_request_interval_seconds=max(float(getattr(config, 'llm_entity_resolution_request_interval_seconds', 0.5) or 0.0), 1.0), wikidata_min_request_interval_seconds=max(float(getattr(config, 'wikidata_min_request_interval_seconds', 2.0) or 0.0), 2.0))
    ensure_output_dirs(config)
    validate_llm_configuration(config)
    print(f'[OnlineTest] Building online test for {base_dataset_dir}: domain={config.semantic_domain}', flush=True)
    (entity_name_map, allowed_entity_codes, event_map) = load_official_mappings(config)
    conceptual_entities = load_conceptual_entity_blacklist()
    latest = latest_gdelt_date or discover_latest_gdelt_date(config)
    online_dates = build_test_window_dates(config, latest)
    online_window = normalize_time_window_from_dates(online_dates, 'day')
    print(f'[OnlineTest] Collecting GDELT online window: {online_window[0]}..{online_window[1]}', flush=True)
    candidate_events = collect_gdelt_events(online_dates, config, event_map, conceptual_entities, entity_name_map)
    candidate_events = normalize_event_times(candidate_events, config)
    all_event_entities = sorted({row['source_name'] for row in candidate_events} | {row['target_name'] for row in candidate_events})
    print(f'[OnlineTest] Resolving {len(all_event_entities)} online event entity/entities', flush=True)
    mappings = match_entities_to_wikidata(all_event_entities, config, candidate_events)
    canonical_events = canonicalize_events_with_mappings(candidate_events, mappings, config)
    (canonical_events, filtered_entity_facts) = filter_events_by_entity_acceptance_and_self_loops(canonical_events, mappings, config.wikidata_match_threshold)
    entity_domain_map = build_entity_domain_map(mappings, config)
    semantic_domain_entities = semantic_domain_entity_rows(mappings, config)
    (canonical_events, filtered_domain_facts) = filter_events_by_semantic_domain(canonical_events, entity_domain_map, config)
    print(f'[OnlineTest] Domain-filtered online events: kept={len(canonical_events)}, filtered={len(filtered_domain_facts)}', flush=True)
    base_train_events = dataset_fact_rows_for_debias(base_dataset_dir, 'facts_train.csv')
    (canonical_events, filtered_online_debias) = debias_test_facts(base_train_events, canonical_events, config)
    for row in filtered_online_debias:
        row['filter_stage'] = 'online_test_debias'
    print(f'[OnlineTest] Debiased online events: kept={len(canonical_events)}, filtered={len(filtered_online_debias)}', flush=True)
    online_entity_names = {row['source_name'] for row in canonical_events} | {row['target_name'] for row in canonical_events}
    interval_seed_mappings = filter_mappings_for_entity_names(mappings, online_entity_names)
    interval_windows = {'test': online_window}
    (intervals, extended_mappings) = build_intervals_from_wikidata_mappings(interval_seed_mappings, interval_windows, config)
    tables = build_online_test_tables(canonical_events, intervals, base_dataset_dir, extended_mappings, config, online_window)
    out = export_online_test_artifact(base_dataset_dir, config, tables, extended_mappings, semantic_domain_entities, filtered_entity_facts, filtered_domain_facts, latest, filtered_online_debias)
    assert_dataset_integrity({'test': {'events': tables['events'], 'intervals': tables['intervals'], 'points': []}}, {row['entity_id'] for row in tables['entities']}, {row['relation_id'] for row in tables['relations']}, {'test': online_window})
    print(f'[OnlineTest] Wrote online_test artifact: {out}', flush=True)
    print(json.dumps(tables['stats'], ensure_ascii=False, indent=2), flush=True)
    return {'output_dir': str(out), 'base_dataset_dir': str(base_dataset_dir), 'config': config, 'latest_gdelt_date': latest, 'online_dates': online_dates, 'window': online_window, 'events': tables['events'], 'intervals': tables['intervals'], 'entities': tables['entities'], 'relations': tables['relations'], 'mappings': extended_mappings, 'filtered_test': filtered_online_debias, 'stats': tables['stats']}

def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    print('[Pipeline] progress-enabled-v2', flush=True)
    setting = validate_pipeline_setting(config)
    print(f'[Pipeline] Starting dataset build: setting={setting}, output_dir={config.output_dir}', flush=True)
    ensure_output_dirs(config)
    validate_llm_configuration(config)
    print('[Pipeline] Loading GDELT lookup mappings', flush=True)
    (entity_name_map, allowed_entity_codes, event_map) = load_official_mappings(config)
    conceptual_entities = load_conceptual_entity_blacklist()
    print(f'[Pipeline] Loaded {len(conceptual_entities)} conceptual blacklist entries', flush=True)
    history_dates = date_range_inclusive(config.history_start_date, config.history_end_date)
    latest = discover_latest_gdelt_date(config) if setting == 'online_testing' else None
    test_dates = build_test_window_dates(config, latest)
    validate_pipeline_time_window_alignment(config, history_dates, test_dates)
    (train_dates_raw, valid_dates_raw) = split_history_dates(history_dates, config.valid_length_days)
    print(f'[Pipeline] Collecting history window: {history_dates[0]}..{history_dates[-1]}', flush=True)
    history_events = collect_gdelt_events(history_dates, config, event_map, conceptual_entities, entity_name_map)
    print(f'[Pipeline] Collecting test window: {test_dates[0]}..{test_dates[-1]}', flush=True)
    candidate_test_events = collect_gdelt_events(test_dates, config, event_map, conceptual_entities, entity_name_map)
    (train_events, valid_events) = split_history_events_by_date_windows(history_events, train_dates_raw, valid_dates_raw)
    train_events = normalize_event_times(train_events, config)
    valid_events = normalize_event_times(valid_events, config)
    candidate_test_events = normalize_event_times(candidate_test_events, config)
    print(f'[Pipeline] Split history: train={len(train_events)}, valid={len(valid_events)}', flush=True)
    (debiased_test_events, filtered_test) = debias_test_facts(train_events, candidate_test_events, config)
    print(f'[Pipeline] Debiased test: kept={len(debiased_test_events)}, filtered={len(filtered_test)}', flush=True)
    final_test_events = debiased_test_events
    print(f'[Pipeline] Final test facts retained for experiment-time transductive/inductive masking: {len(final_test_events)}', flush=True)
    all_event_entities = sorted({row['source_name'] for row in train_events + valid_events + final_test_events} | {row['target_name'] for row in train_events + valid_events + final_test_events})
    mappings = match_entities_to_wikidata(all_event_entities, config, train_events + valid_events + final_test_events)
    canonical_train_events = canonicalize_events_with_mappings(train_events, mappings, config)
    canonical_valid_events = canonicalize_events_with_mappings(valid_events, mappings, config)
    canonical_test_events = canonicalize_events_with_mappings(final_test_events, mappings, config)
    (canonical_train_events, filtered_train_entities) = filter_events_by_entity_acceptance_and_self_loops(canonical_train_events, mappings, config.wikidata_match_threshold)
    (canonical_valid_events, filtered_valid_entities) = filter_events_by_entity_acceptance_and_self_loops(canonical_valid_events, mappings, config.wikidata_match_threshold)
    (canonical_test_events, filtered_test_entities) = filter_events_by_entity_acceptance_and_self_loops(canonical_test_events, mappings, config.wikidata_match_threshold)
    filtered_entity_facts = filtered_train_entities + filtered_valid_entities + filtered_test_entities
    print(f'[Pipeline] Entity/self-loop filter removed {len(filtered_entity_facts)} event fact(s)', flush=True)
    entity_domain_map = build_entity_domain_map(mappings, config)
    semantic_domain_entities = semantic_domain_entity_rows(mappings, config)
    (canonical_train_events, filtered_train_domain) = filter_events_by_semantic_domain(canonical_train_events, entity_domain_map, config)
    (canonical_valid_events, filtered_valid_domain) = filter_events_by_semantic_domain(canonical_valid_events, entity_domain_map, config)
    (canonical_test_events, filtered_test_domain) = filter_events_by_semantic_domain(canonical_test_events, entity_domain_map, config)
    filtered_domain_facts = filtered_train_domain + filtered_valid_domain + filtered_test_domain
    print(f'[Pipeline] Semantic domain filter ({config.semantic_domain}) removed {len(filtered_domain_facts)} event fact(s)', flush=True)
    (canonical_test_events, filtered_test_canonical) = debias_canonical_test_facts(canonical_train_events, canonical_test_events, config)
    if filtered_test_canonical:
        filtered_test.extend(filtered_test_canonical)
        print(f'[Pipeline] Canonical post-filter removed {len(filtered_test_canonical)} test event fact(s)', flush=True)
    train_window = event_or_date_window(canonical_train_events, train_dates_raw, config.dataset_time_granularity, normalize_time_window_from_dates(history_dates[:1], config.dataset_time_granularity))
    valid_window = event_or_date_window(canonical_valid_events, valid_dates_raw, config.dataset_time_granularity, train_window)
    test_window = event_or_date_window(canonical_test_events, test_dates, config.dataset_time_granularity)
    windows = {'train': train_window, 'valid': valid_window, 'test': test_window}
    (intervals, points, mappings) = build_temporal_knowledge_from_wikidata_mappings(mappings, windows, config)
    semantic_domain_entities = semantic_domain_entity_rows(mappings, config)
    print(f'[Pipeline] Built {len(intervals)} raw interval fact(s) and {len(points)} raw point fact(s)', flush=True)
    interval_splits_raw = assign_interval_splits(intervals, windows, config.dataset_time_granularity)
    point_splits_raw = assign_point_splits(points, windows, config.dataset_time_granularity)
    raw_splits = {'train': {'events': canonical_train_events, 'intervals': interval_splits_raw['train'], 'points': point_splits_raw['train']}, 'valid': {'events': canonical_valid_events, 'intervals': interval_splits_raw['valid'], 'points': point_splits_raw['valid']}, 'test': {'events': canonical_test_events, 'intervals': interval_splits_raw['test'], 'points': point_splits_raw['test']}}
    all_events = raw_splits['train']['events'] + raw_splits['valid']['events'] + raw_splits['test']['events']
    all_intervals = raw_splits['train']['intervals'] + raw_splits['valid']['intervals'] + raw_splits['test']['intervals']
    all_points = raw_splits['train']['points'] + raw_splits['valid']['points'] + raw_splits['test']['points']
    (entities, relations, mapped_events) = build_id_tables(all_events, all_intervals, mappings, all_points)
    event_key_to_row = {(row['source_name'], row['relation_label'], row['target_name'], row['t']): row for row in mapped_events}
    mapped_train = [event_key_to_row[row['source_name'], row['relation_label'], row['target_name'], row['date']] for row in raw_splits['train']['events']]
    mapped_valid = [event_key_to_row[row['source_name'], row['relation_label'], row['target_name'], row['date']] for row in raw_splits['valid']['events']]
    mapped_test = [event_key_to_row[row['source_name'], row['relation_label'], row['target_name'], row['date']] for row in raw_splits['test']['events']]
    mapped_test = annotate_test_entity_overlap_types(mapped_train, mapped_test)
    entity_to_id = {row['entity_name']: row['entity_id'] for row in entities}
    relation_to_id = {row['relation_label']: row['relation_id'] for row in relations}
    interval_splits = {split: map_interval_ids(payload['intervals'], entity_to_id, relation_to_id) for (split, payload) in raw_splits.items()}
    point_splits = {split: map_point_ids(payload['points'], entity_to_id, relation_to_id) for (split, payload) in raw_splits.items()}
    splits = {'train': {'events': mapped_train, 'intervals': interval_splits['train'], 'points': point_splits['train']}, 'valid': {'events': mapped_valid, 'intervals': interval_splits['valid'], 'points': point_splits['valid']}, 'test': {'events': mapped_test, 'intervals': interval_splits['test'], 'points': point_splits['test']}}
    entity_ids = {row['entity_id'] for row in entities}
    relation_ids = {row['relation_id'] for row in relations}
    assert_dataset_integrity(splits, entity_ids, relation_ids, windows)
    print('[Pipeline] Exporting CSV/JSON artifacts', flush=True)
    export_dataset(config, splits, entities, relations, mappings, filtered_test, windows, filtered_entity_facts, filtered_domain_facts, semantic_domain_entities)
    print('[Pipeline] Export complete', flush=True)
    return {'splits': splits, 'entities': entities, 'relations': relations, 'mappings': mappings, 'filtered_test': filtered_test, 'filtered_entity_facts': filtered_entity_facts, 'filtered_domain_facts': filtered_domain_facts, 'semantic_domain_entities': semantic_domain_entities, 'windows': windows}
