import inspect
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo  # Python 3.9+


LOG_DIR = "logs"
IST = ZoneInfo("Asia/Kolkata")

def _ensure_logs_dir():
    os.makedirs(LOG_DIR, exist_ok=True)

# get name of the calling script to form log file name
def _resolve_log_file(log_name: Optional[str]) -> str:
    if log_name:
        candidate = Path(log_name)
    else:
        stack = inspect.stack()
        caller_frame = stack[2] if len(stack) > 2 else stack[-1]
        candidate = Path(caller_frame.filename)

    return os.path.join(LOG_DIR, f"{candidate.stem}.jsonl")


def log_run(entry: dict, log_name: Optional[str] = None):
    """Append a JSON line describing this run to logs/<caller>.jsonl."""
    _ensure_logs_dir()
    current_time_iso_ist = datetime.now(IST).replace(microsecond=0).isoformat()
    enriched_entry = {
        "time": current_time_iso_ist,
        **entry,
    }
    log_file = _resolve_log_file(log_name)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(enriched_entry) + "\n")
