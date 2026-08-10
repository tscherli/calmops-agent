"""Google Sheets backend for mood log rows.

OAuth Desktop flow (credentials.json + cached token.json), narrowest workable
scope (drive.file -- limited to files this app itself creates/uses), single
spreadsheet identified by MOOD_SHEET_ID. Any write failure falls back to a
local JSONL file rather than losing the entry; append_mood_row/log_mood never
raise past this module's boundary -- callers get a structured status instead.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_core.tools import tool

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
SHEET_TAB = "Log"
SHEET_COLUMNS = [
    "timestamp",
    "row_id",
    "valence",
    "energy_level",
    "circumstances",
    "measure_taken",
    "helpfulness",
    "note",
]

BASE_DIR = Path(__file__).resolve().parent.parent
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
TOKEN_PATH = BASE_DIR / "token.json"
PENDING_LOG_PATH = BASE_DIR / "state" / "pending_sheet_log.jsonl"

Valence = Literal["good", "neutral", "bad"]
EnergyLevel = Literal["low", "medium", "high"]


def _get_credentials() -> Credentials:
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
    return creds


def get_sheets_service():
    """Build a real Google Sheets API client via OAuth. Only called when no
    mock `service` is injected -- keeps this module importable/testable
    without real credentials on disk."""
    creds = _get_credentials()
    return build("sheets", "v4", credentials=creds)


def _write_pending_fallback(row: dict[str, Any]) -> None:
    PENDING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PENDING_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _build_note(situation: str | None, thoughts: str | None, need: str | None, next_step: str | None) -> str:
    parts = []
    if situation:
        parts.append(f"situation: {situation}")
    if thoughts:
        parts.append(f"thoughts: {thoughts}")
    if need:
        parts.append(f"need: {need}")
    if next_step:
        parts.append(f"next_step: {next_step}")
    return " | ".join(parts)


def append_mood_row(
    valence: Valence,
    circumstances: str,
    energy_level: EnergyLevel | None = None,
    measure_taken: str | None = None,
    helpfulness: str | None = None,
    situation: str | None = None,
    thoughts: str | None = None,
    need: str | None = None,
    next_step: str | None = None,
    *,
    service: Any = None,
    sheet_id: str | None = None,
) -> dict[str, Any]:
    """Append one structured mood row to the Sheet. Never raises -- any
    failure (missing config, API/auth error) falls back to a local JSONL
    file and returns status="error" instead.

    energy_level is optional and orthogonal to valence -- it captures
    fatigue/exhaustion (an emotional-exhaustion-adjacent signal from burnout
    research) as its own trend, distinct from day-to-day mood. See README
    "Why energy_level exists" for the non-clinical framing/sources."""
    row_id = str(uuid.uuid4())
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "row_id": row_id,
        "valence": valence,
        "energy_level": energy_level or "",
        "circumstances": circumstances,
        "measure_taken": measure_taken or "",
        "helpfulness": helpfulness or "",
        "note": _build_note(situation, thoughts, need, next_step),
    }

    sheet_id = sheet_id or os.environ.get("MOOD_SHEET_ID")
    if not sheet_id:
        _write_pending_fallback(row)
        return {"status": "error", "row_id": row_id, "error": "MOOD_SHEET_ID not configured"}

    try:
        svc = service or get_sheets_service()
        values = [[row[col] for col in SHEET_COLUMNS]]
        svc.spreadsheets().values().append(
            spreadsheetId=sheet_id,
            range=f"{SHEET_TAB}!A1",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": values},
        ).execute()
        return {"status": "ok", "row_id": row_id}
    except Exception as exc:
        _write_pending_fallback(row)
        return {"status": "error", "row_id": row_id, "error": str(exc)}


@tool
def log_mood(
    valence: Valence,
    circumstances: str,
    energy_level: str = "",
    measure_taken: str = "",
    helpfulness: str = "",
    situation: str = "",
    thoughts: str = "",
    need: str = "",
    next_step: str = "",
) -> dict[str, Any]:
    """Log a structured mood entry to the user's private Google Sheet.

    valence: "good", "neutral", or "bad".
    circumstances: brief description of what's going on.
    energy_level: "low", "medium", or "high", only if fatigue/energy came up -- distinct from
    valence (you can be low-energy-but-content, or high-energy-but-stressed).
    measure_taken / helpfulness: only if the user named a coping action and how well it worked.
    situation / thoughts / need / next_step: deeper-structuring fields, only for heavier check-ins
    that went through the structuring sequence.
    """
    return append_mood_row(
        valence=valence,
        circumstances=circumstances,
        energy_level=energy_level or None,
        measure_taken=measure_taken or None,
        helpfulness=helpfulness or None,
        situation=situation or None,
        thoughts=thoughts or None,
        need=need or None,
        next_step=next_step or None,
    )
