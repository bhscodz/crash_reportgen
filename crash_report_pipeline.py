"""
crash_report_pipeline.py
PDF crash reports → Gemma 4 (LiteRT-LM) → Excel

Uses pymupdf4llm to convert each PDF page to clean Markdown before sending to
the LLM.  Markdown pipe-tables are far more reliable input than raw text dumps,
and eliminate the need for image uploads entirely.

Speed improvements over the original per-page approach:
  1. pymupdf4llm renders tables as | Label | Value | rows — LLM reads structure
  2. Near-empty pages are filtered before any inference
  3. Pages are batched (MAX_PAGES_PER_CHUNK at a time) — far fewer LLM calls
  4. AccidentTracker deduplicates and merges records incrementally as they arrive

Usage (Colab / Linux):
    python crash_report_pipeline.py report1.pdf report2.pdf -o crash_reports.xlsx
"""

import argparse
import glob
import json
import os
import re

import litert_lm
import openpyxl
import pymupdf4llm
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


# ── Model path ────────────────────────────────────────────────────────────────

_HF_SNAPSHOT = (
    "/root/.cache/huggingface/hub/"
    "models--litert-community--gemma-4-E4B-it-litert-lm/snapshots/"
    "10848a680ad0ab45b556566152371b38c238d6f0"
)


def find_model_path() -> str:
    for pattern in [
        f"{_HF_SNAPSHOT}/**/*.litertlm",
        "/root/.cache/huggingface/hub/models--litert-community--gemma-4-E4B-it-litert-lm/**/*.litertlm",
        "**/*.litertlm",
    ]:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(
        "No .litertlm file found. Download the model with:\n"
        "  hf download litert-community/gemma-4-E4B-it-litert-lm"
    )


# ── PDF → Markdown extraction ─────────────────────────────────────────────────

def extract_pages_as_markdown(pdf_path: str) -> list[dict]:
    """
    Convert every page of a PDF to Markdown using pymupdf4llm.

    pymupdf4llm detects tables and renders them as Markdown pipe-tables
    (| Label | Value |), which gives the LLM clean, structured input instead
    of ambiguous whitespace-delimited text.

    No images are extracted — the Markdown already captures all table data,
    so no vision encoder is needed.

    Returns a list of {"page_num": int, "text": str} dicts, one per page.
    """
    chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    pages  = []
    for i, chunk in enumerate(chunks):
        meta   = chunk.get("metadata", {})
        # pymupdf4llm ≤1.24 uses "page", ≥1.27 uses "page_number" (both 0-indexed).
        page_0 = meta.get("page", meta.get("page_number", i))
        pages.append({"page_num": page_0 + 1, "text": chunk["text"]})
    return pages


# ── Page helpers ──────────────────────────────────────────────────────────────

TEXT_MIN_CHARS      = 100  # pages with fewer chars are skipped (blank / decorative)
MAX_PAGES_PER_CHUNK = 5    # pages sent to the LLM in one conversation turn


def is_near_empty(text: str) -> bool:
    """Return True for blank or near-blank pages that carry no useful data."""
    return len(text.strip()) < TEXT_MIN_CHARS


def chunk_pages(pages: list, max_per_chunk: int = MAX_PAGES_PER_CHUNK) -> list:
    """
    Split pages into consecutive groups of at most max_per_chunk.

    Example: a 9-page report becomes 2 chunks (5 + 4 pages) instead of
    9 separate LLM calls.
    """
    return [pages[i : i + max_per_chunk] for i in range(0, len(pages), max_per_chunk)]


# ── LLM extraction ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a crash report extraction specialist. You receive Markdown-formatted \
text from one or more pages of a crash report document. Extract every unique \
crash incident and return ONLY raw JSON — no markdown fences, no explanation.

Output format:
{
  "crashes": [
    {
      "column_name": "value or null",
      ...
    }
  ]
}

Rules:
- Decide column names yourself based on what data is present in this document.
  Use snake_case (e.g. "report_number", "crash_date", "road_surface").
- Each crash is a flat key-value dict — no nested objects.
- Use the same column name consistently for the same type of data.
- Counts (vehicles, injuries, fatalities) must be integers, not strings.
- Absent or unclear fields → null. Do not guess or invent values.
- If multiple pages describe the same crash, merge them into one record.
- If there is no crash data at all, return: {"crashes": []}"""


def parse_json_response(raw_text: str) -> dict:
    """Strip markdown fences and extract the first valid JSON object."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("`").strip()
    match   = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"crashes": []}


def extract_crashes_from_chunk(engine, chunk: list, pdf_filename: str) -> dict:
    """
    Send a chunk of Markdown pages to Gemma 4 in one conversation turn.

    Pages are separated by HTML comment markers so the model can tell where
    one page ends and the next begins.  No images are sent — pymupdf4llm's
    Markdown output already captures all table structure.
    """
    system_msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    ]

    pages_md = "\n\n".join(
        f"<!-- PAGE {p['page_num']} -->\n{p['text']}"
        for p in chunk
    )

    content = [{
        "type": "text",
        "text": (
            f"Document: {pdf_filename}\n\n"
            f"{pages_md}\n\n"
            "Extract all crash incidents. Return JSON only."
        ),
    }]

    with engine.create_conversation(messages=system_msgs) as conv:
        response = conv.send_message({"role": "user", "content": content})

    return parse_json_response(response["content"][0]["text"])


# ── Accident tracker ──────────────────────────────────────────────────────────

class AccidentTracker:
    """
    Collects crash records and merges duplicates on the fly.

    A record is a duplicate when its fingerprint (built from identifying fields
    like report_number, or date+location) matches an existing record.  On a
    match, new data is merged in: narrative fields are appended, all other
    fields keep the first-seen value.

    Usage:
        tracker = AccidentTracker()
        tracker.add(crash_dict, source="report.pdf, p.1-5", page=1)
        print(tracker.count())      # unique accidents so far
        records = tracker.records() # final list → pass to write_excel()
    """

    # Narrative fields get unique values appended on merge (not first-wins).
    _NARRATIVE = {
        "narrative", "description", "details", "notes",
        "remarks", "summary", "comments", "observation",
    }

    # Field combinations tried in order to fingerprint a crash.
    # First combo where ALL fields are non-empty wins.
    _FINGERPRINT_COMBOS = [
        ["report_number"], ["case_number"], ["incident_number"],
        ["date", "time", "location"], ["crash_date", "crash_time", "location"],
        ["date", "location"],         ["crash_date", "location"],
        ["date", "time", "address"],
    ]

    def __init__(self):
        self._keyed:   dict[str, dict] = {}  # fingerprint → record
        self._unkeyed: list[dict]      = []  # records with no fingerprint

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, crash: dict, source: str, page) -> None:
        """Add a crash dict; merge it if it matches an existing record."""
        fp = self._fingerprint(crash)
        if fp and fp in self._keyed:
            self._merge_into(fp, crash, source)
        elif fp:
            self._keyed[fp] = {"crash": crash, "source": source, "page": page}
        else:
            self._unkeyed.append({"crash": crash, "source": source, "page": page})

    def count(self) -> int:
        """Number of unique accidents tracked so far."""
        return len(self._keyed) + len(self._unkeyed)

    def records(self) -> list[dict]:
        """Deduplicated record list, ready for write_excel()."""
        return list(self._keyed.values()) + self._unkeyed

    # ── private helpers ───────────────────────────────────────────────────────

    def _fingerprint(self, crash: dict) -> str | None:
        for combo in self._FINGERPRINT_COMBOS:
            vals = [str(crash.get(f) or "").strip().lower() for f in combo]
            if all(vals):
                return "|".join(vals)
        return None

    def _merge_into(self, fp: str, new_crash: dict, new_source: str) -> None:
        existing = self._keyed[fp]
        for key, val in new_crash.items():
            if val is None or val == "":
                continue
            if key in self._NARRATIVE:
                current = existing["crash"].get(key, "")
                snippet = str(val).strip()
                if snippet not in current:
                    existing["crash"][key] = f"{current} | {snippet}".strip(" |")
            else:
                existing["crash"].setdefault(key, val)  # first-wins

        if new_source not in existing["source"]:
            existing["source"] += f" + {new_source}"


# ── Dynamic column ordering ───────────────────────────────────────────────────

# Known fields floated to the front of the Excel sheet.
# Anything the model invents that isn't listed goes alphabetically at the end.
_PREFERRED_ORDER = [
    "report_number", "case_number", "incident_number",
    "date", "crash_date", "time", "crash_time",
    "location", "address", "intersection", "road", "highway", "street",
    "city", "state", "city_state", "county",
    "weather", "weather_conditions",
    "road_conditions", "road_surface",
    "light_conditions", "lighting",
    "vehicles_involved", "number_of_vehicles",
    "injuries", "number_of_injuries", "injured",
    "fatalities", "deaths", "killed",
    "vehicle_details", "vehicle_1", "vehicle_2",
    "driver_details", "driver_1", "driver_2",
    "contributing_factors", "cause", "primary_cause",
    "narrative", "description", "details", "notes",
]


def ordered_columns(all_records: list[dict]) -> list[str]:
    """
    Collect every field key used across all records.
    Preferred fields first, then alphabetical remainder,
    then the two fixed citation columns at the end.
    """
    seen: set = set()
    for rec in all_records:
        seen.update(rec["crash"].keys())
    preferred = [f for f in _PREFERRED_ORDER if f in seen]
    remainder = sorted(seen - set(preferred))
    return preferred + remainder + ["Source / Citation", "Page"]


# ── Excel writer ──────────────────────────────────────────────────────────────

def write_excel(all_records: list, output_path: str) -> None:
    columns = ordered_columns(all_records)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Crash Reports"

    hdr_fill   = PatternFill("solid", fgColor="1F4E79")
    hdr_font   = Font(bold=True, color="FFFFFF", size=11)
    hdr_align  = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin_side  = Side(style="thin", color="B0B0B0")
    border     = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
    even_fill  = PatternFill("solid", fgColor="DCE6F1")
    odd_fill   = PatternFill("solid", fgColor="FFFFFF")
    data_align = Alignment(vertical="top", wrap_text=True)

    for c, col_name in enumerate(columns, 1):
        cell = ws.cell(row=1, column=c, value=col_name)
        cell.fill, cell.font, cell.alignment, cell.border = hdr_fill, hdr_font, hdr_align, border
    ws.row_dimensions[1].height = 32

    for r, record in enumerate(all_records, 2):
        crash = record["crash"]
        fill  = even_fill if r % 2 == 0 else odd_fill
        for c, col_name in enumerate(columns, 1):
            if col_name == "Source / Citation":
                val = record["source"]
            elif col_name == "Page":
                val = record["page"]
            else:
                val = crash.get(col_name) or ""
            cell = ws.cell(row=r, column=c, value=val)
            cell.fill, cell.alignment, cell.border = fill, data_align, border

    for c, col_name in enumerate(columns, 1):
        col_letter = get_column_letter(c)
        max_len    = len(col_name)
        for r in range(2, ws.max_row + 1):
            v = ws.cell(row=r, column=c).value
            if v:
                max_len = max(max_len, min(len(str(v)), 60))
        ws.column_dimensions[col_letter].width = min(max_len + 4, 60)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    wb.save(output_path)
    print(f"✓ Saved: {output_path}  ({len(all_records)} record(s), {len(columns)} columns)")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_pdfs(pdf_paths: list, output_excel: str = "crash_reports.xlsx") -> list:
    """
    For each PDF:
      1. Convert all pages to Markdown with pymupdf4llm.
      2. Drop near-empty pages.
      3. Batch remaining pages into chunks → one LLM call per chunk.
      4. Feed results into AccidentTracker for on-the-fly dedup and merge.

    All PDFs share one AccidentTracker, so cross-document duplicates are
    caught too.
    """
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
    model_path = find_model_path()
    tracker    = AccidentTracker()

    with litert_lm.Engine(model_path, backend=litert_lm.Backend.CPU) as engine:
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            print(f"\nProcessing: {filename}")

            pages = extract_pages_as_markdown(pdf_path)
            print(f"  {len(pages)} page(s) converted to Markdown")

            content_pages = [p for p in pages if not is_near_empty(p["text"])]
            skipped = len(pages) - len(content_pages)
            if skipped:
                print(f"  {skipped} near-empty page(s) skipped")

            chunks = chunk_pages(content_pages)
            print(f"  {len(chunks)} LLM call(s) for {len(content_pages)} page(s)")

            before = tracker.count()

            for chunk in chunks:
                page_nums = [p["page_num"] for p in chunk]
                label = (
                    str(page_nums[0]) if len(page_nums) == 1
                    else f"{page_nums[0]}–{page_nums[-1]}"
                )
                print(f"  → p.{label}...", end=" ", flush=True)

                result  = extract_crashes_from_chunk(engine, chunk, filename)
                crashes = result.get("crashes", [])

                for crash in crashes:
                    tracker.add(crash, source=f"{filename}, p.{label}", page=page_nums[0])

                print(f"{len(crashes)} crash(es) found")

            added = tracker.count() - before
            print(f"  {added} unique accident(s) from this PDF  (total: {tracker.count()})")

    all_records = tracker.records()
    print(f"\nTotal unique accidents: {len(all_records)}")
    write_excel(all_records, output_excel)
    return all_records


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract crash reports from PDFs to Excel")
    parser.add_argument("pdfs", nargs="+", help="PDF crash report file path(s)")
    parser.add_argument("-o", "--output", default="crash_reports.xlsx", help="Output Excel file")
    args = parser.parse_args()
    process_pdfs(args.pdfs, args.output)
