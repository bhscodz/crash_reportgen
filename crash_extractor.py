#!/usr/bin/env python3
"""
crash_extractor.py

Extracts vehicle crash report details from a PDF using a local RAG pipeline:
  - PyMuPDF              : reads and extracts text from a.pdf
  - sentence-transformers: creates embeddings for every text chunk (fully offline)
  - FAISS                : vector store for fast similarity search
  - MediaPipe LiteRT     : runs Gemma 3n E4B locally (no internet / no API key)
  - LangGraph            : orchestrates the RAG pipeline as a directed graph
  - pandas / openpyxl    : writes all extracted fields to crash_report.xlsx

Usage:
    python crash_extractor.py
"""

import numpy as np
import pandas as pd
import fitz  # PyMuPDF

from pathlib import Path
from typing import TypedDict, List

from sentence_transformers import SentenceTransformer
import faiss

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.genai import llm_inference

from langgraph.graph import StateGraph, END


# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  –  edit these paths if needed
# ──────────────────────────────────────────────────────────────────────────────

PDF_PATH     = "a.pdf"
MODEL_PATH   = "gemma-3n-E4B-it-int4.task"   # download instructions in HOW_IT_WORKS.md
OUTPUT_EXCEL = "crash_report.xlsx"
EMBED_MODEL  = "all-MiniLM-L6-v2"            # small, fast, runs fully offline

CHUNK_SIZE    = 600   # characters per chunk
CHUNK_OVERLAP = 80    # overlap so no info is cut at a boundary


# ──────────────────────────────────────────────────────────────────────────────
#  FIELDS TO EXTRACT
#  Each key becomes an Excel column; the matching query drives vector retrieval.
# ──────────────────────────────────────────────────────────────────────────────

# fmt: off
FIELD_QUERIES: dict[str, str] = {
    # ── Event basics ──────────────────────────────────────────────────────────
    "crash_date"              : "date when the accident or crash occurred",
    "crash_time"              : "time of the accident or crash",
    "crash_day_of_week"       : "day of the week the crash happened",
    # ── Location ──────────────────────────────────────────────────────────────
    "crash_location_address"  : "street address or location of the crash",
    "city"                    : "city where the crash happened",
    "county"                  : "county where the crash happened",
    "state"                   : "state where the crash happened",
    "zip_code"                : "zip code or postal code of crash location",
    # ── Vehicles ──────────────────────────────────────────────────────────────
    "number_of_vehicles"      : "how many vehicles were involved in the crash",
    "vehicle_1_make"          : "make or brand of first vehicle",
    "vehicle_1_model"         : "model of first vehicle",
    "vehicle_1_year"          : "year of first vehicle",
    "vehicle_1_color"         : "color of first vehicle",
    "vehicle_1_license_plate" : "license plate number of first vehicle",
    "vehicle_1_vin"           : "VIN vehicle identification number of first vehicle",
    "vehicle_1_direction"     : "direction of travel of first vehicle before crash",
    "vehicle_2_make"          : "make or brand of second vehicle",
    "vehicle_2_model"         : "model of second vehicle",
    "vehicle_2_year"          : "year of second vehicle",
    "vehicle_2_color"         : "color of second vehicle",
    "vehicle_2_license_plate" : "license plate number of second vehicle",
    "vehicle_2_vin"           : "VIN of second vehicle",
    "vehicle_2_direction"     : "direction of travel of second vehicle",
    # ── Drivers ───────────────────────────────────────────────────────────────
    "driver_1_name"           : "name of first driver",
    "driver_1_age"            : "age of first driver",
    "driver_1_gender"         : "gender or sex of first driver",
    "driver_1_license_number" : "driver license number of first driver",
    "driver_1_license_state"  : "state that issued license to first driver",
    "driver_1_address"        : "home address of first driver",
    "driver_1_phone"          : "phone number of first driver",
    "driver_2_name"           : "name of second driver",
    "driver_2_age"            : "age of second driver",
    "driver_2_gender"         : "gender or sex of second driver",
    "driver_2_license_number" : "driver license number of second driver",
    "driver_2_license_state"  : "state that issued license to second driver",
    "driver_2_address"        : "home address of second driver",
    "driver_2_phone"          : "phone number of second driver",
    # ── Casualties ────────────────────────────────────────────────────────────
    "total_injuries"          : "total number of injuries or people injured",
    "total_fatalities"        : "number of deaths or fatalities",
    # ── Crash details ─────────────────────────────────────────────────────────
    "crash_type"              : "type of crash or accident",
    "manner_of_collision"     : "manner of collision how the vehicles collided",
    "primary_crash_cause"     : "primary cause of the accident",
    "secondary_crash_cause"   : "secondary or contributing cause of the accident",
    "weather_conditions"      : "weather conditions at the time of crash",
    "lighting_conditions"     : "lighting conditions day or night at crash",
    "road_surface_condition"  : "road surface condition wet dry icy",
    "road_type"               : "type of road highway intersection etc",
    "speed_limit"             : "posted speed limit at crash location",
    "estimated_speed"         : "estimated speed at time of impact",
    "alcohol_suspected"       : "alcohol impairment suspected DUI DWI",
    "drugs_suspected"         : "drugs or substances suspected impairment",
    "airbags_deployed"        : "whether airbags deployed in crash",
    "seatbelts_used"          : "seatbelts or safety belts used",
    # ── Damage ────────────────────────────────────────────────────────────────
    "property_damage"         : "property damage description",
    "estimated_damage_cost"   : "estimated cost of property or vehicle damage",
    # ── Report / Law enforcement ───────────────────────────────────────────────
    "police_report_number"    : "police report number case number incident number",
    "responding_agency"       : "police agency or department that responded",
    "responding_officer"      : "name of responding officer",
    "report_date"             : "date the police report was filed",
    # ── Insurance ─────────────────────────────────────────────────────────────
    "insurance_1_company"     : "insurance company of first vehicle driver",
    "insurance_1_policy"      : "insurance policy number of first vehicle",
    "insurance_2_company"     : "insurance company of second vehicle driver",
    "insurance_2_policy"      : "insurance policy number of second vehicle",
    # ── Witnesses ─────────────────────────────────────────────────────────────
    "witness_1_name"          : "name of first witness",
    "witness_1_contact"       : "contact information phone or address of first witness",
    "witness_2_name"          : "name of second witness",
    "witness_2_contact"       : "contact information phone or address of second witness",
    # ── Narrative ─────────────────────────────────────────────────────────────
    "crash_narrative"         : "narrative or summary description of how the crash happened",
}
# fmt: on

FIELDS: List[str] = list(FIELD_QUERIES.keys())


# ──────────────────────────────────────────────────────────────────────────────
#  STEP 1 — PDF TEXT EXTRACTION  (PyMuPDF)
# ──────────────────────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: str) -> str:
    """
    Open every page of the PDF and concatenate the plain text.
    Each page is prefixed with a [PAGE N] header so the LLM can
    reference page numbers if needed.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(f"[PAGE {num}]\n{text}")
    doc.close()
    return "\n\n".join(pages)


def chunk_text(text: str) -> List[str]:
    """
    Split the full document text into overlapping chunks.
    Overlap ensures that a fact sitting near a chunk boundary
    is never completely absent from all retrieved chunks.
    """
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
#  STEP 2 — VECTOR STORE  (sentence-transformers + FAISS)
# ──────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Embeds every chunk once at startup, then answers top-k similarity
    queries in milliseconds using a FAISS flat inner-product index.
    """

    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        # all-MiniLM-L6-v2 is ~80 MB and downloads once; cached afterwards
        self.embedder = SentenceTransformer(EMBED_MODEL)

        raw = self.embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        # L2-normalise so inner product equals cosine similarity
        normed = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)

        self.index = faiss.IndexFlatIP(normed.shape[1])
        self.index.add(normed)

    def search(self, query: str, top_k: int = 4) -> List[str]:
        """Return the top_k chunks most relevant to the query."""
        q = self.embedder.encode([query], convert_to_numpy=True)
        q = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
        _, idxs = self.index.search(q, top_k)
        return [self.chunks[i] for i in idxs[0] if 0 <= i < len(self.chunks)]


# ──────────────────────────────────────────────────────────────────────────────
#  STEP 3 — LOCAL LLM  (MediaPipe LiteRT + Gemma 3n E4B)
# ──────────────────────────────────────────────────────────────────────────────

def load_gemma(model_path: str) -> llm_inference.LlmInference:
    """
    Load the Gemma .task model from disk using MediaPipe's LLM Inference API.
    temperature=0.1 keeps answers deterministic and factual.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"\nModel file not found: {model_path}\n"
            "Download it from Kaggle (free account required):\n"
            "  https://www.kaggle.com/models/google/gemma-3n/frameworks/litert\n"
            "Choose: gemma-3n-E4B-it-int4.task  and place it next to this script."
        )

    options = llm_inference.LlmInferenceOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        max_tokens=256,   # short, focused answers only
        top_k=40,
        temperature=0.1,  # low = more deterministic, better for extraction
    )
    return llm_inference.LlmInference.create_from_options(options)


def extract_field(model: llm_inference.LlmInference, field: str, context: str) -> str:
    """
    Ask Gemma to extract a single field value from the retrieved context.
    Uses the Gemma instruction-tuned prompt format <start_of_turn>user ... model.
    """
    prompt = (
        "<start_of_turn>user\n"
        "You are a precise vehicle crash report data extractor.\n"
        f'Extract ONLY the value for this field: "{field}"\n'
        "Rules:\n"
        "  - Return just the value, nothing else.\n"
        "  - If the information is absent, return exactly: N/A\n"
        "  - Do not explain or add any extra words.\n\n"
        f"Text:\n{context}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    raw = model.generate_response(prompt)
    # Strip any trailing turn marker the model might emit
    value = raw.split("<end_of_turn>")[0].strip()
    return value if value else "N/A"


# ──────────────────────────────────────────────────────────────────────────────
#  STEP 4 — LANGGRAPH RAG PIPELINE
#
#  The graph has three nodes that run in sequence for every field:
#
#    build_query  →  retrieve  →  generate
#
#  build_query : maps the field name to a natural-language search query
#  retrieve    : finds the top-k most relevant chunks from the vector store
#  generate    : asks Gemma to extract the field value from those chunks
# ──────────────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    """All data passed between LangGraph nodes for a single field extraction."""
    field:           str   # e.g. "crash_date"
    query:           str   # e.g. "date when the accident occurred"
    context:         str   # concatenated retrieved text chunks
    extracted_value: str   # final answer from Gemma


# ── Node functions ─────────────────────────────────────────────────────────────

def node_build_query(state: PipelineState) -> dict:
    """Convert the internal field name to a human-readable retrieval query."""
    query = FIELD_QUERIES.get(state["field"], state["field"].replace("_", " "))
    return {"query": query}


def make_retrieve_node(store: VectorStore):
    """Factory: returns the retrieve node with `store` baked in via closure."""
    def node_retrieve(state: PipelineState) -> dict:
        chunks  = store.search(state["query"], top_k=4)
        context = "\n---\n".join(chunks)
        return {"context": context}
    return node_retrieve


def make_generate_node(model: llm_inference.LlmInference):
    """Factory: returns the generate node with `model` baked in via closure."""
    def node_generate(state: PipelineState) -> dict:
        value = extract_field(model, state["field"], state["context"])
        return {"extracted_value": value}
    return node_generate


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_pipeline(store: VectorStore, model: llm_inference.LlmInference):
    """Wire up and compile the three-node LangGraph."""
    g = StateGraph(PipelineState)
    g.add_node("build_query", node_build_query)
    g.add_node("retrieve",    make_retrieve_node(store))
    g.add_node("generate",    make_generate_node(model))
    g.set_entry_point("build_query")
    g.add_edge("build_query", "retrieve")
    g.add_edge("retrieve",    "generate")
    g.add_edge("generate",    END)
    return g.compile()


def run_all_fields(pipeline, fields: List[str]) -> dict:
    """
    Invoke the pipeline once per field.
    Returns a dict {field_name: extracted_value}.
    """
    results: dict = {}
    for i, field in enumerate(fields, start=1):
        print(f"  [{i:02d}/{len(fields)}] {field}")
        state: PipelineState = {
            "field": field, "query": "", "context": "", "extracted_value": ""
        }
        final = pipeline.invoke(state)
        results[field] = final.get("extracted_value", "N/A")
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  STEP 5 — EXCEL EXPORT  (pandas + openpyxl)
# ──────────────────────────────────────────────────────────────────────────────

def save_excel(results: dict, output_path: str) -> None:
    """
    Write the field → value dict to a single-row Excel table.
    Column headers are title-cased human-friendly names.
    Column widths are auto-fitted (capped at 60 characters).
    """
    # Rename keys to human-readable headers
    renamed = {k.replace("_", " ").title(): v for k, v in results.items()}
    df = pd.DataFrame([renamed])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Crash Report")

        # Auto-fit each column width
        ws = writer.sheets["Crash Report"]
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)

    print(f"\n  Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    sep = "=" * 60
    print(f"\n{sep}\n  Crash Report Extractor\n{sep}")

    # 1 — PDF
    print(f"\n[1/5] Reading '{PDF_PATH}' with PyMuPDF...")
    raw_text = extract_pdf_text(PDF_PATH)
    chunks   = chunk_text(raw_text)
    print(f"      {len(raw_text):,} chars  |  {len(chunks)} chunks")

    # 2 — Vector store
    print("\n[2/5] Building FAISS vector store (embedding chunks)...")
    store = VectorStore(chunks)

    # 3 — LLM
    print(f"\n[3/5] Loading Gemma 3n E4B via MediaPipe LiteRT...")
    model = load_gemma(MODEL_PATH)

    # 4 — RAG pipeline
    print(f"\n[4/5] Extracting {len(FIELDS)} fields via LangGraph RAG pipeline...")
    pipeline = build_pipeline(store, model)
    results  = run_all_fields(pipeline, FIELDS)

    # 5 — Excel
    print(f"\n[5/5] Writing results to Excel...")
    save_excel(results, OUTPUT_EXCEL)

    print(f"\n{sep}\n  Done!  Open '{OUTPUT_EXCEL}' to see the results.\n{sep}\n")


if __name__ == "__main__":
    main()
