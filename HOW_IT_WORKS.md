# Crash Report Extractor — How It Works

Extracts every detail from a vehicle crash-report PDF and writes it to an Excel file.  
Runs **100 % locally** — no internet connection needed after the first setup.

---

## Architecture

```
a.pdf
  │
  ▼  PyMuPDF
Full text (all pages)
  │
  ▼  chunk_text()
 600-char overlapping chunks  ──────────────────────────┐
  │                                                     │
  ▼  SentenceTransformer (all-MiniLM-L6-v2)            │
Embeddings (384-dim vectors)                            │
  │                                                     │
  ▼  FAISS IndexFlatIP                                  │
Vector store (in RAM)                                   │
                                                        │
For each of the 56 fields:                              │
  ┌─────────────────────────────────────────────────┐  │
  │  LangGraph pipeline (3 nodes)                   │  │
  │                                                 │  │
  │  1. build_query  ←  field name                 │  │
  │       │                                         │  │
  │       ▼  FIELD_QUERIES dict                     │  │
  │  2. retrieve  ←  vector store search ───────────┘  │
  │       │  (top-4 most relevant chunks)               │
  │       ▼                                             │
  │  3. generate  ←  Gemma 3n E4B (MediaPipe LiteRT)   │
  │       │  extracts a single value from the chunks    │
  └───────┼─────────────────────────────────────────┘
          ▼
     {field: value, ...}   (56 entries)
          │
          ▼  pandas + openpyxl
     crash_report.xlsx
```

---

## What Each Part Does

| Component | Role |
|---|---|
| **PyMuPDF** | Opens `a.pdf` and extracts plain text from every page |
| **sentence-transformers** | Converts each text chunk to a 384-dim embedding vector for search |
| **FAISS** | Stores all embeddings in memory; answers "which chunks are most relevant to this query?" in milliseconds |
| **LangGraph** | Defines a 3-node directed graph (`build_query → retrieve → generate`) and runs it once per field |
| **MediaPipe LiteRT** | Loads the quantised Gemma 3n E4B model from disk and runs it on CPU |
| **Gemma 3n E4B** | The language model that reads the retrieved chunks and extracts the specific value |
| **pandas / openpyxl** | Collects all results into a DataFrame and writes it to a formatted Excel file |

---

## Setup (one-time)

### 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` (pulled by `sentence-transformers`) is ~2 GB.  
> On a slow connection, allow 10–20 minutes.

### 2 — Download the Gemma model

The model is free but requires a Kaggle account.

1. Go to: <https://www.kaggle.com/models/google/gemma-3n/frameworks/litert>
2. Sign in and accept the Gemma usage terms.
3. Download **`gemma-3n-E4B-it-int4.task`** (~2.5 GB).
4. Place the file in the **same folder** as `crash_extractor.py`.

> `E4B` = Effective 4 Billion parameters.  
> `int4` = 4-bit quantisation (fastest, smallest, best for CPU-only machines).  
> `it` = instruction-tuned (responds to direct extraction prompts).

### 3 — Place your PDF

Copy your crash report PDF into the same folder and name it **`a.pdf`**  
(or change `PDF_PATH` at the top of `crash_extractor.py`).

---

## Running

```bash
python crash_extractor.py
```

Progress is printed step by step:

```
[1/5] Reading 'a.pdf' with PyMuPDF...
      14,823 chars  |  28 chunks

[2/5] Building FAISS vector store (embedding chunks)...
      Batches: 100%|████████████| 1/1 [00:02<00:00]

[3/5] Loading Gemma 3n E4B via MediaPipe LiteRT...

[4/5] Extracting 56 fields via LangGraph RAG pipeline...
  [01/56] crash_date
  [02/56] crash_time
  ...
  [56/56] crash_narrative

[5/5] Writing results to Excel...
  Saved: crash_report.xlsx
```

First run takes longer (model loads into memory, embeddings download once).  
Subsequent runs are faster.

---

## Output

`crash_report.xlsx` contains a **single row** with **56 columns**, one per field:

| Crash Date | Crash Time | City | Driver 1 Name | Total Injuries | ... |
|---|---|---|---|---|---|
| 2024-03-15 | 14:32 | Springfield | John Doe | 2 | ... |

Fields that are **not found in the PDF** are filled with `N/A`.

---

## Fields Extracted

| Category | Fields |
|---|---|
| Event | date, time, day of week |
| Location | address, city, county, state, zip |
| Vehicles (1 & 2) | make, model, year, color, plate, VIN, direction of travel |
| Drivers (1 & 2) | name, age, gender, license number & state, address, phone |
| Casualties | total injuries, total fatalities |
| Crash details | type, manner of collision, primary/secondary cause, weather, lighting, road surface/type, speed limit, estimated speed |
| Impairment | alcohol suspected, drugs suspected |
| Safety | airbags deployed, seatbelts used |
| Damage | property damage description, estimated cost |
| Report | police report number, agency, officer name, report date |
| Insurance (1 & 2) | company, policy number |
| Witnesses (1 & 2) | name, contact info |
| Narrative | full crash description |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: a.pdf` | Put `a.pdf` in the same folder as the script |
| `FileNotFoundError: gemma-3n-E4B-it-int4.task` | Download the model from Kaggle (see Setup step 2) |
| All fields return `N/A` | The PDF may be **scanned/image-based** — PyMuPDF can only read text-layer PDFs. Use OCR first (e.g. Adobe Acrobat, or `pytesseract`) |
| `ImportError: mediapipe` | Run `pip install mediapipe>=0.10.14` |
| Out of memory | Close other applications; Gemma needs ~6 GB RAM at runtime |
| Slow extraction | Normal on CPU — each field takes ~5–15 seconds; 56 fields ≈ 10–15 minutes total |

---

## Customising

- **Add more fields:** Add an entry to `FIELD_QUERIES` in `crash_extractor.py`; no other changes needed.
- **Change the PDF:** Edit `PDF_PATH` at the top of the script.
- **Change chunk size:** Increase `CHUNK_SIZE` if the PDF is very long; decrease if RAM is limited.
- **Faster retrieval:** Increase `top_k` in `store.search()` calls to give Gemma more context (at the cost of speed).
