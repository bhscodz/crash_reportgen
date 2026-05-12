# Crash Report Generator — Pipeline Flow

```mermaid
flowchart TD
    A([Start]) --> B[/"PDF file paths"/]
    B --> C[Load Gemma 4 E4B\nlitert_lm.Engine on CPU]

    C --> D{For each PDF}

    D --> E[PDF → Markdown\npymupdf4llm · tables as pipe-tables]
    E --> F[Filter near-empty pages\nBatch into chunks of 5]

    F --> G{For each chunk}

    G --> H[Gemma 4 inference\ntext-only · one call per chunk]
    H --> I[AccidentTracker.add\nfingerprint · deduplicate · merge]

    I --> J{More chunks?}
    J -- Yes --> G
    J -- No --> K{More PDFs?}
    K -- Yes --> D
    K -- No --> L[write_excel\ndynamic columns · styled output]

    L --> M([crash_reports.xlsx])
```

---

## Component Breakdown

| Component | Library | Role |
|---|---|---|
| PDF → Markdown | `pymupdf4llm` | Converts pages to pipe-table Markdown; no images needed |
| LLM engine | `litert_lm` | Gemma 4 E4B on CPU; one conversation per chunk |
| Deduplication | `AccidentTracker` | Fingerprints crashes; merges duplicates across pages and PDFs |
| Excel writer | `openpyxl` | Dynamic columns, styled headers, auto-filter, frozen row |

---

## Key Design Decisions

- **Page batching** — up to 5 pages per LLM call; a 9-page report needs 2 calls instead of 9.
- **No images** — `pymupdf4llm` Markdown captures all table structure, eliminating image I/O and vision encoder overhead.
- **`AccidentTracker`** — fingerprints each crash on `report_number` / `date+location` / etc.; normal fields are first-wins, narrative fields get snippets appended.
- **Dynamic columns** — LLM picks snake_case field names freely; `_PREFERRED_ORDER` floats known fields to the front, unknowns sort alphabetically after.
