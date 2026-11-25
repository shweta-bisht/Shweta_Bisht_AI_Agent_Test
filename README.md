# CommerceAI (`commerce-ai-agent`)

**Implementation Track:** **Track A — Code-based agent** built with **Python** and **LangGraph**.

A self-hosted, Bring-Your-Own-Key (BYOK) multi-agent e-commerce automation pipeline built with **LangGraph**, **Google Gemini**, **Pandas**, **OpenPyXL**, **Matplotlib**, and **Pydantic v2**.

[Project report and execution results](./PROJECT_REPORT.md)

Given a single multi-sheet Excel workbook (`vendor data` and `sales data`), CommerceAI runs end-to-end in one command to perform:

1. **Automated Data Ingestion & Cleaning** (with full data-quality audit trail, zero silent record drops).
2. **Product Content Generation** (Google Gemini structured JSON validated with Pydantic v2, strictly grounded in vendor attributes).
3. **Lifestyle Image Generation** (downloads reference product image, validates format, and calls Gemini image-to-image generation).
4. **Competitor Analysis** (5 diverse products, confidence-gated reference price matching, and verifiable search-provider research).
5. **Deterministic Sales Analysis** (Pandas KPIs, explicit order-status rules separating `OJ Shipping`, Matplotlib dashboard, and multi-sheet Excel reports).

---

## Approach & Self-Hosted BYOK Model

- **Track & Framework:** **Track A — Code-based agent** implemented in **Python** using **LangGraph** (`StateGraph` + `MemorySaver` checkpointing), with clear separation between ingestion/cleaning (`processors/`), external API services (`services/`), domain agents (`agents/`), Pydantic v2 schemas (`schemas/`), and reporting (`reports/`).
- **Self-Hosted Bring Your Own Key (BYOK):** CommerceAI ships with zero developer-owned API keys. Evaluators configure their own credentials locally in `.env` (loaded via `python-dotenv` in `config.py`). Secrets are validated at startup, masked in logs (`SecretRedactingFilter` + `mask_secret`), and excluded from Git via `.gitignore`.
- **Graceful Optional Integrations:** `GEMINI_API_KEY` is required for live content and image generation (or omitted in `--dry-run` mode). `SEARCH_API_KEY` is optional—when omitted, competitor research explicitly marks comparisons unavailable without fabricating prices or URLs.

---

## Architecture

```text
                       +-------------------+
                       |       START       |
                       +---------+---------+
                                 |
                                 v
                       +-------------------+
                       |  Excel Ingestion  |
                       +---------+---------+
                                 |
                                 v
                       +-------------------+
                       |   Data Cleaning   |
                       +---------+---------+
                                 |
         +-----------------------+-----------------------+
         | (fatal error)         | (valid)               |
         |                       +------------+----------+----------+
         v                       v            v                     v
  [Execution Summary]    +-----------+  +------------+    +--------------------+
         |               |  Content  |  |   Sales    |    |     Competitor     |
         |               |Generation |  |  Analysis  |    |      Research      |
         |               +-----+-----+  +-----+------+    +---------+----------+
         |                     |              |                     |
         |                     v              |                     |
         |               +-----------+        |                     |
         |               |   Image   |        |                     |
         |               |Generation |        |                     |
         |               +-----+-----+        |                     |
         |                     |              |                     |
         |                     +--------------+---------------------+
         |                                    |
         |                                    v
         |                         +---------------------+
         |                         |  Output Generation  |
         |                         +----------+----------+
         |                                    |
         +------------------------------------+
                                              v
                                   +---------------------+
                                   |  Execution Summary  |
                                   +----------+----------+
                                              |
                                              v
                                         +---------+
                                         |   END   |
                                         +---------+
```

### Project Structure

```text
commerce-ai-agent/
├── main.py                        # CLI entrypoint
├── config.py                      # Centralized BYOK & runtime configuration
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template (no secrets committed)
├── .gitignore
├── README.md
├── agents/
│   ├── content_agent.py           # Gemini structured content generation + caching
│   ├── image_agent.py             # Reference download + Gemini lifestyle images
│   ├── competitor_agent.py        # 5-SKU selection, reference matching & search
│   └── sales_agent.py             # Deterministic Pandas sales analytics wrapper
├── workflow/
│   ├── state.py                   # LangGraph TypedDict state with parallel reducers
│   ├── routing.py                 # Conditional routing after data validation
│   ├── nodes.py                   # Node handlers with partial-failure isolation
│   └── graph.py                   # StateGraph compilation with MemorySaver checkpointer
├── processors/
│   ├── excel_reader.py            # Sheet/column/type validation
│   ├── data_cleaner.py            # Vendor & sales cleaning + DataQualityReport
│   ├── vendor_processor.py        # Product-type inference & diverse selection
│   └── sales_processor.py         # Deterministic metrics, status rules & insights
├── services/
│   ├── gemini_service.py          # Google GenAI structured text generation
│   ├── gemini_image_service.py    # Google GenAI multimodal image-to-image service
│   ├── search_service.py          # Configurable search (Serper/Tavily/SerpAPI/Google CSE)
│   └── retry_service.py           # Exponential backoff + jitter + non-retryable detection
├── schemas/
│   ├── product_schema.py          # Pydantic v2 models for vendor & product content
│   ├── competitor_schema.py       # Pydantic v2 models for competitor comparisons
│   └── execution_schema.py        # DataQualityReport, ImageGenerationRecord, ExecutionReport
├── reports/
│   ├── chart_generator.py         # Matplotlib 5-panel sales dashboard
│   ├── excel_report.py            # OpenPyXL single & 7-sheet consolidated workbooks
│   └── summary_generator.py       # ExecutionReport builder
├── utils/
│   ├── logger.py                  # Logging with API key redaction
│   ├── file_utils.py              # Safe filenames and JSON helpers
│   ├── validators.py              # Whitespace & HTTP URL validation
│   └── cost_tracker.py            # Thread-safe token, retry, failure & cost tracking
└── tests/                         # Unit, integration, and optional live pytest suite
```

---

## Setup Instructions

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url> commerce-ai-agent
cd commerce-ai-agent
python -m venv .venv
```

Activate the virtual environment:

- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source .venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure BYOK credentials (`.env`)

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

| Variable | Mandatory? | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes (unless `--dry-run`) | Google Gemini API key for product content and lifestyle images |
| `GEMINI_MODEL` | Optional (default `gemini-2.5-flash`) | Gemini text model identifier |
| `GEMINI_IMAGE_MODEL` | Optional (default `gemini-2.5-flash-image`) | Gemini multimodal image-to-image model identifier |
| `SEARCH_PROVIDER` | Optional (default `serper`) | Search API provider: `serper`, `tavily`, `serpapi`, or `google_cse` |
| `SEARCH_API_KEY` | Optional | Search API key for live competitor pricing (if omitted, competitor prices are marked unavailable, never fabricated) |
| `SEARCH_CSE_ID` | Only if `SEARCH_PROVIDER=google_cse` | Google Programmable Search Engine ID |
| `MAX_RETRIES` | Optional (default `3`) | Max retry attempts per external call |
| `MAX_CONCURRENT_REQUESTS` | Optional (default `3`) | Worker concurrency for Gemini text/image calls |
| `REQUEST_TIMEOUT` | Optional (default `120`) | Request timeout in seconds |
| `IMAGE_LIMIT` | Optional (default unset = all SKUs) | Limit lifestyle image generation to the first N valid products |

---

## Execution Instructions

Place the input workbook (`test data.xlsx`) in the project root (or pass any path via `--input`).

### Full End-to-End Execution

```bash
python main.py --input "test data.xlsx"
```

### Useful CLI Flags

```bash
# Validate workbook, run cleaning & deterministic sales analytics without API calls
python main.py --input "test data.xlsx" --dry-run

# Generate lifestyle images for only the first 5 valid products (supports all 25 by default)
python main.py --input "test data.xlsx" --image-limit 5

# Skip lifestyle image generation (takes precedence over --image-limit)
python main.py --input "test data.xlsx" --skip-images

# Skip competitor research
python main.py --input "test data.xlsx" --skip-competitors

# Write all generated reports to a custom directory (e.g. output/) while keeping images in images/
python main.py --input "test data.xlsx" --output-dir output --no-cache
```

### Exit Codes

- `0`: Completed cleanly (or `--dry-run` succeeded).
- `1`: Fatal configuration, input workbook, or pipeline error.
- `2`: Completed with partial SKU/task failures (all completed branches and failure details are accurately recorded in `<output-dir>/execution_report.json` and `<output-dir>/final_summary.xlsx`).

### Current Gemini limitation

The configured API key can list `gemini-3.8-flash` and `gemini-2.5-flash-image`, but model availability does not guarantee generation quota. Live text-generation requests in this environment were rejected after the Gemini free-tier daily quota was exhausted, and a prior live image-generation attempt was rejected with `RESOURCE_EXHAUSTED`. No lifestyle images were successfully generated. The image branch remains enabled and records failures separately from skipped work; no placeholder or synthetic lifestyle images are written.

Sample outputs from successful offline branches are in [`sample_outputs/`](./sample_outputs/): sales analytics in `sales_summary.json` and `sales_analysis.xlsx`, the dashboard in `sales_dashboard.png`, and the consolidated workbook in `final_summary.xlsx`. `execution_report.json` documents the successful dry-run branches. `live_execution_report.json` captures the latest live attempt: 25 content failures, 10,692 sales records processed, and images marked skipped. A prior image attempt was rejected with `RESOURCE_EXHAUSTED`, as noted above.

---

## Generated Outputs

By default, reports are written to `outputs/` (override with `--output-dir output` or `OUTPUT_DIR`) and images are written to `images/` (override with `--image-dir` or `IMAGE_DIR`):

```text
<output-dir>/                    # e.g., outputs/ or output/ (--output-dir output)
├── product_content.xlsx         # Generated titles, descriptions, 5 features, grounded attributes
├── product_content.json         # Structured Pydantic JSON output for all vendor SKUs
├── competitor_analysis.xlsx     # Competitor comparison + Reference Price Matching audit sheet
├── sales_analysis.xlsx          # Deterministic KPIs, monthly, top/low/non-moving SKUs, vendor/carrier
├── sales_summary.json           # Machine-readable sales analytics JSON
├── sales_dashboard.png          # 5-panel Matplotlib dashboard
├── data_quality_report.json     # Full ingestion & cleaning audit trail
├── execution_report.json        # Run timings, counts, tokens, retries, errors & estimated cost
└── final_summary.xlsx           # Consolidated 7-sheet executive workbook

images/
├── original/<SKU>.jpg           # Downloaded and Pillow-verified reference images
└── lifestyle/<SKU>.jpg          # Gemini reference-preserved lifestyle interior images
```

---

## Data Assumptions & Design Decisions

1. **Zero Manual Data Preparation**: The original Excel workbook is read directly on every run; all whitespace trimming, blank/duplicate bullet removal, date parsing, and quality auditing happen inside `processors/`.
2. **Disjoint SKU Systems**: In `test data.xlsx`, `vendor data` SKUs (`BM349936`–`BM349960`) and `sales data` codes (`MFK-*`, `IR*`, etc.) have 0 overlap. CommerceAI **never assumes equivalence**. `competitor_agent.py` runs an explicit exact -> normalized -> fuzzy matcher with a `0.75` confidence threshold and rejects weak matches, marking `our_reference_price` as `Unavailable`.
3. **Order Status Rules**:
   - `Processed`: Included in Confirmed Sales.
   - `Cancelled` & `Pending`: Excluded from Confirmed Sales (audited separately).
   - `OJ Shipping`: Reported separately in `oj_shipping_summary` and excluded from Confirmed Sales until its business meaning is verified.
4. **No Fabricated Specs or Prices**: Missing vendor dimensions/attributes remain `None`/`Unavailable`. Unconfigured search keys result in explicit `unavailable_unconfigured` records rather than invented competitor prices.

---

## Testing Instructions

Run the automated unit & mocked integration suite (no paid API calls):

```bash
pytest -v
```

Optional live integration tests (consumes real Gemini API quota; requires `GEMINI_API_KEY` in `.env` or environment):

```bash
# PowerShell
$env:RUN_LIVE_TESTS="1"; pytest -v -m live
```

---

## Recommended Git Commit Sequence

1. `chore: initialize CommerceAI project structure, config, schemas, and utilities`
2. `feat(ingestion): add Excel workbook validation and automated vendor/sales data cleaning`
3. `feat(sales): implement deterministic Pandas sales analytics and Matplotlib dashboard`
4. `feat(services): add Gemini text, multimodal image-to-image, search, and retry services`
5. `feat(agents): implement content, lifestyle image, and competitor analysis agents`
6. `feat(workflow): orchestrate parallel pipeline with LangGraph, OpenPyXL reports, and CLI`
7. `test: add comprehensive unit, mocked integration, and optional live pytest suite`
8. `docs: add README, architecture diagram, and BYOK setup guide`
