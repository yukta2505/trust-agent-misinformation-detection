# TRUST-AGENT — End-to-End System Architecture (Implementation-Accurate)

This document is meant to be used to generate a system architecture diagram. It is derived from the *actual code paths* in this repository (FastAPI → LangGraph orchestrator → nodes → agents → aggregator → frontend).

Repo root: `trust-agent-misinformation-detection/`

---

## 1) High-level overview

**Goal:** detect *out-of-context (OOC) misinformation*, where a real image is paired with a misleading claim about *time, place, person, event,* or *source context*.

**Core strategy (5 signals):**
1. **Captioning** (image understanding): GPT‑4o Vision primary, **LLaVA local** fallback.
2. **Entity consistency** (LLM): claim entities vs caption + retrieved evidence (miscaptioning detection).
3. **Temporal consistency** (LLM): claim time references vs evidence + EXIF/claim metadata (TYPE A/B/B_WRONG/C).
4. **Source credibility** (LLM): trust tiering + cross-source agreement; neutral when evidence absent.
5. **CLIP visual consistency** (local): image ↔ claim similarity relative to image ↔ caption baseline.
6. **Claim plausibility** (LLM, soft): suspicious framing when evidence is missing; used for thresholding signals.

**Decision:** a rule-based fusion computes a **final_score** and an **adaptive threshold** (0.60 / 0.56 / 0.52) based on how many OOC “signals” fire. Then an LLM “aggregator” writes the public explanation.

---

## 2) Deployed system components (runtime boxes)

### A) Frontend (React + Vite)
- **User inputs:** image file + claim text.
- **HTTP:** `POST http://localhost:8000/analyse` (multipart form with `image`, `claim`).
- **Outputs shown:** verdict, confidence%, explanation, per-agent numeric scores, evidence list, runtime/errors.

Primary files:
- `frontend/src/App.jsx` (pages + API call + sectioning)
- `frontend/src/components/UploadForm.jsx`
- `frontend/src/components/VerdictCard.jsx`
- `frontend/src/components/AgentScores.jsx`
- `frontend/src/components/EvidenceList.jsx`

### B) API server (FastAPI)
- **Endpoint:** `POST /analyse` saves uploaded image to a temp file, calls orchestrator, deletes temp file.
- **Singleton orchestrator/config:** cached with `lru_cache()` so the LangGraph and heavy models aren’t rebuilt per request.

Primary files:
- `api/main.py`
- `api/dependencies.py`
- `api/schemas.py`

### C) Orchestrator (LangGraph)
- Defines a **stateful graph** using `langgraph.graph.StateGraph`.
- Executes nodes in strict order until evidence retrieval, then fans out to parallel agent nodes, then fans in to aggregator.

Primary files:
- `orchestrator.py`
- `graph_state.py`
- `graph_nodes.py`

### D) Backend modules (captioning, retrieval, metadata)
- Captioning: GPT‑4o Vision primary, LLaVA fallback
- Entity extraction: spaCy NER (blank model fallback)
- Metadata extraction: EXIF + date/year parsing from claim/caption
- Evidence retrieval: reverse image + web/news search + optional historical FAISS index
- Evidence filtering: conservative noise removal + keyword overlap
- LLM client: provider selection + Groq 429 wait/retry + OpenAI fallback

Primary files:
- `backend/captioning2.py`
- `backend/entity_extraction.py`
- `backend/metadata_extractor.py`
- `backend/evidence_retrieval.py`
- `backend/evidence_filter.py`
- `backend/llm_client.py`
- `backend/config.py`

### E) Agents (LLM + local)
- LLM agents: entity, temporal, credibility, plausibility, aggregator (explanation)
- Local agent: CLIP

Primary files:
- `agents/entity_agent.py`
- `agents/temporal_agent.py`
- `agents/credibility_agent.py`
- `agents/plausibility_agent.py`
- `agents/clip_agent.py`
- `agents/aggregator_agent.py`

---

## 3) Request/response path (wire-level)

1. Browser → Frontend (`UploadForm`)
2. Frontend → API: `POST /analyse` (multipart)
3. API writes upload to temp file → calls `TrustAgentOrchestrator.run(image_path, claim)`
4. Orchestrator invokes LangGraph pipeline
5. Graph produces final state
6. API maps orchestrator result to `AnalyseResponse` JSON
7. Frontend renders:
   - Verdict section
   - Agent scores section
   - Evidence/news retrieval section

---

## 4) LangGraph pipeline (exact node graph)

Defined in `orchestrator.py` and backed by node functions from `graph_nodes.py`.

### 4.1 State object

LangGraph state type: `AgentState` in `graph_state.py` (TypedDict, `total=False`).

Important keys (by lifecycle):
- **Inputs:** `image_path`, `claim`, `top_k`, `errors`
- **Preprocessing outputs:** `caption`, `entities`, `metadata`, `clip_result`
- **Evidence:** `evidence`
- **Agent results:** `entity_result`, `temporal_result`, `credibility_result`, `plausibility_result`
- **Final output:** `verdict`, `confidence_percent`, `explanation`, `key_evidence_for_verdict`, `flags`
- **Meta:** `processing_time_sec` (computed outside the graph), `errors` (non-fatal)

Note: `graph_nodes.py` writes `clip_result` into state; `AgentState` doesn’t explicitly list `clip_result` in the schema, but LangGraph state merging still accepts it because `TypedDict(total=False)` is permissive at runtime.

### 4.2 Node ordering and concurrency

**Sequential chain:**
1. `caption` → `extract_entities` → `retrieve_evidence`

**Parallel fan-out after evidence retrieval:**
- `entity_agent`
- `temporal_agent`
- `credibility_agent`
- `plausibility_agent`

**Fan-in:**
- all 4 feed into `aggregator`

### 4.3 Node responsibilities (what each does)

#### Node 1 — `caption` (`node_caption`)
Input:
- `image_path`

Output:
- `caption` (string)
- appends to `errors` on failure

Implementation:
- Uses `backend/captioning2.py::ImageCaptioner`
  - **Primary:** GPT‑4o Vision via OpenAI API if `OPENAI_API_KEY` is set
  - **Fallback:** local LLaVA (`llava-hf/llava-1.5-7b-hf`) if OpenAI missing/fails

Failure behavior:
- If captioning fails: returns empty caption and records error; pipeline continues.

#### Node 2 — `extract_entities` (`node_extract_entities`)
Inputs:
- `claim`, `caption`, `image_path`

Outputs:
- `entities`: spaCy NER over combined claim+caption
- `metadata`: EXIF + temporal clues from claim/caption, plus a human-readable `summary`
- `clip_result`: local CLIP analysis dict (or “SKIPPED/ERROR”)
- appends to `errors` on CLIP failures

Implementation details:
- NER: `backend/entity_extraction.py::EntityExtractor`
  - Loads `SPACY_MODEL` (default `en_core_web_sm`), falls back to `spacy.blank("en")` if missing.
- Metadata: `backend/metadata_extractor.py::MetadataExtractor.extract_all(...)`
  - EXIF capture_date, camera details, optional GPS
  - claim/caption date/year parsing
- CLIP: `agents/clip_agent.py::CLIPVisualAgent.analyse(...)` if `USE_CLIP=true`
  - Computes:
    - image↔claim similarity
    - image↔caption similarity
    - `clip_score = (img_claim_similarity / img_caption_similarity)` clamped to 0..1
  - Interprets clip_score:
    - ≥ 0.80 consistent, ≥ 0.55 uncertain, else inconsistent

#### Node 3 — `retrieve_evidence` (`node_retrieve_evidence`)
Inputs:
- `image_path`, `claim`, `entities`, `caption`, `top_k`

Output:
- `evidence`: list of evidence items (dicts)
- appends to `errors` on failures (upload/search/index)

Evidence sources (cumulative list):
1. **Reverse image search** (`backend/evidence_retrieval.py::ReverseImageSearcher.search`)
   - Only runs if `SERPAPI_API_KEY` exists.
   - Upload strategy:
     - if `IMGBB_API_KEY` exists → upload to ImgBB (3 retries)
     - else → upload to `0x0.st`
   - SerpAPI engine: `google_reverse_image`
   - Normalizes results to dicts:
     - `type="reverse_image"`, `title`, `url`, `snippet`, `source`, `timestamp=None`
2. **Web/news search** (`backend/evidence_retrieval.py::WebSearcher.search`)
   - Builds a short query primarily from the **claim** text
   - Optionally adds up to 2 “high value” entities (GPE/ORG/PERSON/EVENT)
   - Search provider:
     - if `SERPAPI_API_KEY` → SerpAPI `google` engine (`organic_results`)
     - else if `NEWSAPI_KEY` → NewsAPI `everything`
3. **Historical semantic index** (optional) (`historical_index.py::HistoricalEvidenceIndex.search`)
   - Uses sentence-transformers + FAISS
   - Loads persisted index if it exists:
     - `Config.historical_index_path` (default `artifacts/historical.faiss`)
     - `Config.historical_meta_path` (default `artifacts/historical_meta.json`)
   - Query used: `claim + caption`
   - Adds fields like `semantic_distance` and `semantic_similarity`

Filtering step:
- After all retrievals, evidence is passed through:
  - `backend/evidence_filter.py::filter_relevant_evidence`
  - Stage 1: removes obvious garbage (lyrics/music/dictionary sites, specific domains)
  - Stage 2: keeps items with keyword overlap against claim tokens (`min_overlap=1`)
  - Safety net: if overlap-filter removes everything, returns stage1 (don’t lose all evidence)
  - Adds `relevance_overlap` integer per evidence item

Failure behavior:
- If any retrieval fails: it’s logged and appended to `errors`, but pipeline continues with whatever evidence exists (including empty list).

---

## 5) Agent layer (what each agent produces and how it scores)

All LLM agents use the same transport: `backend/llm_client.py::chat_with_fallback(...)`
- Returns JSON text; agents parse it into dicts.
- Uses provider based on `Config.llm_provider`:
  - `openai` (default): requires `OPENAI_API_KEY`
  - `groq`: requires `GROQ_API_KEY`, uses OpenAI-compatible API
  - `ollama`: points to `OLLAMA_BASE_URL` (OpenAI-compatible server)

### 5.1 Entity Analysis Agent (LLM) — `EntityAnalysisAgent`
File: `agents/entity_agent.py`

Inputs:
- claim, caption, top evidence (first 5 items)

Outputs (JSON keys):
- `claim_entities`, `caption_entities`, `evidence_entities` (label→list)
- `matches`: entity overlaps
- `contradictions`: explicit conflicts (caption/evidence contradict claim)
- `miscaptioned_signals`: evidence attributes image to different person/place/event than claim
- `entity_score` (0..1)
- `reasoning` (short)

Score intent:
- High (0.80–1.00): caption/evidence confirm claim entities
- Neutral-ish (0.60–0.79): no contradiction, or claim lacks concrete named entities
- Low (0.10–0.39): specific contradiction or miscaptioning
- Very low (0.00–0.09): severe, multi-source contradiction

Built-in safety override:
- If the model returns `entity_score < 0.30` but there are **no** contradictions or miscaptioned signals, score is overridden to **0.60** (neutral).

### 5.2 Temporal Reasoning Agent (LLM) — `TemporalReasoningAgent`
File: `agents/temporal_agent.py`

Inputs:
- claim, caption, top evidence (first 6 items), `metadata_summary` from EXIF/claim parsing

Outputs:
- `claim_type`: `TYPE_A | TYPE_B | TYPE_B_WRONG | TYPE_C`
  - TYPE_A: “old event claimed as today/now/current”
  - TYPE_B: historical date claim that is consistent
  - TYPE_B_WRONG: claim states a specific date/year but evidence indicates a different one
  - TYPE_C: no date to check → neutral
- `claim_time_reference`, `image_time_reference`, `time_gap_description`
- `is_temporally_consistent` (bool)
- `temporal_score` (0..1)
- `reasoning`

Safety overrides:
- TYPE_C must be ≥ 0.60 (forced to 0.65 if lower)
- TYPE_B must be ≥ 0.60 (forced to 0.65 if lower)
- TYPE_B_WRONG has a floor (forced to 0.20 if < 0.15)
- If score < 0.30 but claim_type is not TYPE_A/B_WRONG → coerces to TYPE_C and score 0.65

### 5.3 Source Credibility Agent (LLM) — `SourceCredibilityAgent`
File: `agents/credibility_agent.py`

Inputs:
- claim, caption, top evidence (first 6 items)

Outputs:
- `sources_evaluated`: list of `{source, credibility_tier, supports_claim, reason}`
- `cross_source_agreement`: `AGREE|DISAGREE|MIXED|INSUFFICIENT`
- `dominant_narrative`
- `credibility_score` (0..1)
- `reasoning`

Critical rule (hardcoded in agent logic):
- If **no evidence_items** are present, returns immediately:
  - `credibility_score = 0.55`
  - agreement = `INSUFFICIENT`
  - narrative = `No relevant sources found`
  - (no LLM call)

Safety overrides after LLM call:
- If agreement is `INSUFFICIENT` and there are no HIGH/MEDIUM contradictions, and score is too low → override to 0.55.
- If score < 0.35 but there are no active HIGH/MEDIUM contradictions → override to 0.55.

Interpretation:
- Below 0.55 is reserved for *active contradictions* from credible sources, not for lack of evidence.

### 5.4 CLIP Visual Agent (local) — `CLIPVisualAgent`
File: `agents/clip_agent.py`

Inputs:
- image_path, claim, caption

Outputs:
- `clip_score` (0..1) plus the raw similarities and a short “interpretation”

Key logic:
- Computes `img_claim_similarity` and `img_caption_similarity`
- Defines `clip_score = img_claim_similarity / img_caption_similarity`, clamped 0..1
  - Intuition: if the claim matches the image almost as well as the caption does, it’s likely consistent.

Failure behavior:
- On import/model errors: returns `clip_score=0.5` and `interpretation="ERROR"`; pipeline continues.

### 5.5 Claim Plausibility Agent (LLM, soft) — `ClaimPlausibilityAgent`
File: `agents/plausibility_agent.py`

Inputs:
- claim, caption

Outputs:
- `plausibility_score` (0..1)
- `red_flags`, `positive_signals`
- `claim_type_assessment` and `reasoning`

Important: This is a **soft** signal. It does not directly contribute a weight in the final_score formula; instead it contributes to the OOC signal count used for thresholding and explanation.

---

## 6) Fusion + verdict logic (Aggregator v2)

File: `agents/aggregator_agent.py`

The aggregator has two roles:
1. **Deterministic fusion**: compute `final_score` and the internal verdict using explicit rules.
2. **Natural language explanation**: call an LLM to produce a public-facing explanation and selected evidence bullets.

### 6.1 Compute `final_score` (weighted average with credibility redistribution)

Inputs:
- `entity_score`, `temporal_score`, `credibility_score`, `clip_score`
- `credibility_result` (to decide “cred absent”)
- `Config` weights:
  - `WEIGHT_ENTITY` default 0.25 (config) but aggregator has internal defaults:
    - entity 0.35, temporal 0.30, credibility 0.20, clip 0.15
  - Actual weights used come from `Config.weight_*` if present; otherwise aggregator defaults.

Credibility-absent detection:
- `_cred_is_absent(cred)` is true if:
  - `cross_source_agreement == "INSUFFICIENT"`, or
  - `dominant_narrative == "No relevant sources found"`, or
  - `sources_evaluated` is empty

If credibility is absent:
- Floors `c_score` to at least 0.55 (neutral)
- Redistributes credibility weight (wc) into other weights:
  - entity gets `wc * 0.40`
  - temporal gets `wc * 0.35`
  - clip gets `wc * 0.25`
  - wc becomes 0.0
- Final:
  - `final_score = normalized_weighted_sum(entity, temporal, credibility, clip)`

### 6.2 Count “OOC signals” (drives adaptive threshold and explanation bullets)

Function: `count_signals(ent, tmp, cred, clip_score, plausibility)`

Signals added when:
- Temporal:
  - TYPE_A → signal
  - TYPE_B_WRONG → signal
  - temporal_score < 0.40 → signal
- Entity:
  - any `miscaptioned_signals` → signal
  - else any `contradictions` → signal
  - else entity_score < 0.35 → signal
- Credibility (only when credibility is NOT absent):
  - DISAGREE and credibility_score < 0.40 → signal
  - or credibility_score < 0.35 → signal
- CLIP:
  - clip_score < 0.40 → signal
- Plausibility (optional):
  - plausibility_score < 0.45, OR
  - plausibility_score < 0.60 and there are plausibility red flags

Output:
- `ooc_signal_count = len(signals)`
- `signals[]` string list used as “flags”/evidence bullets if needed

### 6.3 Adaptive threshold

Threshold selection:
- 0 or 1 signal → threshold **0.60**
- 2 signals → threshold **0.56**
- 3+ signals → threshold **0.52**

### 6.4 Absolute overrides

Even if score is high, the aggregator forces `OUT-OF-CONTEXT` when:
- Temporal claim_type is TYPE_A and temporal_score ≤ 0.20
- Entity has miscaptioned_signals and entity_score ≤ 0.25

Otherwise:
- verdict = `PRISTINE` if `final_score >= threshold`, else `OUT-OF-CONTEXT`

### 6.5 Public explanation generation (LLM call)

The aggregator then sends a compact “agent_block” + top evidence to an LLM with a strict JSON schema:
- `explanation` (2–3 sentences)
- `confidence_percent` (int)
- `key_evidence_for_verdict` (up to 3 bullets)
- `flags` (red flags, empty if PRISTINE)
- `ooc_category` (`temporal_trick|miscaptioned|false_context|none`)

If that LLM call fails:
- It generates a fallback explanation and a confidence derived from `abs(final_score - threshold)`.

---

## 7) Fallbacks and graceful degradation (what happens when things are missing)

### 7.1 Captioning fallbacks
- If `OPENAI_API_KEY` exists and GPT‑4o Vision succeeds → caption is from GPT‑4o Vision.
- If OpenAI is missing or fails/quota → caption is from **local LLaVA**.
- If both fail → caption becomes empty, error recorded, pipeline continues.

### 7.2 Evidence retrieval fallbacks
- If `SERPAPI_API_KEY` missing → reverse-image search and SerpAPI web search are skipped.
- If SerpAPI fails → tries NewsAPI (if `NEWSAPI_KEY` exists).
- If no external keys exist → evidence list is empty (not an error).
- Historical index:
  - If the FAISS index files do not exist → returns empty list (not an error).

### 7.3 LLM provider fallbacks (Groq 429 handling)
Used by all LLM-based agents through `chat_with_fallback(...)`:
- If provider is `groq` and a 429 occurs:
  - parses wait time from error message
  - sleeps and retries up to `max_rate_limit_waits` (default 3)
  - then falls back to OpenAI if `OPENAI_API_KEY` is configured
  - else raises a clear error

### 7.4 Agent-level safety defaults
- Entity agent: avoids very low scores without explicit contradictions.
- Temporal agent: floors TYPE_C/TYPE_B to neutral, prevents spurious zeros.
- Credibility agent: hard neutral 0.55 when no evidence; overrides low scores if no credible contradictions.
- CLIP: returns neutral 0.5 on failure.
- Plausibility: returns neutral 0.70 on failure.
- Aggregator: always returns a verdict (rule-based), even if its explanation LLM fails.

### 7.5 “No evidence” behavior (intentional)
When retrieval returns nothing:
- Credibility defaults to 0.55 neutral (and does not penalize)
- Entity/Temporal tend to neutral unless caption contradicts claim strongly
- CLIP still provides an independent visual signal
- Plausibility can add a “soft suspicion” signal
- Thresholding can still classify OOC if multiple non-evidence signals fire

---

## 8) Data contracts (what the API returns)

Response model: `api/schemas.py::AnalyseResponse`

Returned fields include:
- `verdict`, `confidence_percent`, `explanation`
- `entity_score`, `temporal_score`, `credibility_score`, `final_score`
- `plausibility_score` (optional)
- `ooc_signal_count`, `threshold_used`, `ooc_category`
- `caption`
- `key_evidence_for_verdict`, `flags`
- `evidence[]` (retrieved items after filtering)
- `processing_time_sec`
- `errors[]`

Not currently returned by the API (even though computed in the pipeline):
- the full `clip_result` object and/or a `clip_score` field (it is computed in-state and used in aggregation).

---

## 9) Configuration map (environment → runtime behavior)

Central config: `backend/config.py::Config`

Key environment variables:
- `LLM_PROVIDER`: `openai` | `groq` | `ollama`
- `OPENAI_API_KEY`, `OPENAI_MODEL`
- `GROQ_API_KEY`, `GROQ_MODEL`
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- `SERPAPI_API_KEY`, `NEWSAPI_KEY`, `IMGBB_API_KEY`
- `USE_CLIP` (true/false)
- `DEFAULT_TOP_K`, `REQUEST_TIMEOUT`
- `PRISTINE_THRESHOLD` (note: v2 aggregator uses adaptive thresholds internally; older math-verdict logic may still exist in legacy code paths)
- `WEIGHT_ENTITY`, `WEIGHT_TEMPORAL`, `WEIGHT_CREDIBILITY`, `WEIGHT_CLIP` (used by fusion)

---

## 10) Suggested diagram building blocks (for your architecture diagram)

Use these as boxes/edges:

### Frontend lane
- Page: Home (UploadForm)
- Results sections: Verdict / Agent Scores / Evidence
- HTTP client: Axios

### API lane
- FastAPI `/analyse`
- Temp file store (OS temp)
- Orchestrator singleton

### Pipeline lane (LangGraph)
- Node: Captioning (GPT‑4o Vision → LLaVA fallback)
- Node: Entity+Metadata+CLIP (spaCy + EXIF + local CLIP)
- Node: Evidence Retrieval (Reverse image + Web/News + Historical FAISS)
- Node: Evidence Filter
- Parallel nodes:
  - Entity Agent (LLM)
  - Temporal Agent (LLM)
  - Credibility Agent (LLM)
  - Plausibility Agent (LLM)
- Node: Aggregator v2
  - deterministic fusion + thresholding
  - LLM explanation generator

### External services (optional boxes)
- OpenAI API (Vision + LLM)
- Groq API (OpenAI-compatible)
- Ollama server (OpenAI-compatible, local)
- SerpAPI
- NewsAPI
- ImgBB or 0x0.st (image upload for reverse image)

### Local model/runtime boxes
- CLIP model (transformers + torch)
- LLaVA model (transformers)
- spaCy model
- sentence-transformers embedder + FAISS index (optional)

---

## 11) End-to-end pipeline narrative (single pass, step-by-step)

1. User uploads image + claim in the frontend.
2. Frontend sends multipart request to FastAPI `/analyse`.
3. FastAPI writes the image to a temp path.
4. Orchestrator creates initial LangGraph state:
   - `{image_path, claim, top_k, errors: []}`
5. Caption node:
   - tries GPT‑4o Vision → else LLaVA → returns caption.
6. Entity/metadata/CLIP node:
   - spaCy extracts entities from (claim+caption)
   - EXIF/temporal metadata extracted and summarized
   - local CLIP checks image↔claim consistency (if enabled)
7. Retrieval node:
   - reverse image search (if SerpAPI key)
   - web/news search (SerpAPI or NewsAPI)
   - optional historical FAISS semantic search
   - conservative filtering to remove obvious garbage and keep relevant items
8. Parallel agent fan-out:
   - entity agent scores entity consistency and miscaptioning signals
   - temporal agent classifies claim time logic and scores it
   - credibility agent tiers sources and scores agreement (neutral when absent)
   - plausibility agent flags suspicious framing (soft)
9. Aggregator:
   - computes final_score from e/t/c/clip (with credibility redistribution)
   - counts OOC signals → selects adaptive threshold
   - applies absolute overrides (TYPE_A severe; miscaptioning severe)
   - chooses final verdict PRISTINE vs OUT-OF-CONTEXT
   - asks LLM to produce explanation + confidence + evidence bullets (with fallback)
10. API maps final orchestrator result into response JSON.
11. Frontend displays verdict, scores, evidence, and any non-fatal errors.

