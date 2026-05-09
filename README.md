# TRUST-AGENT
### Out-of-Context Misinformation Detection System

**Advanced AI-powered detection system that identifies when real images are paired with false or misleading claims about different times, locations, or events.**

---

## 📋 Table of Contents
1. [What is TRUST-AGENT?](#what-is-trust-agent)
2. [Project Structure](#project-structure)
3. [System Architecture](#system-architecture)
4. [Pipeline Workflow](#pipeline-workflow)
5. [Core Components](#core-components)
6. [Agent Details](#agent-details)
7. [Scoring System](#scoring-system)
8. [Setup & Installation](#setup--installation)
9. [Running the System](#running-the-system)
10. [API Endpoints](#api-endpoints)
11. [Frontend Dashboard](#frontend-dashboard)

---

## What is TRUST-AGENT?

Out-of-context (OOC) misinformation is one of the most common and dangerous forms of fake news. It uses **real, genuine images** paired with **false or misleading claims** about a different time, location, or event.

**Example:** 
- A real photograph of the 2020 Beirut explosion is shared with the caption *"Massive explosion rocks Beirut today (2026)"*
- The image is authentic — but the context/claim is false
- TRUST-AGENT detects this mismatch automatically

### Key Features

| Feature | Detail |
|---|---|
| **Multi-Agent Reasoning** | 5 specialized agents (Entity, Temporal, Credibility, CLIP, Plausibility) run in parallel via LangGraph |
| **Image Captioning** | BLIP model (local) reads text, flags, landmarks, and context from images |
| **Local CLIP Analysis** | CPU-based visual consistency check — no API key or internet required |
| **Temporal Classification** | TYPE A/B/C logic detects "old image as recent" tricks |
| **Credibility Scoring** | HIGH/MEDIUM/LOW tier source evaluation (BBC vs unknown blog) |
| **Explainable Output** | Plain-English verdicts + per-agent scores + visual dashboards + red flags |
| **Deployment Flexibility** | Online (OpenAI) / Hybrid (Groq) / Offline (Ollama) modes |
| **Web Dashboard** | React frontend + FastAPI backend with real-time processing |
| **Zero-Shot** | No training required — works on any image+claim immediately |
| **CPU-Only** | No GPU required — runs entirely on CPU |

---

## 📁 Project Structure

```
trust-agent-misinformation-detection/
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── orchestrator.py                        # LangGraph pipeline coordinator
├── graph_state.py                         # Shared state schema for all agents
├── graph_nodes.py                         # Individual processing nodes
│
├── agents/                                # Specialized detection agents
│   ├── __init__.py
│   ├── entity_agent.py                   # Entity consistency checker (25% weight)
│   ├── temporal_agent.py                 # Temporal reasoning (30% weight)
│   ├── credibility_agent.py              # Source credibility evaluator (20% weight)
│   ├── plausibility_agent.py             # Claim plausibility evaluator (soft signal)
│   ├── clip_agent.py                     # Visual consistency checker (25% weight)
│   ├── aggregator_agent.py               # Final verdict synthesizer
│   ├── entity_agent_old.py               # Legacy versions
│   ├── temporal_agent_old.py
│   ├── credibility_agent_old.py
│   └── aggregator_agent_old.py
│
├── backend/                               # Core processing modules
│   ├── __init__.py
│   ├── config.py                         # Configuration management
│   ├── captioning.py                     # BLIP image captioning
│   ├── entity_extraction.py              # spaCy NER extraction
│   ├── metadata_extractor.py             # EXIF + temporal metadata
│   ├── evidence_retrieval.py             # Reverse image + web search
│   ├── evidence_filter.py                # Noise filtering for evidence
│   ├── llm_client.py                     # LLM API client with fallbacks
│   └── utils.py                          # Text cleaning, keyword extraction
│
├── api/                                   # FastAPI application
│   ├── __init__.py
│   ├── main.py                           # API endpoints (/health, /analyse)
│   ├── schemas.py                        # Pydantic models (request/response)
│   └── dependencies.py                   # Dependency injection
│
├── frontend/                              # React dashboard
│   ├── package.json                      # Frontend dependencies
│   ├── vite.config.js                    # Vite build config
│   ├── index.html                        # HTML entry point
│   ├── src/
│   │   ├── main.jsx                      # React app entry
│   │   ├── App.jsx                       # Main app component
│   │   ├── index.css                     # Global styles
│   │   └── components/
│   │       ├── UploadForm.jsx            # Image + claim upload form
│   │       ├── VerdictCard.jsx           # Verdict display (PRISTINE/OOC)
│   │       ├── AgentScores.jsx           # Score visualization
│   │       ├── EvidenceList.jsx          # Evidence ranking display
│   │       ├── LoadingSpinner.jsx        # Loading indicator
│   │       ├── AboutPage.jsx             # Info about TRUST-AGENT
│   │       ├── HowItWorksPage.jsx        # Pipeline explanation
│   │       ├── AnalyzePage.jsx           # Main analysis page
│   │       └── Navbar.jsx                # Navigation
│
├── dataset/                               # Test data
│   ├── test_dataset.csv                  # Test cases
│   └── images/                           # Test images
│
├── image-text-verification/              # VERITE benchmark integration
│   ├── requirements.txt
│   ├── main.py
│   ├── experiment.py
│   ├── model.py
│   ├── prepare_datasets.py
│   ├── extract_features.py
│   ├── utils.py
│   ├── download_verite.py
│   └── VERITE/                          # VERITE dataset
│
├── historical_index.py                   # Historical evidence indexing
├── evaluate.py                           # Evaluation metrics
├── verite_eval.py                        # VERITE benchmark evaluation
├── test_pipeline.py                      # Unit tests
│
└── results/                               # Output files
    ├── eval_results.csv
    ├── verite_results.csv
    └── summaries (txt files)
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INPUT                           │
│            (Image File + Text Claim)                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   FastAPI /analyse Route   │
        │  (Temp file handling)      │
        └────────┬───────────────────┘
                 │
        ┌────────▼──────────────────────────┐
        │   TrustAgentOrchestrator.run()    │
        │   (LangGraph state machine)       │
        └────────┬──────────────────────────┘
                 │
   ══════════════╪════════════════════════════════════════
   PREPROCESSING PHASE (Sequential)
   ══════════════╪════════════════════════════════════════
        │
        ├─► [node_caption]
        │   BLIP model reads image
        │   → caption: str (e.g., "crowd waving flags")
        │
        ├─► [node_extract_entities]
        │   spaCy NER + EXIF metadata
        │   CLIP local analysis
        │   → entities, metadata, clip_result
        │
        └─► [node_retrieve_evidence]
            Reverse image search (SerpAPI/ImgBB)
            Web search (SerpAPI/NewsAPI)
            Evidence filtering
            → evidence: List[Dict]
   
   ══════════════╪════════════════════════════════════════
   PARALLEL AGENT PHASE (5 Agents in Parallel)
   ══════════════╪════════════════════════════════════════
        │
        ├─► [node_entity_agent]
        │   Input: claim, caption, entities
        │   Task: Check people/place/org/event consistency
        │   Output: entity_result, entity_score
        │
        ├─► [node_temporal_agent]
        │   Input: claim, caption, metadata
        │   Task: Classify TYPE A/B/C, check date consistency
        │   Output: temporal_result, temporal_score
        │
        ├─► [node_credibility_agent]
        │   Input: claim, evidence items
        │   Task: Evaluate source trustworthiness
        │   Output: credibility_result, credibility_score
        │
        ├─► [node_plausibility_agent]
        │   Input: claim, caption
        │   Task: Assess whether the claim sounds plausible given the image caption
        │   Output: plausibility_result, plausibility_score
        │
        └─► [node_clip_agent_wrapper]
            Input: image, claim, caption
            Task: Visual consistency (local, no API)
            Output: clip_result, clip_score
   
   ══════════════╪════════════════════════════════════════
   FUSION & AGGREGATION PHASE
   ══════════════╪════════════════════════════════════════
        │
        ├─► [node_aggregator_agent]
        │   Input: entity_result, temporal_result, credibility_result,
        │          clip_result, plausibility_result, evidence
        │   Task: Combine signals, adjust threshold via plausibility,
        │         and generate final verdict + explanation
        │   Output: verdict, final_score, confidence_percent,
        │           key_evidence_for_verdict, flags, ooc_category
        │
        └─► [node_aggregator_agent]
            Input: All agent results + final_score + verdict
            Task: Generate plain-English explanation
            Output: explanation, confidence%, key_evidence, flags
   
   ══════════════╪════════════════════════════════════════
                 │
                 ▼
   ┌─────────────────────────────────────────┐
   │      TrustAgentResult Object            │
   ├─────────────────────────────────────────┤
   │ • verdict: PRISTINE / OUT-OF-CONTEXT    │
   │ • confidence_percent: 0-100             │
   │ • explanation: plain English reason     │
   │ • caption: image description            │
   │ • entity_score, temporal_score, etc.    │
   │ • key_evidence_for_verdict: [...]       │
   │ • flags: red flags detected             │
   │ • evidence: ranked sources              │
   └─────────────────────────────────────────┘
                 │
                 ▼
   ┌─────────────────────────────────────────┐
   │    FastAPI Response (JSON)              │
   │    to React Frontend Dashboard          │
   └─────────────────────────────────────────┘
                 │
                 ▼
   ┌─────────────────────────────────────────┐
   │   React Components Render:              │
   │   • VerdictCard (PRISTINE/OOC)          │
   │   • AgentScores (score visualization)   │
   │   • EvidenceList (ranked sources)       │
   │   • Red flags & explanation             │
   └─────────────────────────────────────────┘
```

---

## Pipeline Workflow

### Phase 1: Preprocessing (Sequential)

#### Step 1.1 — Image Captioning
- **Module:** `backend/captioning.py` → `ImageCaptioner.caption()`
- **Model:** BLIP (`Salesforce/blip-image-captioning-large`)
- **Process:**
  - Load image from disk
  - Convert to RGB (PIL Image)
  - Tokenize with BlipProcessor
  - Generate caption (max 80 tokens, beam search=5)
- **Output:** `caption: str`
  - Example: *"crowd of people waving iraqi flags on a street"*
- **Error handling:** Returns empty string + logs error
- **Configuration:** `config.blip_model`, `config.use_gpt4_vision` (for fallback)

#### Step 1.2 — Entity Extraction & Metadata
- **Module:** `backend/entity_extraction.py` → `EntityExtractor.extract()`
- **Model:** spaCy NER (`en_core_web_sm`)
- **Process:**
  - Combined text: `claim + caption`
  - spaCy pipeline: tokenization → NER
  - Extract: PERSON, ORG, GPE (place), DATE, EVENT, LOC, NORP
- **Output:** `entities: Dict[str, List[str]]`
  - Example: `{"PERSON": ["John"], "GPE": ["Baghdad", "Iraq"], "DATE": ["2026"]}`

**Parallel: CLIP Visual Consistency (LOCAL)**
- **Module:** `agents/clip_agent.py` → `CLIPVisualAgent.compute_similarity()`
- **Model:** OpenAI CLIP (`openai/clip-vit-base-patch32`)
- **Process:**
  - Load image + process claim text
  - Compute embedding similarity (cosine)
  - Score: 0.0-1.0 (higher = more consistent)
- **Output:** `clip_result: Dict` with `clip_score`, `interpretation`
- **Note:** Fully local, no API key needed, runs on CPU

**Metadata Extraction:**
- **Module:** `backend/metadata_extractor.py` → `MetadataExtractor.extract_all()`
- **Process:**
  - Extract EXIF data (capture date, GPS)
  - Parse temporal clues from claim text (e.g., "today", "2026")
  - Extract keywords for web search
- **Output:** `metadata: Dict` with image date, claim date, keywords

#### Step 1.3 — Evidence Retrieval
- **Module:** `backend/evidence_retrieval.py`
- **Two parallel searches:**

**A) Reverse Image Search:**
  - Upload image to ImgBB or 0x0.st
  - SerpAPI reverse-image-search: Google's reverse image engine
  - Parse results: `[{source, title, url, snippet, domain}, ...]`
  - Timeout per request: 30s

**B) Web/News Search:**
  - Query: `claim + top entities + keywords`
  - SerpAPI Google Search + NewsAPI
  - Parse results: news articles, blog posts, official sources
  - Filter noise: remove music videos, unrelated content

**Evidence Filtering:**
- **Module:** `backend/evidence_filter.py`
- **Rules:**
  - Remove music videos, social media posts without context
  - Prioritize official sources (.gov, .edu, major outlets)
  - Keep top-k most relevant items
- **Output:** `evidence: List[Dict[str, Any]]` (ranked by relevance)

---

### Phase 2: Parallel Agent Execution

All 4 agents receive the state simultaneously. Each runs independently and returns results.

#### Agent 1: Entity Analysis Agent (35% Weight)

**Location:** `agents/entity_agent.py` → `EntityAnalysisAgent.analyse()`

**Input:**
- `claim`: Text claim
- `caption`: Image description from BLIP
- `entities`: Extracted entities (claim + caption)

**Logic:**
1. Extract entities from claim AND caption
2. Find overlaps (entities in both)
3. Find contradictions (specific mismatches)
4. **Default Rules (CRITICAL):**
   - No entities in claim → score 0.65 (neutral)
   - Caption too generic → score 0.60 (neutral, benefit of doubt)
   - Entities match → score 0.80-1.0 (consistent)
   - Specific contradiction → score 0.0-0.25 (conflict)
   - Missing info ≠ contradiction (NOT scored as 0.0)

**Output:** `entity_result: Dict`
```json
{
  "claim_entities": {"PERSON": [...], "GPE": [...], ...},
  "caption_entities": {...},
  "matches": ["entity1", "entity2"],
  "contradictions": ["specific factual conflicts or empty"],
  "entity_score": 0.0,
  "reasoning": "explanation"
}
```

**Examples:**
- Claim: "Protesters in Baghdad wave Iraqi flags"  
  Caption: "crowd waving Iraqi flags in urban area"  
  → Score: 0.85 (entities match)

- Claim: "Beirut explosion 2026"  
  Caption: "destroyed European-style buildings"  
  → Score: 0.15 (location contradiction)

---

#### Agent 2: Temporal Reasoning Agent (30% Weight)

**Location:** `agents/temporal_agent.py` → `TemporalReasoningAgent.analyse()`

**Input:**
- `claim`: Text claim
- `caption`: Image description
- `metadata`: Image capture date + claim temporal clues

**Classification Logic:**

**TYPE A — Old Event, False "Recent" Claim**
- Claim uses: "today", "now", "2025", "2026", "currently"
- Evidence shows: Event actually happened years earlier
- Example: "Beirut explosion today 2026" (happened Aug 2020)
- **Score:** 0.0-0.2 (MISINFORMATION SIGNAL)

**TYPE B — Historically Accurate Date**
- Claim states specific year (2006, 2013, 2015, 2020, 2022, etc.)
- Describes factual event with correct date
- NO evidence contradicting the date
- Example: "Paris Hilton selfie from 2006", "NASA Mars Rover photo May 7, 2022"
- **Score:** 0.65-0.85 (date is correct, not a trick)

**TYPE C — No Specific Date**
- Claim has NO year/date reference
- Example: "Image shows electric scooters", "Athletes highlining"
- **Score:** 0.65 (neutral, nothing to verify)

**Output:** `temporal_result: Dict`
```json
{
  "claim_type": "TYPE_A|TYPE_B|TYPE_C",
  "claim_time_reference": "year or 'none'",
  "image_time_reference": "actual time period or 'unknown'",
  "time_gap_description": "consistent / X years apart",
  "is_temporally_consistent": true,
  "temporal_score": 0.0,
  "reasoning": "explanation"
}
```

**CRITICAL RULE:**
- TYPE B (historical dates) should NOT be scored as 0.0 just because you can't verify the exact year
- 0.0 = confirmed date IS WRONG (e.g., claim says 2026 but evidence proves 2020)
- Missing verification ≠ contradiction

---

#### Agent 3: Source Credibility Agent (20% Weight)

**Location:** `agents/credibility_agent.py` → `SourceCredibilityAgent.analyse()`

**Input:**
- `claim`: Text claim
- `evidence_items`: List of retrieved sources

**Credibility Tiers:**
- **HIGH:** BBC, Reuters, AP, AFP, NYT, Washington Post, Al Jazeera, NASA, .gov, major news outlets
- **MEDIUM:** Regional news, Wikipedia, established outlets
- **LOW:** Social media, unknown blogs, tabloids, forums
- **UNKNOWN:** Unidentifiable source

**Scoring Guide:**
- 0.80–1.00: Multiple HIGH-credibility sources confirm claim
- 0.65–0.79: Some credible sources support claim
- 0.55: NO relevant sources found (NEUTRAL DEFAULT — not suspicious)
- 0.35–0.54: Mixed signals, some doubt from credible sources
- 0.00–0.34: Credible sources ACTIVELY AND SPECIFICALLY contradict claim

**CRITICAL DEFAULT:**
- If NO evidence retrieved → score 0.55 (neutral)
- Absence of evidence ≠ proof of misinformation
- Only score low if credible sources ACTIVELY contradict

**Output:** `credibility_result: Dict`
```json
{
  "sources_evaluated": [
    {
      "source": "BBC News",
      "credibility_tier": "HIGH",
      "supports_claim": true,
      "reason": "official report confirms event"
    }
  ],
  "cross_source_agreement": "AGREE|DISAGREE|MIXED|INSUFFICIENT",
  "dominant_narrative": "what credible sources say",
  "credibility_score": 0.55,
  "reasoning": "explanation"
}
```

---

#### Agent 4: CLIP Visual Consistency Agent (15% Weight)

**Location:** `agents/clip_agent.py` → `CLIPVisualAgent.compute_similarity()`

**What is CLIP?**
- OpenAI's CLIP model trained on 400M image-text pairs
- Computes semantic similarity between images and text
- Runs locally on CPU (no API key)

**Input:**
- `image_path`: Image file
- `claim`: Text claim
- `caption`: BLIP-generated caption

**Process:**
1. Load CLIP model (`openai/clip-vit-base-patch32`)
2. Encode image to embedding
3. Encode claim text to embedding
4. Compute cosine similarity: 0.0-1.0
5. Threshold interpretation:
   - \>0.25: High similarity (visually consistent)
   - 0.18-0.25: Medium (ambiguous)
   - <0.18: Low (visually inconsistent)

**Output:** `clip_result: Dict`
```json
{
  "clip_score": 0.32,
  "interpretation": "CONSISTENT|AMBIGUOUS|INCONSISTENT",
  "reasoning": "image and claim visual match"
}
```

**Advantages:**
- Works even when NO external evidence found
- Independent signal (cannot be faked by web evidence)
- Runs on CPU, no API calls

---

### Phase 3: Aggregation & Verdict

#### Step 3.1 — Aggregator Decision

**Module:** `agents/aggregator_agent.py` → `AggregatorAgent.aggregate()`

**Calculation:**
```
final_score = (0.35 × entity_score) 
            + (0.30 × temporal_score)
            + (0.20 × credibility_score)
            + (0.15 × clip_score)
```

**Weights Rationale:**
- Entity + Temporal: **65%** — Core mismatch detection (people, place, time)
- Credibility: **20%** — Source reliability
- CLIP: **15%** — Visual consistency double-check
- Plausibility: **soft signal** — adjusts OOC threshold, not raw score

**Threshold Logic:**
- Starts at `0.60`
- Lowers adaptively when multiple OOC signals appear
- Uses claim plausibility to make the final decision more robust

**Final Decision:**
- `final_score >= threshold` → **"PRISTINE"**
- `final_score < threshold` → **"OUT-OF-CONTEXT"**

**Output:**
```
final_score: float (0.0-1.0)
math_verdict: str ("PRISTINE" or "OUT-OF-CONTEXT")
```

---

#### Step 3.2 — Aggregator Agent (Final Explanation)

**Module:** `agents/aggregator_agent.py` → `AggregatorAgent.aggregate()`

**Input:**
- All agent results (entity, temporal, credibility)
- `final_score` and `math_verdict`
- Extracted contradictions, mismatches

**Task:**
1. Use LLM (Claude/GPT-4) to synthesize explanation
2. Generate plain-English reason for verdict
3. Extract key evidence points
4. Identify red flags
5. MUST enforce mathematical verdict (cannot override)

**Output:** `aggregator_result: Dict`
```json
{
  "verdict": "PRISTINE or OUT-OF-CONTEXT",
  "confidence_percent": 75,
  "explanation": "2-3 sentence plain English explanation",
  "key_evidence_for_verdict": [
    "specific finding 1",
    "specific finding 2"
  ],
  "flags": ["red flag 1", "red flag 2"]
}
```

**Example Output:**
```
Verdict: OUT-OF-CONTEXT
Confidence: 78%
Explanation: "The image shows a destroyed building typical of 2020s Middle Eastern 
conflicts, but the claim alleges this is from today in Baghdad. Reverse image search 
confirms this is from the August 2020 Beirut explosion, not a current event."

Key Evidence:
- Reverse image search confirms 2020 Beirut explosion
- Temporal mismatch: claimed 2026, actually 2020 (6 years gap)
- Caption shows building destruction consistent with 2020 event

Flags:
- Major temporal discrepancy (6 years)
- Date claim is false
- Potentially misleading timeline
```

---

## Core Components

### Backend Modules

#### `backend/config.py` — Configuration Management
- Loads API keys from environment: OpenAI, SerpAPI, ImgBB
- Default models: BLIP, spaCy, CLIP
- Timeouts and retry logic
- Supports 3 deployment modes (online, hybrid, offline)

#### `backend/llm_client.py` — LLM API Client
- Wraps OpenAI / Groq / Ollama APIs
- Fallback logic: tries primary LLM, falls back to alternative
- Handles JSON parsing, retries on failure
- Logging for debugging

#### `backend/utils.py` — Utilities
- Text cleaning (remove URLs, special chars, extra whitespace)
- Keyword extraction (TF-IDF based)
- Entity flattening for comparison

#### `backend/metadata_extractor.py` — EXIF & Temporal
- Extracts EXIF data: capture date, GPS coordinates, camera model
- Parses temporal clues from text (e.g., "today", "2026")
- Generates keywords for web search

#### `backend/evidence_filter.py` — Noise Filtering
- Removes irrelevant sources (music videos, spam)
- Prioritizes official sources
- Ranks by credibility + relevance

---

### Orchestration

#### `orchestrator.py` — TrustAgentOrchestrator
- Main class: coordinates entire pipeline
- Uses LangGraph StateGraph
- Node registration and execution
- State management and merging
- Timing and error collection
- Returns `TrustAgentResult` object

#### `graph_state.py` — AgentState Schema
- TypedDict defining all shared state keys
- Input keys: image_path, claim, top_k
- Processing keys: caption, entities, evidence
- Agent result keys: entity_result, temporal_result, etc.
- Output keys: verdict, explanation, flags
- Meta keys: errors, processing_time_sec

#### `graph_nodes.py` — Node Functions
- `node_caption`: Run BLIP
- `node_extract_entities`: Run spaCy + CLIP
- `node_retrieve_evidence`: Reverse image + web search
- `node_entity_agent`: Entity analysis
- `node_temporal_agent`: Temporal reasoning
- `node_credibility_agent`: Source credibility
- `node_score_fusion`: Calculate final score
- `node_aggregator_agent`: Generate explanation

---

### API Layer

#### `api/main.py` — FastAPI Application
- **GET /health** → Server status, model info
- **POST /analyse** → Main endpoint
  - Accepts multipart form: image file + claim text
  - Calls orchestrator
  - Returns JSON response
  - CORS enabled for React frontend

#### `api/schemas.py` — Pydantic Models
- Request: image file + claim text
- Response: verdict, scores, explanation, evidence, flags, errors

#### `api/dependencies.py` — Dependency Injection
- Provides singleton Config
- Provides singleton TrustAgentOrchestrator
- Handles initialization and cleanup

---

### Frontend

#### React Architecture (Vite)
- Single-page app (SPA)
- Component-based UI
- Axios for API calls
- Real-time result display

#### Key Components
- `UploadForm.jsx`: Image + claim input
- `VerdictCard.jsx`: Large verdict display + confidence
- `AgentScores.jsx`: Score bars for each agent
- `EvidenceList.jsx`: Ranked evidence sources
- `LoadingSpinner.jsx`: Loading animation
- `AboutPage.jsx`: System explanation
- `HowItWorksPage.jsx`: Pipeline visualization

---

## Scoring System

### Individual Agent Scores

| Agent | Weight | Range | Interpretation |
|-------|--------|-------|---|
| **Entity** | 35% | 0.0-1.0 | People/place/org consistency |
| **Temporal** | 30% | 0.0-1.0 | Date/time consistency |
| **Credibility** | 20% | 0.0-1.0 | Source trustworthiness |
| **CLIP** | 15% | 0.0-1.0 | Visual consistency |

### Score Ranges

**0.80–1.00 (STRONG)**
- Entity: Entities match perfectly
- Temporal: Date is correct
- Credibility: Multiple HIGH-tier sources confirm
- CLIP: High visual similarity

**0.60–0.79 (CONSISTENT)**
- Entity: No contradictions found
- Temporal: Generic time reference, no mismatch
- Credibility: Some credible sources support
- CLIP: Partial visual match

**0.55 (NEUTRAL)**
- Credibility default when no evidence
- No clear signal for or against

**0.40–0.59 (DOUBT)**
- Some inconsistency or unverifiable claims
- Mixed signals

**0.0–0.39 (CONTRADICTION)**
- Entity: Specific factual mismatch
- Temporal: Date provably wrong
- Credibility: Sources actively contradict
- CLIP: Low visual consistency

### Final Verdict Logic

```
final_score = weighted_avg(all_agents)

if final_score >= 0.60:
    verdict = "PRISTINE"           # ✓ Image consistent with claim
    confidence = ceil(final_score * 100)
else:
    verdict = "OUT-OF-CONTEXT"     # ✗ Detected mismatch
    confidence = ceil((1 - final_score) * 100)
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip or conda
- (Optional) GPU with CUDA for faster inference
- API Keys: OpenAI (or Groq/Ollama), SerpAPI, ImgBB

### Step 1: Clone Repository
```bash
git clone <repo-url>
cd trust-agent-misinformation-detection
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\Activate
```

### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### Step 5: Environment Configuration
Create `.env` file in project root:
```env
# LLM Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo

# Alternative (Groq for free option)
GROQ_API_KEY=gsk-...

# Search APIs
SERPAPI_KEY=...
IMGBB_API_KEY=...

# Optional
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
```

### Step 6: Frontend Setup
```bash
cd frontend
npm install
cd ..
```

---

## Running the System

### Option 1: Full Stack (Recommended)

**Terminal 1 — Start Backend API**
```bash
python -m uvicorn api.main:app --reload --port 8000
```
- Swagger docs at: http://localhost:8000/docs

**Terminal 2 — Start Frontend**
```bash
cd frontend
npm run dev
```
- Frontend at: http://localhost:5173

**Terminal 3 — Test**
```bash
curl -X POST http://localhost:8000/health
```

### Option 2: Backend Only (Programmatic)
```python
from orchestrator import TrustAgentOrchestrator
from backend.config import Config

config = Config()
orc = TrustAgentOrchestrator(config)

result = orc.run(
    image_path="/path/to/image.jpg",
    claim="Image shows Beirut explosion today 2026"
)

print(f"Verdict: {result.verdict}")
print(f"Confidence: {result.confidence_percent}%")
print(f"Explanation: {result.explanation}")
```

### Option 3: Offline Mode (Ollama)
```env
OPENAI_MODEL=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
```
Download Ollama: https://ollama.ai

---

## API Endpoints

### GET /health
**Check server status**
```bash
curl http://localhost:8000/health
```
**Response:**
```json
{
  "status": "ok",
  "model": "gpt-4-turbo"
}
```

### POST /analyse
**Main misinformation detection endpoint**

**Request:**
```bash
curl -X POST http://localhost:8000/analyse \
  -F "image=@image.jpg" \
  -F "claim=Image shows Beirut explosion today 2026"
```

**Response:**
```json
{
  "verdict": "OUT-OF-CONTEXT",
  "confidence_percent": 82,
  "explanation": "The image shows a building destroyed in the August 2020 Beirut explosion...",
  "caption": "destroyed building with debris and smoke",
  "entity_score": 0.75,
  "temporal_score": 0.15,
  "credibility_score": 0.65,
  "final_score": 0.5341,
  "key_evidence_for_verdict": [
    "Reverse image search confirms August 2020 Beirut explosion",
    "Temporal mismatch: claimed 2026, actually 2020",
    "Caption shows typical explosion destruction"
  ],
  "flags": [
    "Major temporal discrepancy (6 years)",
    "False date claim",
    "Potentially misleading context"
  ],
  "evidence": [
    {
      "source": "Reuters",
      "title": "Beirut explosion August 2020",
      "snippet": "...",
      "url": "...",
      "credibility_tier": "HIGH"
    }
  ],
  "processing_time_sec": 12.34,
  "errors": []
}
```

---

## Frontend Dashboard

### Main Interface

**Upload Section:**
- Drag-and-drop or click to upload image
- Text area for claim input
- "Analyze" button

**Results Display:**

1. **Verdict Card** (Large, prominent)
   - PRISTINE (green checkmark) or OUT-OF-CONTEXT (red warning)
   - Confidence percentage

2. **Agent Scores** (Visual bars)
   - Entity score bar (green → red)
   - Temporal score bar
   - Credibility score bar
   - CLIP score bar
   - Final weighted score

3. **Evidence List** (Ranked)
   - Source name + credibility tier
   - Article title
   - Snippet preview
   - URL link

4. **Red Flags** (If any)
   - Key findings
   - Specific mismatches
   - Warnings

5. **Explanation** (Plain English)
   - 2-3 sentence summary
   - Why the verdict was reached

---

## Evaluation & Testing

### Test Suite
Run unit tests:
```bash
python -m pytest test_pipeline.py -v
```

### VERITE Benchmark
Evaluate against VERITE dataset (1,231 image+claim pairs):
```bash
cd image-text-verification
python verite_eval.py
```

### Performance Metrics
- Accuracy, Precision, Recall, F1 score
- Per-agent contribution analysis
- Processing time breakdown

---

## Deployment Notes

### Docker (Future)
```dockerfile
FROM python:3.10-slim
# ... Dockerfile setup
```

### Scaling Considerations
- Async processing for concurrent requests
- Evidence caching layer (Redis)
- Model quantization for faster inference
- Distributed LangGraph execution

---

## License & Attribution
- BLIP: Salesforce Research
- spaCy: Explosion AI
- CLIP: OpenAI
- LangGraph: LangChain
- VERITE Dataset: Research Community

---

## Support & Contact
For issues, questions, or improvements:
- Open GitHub issues
- Submit pull requests
- Contact: [your-email]

---

**Last Updated:** April 2026  
**Version:** 1.0.0  
**Status:** Production-Ready

- **1.0** → entities match perfectly (e.g. Baghdad + Iraqi flags both match)
- **0.6** → caption doesn't contradict claim (neutral)
- **0.0** → specific contradiction (e.g. claim says Tokyo, image shows Paris)

### Agent 2 — Temporal Reasoning Agent
Detects the most common OOC trick: old photo claimed as a recent event.

**Three claim types:**
- **TYPE A** — old image falsely claimed as recent → score 0.0–0.2
  - *"Beirut explosion today 2026"* but explosion was August 2020
- **TYPE B** — historical subject + recent action → score 0.7–0.9 (NOT a trick)
  - *"19th century painting sells for $17.9m"* — painting is old, sale is recent
- **TYPE C** — no date information → score 0.6 (benefit of doubt)

### Agent 3 — Source Credibility Agent
Evaluates how trustworthy retrieved sources are and whether they support or contradict the claim.

- **HIGH** tier: BBC, Reuters, AP, AFP, NYT, Al Jazeera, .gov sites
- **MEDIUM** tier: regional news, Wikipedia, established outlets
- **LOW** tier: social media, unknown blogs, tabloids

### Agent 4 — CLIP Visual Consistency Agent *(local, no API)*
Uses OpenAI's CLIP model running entirely on your CPU to measure how visually consistent the image is with the claim text — without any internet connection after the first download.

- Computes image ↔ claim similarity
- Computes image ↔ caption similarity as a baseline
- Relative score determines visual consistency
- **Runs offline, ~1-2 seconds on CPU, 600MB download once**

---

## Confidence Score Formula

```
Final Score = (0.35 × Entity) + (0.30 × Temporal) + (0.20 × Credibility) + (0.15 × CLIP)

Final Score ≥ threshold  →  PRISTINE
Final Score < threshold  →  OUT-OF-CONTEXT
```

The threshold is adjusted based on out-of-context signals and claim plausibility.
Confidence % = |Final Score − 0.5| × 200
```

---

## Project Structure

```
trust-agent-misinformation-detection/
│
├── .env                          ← your API keys (never commit this)
├── .env.example                  ← template — copy to .env
├── requirements.txt
├── README.md
│
├── backend/                      ← preprocessing + retrieval
│   ├── config.py                 ← all settings from .env
│   ├── utils.py                  ← shared helpers
│   ├── captioning.py             ← GPT-4o Vision (primary) + BLIP (fallback)
│   ├── entity_extraction.py      ← spaCy NER
│   ├── metadata_extractor.py     ← EXIF + date extraction (AVIF/JPEG/PNG/WEBP)
│   ├── evidence_retrieval.py     ← SerpAPI + NewsAPI with clean query builder
│   ├── evidence_filter.py        ← removes noise/music videos from evidence
│   └── llm_client.py             ← OpenAI / Groq / Ollama provider factory
│
├── agents/                       ← all 4 reasoning agents
│   ├── entity_agent.py           ← entity match/contradiction check
│   ├── temporal_agent.py         ← TYPE A/B/C temporal reasoning
│   ├── credibility_agent.py      ← source trustworthiness scoring
│   ├── aggregator_agent.py       ← final verdict + explanation
│   └── clip_agent.py             ← local CLIP visual consistency (no API)
│
├── graph_state.py                ← LangGraph shared state (TypedDict)
├── graph_nodes.py                ← all pipeline node functions
├── orchestrator.py               ← LangGraph StateGraph (parallel agents)
├── historical_index.py           ← optional FAISS semantic index
│
├── api/                          ← FastAPI REST backend
│   ├── main.py                   ← POST /analyse, GET /health
│   ├── schemas.py                ← Pydantic models
│   └── dependencies.py           ← shared orchestrator instance
│
├── frontend/                     ← React + Vite dashboard
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── UploadForm.jsx
│           ├── VerdictCard.jsx
│           ├── AgentScores.jsx
│           ├── EvidenceList.jsx
│           └── LoadingSpinner.jsx
│
├── dataset/                      ← manual test dataset
│   ├── images/
│   └── test_dataset.csv
│
├── evaluate.py                   ← evaluation on manual dataset
├── benchmark_eval.py             ← evaluation on NewsCLIPpings benchmark
└── test_pipeline.py              ← quick single-image test
```

---

## Technology Stack

| Component | Technology | Local/API |
|---|---|---|
| Image captioning | GPT-4o Vision / Salesforce BLIP-large | API / Local |
| Visual consistency | OpenAI CLIP ViT-B/32 | **Local** |
| Named entity recognition | spaCy en_core_web_sm | **Local** |
| EXIF metadata | Pillow (PIL) | **Local** |
| Semantic search | FAISS + sentence-transformers | **Local** |
| Agent reasoning | OpenAI GPT-4o-mini / Groq LLaMA-3.3-70B / Ollama LLaMA-3.2 | API / Free / **Local** |
| Agent orchestration | LangGraph + LangChain Core | **Local** |
| Reverse image search | SerpAPI + ImgBB | API (optional) |
| Web/news search | SerpAPI / NewsAPI | API (optional) |
| Backend API | FastAPI + Uvicorn | **Local** |
| Frontend | React + Vite | **Local** |

---

## Three Deployment Modes

### Mode 1 — Online (Best Quality)
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o-mini
SERPAPI_API_KEY=...
IMGBB_API_KEY=...
USE_CLIP=true
```
Cost: ~$0.003–0.005 per analysis

### Mode 2 — Hybrid (Cheap, Recommended)
```env
LLM_PROVIDER=groq
OPENAI_API_KEY=sk-proj-...     # only for GPT-4o Vision captioning
GROQ_API_KEY=gsk_...           # reasoning agents — FREE
GROQ_MODEL=llama-3.3-70b-versatile
SERPAPI_API_KEY=...
USE_CLIP=true
```
Cost: ~$0.001 per analysis (OpenAI vision only, Groq is free)

### Mode 3 — Fully Offline (No Internet After Setup)
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434/v1
USE_CLIP=true
# All API keys left blank
```
Cost: $0 — runs entirely on CPU

---

## Setup Instructions

### Step 1 — Prerequisites
```bash
# Python 3.10+
python --version

# Node.js 18+ (for frontend)
node --version
```

### Step 2 — Virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> First run downloads BLIP (~900MB) and CLIP (~600MB) to HuggingFace cache.
> This is a one-time download. All subsequent runs use the cached models.

### Step 4 — For offline mode: install Ollama
```bash
# Download from https://ollama.com/download, then:
ollama pull llama3.2
ollama serve   # keep this running in a separate terminal
```

### Step 5 — Create .env file
```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

Open `.env` and fill in your keys. Minimum required:
```env
OPENAI_API_KEY=sk-proj-your-key-here
```

Verify it works:
```bash
python -c "from backend.config import Config; c = Config(); print('Key set:', bool(c.openai_api_key))"
```

### Step 6 — Run API server
```bash
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** — interactive Swagger UI for testing.

### Step 7 — Run React frontend
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** — web dashboard.

---

## API Reference

### GET /health
```json
{
  "status": "ok",
  "model": "gpt-4o-mini",
  "version": "1.0.0"
}
```

### POST /analyse
```
Content-Type: multipart/form-data
Fields:
  image : <image file>   (JPEG, PNG, WEBP, AVIF supported)
  claim : <text string>  (the caption being fact-checked)
```

**Response:**
```json
{
  "verdict": "OUT-OF-CONTEXT",
  "confidence_percent": 82,
  "explanation": "The image shows the 2020 Beirut port explosion. The claim states this happened in 2026 — a 6-year discrepancy confirmed by multiple credible sources.",
  "caption": "A massive explosion destroys buildings near a port. Grain silos visible in background, labelled 'Port of Beirut'.",
  "entity_score": 0.75,
  "temporal_score": 0.05,
  "credibility_score": 0.65,
  "clip_score": 0.60,
  "final_score": 0.42,
  "key_evidence_for_verdict": [
    "Temporal agent detected TYPE A time-trick — event was 2020, claim says 2026",
    "Al Jazeera and BBC report Beirut explosion as August 4, 2020",
    "6-year time gap between image origin and claimed date"
  ],
  "flags": [
    "Clear temporal mismatch — image is from 2020, claim says 2026",
    "Credible sources date this event to August 2020"
  ],
  "evidence": [...],
  "processing_time_sec": 22.4,
  "errors": []
}
```

---

## Testing

### Quick test (single image)
```bash
# Place any image as img1.jpg in your project root
python test_pipeline.py
```

### Evaluate on your manual dataset
```bash
# First 5 samples (quick check)
python evaluate.py --limit 5

# Full dataset
python evaluate.py

# Resume after crash
python evaluate.py --skip-done
```

### Evaluate on NewsCLIPpings benchmark (standard academic comparison)
```bash
# Dry run — download data only, no API calls
python benchmark_eval.py --limit 20 --dry

# 50 samples (~$0.10 cost with gpt-4o-mini + Groq)
python benchmark_eval.py --limit 50

# 100 samples (recommended for report)
python benchmark_eval.py --limit 100

# Resume if interrupted
python benchmark_eval.py --limit 100 --skip-done
```

The benchmark script downloads only the test split annotations (few MB) and
fetches only the specific images needed. **No 1TB download required.**

### Verify offline mode is working
Look for this in your terminal logs:
```
=== RUNNING IN FULLY OFFLINE MODE ===
  Captioning : BLIP (local)
  Reasoning  : Ollama llama3.2 (local)
  CLIP agent : enabled
  Evidence   : disabled (no API keys)
```

---

## Evaluation Metrics

| Metric | Formula | What it measures |
|---|---|---|
| Accuracy | (TP + TN) / Total | Overall correct predictions |
| Precision | TP / (TP + FP) | Of all OOC flags, how many were correct |
| Recall | TP / (TP + FN) | Of all actual OOC cases, how many were caught |
| F1-Score | 2×P×R / (P+R) | Balance between precision and recall |

Where:
- **TP** = correctly detected as OUT-OF-CONTEXT
- **TN** = correctly identified as PRISTINE
- **FP** = PRISTINE wrongly flagged as OOC (false alarm)
- **FN** = OOC case missed (predicted PRISTINE)

---

## Comparison with State-of-the-Art

| System | Accuracy | Training | GPU | Explainability | Temporal Agent | Credibility Agent | Confidence Score |
|---|---|---|---|---|---|---|---|
| COSMOS (2020) | ~73% | YES | YES | NO | NO | NO | NO |
| CCN (2022) | ~78% | YES | YES | NO | NO | NO | NO |
| SNIFFER (2024) | ~88% | YES | YES | Partial | NO | NO | NO |
| EXCLAIM (2025) | ~89% | NO | YES | YES | Partial | NO | NO |
| E2LVLM (2025) | 90.3% | YES | YES | YES | NO | NO | NO |
| **TRUST-AGENT** | **TBD** | **NO** | **NO** | **YES** | **YES** | **YES** | **YES** |

**Research gaps filled by TRUST-AGENT that no prior system addresses simultaneously:**
1. Dedicated Temporal Agent with TYPE A/B/C classification
2. Source Credibility Agent with tiered trustworthiness scoring
3. Local CLIP agent — visual consistency without any API
4. Per-agent confidence score breakdown shown to user
5. LLM-provider agnostic — OpenAI / Groq / Ollama in one config line
6. Fully offline deployment mode
7. End-to-end web application (no prior paper demonstrates this)

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | For online/hybrid | — | OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI model |
| `LLM_PROVIDER` | No | `openai` | `openai` / `groq` / `ollama` |
| `GROQ_API_KEY` | For hybrid | — | Groq API key (free) |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model |
| `OLLAMA_MODEL` | For offline | `llama3.2` | Local Ollama model |
| `OLLAMA_BASE_URL` | For offline | `http://localhost:11434/v1` | Ollama server URL |
| `USE_CLIP` | No | `true` | Enable local CLIP agent |
| `SERPAPI_API_KEY` | No | — | SerpAPI key (evidence retrieval) |
| `NEWSAPI_KEY` | No | — | NewsAPI key |
| `IMGBB_API_KEY` | No | — | ImgBB key (image upload for reverse search) |
| `WEIGHT_ENTITY` | No | `0.25` | Entity agent weight |
| `WEIGHT_TEMPORAL` | No | `0.30` | Temporal agent weight |
| `WEIGHT_CREDIBILITY` | No | `0.20` | Credibility agent weight |
| `WEIGHT_CLIP` | No | `0.25` | CLIP agent weight |
| `PRISTINE_THRESHOLD` | No | `0.60` | Score above which → PRISTINE |
| `REQUEST_TIMEOUT` | No | `45` | API timeout in seconds |

> **Minimum to run:** Only `OPENAI_API_KEY` is required for online mode.
> The pipeline degrades gracefully — if SerpAPI is missing, evidence
> retrieval is skipped. If OpenAI is missing but Groq is set, Groq handles
> reasoning and BLIP handles captioning.

---

## Research Background

This project addresses limitations identified across 7 research papers:

| Gap | Existing systems | TRUST-AGENT |
|---|---|---|
| No temporal reasoning | All prior work | TYPE A/B/C Temporal Agent |
| No source credibility | All prior work | Tiered credibility scoring |
| Binary output only | All prior work | Confidence % + per-agent breakdown |
| Requires training data | COSMOS, CCN, SNIFFER, E2LVLM | Zero-shot — no training |
| Requires GPU | All prior work | CPU only |
| API-dependent | EXCLAIM, E2LVLM | CLIP local + Ollama offline mode |
| No web deployment | All prior work | React + FastAPI live app |

**Key references:**
- Wu et al. E2LVLM. arXiv:2502.10455, Feb 2025
- Wu et al. EXCLAIM. arXiv:2504.06269, Mar 2025
- Qi et al. SNIFFER. CVPR 2024
- Abdelnabi et al. NewsCLIPpings. CVPR 2022
- Aneja et al. COSMOS. CVPR 2020

---

## Limitations

- First request is slow (30-60s) as BLIP and CLIP load into memory
- Without SerpAPI/NewsAPI, credibility agent defaults to neutral (0.55)
- Groq free tier: 100k tokens/day — switch to `llama-3.1-8b-instant` if exceeded
- BLIP captions are weaker than GPT-4o Vision for fine-grained scene details
- Offline mode (Ollama) has lower accuracy than online mode (~60-70% vs ~76-80%)
- CLIP has a 77-token text limit — long claims are automatically truncated

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'trust_agent'` | Run `uvicorn api.main:app` not `uvicorn trust_agent.api.main:app` |
| `ImportError: attempted relative import beyond top-level` | Your agent files have old `..backend` imports — replace with `backend` |
| `spaCy model not found` | Run `python -m spacy download en_core_web_sm` |
| `EXIF: AvifImageFile has no attribute _getexif` | Replace `backend/metadata_extractor.py` with the latest version |
| `429 Rate limit (Groq)` | Daily limit reached — change `GROQ_MODEL=llama-3.1-8b-instant` |
| `SerpAPI timeout` | Set `REQUEST_TIMEOUT=45` in `.env` |
| `ImgBB upload failed` | Intermittent — script retries 3 times automatically |
| `0x0.st: 403 Forbidden` | Set `IMGBB_API_KEY` in `.env` instead |

---

## Project Info

| Field | Detail |
|---|---|
| Institution | RCOEM, Nagpur |
| Department | Artificial Intelligence and Machine Learning |
| Programme | B.Tech CSE (AIML) |
| Semester | VI |
| Guide | Dr. Nisarg Gandhewar |
| Academic Year | 2024–25 |