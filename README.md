# TRUST-AGENT: Out-of-Context Misinformation Detection System


---

## What is TRUST-AGENT?

TRUST-AGENT is an AI-powered multi-agent system that detects **out-of-context (OOC) misinformation** — a form of fake news where a **real, genuine image** is paired with a **false or misleading claim** about a different time, location, or event.

For example: a real flood photograph from 2015 being shared on social media with the claim *"This is from the 2024 Kerala floods."* The image is real, the claim is false — this is out-of-context misinformation, and it is very hard for existing AI systems to detect.

TRUST-AGENT tackles this by imitating how a human fact-checker thinks — using three independent AI agents to analyse entity consistency, temporal consistency, and source credibility, then synthesising a final verdict with a plain-English explanation.

---

## Key Features

- **Multi-agent reasoning** — three specialist agents run in parallel (LangGraph)
- **GPT-4o powered** — all reasoning agents use OpenAI GPT-4o via API
- **BLIP image captioning** — automatically generates a description of any uploaded image
- **Evidence retrieval** — reverse image search + web/news search via SerpAPI and NewsAPI
- **Explainable output** — every verdict comes with a confidence score and human-readable explanation
- **FastAPI backend** — REST API with Swagger UI for easy testing
- **React frontend** — clean web dashboard for uploading images and viewing results
- **No training required** — no dataset download, no GPU needed

---

## System Architecture

```
User Input (Image + Text Claim)
          │
          ▼
┌─────────────────────────────┐
│     Evidence Retrieval      │
│  • Reverse Image Search     │  ← SerpAPI Google Reverse Image
│  • Web & News Search        │  ← SerpAPI / NewsAPI
│  • Historical FAISS Index   │  ← Optional semantic search
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Preprocessing & Extraction │
│  • BLIP Image Captioning    │  ← Salesforce BLIP model
│  • spaCy Entity Extraction  │  ← Named entity recognition
│  • EXIF Metadata Extraction │  ← Capture date, GPS, camera
└─────────────┬───────────────┘
              │
    ┌─────────┴──────────┐
    │   FAN-OUT          │  (3 agents run in PARALLEL via LangGraph)
    ▼         ▼          ▼
┌────────┐ ┌────────┐ ┌────────────┐
│ Entity │ │Temporal│ │ Source     │
│ Agent  │ │ Agent  │ │ Credibility│  ← All powered by GPT-4o
└───┬────┘ └───┬────┘ └─────┬──────┘
    │          │             │
    └──────────┴─────────────┘
              │   FAN-IN
              ▼
┌─────────────────────────────┐
│      Score Fusion           │
│  Entity(35%) + Temporal(35%)│
│  + Credibility(30%)         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│     Aggregator Agent        │  ← GPT-4o generates explanation
│  Final Verdict + Confidence │
│  Human-readable Explanation │
└─────────────┬───────────────┘
              │
              ▼
    PRISTINE / OUT-OF-CONTEXT
    Confidence % + Explanation
```

---

## Agents Explained

### Agent 1 — Entity Analysis Agent
Checks whether the **people, places, organisations, and dates** mentioned in the claim are consistent with what the image actually shows (via BLIP caption + evidence).

- Score 1.0 → entities match perfectly
- Score 0.5 → partial match or ambiguous
- Score 0.0 → clear contradiction (wrong person, wrong country, wrong event)

### Agent 2 — Temporal Reasoning Agent
Checks whether the **time period** implied by the claim matches the time period the image actually comes from. This is the most important agent for catching OOC misinformation, since the most common pattern is a real old photo being claimed to show a recent event.

- Uses EXIF capture date from the image
- Uses publication dates from retrieved news articles
- Uses years/dates mentioned in the claim text

### Agent 3 — Source Credibility Agent
Checks **how trustworthy the sources** in the evidence are, and whether credible sources support or contradict the claim.

- HIGH credibility: Reuters, BBC, AP, NYT, .gov sites
- MEDIUM: regional news, Wikipedia
- LOW: social media, unknown domains

### Agent 4 — Aggregator Agent
Synthesises the three agent outputs into a final verdict with a plain-English explanation that a non-expert can understand.

---

## Confidence Score Formula

```
Final Score = (0.35 × Entity Score) + (0.35 × Temporal Score) + (0.30 × Credibility Score)

If Final Score ≥ 0.60  →  PRISTINE
If Final Score < 0.60  →  OUT-OF-CONTEXT

Confidence % = |Final Score − 0.5| × 200
```

---

## Project Structure

```
trust-agent-misinformation-detection/
│
├── .env                        ← your API keys (never commit this)
├── .env.example                ← template
├── requirements.txt            ← all Python dependencies
├── __init__.py
│
├── backend/                    ← preprocessing + evidence retrieval
│   ├── config.py               ← reads all settings from .env
│   ├── utils.py                ← shared helpers
│   ├── captioning.py           ← BLIP: image → caption
│   ├── entity_extraction.py    ← spaCy: text → named entities
│   ├── metadata_extractor.py   ← EXIF + date extraction
│   └── evidence_retrieval.py   ← SerpAPI + NewsAPI search
│
├── agents/                     ← all 4 Claude-powered agents
│   ├── entity_agent.py         ← entity match/contradiction
│   ├── temporal_agent.py       ← time consistency check
│   ├── credibility_agent.py    ← source trust scoring
│   └── aggregator_agent.py     ← final verdict + explanation
│
├── graph_state.py              ← LangGraph shared state schema
├── graph_nodes.py              ← all 6 pipeline node functions
├── orchestrator.py             ← LangGraph pipeline (parallel agents)
├── historical_index.py         ← optional FAISS semantic index
│
├── api/                        ← FastAPI REST backend
│   ├── main.py                 ← POST /analyse, GET /health
│   ├── schemas.py              ← request/response models
│   └── dependencies.py        ← shared orchestrator instance
│
├── frontend/                   ← React + Vite dashboard
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── UploadForm.jsx
│           ├── VerdictCard.jsx
│           ├── AgentScores.jsx
│           └── EvidenceList.jsx
│
├── dataset/                    ← your manual test dataset
│   ├── images/                 ← test images (img001.jpg, ...)
│   └── test_dataset.csv        ← ground truth labels
│
└── test_pipeline.py            ← quick end-to-end test script
```

---

## Technology Stack

| Component | Technology |
|---|---|
| LLM / Reasoning Agents | OpenAI GPT-4o |
| Agent Orchestration | LangGraph + LangChain Core |
| Image Captioning | Salesforce BLIP (via HuggingFace Transformers) |
| Named Entity Recognition | spaCy `en_core_web_sm` |
| Semantic Search | FAISS + sentence-transformers |
| Reverse Image Search | SerpAPI Google Reverse Image |
| Web / News Search | SerpAPI Google Search + NewsAPI |
| Image Upload (for search) | ImgBB API |
| Backend API | FastAPI + Uvicorn |
| Frontend | React + Vite |
| Language | Python 3.10+ |

---

## Setup Instructions

### Step 1 — Clone / download the project

Place all files inside a single folder, e.g.:
```
trust-agent-misinformation-detection/
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Activate — Windows:
venv\Scripts\activate

# Activate — Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> Note: First install downloads BLIP (~900 MB) and sentence-transformers models.
> This is a one-time download. Subsequent runs are fast.

### Step 4 — Create your .env file

Copy `.env.example` to `.env` and fill in your keys:

```bash
copy .env.example .env       # Windows
cp .env.example .env         # Mac/Linux
```

Open `.env` and set at minimum:

```env
OPENAI_API_KEY=sk-proj-your-key-here
```

Get your key from: https://platform.openai.com/api-keys

### Step 5 — Run the API server

```bash
uvicorn api.main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** to see the interactive Swagger UI.

### Step 6 — Run the React frontend (optional)

```bash
cd frontend
npm install
npm run dev
```

Visit **http://localhost:5173** for the web dashboard.

---

## API Usage

### Health check
```
GET http://localhost:8000/health
```
```json
{"status": "ok", "model": "gpt-4o", "version": "1.0.0"}
```

### Analyse an image + claim
```
POST http://localhost:8000/analyse
Content-Type: multipart/form-data

image: <image file>
claim: "This photo shows protesters in New Delhi, 2024."
```

**Response:**
```json
{
  "verdict": "OUT-OF-CONTEXT",
  "confidence_percent": 87,
  "explanation": "The image appears to show a protest outside the Washington State Capitol building in the United States, not in New Delhi. Evidence sources confirm this location, and no temporal or entity matches support the claim.",
  "caption": "a crowd of people holding signs outside a government building",
  "entity_score": 0.1,
  "temporal_score": 0.45,
  "credibility_score": 0.2,
  "final_score": 0.25,
  "key_evidence_for_verdict": [
    "BLIP caption identifies a Western government building, not Indian Parliament",
    "No evidence sources mention New Delhi in connection with this image",
    "Entity contradiction: claim says India, image shows US Capitol dome"
  ],
  "flags": [
    "Location mismatch: image shows US Capitol, claim says New Delhi",
    "No corroborating sources found for claimed event"
  ],
  "evidence": [...],
  "processing_time_sec": 12.4,
  "errors": []
}
```

---

## Evaluation / Testing

### Quick test (no dataset needed)
```bash
python test_pipeline.py
```

Place any image as `img1.jpg` in the project root. The script tests two claims — one truthful, one misleading — and prints the verdict.

### Evaluation on your test dataset

Create `dataset/test_dataset.csv`:
```csv
id,image_path,claim,ground_truth
1,dataset/images/img001.jpg,"Floods from Kerala 2024",OUT-OF-CONTEXT
2,dataset/images/img002.jpg,"Protest outside Washington Capitol 2023",PRISTINE
```

Then run the evaluation script (generates Accuracy / Precision / Recall / F1):
```bash
python evaluate.py --dataset dataset/test_dataset.csv
```

### How to build the test dataset

Collect **30–50 real examples** from verified fact-checking websites:

| Source | URL |
|---|---|
| BoomLive | boomlive.in/fact-check |
| AltNews | altnews.in |
| AFP Fact Check | factcheck.afp.com |
| Snopes | snopes.com |
| Vishvas News | vishvasnews.com |

For each case:
1. Download the image → save as `dataset/images/img001.jpg`
2. Copy the false claim that was spreading
3. Label as `OUT-OF-CONTEXT`
4. Use the same image with the correct caption → label as `PRISTINE`

**Target: 15 OUT-OF-CONTEXT + 15 PRISTINE = 30 samples minimum**

---

## Evaluation Metrics

| Metric | Formula | What it measures |
|---|---|---|
| Accuracy | (TP + TN) / Total | Overall correct predictions |
| Precision | TP / (TP + FP) | How many flagged as OOC were actually OOC |
| Recall | TP / (TP + FN) | How many actual OOC cases were caught |
| F1-Score | 2 × P × R / (P + R) | Balance of precision and recall |

Where:
- **TP** = correctly identified as OUT-OF-CONTEXT
- **TN** = correctly identified as PRISTINE
- **FP** = incorrectly flagged as OUT-OF-CONTEXT (was actually PRISTINE)
- **FN** = missed OOC case (predicted PRISTINE but was OUT-OF-CONTEXT)

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | Your OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o` | Model to use (`gpt-4o` or `gpt-4o-mini`) |
| `SERPAPI_API_KEY` | No | — | SerpAPI key for evidence retrieval |
| `NEWSAPI_KEY` | No | — | NewsAPI key for news search |
| `IMGBB_API_KEY` | No | — | ImgBB key for image upload |
| `WEIGHT_ENTITY` | No | `0.35` | Entity agent score weight |
| `WEIGHT_TEMPORAL` | No | `0.35` | Temporal agent score weight |
| `WEIGHT_CREDIBILITY` | No | `0.30` | Credibility agent score weight |
| `PRISTINE_THRESHOLD` | No | `0.60` | Score above which verdict = PRISTINE |
| `DEFAULT_TOP_K` | No | `5` | Number of evidence items to retrieve |

> The pipeline works with only `OPENAI_API_KEY` set. Evidence retrieval APIs are optional — agents will reason based on the image caption and claim alone if no evidence is found.

---

## Research Background

This project addresses identified gaps in existing out-of-context misinformation detection research:

- **COSMOS (Aneja et al., 2020)** — established the OOC benchmark but provides only binary decisions without explanatory reasoning
- **NewsCLIPpings (Abdelnabi et al., 2022)** — large-scale dataset for training multimodal models, focuses on detection accuracy not explainability
- **EXCLAIM (Wu et al., 2025)** — agentic system achieving 4.3% higher accuracy than SOTA but requires extensive retrieval
- **E2LVLM (Wu et al., 2025)** — evidence-enhanced approach but raw evidence transmission can introduce false information

TRUST-AGENT addresses these gaps by combining multi-agent reasoning with explainability, without requiring model training or a large dataset.

---

## Limitations

- First request is slow (~30s) as BLIP model loads into memory
- Without SerpAPI/NewsAPI keys, evidence retrieval is disabled and accuracy may be lower
- GPT-4o API calls cost money — use `gpt-4o-mini` in `.env` for cheaper testing
- BLIP captions are short and may miss fine-grained details in complex images
- System works best with English-language claims

---

## Future Work

- Add video frame analysis for video-based OOC misinformation
- Integrate Google Fact Check Tools API for direct fact-check lookup
- Add Hindi/Marathi language support for Indian regional misinformation
- Build a browser extension for real-time social media checking
- Deploy on Hugging Face Spaces for public access

---

