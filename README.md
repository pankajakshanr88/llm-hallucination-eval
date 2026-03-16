# LLM Hallucination Evaluation System

A systematic evaluation pipeline for detecting and measuring hallucinations in AI-generated legal document analysis. Built to answer a critical product question: *"How do we know our AI isn't making things up about users' contracts?"*

## Why This Matters for AI Products

Shipping an AI feature that analyzes legal documents carries real risk — a hallucinated clause or fabricated obligation could lead users to make wrong decisions. Traditional QA (unit tests, manual review) doesn't work for non-deterministic AI outputs. This project establishes a repeatable, automated evaluation framework that gives product teams quantitative confidence in AI quality before launch and continuous monitoring after.

The system uses LangSmith for tracing and a golden dataset of contract Q&A pairs with known-correct answers, evaluated by LLM-as-Judge pipelines that measure faithfulness, hallucination rate, and accuracy.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Golden Dataset │ ──► │  Evaluation      │ ──► │  LangSmith      │
│  (Questions +   │     │  Pipeline        │     │  Dashboard      │
│   Contracts)    │     │  (LLM-as-Judge)  │     │  (Metrics)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Key Thresholds

| Metric | Launch Requirement |
|--------|-------------------|
| Hallucination Rate | ≤ 5% |
| Answer Accuracy | ≥ 85% |
| Judge-Human Agreement | ≥ 80% |

## Setup

### 1. Create a LangSmith Account

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Sign up for a free account
3. Navigate to **Settings** > **API Keys**
4. Create a new API key and copy it

### 2. Install Dependencies

```bash
cd hallucination-tracking
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# LANGCHAIN_API_KEY=your_langsmith_api_key
# OPENAI_API_KEY=your_openai_api_key
```

### 4. Verify Setup

```bash
python -c "from src.config import settings; print(settings)"
```

## Usage

### Run Full Evaluation

```bash
python scripts/run_evaluation.py
```

This will:
- Load the golden dataset
- Run the mock assistant against test cases
- Apply LLM-as-Judge evaluators
- Calculate and display metrics

### Generate Report

```bash
python scripts/generate_report.py
```

### Run Tests

```bash
pytest tests/
```

## Project Structure

```
hallucination-tracking/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── .env.example                 # Environment variables template
├── src/
│   ├── __init__.py
│   ├── config.py               # LangSmith configuration
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── golden_dataset.py   # Dataset loader and structure
│   │   └── sample_data.json    # Sample contracts + Q&A pairs
│   ├── evaluators/
│   │   ├── __init__.py
│   │   ├── faithfulness.py     # Faithfulness evaluator
│   │   ├── hallucination.py    # Hallucination detector
│   │   └── accuracy.py         # Accuracy scorer
│   ├── mock_assistant.py       # Mock Document Assistant
│   └── pipeline.py             # Main evaluation pipeline
├── scripts/
│   ├── run_evaluation.py       # Run full evaluation
│   └── generate_report.py      # Generate metrics report
└── tests/
    └── test_evaluators.py      # Unit tests for evaluators
```

## Evaluators

### Faithfulness Evaluator
Checks if the assistant's answer only uses facts from the contract context.

### Hallucination Detector
Identifies specific unsupported claims in the assistant's response.

### Accuracy Scorer
Rates the overall correctness of the response on a 0-1 scale.

## Metrics Tracked

- **Hallucination Rate**: Percentage of responses containing hallucinated content
- **Accuracy Rate**: Percentage of accurate responses
- **Judge-Human Agreement**: How often the LLM judge agrees with human labels
- **Category Breakdown**: Metrics split by question type (Summarization, Clause Deep Dive, etc.)

## Thresholds and Actions

| Hallucination Rate | Action |
|-------------------|--------|
| ≤ 5% | Ready for launch |
| 5-10% | Investigate and improve; pause launch |
| > 10% | Severe issues; block launch or disable feature |

