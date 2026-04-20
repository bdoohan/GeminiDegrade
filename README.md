# GeminiDegrade

Eval harness for comparing Gemini models on structured data extraction from OCR'd documents. Measures latency, cost, precision, recall, and F1 across configurable test cases.

## Setup

```bash
git clone https://github.com/<your-username>/GeminiDegrade.git
cd GeminiDegrade
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Edit `config.yaml` — add your API key and confirm model IDs:

```yaml
gemini:
  api_key: "YOUR_GEMINI_API_KEY_HERE"
  models:
    - label: "gemini-3.0"
      model_id: "gemini-3-pro-preview"
    - label: "gemini-3.1"
      model_id: "gemini-3.1-pro-preview"
```

Get a key at [aistudio.google.com](https://aistudio.google.com). List available model IDs:

```bash
python3 -c "from google import genai; [print(m.name) for m in genai.Client(api_key='YOUR_KEY').models.list()]"
```

## Run

```bash
# Full suite (all models × all test cases)
python3 eval.py

# Single test case
python3 eval.py --test-id hospital_discharge_001

# Single model
python3 eval.py --model-label gemini-3.1

# Both filters
python3 eval.py --test-id invoice_001 --model-label gemini-3.0
```

Results are written to `results/eval_results_<timestamp>.csv`. Raw model responses saved to `results/raw_responses/` for inspection.

## Metrics

| Metric | Description |
|---|---|
| `latency_avg_s` | Average wall-clock time per call (over N runs) |
| `latency_min/max_s` | Latency spread |
| `input_tokens` / `output_tokens` | Token counts from API metadata |
| `cost_usd` | Estimated cost based on pricing in `config.yaml` |
| `json_valid` | 1 if response parsed as valid JSON |
| `schema_valid` | 1 if output passes JSON Schema validation |
| `field_precision` | Of extracted fields, fraction that are correct |
| `field_recall` | Of expected fields, fraction that were found |
| `field_f1` | Harmonic mean of precision and recall |
| `field_accuracy` | TP / total expected fields |

## Test cases

| ID | Document type | Fields |
|---|---|---|
| `invoice_001` | Vendor invoice | ~30 |
| `medical_form_001` | Patient intake form | ~50 |
| `contract_001` | Software services agreement | ~40 |
| `hospital_discharge_001` | Hospital discharge summary | ~120 |

## Add a test case

1. Add a document to `documents/`
2. Add a JSON Schema to `schemas/`
3. Add ground truth JSON to `ground_truth/`
4. Add an entry to `test_cases.yaml`

No changes to `eval.py` needed.

## Config options

```yaml
eval:
  runs_per_test: 3       # runs per test case per model (latency is averaged)

extraction:
  temperature: 1.0
  max_output_tokens: 10000
  thinking_budget: 5000  # caps thinking tokens on gemini-3.x models
```
