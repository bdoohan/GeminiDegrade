"""
Gemini OCR Extraction Eval Harness
===================================
Compares two (or more) Gemini models on structured data extraction from OCR'd documents.

Metrics per test case per model:
  - latency_avg_s      : average seconds per call (over N runs)
  - latency_min_s      : fastest run
  - latency_max_s      : slowest run
  - input_tokens       : prompt token count (last run)
  - output_tokens      : completion token count (last run)
  - cost_usd           : estimated cost (last run)
  - json_valid         : 1 if response parsed as JSON, else 0
  - schema_valid       : 1 if output passes jsonschema validation, else 0
  - field_precision    : TP / (TP + FP)  — of extracted fields, how many are right
  - field_recall       : TP / (TP + FN)  — of expected fields, how many were found
  - field_f1           : harmonic mean of precision and recall
  - field_accuracy     : TP / total expected fields

Usage:
  pip install -r requirements.txt
  python eval.py                         # run all test cases, both models
  python eval.py --test-id invoice_001  # single test case
  python eval.py --model-label gemini-3.0  # single model
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema
import yaml
from google import genai
from google.genai import types as genai_types


# ---------------------------------------------------------------------------
# Config & loading helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_test_cases(path: str = "test_cases.yaml") -> List[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["test_cases"]


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_prompt_template(path: str = "prompts/extraction_prompt.txt") -> str:
    return Path(path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

def call_gemini(client: genai.Client, model_id: str, prompt: str, cfg: dict) -> tuple[str, int, int, float, str]:
    """
    Returns (response_text, input_tokens, output_tokens, latency_seconds, finish_reason).
    """
    extraction_cfg = cfg["extraction"]
    thinking_budget = extraction_cfg.get("thinking_budget", 0)

    gen_cfg = genai_types.GenerateContentConfig(
        temperature=extraction_cfg["temperature"],
        max_output_tokens=extraction_cfg["max_output_tokens"],
        thinking_config=genai_types.ThinkingConfig(thinking_budget=thinking_budget) if thinking_budget else None,
    )
    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=gen_cfg,
    )
    latency = time.perf_counter() - t0

    text = response.text if response.text else ""
    input_tokens = response.usage_metadata.prompt_token_count or 0
    output_tokens = response.usage_metadata.candidates_token_count or 0
    finish_reason = str(response.candidates[0].finish_reason) if response.candidates else "UNKNOWN"
    return text, input_tokens, output_tokens, latency, finish_reason


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> tuple[Any, bool]:
    """
    Attempt to parse JSON from the model response.
    Strips markdown code fences if present.
    Returns (parsed_object_or_None, is_valid_json).
    """
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        return None, False


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_schema(obj: Any, schema: dict) -> bool:
    try:
        jsonschema.validate(instance=obj, schema=schema)
        return True
    except jsonschema.ValidationError:
        return False


# ---------------------------------------------------------------------------
# Flatten JSON to dot-notation for field-level comparison
# ---------------------------------------------------------------------------

def flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict/list into {dot.notation.key: value} pairs."""
    result = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            result.update(flatten(v, full_key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            full_key = f"{prefix}[{i}]"
            result.update(flatten(v, full_key))
    else:
        result[prefix] = obj
    return result


def normalize_value(v: Any) -> str:
    """Normalize a leaf value for comparison."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v).strip().lower()


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_field_metrics(extracted: Any, ground_truth: Any) -> Dict[str, float]:
    """
    Compare flattened extracted JSON against flattened ground truth.

    TP: key present in both, values match after normalization
    FP: key present in extracted but value wrong OR key not in ground truth
    FN: key present in ground truth but missing or wrong in extracted
    """
    gt_flat = flatten(ground_truth)
    ex_flat = flatten(extracted) if extracted is not None else {}

    tp = 0
    fp = 0
    fn = 0

    for key, gt_val in gt_flat.items():
        if key in ex_flat:
            if normalize_value(ex_flat[key]) == normalize_value(gt_val):
                tp += 1
            else:
                fn += 1  # found but wrong → missed the right value
                fp += 1  # extracted a wrong value
        else:
            fn += 1  # missing entirely

    # FP for keys in extracted that aren't in ground truth at all
    for key in ex_flat:
        if key not in gt_flat:
            fp += 1

    total_expected = len(gt_flat)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = tp / total_expected if total_expected > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_expected_fields": total_expected,
        "field_precision": round(precision, 4),
        "field_recall": round(recall, 4),
        "field_f1": round(f1, 4),
        "field_accuracy": round(accuracy, 4),
    }


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(input_tokens: int, output_tokens: int, pricing: dict) -> float:
    input_cost  = input_tokens  / 1_000_000 * pricing["input_per_million_tokens"]
    output_cost = output_tokens / 1_000_000 * pricing["output_per_million_tokens"]
    return round(input_cost + output_cost, 6)


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(cfg: Dict, test_cases: List[dict], prompt_template: str,
             filter_test_id: Optional[str] = None,
             filter_model_label: Optional[str] = None) -> List[dict]:

    client = genai.Client(api_key=cfg["gemini"]["api_key"])
    runs_per_test = cfg["eval"]["runs_per_test"]
    results = []

    models_cfg = cfg["gemini"]["models"]
    if filter_model_label:
        models_cfg = [m for m in models_cfg if m["label"] == filter_model_label]
        if not models_cfg:
            print(f"[ERROR] No model with label '{filter_model_label}' found in config.")
            sys.exit(1)

    for tc in test_cases:
        if filter_test_id and tc["id"] != filter_test_id:
            continue

        print(f"\n{'='*60}")
        print(f"Test case: {tc['id']}  —  {tc['description']}")
        print(f"{'='*60}")

        document_text = load_text(tc["document"])
        schema        = load_json(tc["schema"])
        ground_truth  = load_json(tc["ground_truth"])

        prompt = prompt_template.replace("{schema}", json.dumps(schema, indent=2))
        prompt = prompt.replace("{document}", document_text)

        for model_cfg in models_cfg:
            label    = model_cfg["label"]
            model_id = model_cfg["model_id"]
            pricing  = model_cfg["pricing"]

            print(f"\n  Model: {label} ({model_id})")

            latencies = []
            last_text = ""
            last_input_tokens = 0
            last_output_tokens = 0

            for run in range(runs_per_test):
                try:
                    text, in_tok, out_tok, lat, finish = call_gemini(client, model_id, prompt, cfg)
                    latencies.append(lat)
                    last_text = text
                    last_input_tokens = in_tok
                    last_output_tokens = out_tok
                    status = "ok"
                except Exception as e:
                    print(f"    Run {run+1}: ERROR — {e}")
                    latencies.append(None)
                    finish = "ERROR"
                    status = "error"
                    continue

                print(f"    Run {run+1}: {lat:.2f}s  |  in={in_tok} out={out_tok}  finish={finish}  [{status}]")

            valid_latencies = [l for l in latencies if l is not None]
            latency_avg = sum(valid_latencies) / len(valid_latencies) if valid_latencies else None
            latency_min = min(valid_latencies) if valid_latencies else None
            latency_max = max(valid_latencies) if valid_latencies else None

            parsed, json_valid = parse_json_response(last_text)
            schema_valid = validate_schema(parsed, schema) if json_valid else False
            cost = estimate_cost(last_input_tokens, last_output_tokens, pricing)

            if json_valid:
                metrics = compute_field_metrics(parsed, ground_truth)
            else:
                metrics = {
                    "tp": 0, "fp": 0, "fn": len(flatten(ground_truth)),
                    "total_expected_fields": len(flatten(ground_truth)),
                    "field_precision": 0.0, "field_recall": 0.0,
                    "field_f1": 0.0, "field_accuracy": 0.0,
                }

            row = {
                "test_id":            tc["id"],
                "model_label":        label,
                "model_id":           model_id,
                "runs":               runs_per_test,
                "latency_avg_s":      round(latency_avg, 3) if latency_avg is not None else None,
                "latency_min_s":      round(latency_min, 3) if latency_min is not None else None,
                "latency_max_s":      round(latency_max, 3) if latency_max is not None else None,
                "input_tokens":       last_input_tokens,
                "output_tokens":      last_output_tokens,
                "cost_usd":           cost,
                "json_valid":         int(json_valid),
                "schema_valid":       int(schema_valid),
                **metrics,
            }
            results.append(row)

            print(f"    → JSON valid={json_valid}  schema_valid={schema_valid}")
            print(f"    → precision={metrics['field_precision']:.3f}  "
                  f"recall={metrics['field_recall']:.3f}  "
                  f"F1={metrics['field_f1']:.3f}  "
                  f"cost=${cost:.5f}")

            # Dump last raw response for debugging
            debug_dir = Path(cfg["eval"]["output_dir"]) / "raw_responses"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"{tc['id']}__{label.replace('/', '_')}.txt"
            debug_path.write_text(last_text, encoding="utf-8")

    return results


# ---------------------------------------------------------------------------
# Output: CSV + summary table
# ---------------------------------------------------------------------------

def write_csv(results: List[dict], output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = str(Path(output_dir) / f"eval_results_{timestamp}.csv")
    if not results:
        return path
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    return path


def print_summary(results: List[dict]) -> None:
    if not results:
        print("\nNo results to summarize.")
        return

    # Group by model
    from collections import defaultdict
    by_model: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_model[r["model_label"]].append(r)

    col_w = 22
    metric_keys = [
        "latency_avg_s", "cost_usd", "json_valid", "schema_valid",
        "field_precision", "field_recall", "field_f1", "field_accuracy",
    ]

    print(f"\n{'='*70}")
    print("SUMMARY — averages across all test cases per model")
    print(f"{'='*70}")
    header = f"{'metric':<25}" + "".join(f"{m:<{col_w}}" for m in by_model.keys())
    print(header)
    print("-" * len(header))

    for key in metric_keys:
        row = f"{key:<25}"
        for label, rows in by_model.items():
            vals = [r[key] for r in rows if r.get(key) is not None]
            avg = sum(vals) / len(vals) if vals else float("nan")
            row += f"{avg:<{col_w}.4f}"
        print(row)

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gemini OCR extraction eval harness")
    parser.add_argument("--config",      default="config.yaml",            help="Path to config.yaml")
    parser.add_argument("--test-cases",  default="test_cases.yaml",        help="Path to test_cases.yaml")
    parser.add_argument("--prompt",      default="prompts/extraction_prompt.txt")
    parser.add_argument("--test-id",     default=None,  help="Run only this test case ID")
    parser.add_argument("--model-label", default=None,  help="Run only this model label")
    args = parser.parse_args()

    cfg            = load_config(args.config)
    test_cases     = load_test_cases(args.test_cases)
    prompt_template = load_prompt_template(args.prompt)

    results = run_eval(
        cfg, test_cases, prompt_template,
        filter_test_id=args.test_id,
        filter_model_label=args.model_label,
    )

    csv_path = write_csv(results, cfg["eval"]["output_dir"])
    print(f"\nResults written to: {csv_path}")
    print_summary(results)


if __name__ == "__main__":
    main()
