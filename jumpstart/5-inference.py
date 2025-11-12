"""Run inference against multiple SageMaker endpoints and capture responses.

The script reads an evaluation dataset plus prompt template, invokes every
configured endpoint in parallel for each example, and writes a single CSV with
columns: ``instruction``, ``context``, ``expected_answer`` (taken from the
dataset's ``response`` field), and one column per endpoint label containing the
generated answer. Downstream evaluation should operate entirely on that CSV.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from botocore.exceptions import ClientError
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer


@dataclass
class ModelEndpoint:
    label: str
    endpoint_name: str


class EndpointInvoker:
    def __init__(self, spec: ModelEndpoint) -> None:
        self.spec = spec
        self.predictor = Predictor(
            endpoint_name=spec.endpoint_name,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )

    def invoke(self, payload: Dict) -> str:
        parameters = dict(payload.get("parameters", {}))
        request = {
            "inputs": payload["inputs"],
            "parameters": parameters,
        }
        response = self.predictor.predict(request, custom_attributes="accept_eula=true")
        if isinstance(response, list):
            response = response[0]
        generated = response.get("generated_text") or response.get("generated_texts")
        if isinstance(generated, list):
            generated = generated[0]
        return generated or ""


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_template(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_payload(example: Dict, template: Dict[str, str], *, max_new_tokens: int) -> Dict:
    prompt = template["prompt"].format(
        instruction=example.get("instruction", ""),
        context=example.get("context", ""),
    )
    return {
        "inputs": f"{prompt}\n\n### Response:\n",
        "parameters": {"max_new_tokens": max_new_tokens},
    }


def expected_answer(example: Dict) -> str:
    return example.get("response", "")


def generate_responses(
    eval_rows: Iterable[Dict],
    template: Dict[str, str],
    invokers: List[EndpointInvoker],
    *,
    max_new_tokens: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for example in eval_rows:
        payload = build_payload(example, template, max_new_tokens=max_new_tokens)
        instruction = example.get("instruction", "")
        context = example.get("context", "")
        expected = expected_answer(example)

        row: Dict[str, object] = {
            "instruction": instruction,
            "context": context,
            "expected_answer": expected,
        }

        with ThreadPoolExecutor(max_workers=len(invokers)) as executor:
            future_map = {
                executor.submit(invoker.invoke, payload): invoker.spec.label for invoker in invokers
            }
            for future in as_completed(future_map):
                label = future_map[future]
                try:
                    answer = future.result()
                except ClientError as exc:  # pragma: no cover - network side effect
                    answer = f"ERROR: {exc}"
                row[label] = answer

        rows.append(row)

    df = pd.DataFrame(rows)
    label_cols: List[str] = []
    for invoker in invokers:
        label = invoker.spec.label
        if label not in label_cols:
            label_cols.append(label)
    ordered_cols = ["instruction", "context", "expected_answer"] + label_cols
    return df[ordered_cols]


if __name__ == "__main__":
    EVAL_PATH = Path("test.jsonl")
    TEMPLATE_PATH = Path("template.json")
    OUTPUT_DIR = Path("logs") / "inference_runs"
    MAX_NEW_TOKENS = 100

    MODEL_ENDPOINTS: List[ModelEndpoint] = [
        ModelEndpoint(label="good", endpoint_name="jumpstart-ft-1-8117cd07-endpoint"),
        ModelEndpoint(label="base", endpoint_name="jumpstart-ft-2-eff48b5d-endpoint"),
        ModelEndpoint(label="same", endpoint_name="jumpstart-ft-1-43ae7938-endpoint"),
    ]

    if not EVAL_PATH.exists():
        raise SystemExit(f"Evaluation dataset not found: {EVAL_PATH}")

    if not TEMPLATE_PATH.exists():
        raise SystemExit(f"Prompt template not found: {TEMPLATE_PATH}")

    if not MODEL_ENDPOINTS:
        raise SystemExit("Populate MODEL_ENDPOINTS with the endpoints to query.")

    template = load_template(TEMPLATE_PATH)
    eval = load_jsonl(EVAL_PATH)
    invokers = [EndpointInvoker(spec) for spec in MODEL_ENDPOINTS]

    responses_df = generate_responses(
        eval,
        template,
        invokers,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    from zoneinfo import ZoneInfo

    timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%d-%H%M%S")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    responses_path = run_dir / "model_responses.csv"
    responses_df.to_csv(responses_path, index=False)

    latest_dir = OUTPUT_DIR / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    latest_path = latest_dir / "model_responses.csv"
    responses_df.to_csv(latest_path, index=False)

    print(f"Inference responses written to: {responses_path}")
    print(f"Latest responses available at: {latest_path}")
