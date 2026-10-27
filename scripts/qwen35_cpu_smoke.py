"""Real CPU Qwen3.5-0.8B smoke: one writer decision and one career plan.

Runs inference without API credentials. Outputs raw answers and latency;
never substitutes deterministic answers for model generations.
"""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from progression_memory.v2.model_writer import ModelWriter
from progression_memory.v2.models import Observation

MODEL = "Qwen/Qwen3.5-0.8B"


class LocalQwen:
    def __init__(self):
        torch.set_num_threads(2)
        self.processor = AutoProcessor.from_pretrained(MODEL)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).eval()
        self.calls = []

    def generate_messages(self, messages, json_schema=None):
        ids = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
            enable_thinking=False,
        )
        started = time.monotonic()
        with torch.inference_mode():
            out = self.model.generate(
                **ids, max_new_tokens=192, do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        text = self.processor.decode(
            out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        self.calls.append({"seconds": round(time.monotonic() - started, 2),
                           "input_tokens": int(ids["input_ids"].shape[-1]),
                           "output_tokens": int(out.shape[-1] - ids["input_ids"].shape[-1])})
        return text


def main():
    started = time.monotonic()
    backend = LocalQwen()
    observation = Observation(
        id="person-2026-10", source_id="conversation-2026-10",
        subject_id="fictional:alex",
        text="I still want to found an AI infrastructure startup eventually. "
             "But I have decided not to pursue a master's this year. "
             "For the next two years I prioritize stable engineering income "
             "to build financial runway. I cannot relocate outside Toronto.",
        recorded_at=datetime(2026, 10, 26, tzinfo=UTC),
    )
    result = {"model": MODEL}
    try:
        result["model_writer"] = {
            "status": "parsed",
            "proposal": ModelWriter(backend).decide(observation, []).model_dump(mode="json"),
        }
    except (ValueError, TypeError, RuntimeError, KeyError) as error:
        result["model_writer"] = {"status": "failed",
                                  "error_type": type(error).__name__,
                                  "message": str(error)[:400]}
    planning = [
        {"role": "system", "content":
            "You are a career planning assistant. Return ONLY valid JSON with keys "
            "recommendation, goal, priority, constraint, tradeoffs (array), "
            "next_steps (array). Do not invent facts."},
        {"role": "user", "content":
            "Memory: Eventually found an AI infrastructure startup. "
            "Current priority: stable engineering income for two years to build runway. "
            "Graduate school explicitly deferred this year. Must stay in Toronto. "
            "Decision: Option A, Toronto backend role with stable salary. "
            "Option B, full-time master's requiring relocation and no salary. "
            "Which better fits the current goals, and what should be done next?"},
    ]
    raw = backend.generate_messages(planning)
    try:
        payload = json.loads(raw)
        result["planner"] = {"status": "parsed", "answer": payload}
    except json.JSONDecodeError:
        result["planner"] = {"status": "invalid_json", "raw": raw}
    result["inference"] = backend.calls
    result["wall_seconds"] = round(time.monotonic() - started, 2)
    output = Path("results/qwen35_smoke.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
