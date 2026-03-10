from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.agents.agent_pipeline import AgentPipeline
from backend.agents.schema import AgentInput
from backend.newsclippings_loader import load_examples


def build_agent_input_from_dataset(example: Any, claim_text: str | None = None) -> AgentInput:
    return AgentInput(
        claim_text=claim_text or example.caption,
        caption=example.caption,
        timestamp=example.timestamp,
        source=example.source,
        entities=example.caption_entities_spacy,
        similarity_score=example.similarity_score,
        metadata={
            "id": example.id,
            "image_id": example.image_id,
            "topic": example.topic,
            "title": example.title,
            "falsified_label": example.falsified,
        },
    )


def evaluate_sample(split: str = "val", index: int = 0, claim_text: str | None = None) -> dict[str, Any]:
    examples = load_examples(split=split, limit=index + 1)
    if index >= len(examples):
        raise IndexError(f"Index {index} out of range for split '{split}'.")

    sample = examples[index]
    payload = build_agent_input_from_dataset(sample, claim_text=claim_text)
    result = AgentPipeline().run(payload)
    return {
        "dataset_example": asdict(sample),
        "agent_input": asdict(payload),
        "pipeline_result": asdict(result),
    }


if __name__ == "__main__":
    # Local smoke run without API server.
    output = evaluate_sample(split="val", index=0)
    print(output["pipeline_result"])
