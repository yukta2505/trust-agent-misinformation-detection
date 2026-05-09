"""
Quick test script for TRUST-AGENT pipeline.
Run from inside the trust-agent-misinformation-detection folder:
    python test_pipeline.py
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

from orchestrator import TrustAgentOrchestrator
from backend.config import Config


def main() -> None:
    image_path = "img1.jpg"   # put any test image in the same folder

    test_cases = [
        {
            "name": "TRUTHFUL CLAIM",
            "claim": "Protesters gather outside the Washington State Capitol building.",
        },
        {
            "name": "MISLEADING CLAIM (OOC)",
            "claim": "These protesters are rallying outside the Indian Parliament in New Delhi, 2024.",
        },
    ]

    config = Config()

    if not config.openai_api_key:
        print("ERROR: Set OPENAI_API_KEY in your .env file.")
        sys.exit(1)

    orchestrator = TrustAgentOrchestrator(config)

    for tc in test_cases:
        print("\n" + "=" * 70)
        print(f"TEST: {tc['name']}")
        print(f"CLAIM: {tc['claim']}")
        print("=" * 70)

        result = orchestrator.run(image_path=image_path, claim=tc["claim"])

        print(f"\n  VERDICT          : {result.verdict}")
        print(f"  CONFIDENCE       : {result.confidence_percent}%")
        print(f"  FINAL SCORE      : {result.final_score:.3f}")
        print(f"  CAPTION          : {result.caption}")
        print(f"\n  EXPLANATION:\n  {result.explanation}")

        if result.flags:
            print(f"\n  RED FLAGS:")
            for flag in result.flags:
                print(f"    * {flag}")

        print(f"\n  KEY EVIDENCE:")
        for ev in result.key_evidence_for_verdict:
            print(f"    - {ev}")

        print(f"\n  AGENT SCORES:")
        print(f"    Entity      : {result.entity_score:.3f}")
        print(f"    Temporal    : {result.temporal_score:.3f}")
        print(f"    Credibility : {result.credibility_score:.3f}")
        print(f"\n  Processing time : {result.processing_time_sec:.1f}s")


if __name__ == "__main__":
    main()
