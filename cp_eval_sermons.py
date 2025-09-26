"""Sermon Evaluation CLI (two-step, language-agnostic).

Usage examples:
  # Step 1 + Step 2 from text transcript using OpenAI evaluator model
  python cp_eval_sermons.py --mode text --provider openai --model gpt-5-mini \
    --transcript resources/Example_of_other_sermon_evals/eval_colossians_3.18-21.md \
    --out-dir data/sermons_evals --label col318-21

  # Step 1 from audio and Step 2 using Gemini
  python cp_eval_sermons.py --mode audio --provider gemini --model gemini-2.5-flash \
    --audio data/sermons/Ephesians_4_7-16.mp3 --out-dir data/sermons_evals --label eph4_7-16

Outputs:
  - step1 JSONL and step2 JSONL files under out-dir
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from parrot_ai.sermon_evaluation import SermonEvaluationEngine


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Two-step sermon evaluation")
    p.add_argument("--mode", choices=["text", "audio"], default="text", help="Input mode: text transcript or audio (Gemini only)")
    p.add_argument("--provider", choices=["openai", "gemini"], default="openai", help="Provider for evaluation prompts")
    p.add_argument("--model", default="gpt-5-mini", help="Model to use for prompts (e.g., gpt-5-mini, gemini-2.5-flash)")
    p.add_argument("--transcript", help="Path to sermon transcript (text or markdown) for --mode text")
    p.add_argument("--audio", help="Path to sermon audio file for --mode audio (Gemini only)")
    p.add_argument("--out-dir", default="data/sermons_evals", help="Output directory for JSONL artifacts")
    p.add_argument("--label", required=True, help="Run label to tag outputs (e.g., eph2_11-13)")
    return p.parse_args(argv)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step1_path = out_dir / f"{args.label}_step1_extraction.json"
    step2_path = out_dir / f"{args.label}_step2_scoring.json"

    engine = SermonEvaluationEngine(provider=args.provider, model=args.model)
    print(f"[init] Provider={args.provider} | Model={args.model} | Mode={args.mode}")

    # Step 1
    if args.mode == "text":
        if not args.transcript:
            raise SystemExit("--transcript is required for --mode text")
        transcript = read_text_file(args.transcript)
        step1 = engine.extract_structure_from_text(transcript)
    else:
        if args.provider != "gemini":
            raise SystemExit("Audio mode requires --provider gemini")
        if not args.audio:
            raise SystemExit("--audio is required for --mode audio")
        step1 = engine.extract_structure_from_audio(args.audio)

    # Save Step 1
    with step1_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(step1.model_dump(), ensure_ascii=False) + "\n")
    print(f"[write] Step 1 extraction -> {step1_path}")

    # Step 2
    step2 = engine.score_from_extraction(step1)
    with step2_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(step2.model_dump(), ensure_ascii=False) + "\n")
    print(f"[write] Step 2 scoring -> {step2_path}")

    print("[done] Sermon evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
