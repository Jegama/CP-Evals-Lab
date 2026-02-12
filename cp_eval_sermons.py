"""Sermon Evaluation CLI (two-step, audio-only, language-agnostic).

Usage examples:
  # Single-run evaluation
  python cp_eval_sermons.py --model gemini-3-flash-preview \
        --audio data/sermons/Ephesians_4_7-16.mp3 --out-dir data/sermons_evals \
        --label eph4_7-16 --preacher "Josh" --markdown

  # Multi-run evaluation with self-consistency (3 parallel runs)
  python cp_eval_sermons.py --model gemini-3-flash-preview --num-scoring-runs 3 \
        --audio data/sermons/Ephesians_4_7-16.mp3 --out-dir data/sermons_evals \
        --label eph4_7-16_multi --preacher "Josh" --markdown

Outputs:
  - step1 JSONL and step2 JSONL files under out-dir
  - aggregated summary CSV grouped by preacher under out-dir
  - optional Markdown report (with --markdown flag)
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from parrot_ai.evaluation_schemas import SermonExtractionStep1, SermonScoringStep2
from parrot_ai.sermon_evals import SermonEvaluationEngine, render_markdown


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Two-step sermon evaluation (audio-only)")
    p.add_argument(
        "--audio", required=True, help="Path to sermon audio file (mp3, wav, m4a)"
    )
    p.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model to use for evaluation (default: gemini-3-flash-preview)",
    )
    p.add_argument(
        "--out-dir",
        default="data/sermons_evals",
        help="Output directory for JSONL artifacts",
    )
    p.add_argument(
        "--label", required=True, help="Run label to tag outputs (e.g., eph2_11-13)"
    )
    p.add_argument(
        "--md-file",
        help="Optional explicit Markdown output path; defaults to <out-dir>/<label>.md",
    )
    p.add_argument(
        "--preacher",
        required=True,
        help="Name or identifier for the preacher being evaluated (stored in summary CSV)",
    )
    p.add_argument(
        "--markdown",
        action="store_true",
        help="Also emit a human-friendly Markdown report",
    )
    p.add_argument(
        "--num-scoring-runs",
        type=int,
        default=1,
        help="Number of parallel scoring runs for self-consistency; default 1 (single run)",
    )
    return p.parse_args(argv)


def append_aggregated_summary_csv(
    csv_path: Path,
    *,
    preacher: str,
    label: str,
    scoring: SermonScoringStep2,
    model: str,
    extraction: SermonExtractionStep1 | None = None,
    num_scoring_runs: int = 1,
) -> None:
    summary = getattr(scoring, "Aggregated_Summary", None)
    if summary is None:
        print("[warn] Aggregated summary missing; skipping CSV append.")
        return

    # Calculate duration in minutes if available
    duration_minutes = None
    duration_penalty = None
    if extraction and extraction.audio_duration:
        duration_minutes = round(extraction.audio_duration / 60.0, 2)
        duration_penalty = summary.duration_penalty

    fieldnames = [
        "timestamp",
        "label",
        "model",
        "preacher",
        "Textual_Fidelity",
        "Proposition_Clarity",
        "Introduction",
        "Application_Effectiveness",
        "Structure_Cohesion",
        "Illustrations",
        "Overall_Impact",
        "audio_duration_minutes",
        "duration_penalty",
        "num_scoring_runs",
    ]

    row = {
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "model": model,
        "preacher": preacher.replace("_", " "),
        "Textual_Fidelity": summary.Textual_Fidelity,
        "Proposition_Clarity": summary.Proposition_Clarity,
        "Introduction": summary.Introduction,
        "Application_Effectiveness": summary.Application_Effectiveness,
        "Structure_Cohesion": summary.Structure_Cohesion,
        "Illustrations": summary.Illustrations,
        "Overall_Impact": summary.Overall_Impact,
        "audio_duration_minutes": duration_minutes or "",
        "duration_penalty": duration_penalty or "",
        "num_scoring_runs": num_scoring_runs,
    }

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[write] Aggregated summary CSV -> {csv_path}")


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step1_path = out_dir / f"{args.label}_step1_extraction.json"
    step2_path = out_dir / f"{args.label}_step2_scoring.json"

    engine = SermonEvaluationEngine(model=args.model)
    print(f"[init] Model={args.model} | Mode=audio | Runs={args.num_scoring_runs}")

    # Step 1: Extract from audio (also uploads and caches file internally)
    step1 = engine.extract_structure_from_audio(args.audio)
    
    # Get audio file object for Step 2 scoring (multi-run mode needs it)
    _, audio_file_obj = engine.audio_manager.upload_or_get_gemini_file(
        args.audio, engine.provider
    )

    if step1.audio_duration:
        minutes = step1.audio_duration / 60.0
        print(f"[info] Audio duration: {minutes:.2f} minutes")
    else:
        print(
            "[warn] Audio duration could not be extracted (check file format or 'mutagen' install)"
        )

    # Save Step 1
    with step1_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(step1.model_dump(), ensure_ascii=False) + "\n")
    print(f"[write] Step 1 extraction -> {step1_path}")

    # Step 2 (multi-run if requested, else single run)
    if args.num_scoring_runs > 1:
        step2 = engine.score_from_extraction_multi_run(
            step1, audio_file_obj=audio_file_obj, num_runs=args.num_scoring_runs
        )
    else:
        step2 = engine.score_from_extraction(step1, audio_file_obj=audio_file_obj)
    with step2_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(step2.model_dump(), ensure_ascii=False) + "\n")
    print(f"[write] Step 2 scoring -> {step2_path}")

    # Aggregated summary CSV (per preacher)
    summary_csv_path = out_dir / "sermon_aggregated_summary.csv"
    append_aggregated_summary_csv(
        summary_csv_path,
        preacher=args.preacher,
        label=args.label,
        scoring=step2,
        model=args.model,
        extraction=step1,
        num_scoring_runs=args.num_scoring_runs,
    )

    # Optional Markdown report
    if args.markdown:
        md = render_markdown(
            step1,
            step2,
            label=args.label,
            model=args.model,
            num_scoring_runs=args.num_scoring_runs,
        )
        md_path = Path(args.md_file) if args.md_file else out_dir / f"{args.label}.md"
        md_path.write_text(md, encoding="utf-8")
        print(f"[write] Markdown report -> {md_path}")

    print("[done] Sermon evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
