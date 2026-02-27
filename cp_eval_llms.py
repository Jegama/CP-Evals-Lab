"""Unified evaluation / generation CLI.

Features:
1. Evaluate an existing JSONL dataset of messages (system,user,assistant) pairs.
2. Or generate answers from a questions file using various API providers, save a dataset JSONL, then evaluate.
3. Append aggregated rubric scores to a wide comparison CSV (criteria rows, model columns).
4. Append raw evaluation result lines to a JSONL results file (one JSON object per line) preserving previous runs.

Dataset JSONL Format (expected): each line is a JSON object with key "messages" whose value
is a list like: [ {role: system, content: ...}, {role: user, content: ...}, {role: assistant, content: ...} ].

Usage examples (Windows CMD):
  1) Evaluate existing dataset produced by model "google:gemma-3-7b" (judge defaults to gpt-5-mini):
      python cp_eval_llms.py --mode dataset --dataset data/arabic/ar_training_dataset_small_model.jsonl --answers-label google:gemma-3-7b

  2) Evaluate existing dataset overriding judge model:
      python cp_eval_llms.py --mode dataset --dataset data/arabic/ar_training_dataset_small_model.jsonl --answers-label google:gemma-3-7b --judge-model gpt-5

  3) Generate answers (OpenAI) for API evaluation with system prompt:
      python cp_eval_llms.py --mode generate-api_evals --provider openai --gen-model gpt-5-mini --answers-label gpt-5-mini-prompt --use-system-prompt

  4) Generate answers (Together) for fine-tuned model evaluation:
      python cp_eval_llms.py --mode generate-ft_evals --provider together --gen-model meta-llama/Meta-Llama-3-8B-Instruct --answers-label llama8

  5) Generate answers using Gemini without system prompt:
      python cp_eval_llms.py --mode generate-api_evals --provider google --gen-model gemini-2.5-flash --answers-label gemini-2.5-flash-vanilla

  6) Generate answers using Grok without system prompt:
      python cp_eval_llms.py --mode generate-api_evals --provider grok --gen-model grok-3-mini --answers-label grok-3-mini-vanilla

If the comparison CSV exists, a new model column is appended. If the model name already exists,
either pass --overwrite to replace it, or a numeric suffix will be added automatically.
"""

from __future__ import annotations
import argparse
import csv
import json
import re
import sys
import random
import math
import importlib
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Sequence
from parrot_ai.llm_evaluation import (
    EvaluationEngine,
    load_qa_pairs,
    load_eval_questions,
)
from parrot_ai.llm_evals import compute_weighted_final_score
from parrot_ai.evaluation_schemas import (
    SUBCRITERIA_FLAG_MAP,
    ALWAYS_ON_SUBCRITERIA,
)

CORE_SECTION_ORDER = [
    "Adherence",
    "Kindness_and_Gentleness",
    "Interfaith_Sensitivity",
]

CORE_SECTION_SUBCRITERIA = {
    "Adherence": [
        "Core",
        "Secondary",
        "Tertiary_Handling",
        "Biblical_Basis",
        "Consistency",
        "Overall",
    ],
    "Kindness_and_Gentleness": [
        "Core_Clarity_with_Kindness",
        "Pastoral_Sensitivity",
        "Secondary_Fairness",
        "Tertiary_Neutrality",
        "Tone",
        "Overall",
    ],
    "Interfaith_Sensitivity": [
        "Respect_and_Handling_Objections",
        "Objection_Acknowledgement",
        "Evangelism",
        "Gospel_Boldness",
        "Overall",
    ],
}

ARABIC_ACCURACY_SUBCRITERIA = [
    "Grammar_and_Syntax",
    "Theological_Nuance",
    "Contextual_Clarity",
    "Consistency_of_Terms",
    "Arabic_Purity",
    "Overall",
]

FINAL_OVERALL_ROW = ("", "Final_Overall")
WEIGHTED_SCORE_ROW = ("", "Weighted_Production_Score")

META_ROWS = [
    ("Meta", "System_Prompt_Label"),
    ("Meta", "Judge_Model"),
]


def build_rows_order(include_arabic: bool) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for section in CORE_SECTION_ORDER:
        for sub in CORE_SECTION_SUBCRITERIA[section]:
            rows.append((section, sub))
    if include_arabic:
        for sub in ARABIC_ACCURACY_SUBCRITERIA:
            rows.append(("Arabic_Accuracy", sub))
    rows.append(FINAL_OVERALL_ROW)
    if not include_arabic:
        rows.append(WEIGHTED_SCORE_ROW)
    for section, sub in META_ROWS:
        rows.append((section, sub))
    return rows


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def _load_system_prompt(engine: EvaluationEngine, use_system_prompt: bool) -> Optional[str]:
    """Load the language-specific system prompt if requested."""
    if not use_system_prompt:
        return None
    try:
        prompt_module = importlib.import_module(
            f"parrot_ai.prompts.{engine.language}"
        )
        return getattr(prompt_module, "MAIN_SYSTEM_PROMPT", None)
    except (ImportError, AttributeError):
        print(
            f"[warning] Could not load system prompt for language '{engine.language}', proceeding without system prompt"
        )
        return None


def generate_dataset(
    provider: str,
    questions_file: str,
    gen_model: str,
    engine: EvaluationEngine,
    output_dataset: str,
    use_system_prompt: bool = False,
    limit: int = 100,
    system_prompt_label: Optional[str] = None,
) -> str:
    # Load questions (limit=None means all if limit is 0)
    load_limit = None if limit == 0 else limit
    questions = load_eval_questions(questions_file, limit=load_limit)
    if not questions:
        raise SystemExit("No questions loaded for generation.")

    # Get system prompt if requested
    system_prompt = _load_system_prompt(engine, use_system_prompt)

    # Generate responses using the unified method
    responses = engine.generate_responses(
        questions,
        provider=provider,
        model=gen_model,
        system=system_prompt if use_system_prompt else None,
    )

    out_path = Path(output_dataset)
    mode_flag = "a" if out_path.exists() else "w"
    print(
        f"[generate] {'Appending to' if mode_flag == 'a' else 'Creating'} dataset: {out_path}"
    )
    with out_path.open(mode_flag, encoding="utf-8") as f:
        if use_system_prompt and system_prompt:
            f.write(
                json.dumps(
                    {"role": "system", "content": system_prompt}, ensure_ascii=False
                )
                + "\n"
            )

        for r in responses:
            if "error" in r:
                print(
                    f"[warning] Error generating response for question {r['index']}: {r['error']}"
                )
                continue

            msgs = [
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["answer"]},
            ]

            obj = {
                "messages": msgs,
                "gen_model": gen_model,
                "provider": r.get("provider"),
                "timestamp": dt.now().isoformat(),
                "use_system_prompt": use_system_prompt,
            }
            if system_prompt_label:
                obj["system_prompt_label"] = system_prompt_label
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return str(out_path)


def _build_flag_to_subcriteria_index() -> Dict[tuple, str]:
    """Build reverse map: (section, subcriteria) -> flag_name."""
    reverse: Dict[tuple, str] = {}
    for flag_name, pairs in SUBCRITERIA_FLAG_MAP.items():
        for pair in pairs:
            reverse[pair] = flag_name
    return reverse


_SUBCRITERIA_TO_FLAG = _build_flag_to_subcriteria_index()


def _is_applicable(
    section: str, key: str, question_tag: Optional[dict]
) -> bool:
    """Check if a (section, key) subcriteria is applicable for a given question tag.

    Returns True (include score) if:
    - No question tag provided (backward compat)
    - The subcriteria is always-on (Biblical_Basis, Consistency, Tone)
    - The subcriteria's controlling flag is True in the tag
    - The key is "Overall" (handled separately in recomputation)
    """
    if question_tag is None:
        return True
    if key == "Overall":
        return False  # Overalls are recomputed from applicable subcriteria
    pair = (section, key)
    if pair in ALWAYS_ON_SUBCRITERIA:
        return True
    flag_name = _SUBCRITERIA_TO_FLAG.get(pair)
    if flag_name is None:
        return True  # Unknown subcriteria -> include (fail open)
    return bool(question_tag.get(flag_name, True))


def aggregate_scores(
    results: List[dict],
    include_arabic_accuracy: bool,
    question_tags: Optional[Dict[str, dict]] = None,
) -> Dict[tuple, float]:
    agg: Dict[tuple, float] = {}
    counts: Dict[tuple, int] = {}
    # For section overall recomputation: track per-section subcriteria sums
    section_sub_agg: Dict[str, Dict[str, float]] = {}
    section_sub_counts: Dict[str, Dict[str, int]] = {}

    target_sections = ["Adherence", "Kindness_and_Gentleness", "Interfaith_Sensitivity"]
    if include_arabic_accuracy:
        target_sections.append("Arabic_Accuracy")

    use_tags = question_tags is not None and len(question_tags) > 0

    for item in results:
        ev = item.get("evaluation")
        if not ev:
            continue
        # Look up question tag if available
        question_text = item.get("question", "")
        q_tag = question_tags[question_text] if (use_tags and question_tags and question_text in question_tags) else None

        for section in target_sections:
            section_obj = ev.get(section, {})
            if not isinstance(section_obj, dict):
                continue
            for key, val in section_obj.items():
                if key in ("Penalty_Reason", "Heuristic_Arabic_Purity_Pct"):
                    continue
                if not isinstance(val, int):
                    continue

                if use_tags and section != "Arabic_Accuracy":
                    if not _is_applicable(section, key, q_tag):
                        continue

                agg[(section, key)] = agg.get((section, key), 0) + val
                counts[(section, key)] = counts.get((section, key), 0) + 1

                # Track for section overall recomputation
                if key != "Overall":
                    if section not in section_sub_agg:
                        section_sub_agg[section] = {}
                        section_sub_counts[section] = {}
                    section_sub_agg[section][key] = (
                        section_sub_agg[section].get(key, 0) + val
                    )
                    section_sub_counts[section][key] = (
                        section_sub_counts[section].get(key, 0) + 1
                    )

    means = {k: round(agg[k] / counts[k], 2) for k in agg if counts.get(k)}

    # Recompute section Overalls from applicable subcriteria means when tags are used
    if use_tags:
        for section in target_sections:
            if section == "Arabic_Accuracy":
                continue  # Arabic Accuracy keeps raw Overall
            sub_means = []
            for key, total in section_sub_agg.get(section, {}).items():
                cnt = section_sub_counts.get(section, {}).get(key, 0)
                if cnt > 0:
                    sub_means.append(total / cnt)
            if sub_means:
                means[(section, "Overall")] = round(
                    sum(sub_means) / len(sub_means), 2
                )

    # Compute Final_Overall as flat average of all section Overalls
    overall_values = [
        means[(s, "Overall")]
        for s in target_sections
        if (s, "Overall") in means
    ]
    if overall_values:
        means[("", "Final_Overall")] = round(
            sum(overall_values) / len(overall_values), 2
        )

    # Compute Weighted_Production_Score for English runs only
    # (Adherence 40%, Interfaith_Sensitivity 35%, Kindness_and_Gentleness 25%)
    if not include_arabic_accuracy:
        means[("", "Weighted_Production_Score")] = compute_weighted_final_score(means)

    return means


def ensure_csv_structure(
    csv_path: Path, rows_order: Sequence[tuple[str, Optional[str]]]
) -> tuple[list[list[str]], list[str]]:
    header_models: list[str] = []
    rows_by_key: Dict[tuple[str, str], list[str]] = {}
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header and len(header) > 2:
                header_models = header[2:]
            for r in reader:
                if not r or len(r) < 2:
                    continue
                rows_by_key[(r[0], r[1])] = r

    rows: list[list[str]] = []
    for section, sub in rows_order:
        sub_label = sub or "N/A"
        row = rows_by_key.pop((section, sub_label), None)
        if row is None:
            row = [section, sub_label] + ["" for _ in header_models]
        elif len(row) < 2 + len(header_models):
            row.extend([""] * (2 + len(header_models) - len(row)))
        rows.append(row)

    # Preserve any legacy rows that are not in the new ordering
    for row in rows_by_key.values():
        if len(row) < 2 + len(header_models):
            row.extend([""] * (2 + len(header_models) - len(row)))
        rows.append(row)

    # New file case: header_models still empty -> rows only have first two columns
    return rows, header_models


def update_comparison_csv(
    csv_path: Path,
    answers_label: str,
    aggregated: Dict[tuple, float],
    overwrite: bool,
    rows_order: Sequence[tuple[str, Optional[str]]],
    meta_values: Optional[Dict[tuple[str, str], str]] = None,
) -> None:
    rows, existing_header_models = ensure_csv_structure(csv_path, rows_order)
    header_models = existing_header_models.copy()
    final_model_name = answers_label

    if answers_label in header_models and not overwrite:
        suffix = 2
        while f"{answers_label}_{suffix}" in header_models:
            suffix += 1
        final_model_name = f"{answers_label}_{suffix}"
        print(
            f"[csv] Answers label exists; using '{final_model_name}' (use --overwrite to replace)."
        )

    overwrite_existing = overwrite and answers_label in header_models
    if overwrite_existing:
        col_index = header_models.index(answers_label) + 2
    else:
        header_models.append(final_model_name)
        target_width = 2 + len(header_models)
        for row in rows:
            if len(row) < target_width:
                row.extend([""] * (target_width - len(row)))
        col_index = target_width - 1

    meta_values = meta_values or {}
    for row in rows:
        criterion, subcrit = row[0], row[1]
        key = (criterion, "Overall") if subcrit == "N/A" else (criterion, subcrit)
        value = meta_values.get((criterion, subcrit))
        if value is None:
            value = aggregated.get(key, "")
        if len(row) <= col_index:
            row.extend([""] * (col_index + 1 - len(row)))
        row[col_index] = "" if value == "" else str(value)

    header = ["Criterion", "Sub-criterion"] + header_models
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[csv] Updated -> {csv_path}")


def append_results_jsonl(path: Path, results: List[dict], meta: dict) -> None:
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for r in results:
            r_out = {**r, **meta}
            f.write(json.dumps(r_out, ensure_ascii=False) + "\n")
    print(f"[results] Appended {len(results)} -> {path}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse CLI arguments.

    Key organizational conventions (auto inference when omitted):
        data/<language>/
            ├─ training_datasets/
            │    └─ evals/ (dataset & extended modes comparison + results JSONL)
            ├─ ft_evals/  (generate-ft_evals mode: generated datasets, comparison + results JSONL)
            └─ api_evals/ (generate-api_evals mode: generated datasets, comparison + results JSONL)
    """
    p = argparse.ArgumentParser(description="Generate and/or evaluate QA datasets.")
    p.add_argument(
        "--language",
        choices=["arabic", "english"],
        default="arabic",
        help="Language namespace: chooses data/<language>/ tree (default: arabic)",
    )
    p.add_argument(
        "--mode",
        choices=["dataset", "extended", "generate-ft_evals", "generate-api_evals"],
        default="dataset",
        help="dataset: evaluate existing training dataset using fixed 100-question file; extended: random sample of max(500,10%%) questions from dataset; generate-ft_evals: generate answers for fine-tuned models; generate-api_evals: generate answers for API models",
    )
    # dataset only required for dataset/extended modes; we validate later
    p.add_argument(
        "--dataset",
        help="(dataset/extended modes) Existing dataset JSONL to evaluate (training_datasets). Not used for generate-* modes.",
    )
    # questions file (not required for extended mode)
    p.add_argument(
        "--questions-file",
        help="Evaluation questions file (default auto for dataset mode: data/<language>/<prefix>eval_questions.txt). Not required for extended mode unless supplied.",
    )
    p.add_argument(
        "--provider",
        choices=["openai", "together", "hf", "google", "xai", "anthropic"],
        help="(generation modes only) API provider to use for generation (required for generate-* modes)",
    )
    p.add_argument(
        "--gen-model",
        help="(generation modes only) Provider model used to generate answers (required for generate-* modes)",
    )
    p.add_argument(
        "--answers-label",
        help="Human-friendly label for the answers column (defaults: gen-model or inferred from dataset)",
    )
    p.add_argument(
        "--judge-model",
        default="gpt-5-mini",
        help="Model used as evaluator (default: gpt-5-mini)",
    )
    p.add_argument(
        "--use-system-prompt",
        action="store_true",
        help="Use MAIN_SYSTEM_PROMPT from language prompts module for generation (mainly for API evals)",
    )
    p.add_argument(
        "--system-prompt-label",
        help="Optional label describing the system prompt or prompt version used for answer generation/evaluation",
    )
    p.add_argument(
        "--comparison-csv",
        help="Override comparison CSV filename (placed automatically in proper directory if relative)",
    )
    p.add_argument(
        "--results-jsonl",
        help="Override results JSONL filename (auto directory based on mode & language if relative)",
    )
    p.add_argument(
        "--output-dataset",
        help="(generation modes only) Output dataset filename (auto placed in ft_evals/api_evals if relative; default auto name)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite comparison CSV column if answers-label already present",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Silence progress ticks during evaluation",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of questions to evaluate (default: 100). Set to 0 for all.",
    )
    p.add_argument(
        "--question-tags",
        help="Path to question tags JSON file for selective scoring (auto-discovered for English if omitted)",
    )
    return p.parse_args(argv)


def infer_answers_label_from_dataset(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                gen_model = obj.get("gen_model")
                if isinstance(gen_model, str) and gen_model:
                    return gen_model
                # fallback: maybe system message includes model label? skip for now
                break
    except FileNotFoundError:
        return None
    return None


def infer_system_prompt_label_from_dataset(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    label = obj.get("system_prompt_label")
                    if isinstance(label, str) and label:
                        return label
    except FileNotFoundError:
        return None
    return None


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    # Directory scaffolding based on language & mode
    base_lang_dir = Path("data") / args.language
    training_dir = base_lang_dir / "training_datasets"
    training_evals_dir = training_dir / "evals"
    ft_evals_dir = base_lang_dir / "ft_evals"
    api_evals_dir = base_lang_dir / "api_evals"
    for d in (training_evals_dir, ft_evals_dir, api_evals_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Infer questions file if not provided (not required for extended mode)
    prefix = "ar_" if args.language == "arabic" else "en_"
    questions_file = args.questions_file or str(
        base_lang_dir / f"{prefix}eval_questions.txt"
    )
    if args.mode != "extended" and not Path(questions_file).exists():
        raise SystemExit(f"Questions file not found: {questions_file}")

    # Determine answers label (may be overridden later if inferred from dataset)
    answers_label = args.answers_label
    system_prompt_label = args.system_prompt_label

    # Comparison CSV path resolution
    if args.mode in ("dataset", "extended"):
        default_comp_csv = training_evals_dir / "dataset_eval_comparison.csv"
    elif args.mode == "generate-ft_evals":
        default_comp_csv = ft_evals_dir / "ft_evals_comparison.csv"
    else:  # generate-api_evals
        default_comp_csv = api_evals_dir / "api_evals_comparison.csv"

    if args.comparison_csv:
        supplied = Path(args.comparison_csv)
        if supplied.is_absolute():
            comparison_csv_path = supplied
        else:
            comparison_csv_path = default_comp_csv.parent / supplied.name
    else:
        comparison_csv_path = default_comp_csv

    # Build judge engine
    engine = EvaluationEngine(model=args.judge_model, language=args.language)
    print(f"[init] Judge model: {args.judge_model}")
    print(f"[init] Language: {args.language} | Mode: {args.mode}")

    generation_mode = args.mode.startswith("generate")

    # MODE: generation (ft_evals or api_evals)
    if generation_mode:
        if not args.provider:
            raise SystemExit("--provider required for generation modes")
        if not args.gen_model:
            raise SystemExit("--gen-model required for generation modes")
        if not answers_label:
            answers_label = args.gen_model

        # Output dataset naming & placement
        if args.mode == "generate-ft_evals":
            output_dir = ft_evals_dir
            auto_name = f"generated_ft_{args.provider}_{sanitize_filename(answers_label)}.jsonl"
        else:  # generate-api_evals
            output_dir = api_evals_dir
            auto_name = f"generated_api_{args.provider}_{sanitize_filename(answers_label)}.jsonl"

        output_dataset_name = args.output_dataset or auto_name
        output_dataset_path = Path(output_dataset_name)
        if not output_dataset_path.is_absolute():
            output_dataset_path = output_dir / output_dataset_path.name

        if output_dataset_path.exists():
            dataset_path = output_dataset_path
            print(
                f"[generate] Re-using existing generated dataset (no regeneration): {dataset_path}"
            )
        else:
            dataset_path = Path(
                generate_dataset(
                    args.provider,
                    questions_file,
                    args.gen_model,
                    engine,
                    str(output_dataset_path),
                    args.use_system_prompt,
                    limit=args.limit,
                    system_prompt_label=args.system_prompt_label,
                )
            )
            print(f"[generate] Dataset ready at {dataset_path}")
    else:  # MODE: dataset
        if not args.dataset:
            raise SystemExit("--dataset required for dataset mode")
        dataset_path = Path(args.dataset)
        if not dataset_path.is_absolute() and not dataset_path.exists():
            # Try resolve inside training_datasets automatically
            candidate = training_dir / dataset_path.name
            if candidate.exists():
                dataset_path = candidate
        if not dataset_path.exists():
            raise SystemExit(f"Dataset not found: {dataset_path}")
        if not answers_label:
            inferred = infer_answers_label_from_dataset(dataset_path)
            if inferred:
                answers_label = inferred
                print(f"[infer] Using inferred answers label: {answers_label}")
            else:
                raise SystemExit(
                    "Provide --answers-label (could not infer from dataset)."
                )

    if not system_prompt_label and dataset_path.exists():
        inferred_prompt = infer_system_prompt_label_from_dataset(dataset_path)
        if inferred_prompt:
            system_prompt_label = inferred_prompt
            print(f"[infer] Using inferred system prompt label: {system_prompt_label}")

    if not answers_label:
        answers_label = "answers"

    if args.mode == "extended":
        # Extended mode: random sample of questions directly from dataset
        raw_pairs = load_qa_pairs(
            str(dataset_path),
            question_list_path=None,
            limit=0,
        )
        if not raw_pairs:
            raise SystemExit("Dataset appears empty or unreadable for extended mode.")
        q_to_a: Dict[str, str] = {}
        for q, a in raw_pairs:
            if q not in q_to_a:
                q_to_a[q] = a
        total_q = len(q_to_a)
        sample_target = max(500, math.ceil(0.10 * total_q))
        if sample_target > total_q:
            sample_target = total_q
        sample_questions = random.sample(list(q_to_a.keys()), sample_target)
        pairs = [(q, q_to_a[q]) for q in sample_questions]
        print(
            f"[extended] Selected random sample of {len(pairs)} questions (total available: {total_q}; target rule: max(500,10%={math.ceil(0.10 * total_q)}))"
        )
    elif generation_mode:
        # Generation evaluation: use question list; penalize missing answers by inserting empty answer strings
        limit_val = None if args.limit == 0 else args.limit
        eval_questions = load_eval_questions(questions_file, limit=limit_val)
        if not eval_questions:
            raise SystemExit(
                "Questions file empty or unreadable for generation evaluation."
            )
        raw_pairs = load_qa_pairs(
            str(dataset_path),
            question_list_path=None,
            limit=0,
        )
        q_to_a: Dict[str, str] = {}
        for q, a in raw_pairs:
            if q not in q_to_a:
                q_to_a[q] = a
        missing_questions = [q for q in eval_questions if q not in q_to_a]
        if missing_questions:
            print(
                f"[retry] {len(missing_questions)} missing answers detected, retrying generation..."
            )
            retry_responses = engine.generate_responses(
                missing_questions,
                provider=args.provider,
                model=args.gen_model,
                system=_load_system_prompt(engine, args.use_system_prompt),
            )
            # Append retried answers to the JSONL and update q_to_a
            retried = 0
            with dataset_path.open("a", encoding="utf-8") as f:
                for r in retry_responses:
                    if "error" in r:
                        print(f"[retry] Still failed: {r['question'][:60]}... -> {r['error']}")
                        continue
                    q_to_a[r["question"]] = r["answer"]
                    retried += 1
                    msgs = [
                        {"role": "user", "content": r["question"]},
                        {"role": "assistant", "content": r["answer"]},
                    ]
                    obj = {
                        "messages": msgs,
                        "gen_model": args.gen_model,
                        "provider": r.get("provider"),
                        "timestamp": dt.now().isoformat(),
                        "use_system_prompt": args.use_system_prompt,
                    }
                    if args.system_prompt_label:
                        obj["system_prompt_label"] = args.system_prompt_label
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            print(f"[retry] Recovered {retried}/{len(missing_questions)} missing answers")

        missing = 0
        pairs: List[Tuple[str, str]] = []
        for q in eval_questions:
            if q in q_to_a:
                pairs.append((q, q_to_a[q]))
            else:
                missing += 1
                pairs.append(
                    (q, "")
                )  # Placeholder blank answer -> evaluation engine will penalize
        if missing:
            print(
                f"[generate-eval] WARNING: {missing} missing answers still present (will be penalized)."
            )
        print(
            f"[generate-eval] Prepared {len(pairs)} question/answer pairs for evaluation."
        )
    else:
        # Standard dataset mode: strict curated list (default 100)
        limit_val = None if args.limit == 0 else args.limit
        eval_questions = load_eval_questions(questions_file, limit=limit_val)
        eval_set: Set[str] = set(eval_questions)
        if args.limit == 100 and len(eval_questions) != 100:
            raise SystemExit(
                f"Evaluation questions file must contain 100 questions (got {len(eval_questions)})."
            )

        raw_pairs = load_qa_pairs(
            str(dataset_path),
            question_list_path=None,
            limit=0,
        )
        q_to_a: Dict[str, str] = {}
        for q, a in raw_pairs:
            if q in eval_set and q not in q_to_a:
                q_to_a[q] = a
        
        missing_count = 0
        pairs: List[Tuple[str, str]] = []
        for q in eval_questions:
            if q in q_to_a:
                pairs.append((q, q_to_a[q]))
            else:
                missing_count += 1
                pairs.append((q, ""))  # Empty answer triggers penalty
        
        if missing_count:
            print(
                f"[dataset] WARNING: {missing_count} missing answers inserted as empty strings (will be penalized)."
            )
        else:
            print(
                f"[load] Filtered {len(pairs)} evaluation pairs from dataset (strict 100-question set)."
            )

    # Results JSONL placement (mode dependent) -- resolve path BEFORE evaluation
    if args.mode in ("dataset", "extended"):
        default_results_dir = training_evals_dir
    elif args.mode == "generate-ft_evals":
        default_results_dir = ft_evals_dir
    else:  # generate-api_evals
        default_results_dir = api_evals_dir

    default_results_dir.mkdir(parents=True, exist_ok=True)
    if args.results_jsonl:
        supplied = Path(args.results_jsonl)
        if supplied.is_absolute():
            results_jsonl = supplied
        else:
            results_jsonl = default_results_dir / supplied.name
    else:
        filename = f"eval_results_{sanitize_filename(answers_label)}__judged_by_{sanitize_filename(args.judge_model)}.jsonl"
        results_jsonl = default_results_dir / filename

    # Load existing eval results for incremental evaluation
    existing_results: Dict[str, dict] = {}
    if results_jsonl.exists():
        with results_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = obj.get("question", "")
                ev = obj.get("evaluation")
                answer = obj.get("answer", "")
                # Keep result if it has a valid evaluation and the answer wasn't empty
                if q and ev and answer:
                    existing_results[q] = obj
        print(f"[eval] Found {len(existing_results)} existing evaluations in {results_jsonl}")

    # Determine which pairs still need evaluation
    pairs_to_eval = []
    for q, a in pairs:
        if q not in existing_results:
            pairs_to_eval.append((q, a))

    if pairs_to_eval:
        print(f"[eval] Evaluating {len(pairs_to_eval)} new/missing pairs (skipping {len(pairs) - len(pairs_to_eval)} already evaluated)...")
        new_results = engine.batch_evaluate(pairs_to_eval, limit=None, progress=not args.no_progress)
        print("[eval] Done.")
    else:
        new_results = []
        print("[eval] All pairs already evaluated, skipping evaluation.")

    # Merge: existing results + new results, rebuild full list in pairs order
    for item in new_results:
        q = item.get("question", "")
        if q:
            existing_results[q] = item

    results: List[dict] = []
    for q, a in pairs:
        if q in existing_results:
            results.append(existing_results[q])
        else:
            # Should not happen, but fail gracefully
            results.append({"question": q, "answer": a, "error": "no evaluation"})

    # Load question tags for selective scoring (English only)
    question_tags: Optional[Dict[str, dict]] = None
    if args.question_tags:
        tags_path = Path(args.question_tags)
    else:
        tags_path = base_lang_dir / f"{prefix}question_tags.json"
    if tags_path.exists():
        with tags_path.open("r", encoding="utf-8") as f:
            tags_data = json.load(f)
        question_tags = {}
        for tag in tags_data.get("tags", []):
            question_tags[tag["question"]] = tag
        print(f"[tags] Loaded {len(question_tags)} question tags from {tags_path}")
    elif args.question_tags:
        print(f"[warning] Question tags file not found: {tags_path}")

    # Aggregate
    include_arabic_accuracy = args.language == "arabic"
    aggregated = aggregate_scores(results, include_arabic_accuracy, question_tags=question_tags)
    print("[summary] Aggregated means:")
    for k in sorted(aggregated):
        print(f"  {k}: {aggregated[k]}")

    # Update comparison CSV
    rows_order = build_rows_order(include_arabic_accuracy)
    csv_meta_values: Dict[tuple[str, str], str] = {
        ("Meta", "Judge_Model"): args.judge_model,
        ("Meta", "Gen_Model"): args.gen_model if generation_mode else "N/A",
        ("Meta", "Provider"): args.provider if generation_mode else "N/A",
    }
    if system_prompt_label:
        csv_meta_values[("Meta", "System_Prompt_Label")] = system_prompt_label
    update_comparison_csv(
        comparison_csv_path,
        answers_label,
        aggregated,
        overwrite=args.overwrite,
        rows_order=rows_order,
        meta_values=csv_meta_values,
    )

    # Write full results JSONL (overwrite with clean merged set)
    meta = {
        "dataset": str(dataset_path),
        "answers_label": answers_label,
        "judge_model": args.judge_model,
        "gen_model": args.gen_model,
        "provider": args.provider if generation_mode else None,
        "use_system_prompt": args.use_system_prompt if generation_mode else None,
        "system_prompt_label": system_prompt_label,
        "questions_file": questions_file,
        "language": args.language,
        "mode": args.mode,
        "extended_sample_size": len(pairs) if args.mode == "extended" else None,
        "comparison_csv": str(comparison_csv_path),
        "timestamp": dt.now().isoformat(),
    }
    if new_results:
        append_results_jsonl(results_jsonl, new_results, meta)
    else:
        print(f"[results] No new results to append to {results_jsonl}")
    print("[done] Completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
