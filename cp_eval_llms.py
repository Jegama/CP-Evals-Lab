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
      python cp_eval_llms.py --mode generate-api_evals --provider gemini --gen-model gemini-2.5-flash --answers-label gemini-2.5-flash-vanilla

  6) Generate answers using Grok without system prompt:
      python cp_eval_llms.py --mode generate-api_evals --provider grok --gen-model grok-3-mini --answers-label grok-3-mini-vanilla

If the comparison CSV exists, a new model column is appended. If the model name already exists,
either pass --overwrite to replace it, or a numeric suffix will be added automatically.
"""
from __future__ import annotations
import argparse, csv, json, re, sys, random, math, importlib
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from parrot_ai.llm_evaluation import (
    EvaluationEngine,
    load_qa_pairs as base_load_qa_pairs,
    load_eval_questions,
)
BASE_CSV_ROWS = [
    ("Adherence", None),
    ("Kindness_and_Gentleness", None),
    ("Interfaith_Sensitivity", "Respect_and_Handling_Objections"),
    ("Interfaith_Sensitivity", "Objection_Acknowledgement"),
    ("Interfaith_Sensitivity", "Evangelism"),
    ("Interfaith_Sensitivity", "Gospel_Boldness"),
]
ARABIC_EXTRA_ROWS = [
    ("Arabic_Accuracy", "Grammar_and_Syntax"),
    ("Arabic_Accuracy", "Theological_Nuance"),
    ("Arabic_Accuracy", "Contextual_Clarity"),
    ("Arabic_Accuracy", "Consistency_of_Terms"),
    ("Arabic_Accuracy", "Arabic_Purity"),
]

def sanitize_filename(
    name: str
) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)

def load_dataset_pairs(jsonl_path: str) -> List[Tuple[str, str]]:
    """Load all pairs from dataset (no filtering here)."""
    return base_load_qa_pairs(jsonl_path, question_list_path=None, limit=0)

def generate_dataset(
    provider: str,
    questions_file: str,
    gen_model: str,
    engine: EvaluationEngine,
    output_dataset: str,
    use_system_prompt: bool = False
) -> str:
    # Always use full questions file
    questions = load_eval_questions(questions_file)
    if not questions:
        raise SystemExit("No questions loaded for generation.")
    
    # Get system prompt if requested
    system_prompt = None
    if use_system_prompt:
        try:
            prompt_module = importlib.import_module(f"parrot_ai.prompts.{engine.language}")
            system_prompt = getattr(prompt_module, "MAIN_SYSTEM_PROMPT", None)
        except (ImportError, AttributeError):
            print(f"[warning] Could not load system prompt for language '{engine.language}', proceeding without system prompt")
    
    # Generate responses using the unified method
    responses = engine.generate_responses(
        questions, 
        provider=provider, 
        model=gen_model,
        system=system_prompt
    )
    
    out_path = Path(output_dataset)
    mode_flag = "a" if out_path.exists() else "w"
    print(f"[generate] {'Appending to' if mode_flag=='a' else 'Creating'} dataset: {out_path}")
    with out_path.open(mode_flag, encoding="utf-8") as f:
        for r in responses:
            if "error" in r:
                print(f"[warning] Error generating response for question {r['index']}: {r['error']}")
                continue
            obj = {"messages": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["answer"]},
            ],
            "gen_model": gen_model,
            "provider": r.get("provider"),
            "timestamp": dt.now().isoformat(),
            "use_system_prompt": use_system_prompt,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return str(out_path)

def aggregate_scores(
    results: List[dict],
    include_arabic_accuracy: bool
) -> Dict[tuple, float]:
    agg: Dict[tuple, float] = {}
    counts: Dict[tuple, int] = {}
    target_sections = ["Adherence", "Kindness_and_Gentleness", "Interfaith_Sensitivity"]
    if include_arabic_accuracy:
        target_sections.append("Arabic_Accuracy")
    for item in results:
        ev = item.get("evaluation")
        if not ev:
            continue
        for section in target_sections:
            section_obj = ev.get(section, {})
            if not isinstance(section_obj, dict):
                continue
            for key, val in section_obj.items():
                if key in ("Penalty_Reason", "Heuristic_Arabic_Purity_Pct"):
                    continue
                if not isinstance(val, int):
                    continue
                agg[(section, key)] = agg.get((section, key), 0) + val
                counts[(section, key)] = counts.get((section, key), 0) + 1
    return {k: round(agg[k] / counts[k], 2) for k in agg if counts.get(k)}

def ensure_csv_structure(
    csv_path: Path,
    rows_order: list[tuple[str, Optional[str]]]
) -> list[list[str]]:
    if not csv_path.exists():
        rows: list[list[str]] = []
        for section, sub in rows_order:
            rows.append([section, sub or "N/A"])  # no score columns yet
        return rows
    rows: list[list[str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for r in reader: rows.append(r)
    return rows

def update_comparison_csv(
    csv_path: Path,
    answers_label: str,
    aggregated: Dict[tuple, float],
    overwrite: bool,
    rows_order: list[tuple[str, Optional[str]]]
) -> None:
    rows = ensure_csv_structure(csv_path, rows_order)
    existing_header_models: list[str] = []
    existing_header: list[str] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                existing_header = header
                existing_header_models = header[2:]
    final_model_name = answers_label
    if answers_label in existing_header_models and not overwrite:
        suffix = 2
        while f"{answers_label}_{suffix}" in existing_header_models: suffix += 1
        final_model_name = f"{answers_label}_{suffix}"
        print(f"[csv] Answers label exists; using '{final_model_name}' (use --overwrite to replace).")
    if existing_header_models and overwrite and answers_label in existing_header_models:
        col_index = existing_header_models.index(answers_label) + 2
        for row in rows:
            criterion, subcrit = row[0], row[1]
            key = (criterion, "Overall") if subcrit == "N/A" else (criterion, subcrit)
            val = aggregated.get(key, "")
            row[col_index] = "" if val == "" else str(val)
        header = existing_header
    else:
        for row in rows:
            criterion, subcrit = row[0], row[1]
            key = (criterion, "Overall") if subcrit == "N/A" else (criterion, subcrit)
            val = aggregated.get(key, "")
            row.append("" if val == "" else str(val))
        header = ["Criterion", "Sub-criterion"] + existing_header_models
        if not (overwrite and answers_label in existing_header_models): header.append(final_model_name)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)
    print(f"[csv] Updated -> {csv_path}")

def append_results_jsonl(
    path: Path,
    results: List[dict],
    meta: dict
) -> None:
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
    p.add_argument("--language", choices=["arabic", "english"], default="arabic",
                   help="Language namespace: chooses data/<language>/ tree (default: arabic)")
    p.add_argument("--mode", choices=["dataset", "extended", "generate-ft_evals", "generate-api_evals"], default="dataset",
                   help="dataset: evaluate existing training dataset using fixed 100-question file; extended: random sample of max(500,10%%) questions from dataset; generate-ft_evals: generate answers for fine-tuned models; generate-api_evals: generate answers for API models")
    # dataset only required for dataset/extended modes; we validate later
    p.add_argument("--dataset", help="(dataset/extended modes) Existing dataset JSONL to evaluate (training_datasets). Not used for generate-* modes.")
    # questions file (not required for extended mode)
    p.add_argument("--questions-file", help="Evaluation questions file (default auto for dataset mode: data/<language>/<prefix>eval_questions.txt). Not required for extended mode unless supplied.")
    p.add_argument("--provider", choices=["openai", "together", "hf", "gemini", "grok"], help="(generation modes only) API provider to use for generation (required for generate-* modes)")
    p.add_argument("--gen-model", help="(generation modes only) Provider model used to generate answers (required for generate-* modes)")
    p.add_argument("--answers-label", help="Human-friendly label for the answers column (defaults: gen-model or inferred from dataset)")
    p.add_argument("--judge-model", default="gpt-5-mini", help="Model used as evaluator (default: gpt-5-mini)")
    p.add_argument("--use-system-prompt", action="store_true", help="Use MAIN_SYSTEM_PROMPT from language prompts module for generation (mainly for API evals)")
    p.add_argument("--comparison-csv", help="Override comparison CSV filename (placed automatically in proper directory if relative)")
    p.add_argument("--results-jsonl", help="Override results JSONL filename (auto directory based on mode & language if relative)")
    p.add_argument("--output-dataset", help="(generation modes only) Output dataset filename (auto placed in ft_evals/api_evals if relative; default auto name)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite comparison CSV column if answers-label already present")
    p.add_argument("--no-progress", action="store_true", help="Silence progress ticks during evaluation")
    return p.parse_args(argv)

def infer_answers_label_from_dataset(path: Path) -> str | None:
    try:
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                gen_model = obj.get('gen_model')
                if isinstance(gen_model, str) and gen_model:
                    return gen_model
                # fallback: maybe system message includes model label? skip for now
                break
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
    questions_file = args.questions_file or str(base_lang_dir / f"{prefix}eval_questions.txt")
    if args.mode != "extended" and not Path(questions_file).exists():
        raise SystemExit(f"Questions file not found: {questions_file}")

    # Determine answers label (may be overridden later if inferred from dataset)
    answers_label = args.answers_label

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
            auto_name = f"generated_ft_{args.provider}_{sanitize_filename(args.gen_model)}.jsonl"
        else:  # generate-api_evals
            output_dir = api_evals_dir
            auto_name = f"generated_api_{args.provider}_{sanitize_filename(args.gen_model)}.jsonl"
            
        output_dataset_name = args.output_dataset or auto_name
        output_dataset_path = Path(output_dataset_name)
        if not output_dataset_path.is_absolute():
            output_dataset_path = output_dir / output_dataset_path.name
            
        if output_dataset_path.exists():
            dataset_path = output_dataset_path
            print(f"[generate] Re-using existing generated dataset (no regeneration): {dataset_path}")
        else:
            dataset_path = Path(
                generate_dataset(
                    args.provider,
                    questions_file,
                    args.gen_model,
                    engine,
                    str(output_dataset_path),
                    args.use_system_prompt,
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
                raise SystemExit("Provide --answers-label (could not infer from dataset).")

    if not answers_label:
        answers_label = 'answers'

    if args.mode == "extended":
        # Extended mode: random sample of questions directly from dataset
        raw_pairs = load_dataset_pairs(str(dataset_path))
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
        print(f"[extended] Selected random sample of {len(pairs)} questions (total available: {total_q}; target rule: max(500,10%={math.ceil(0.10*total_q)}))")
    elif generation_mode:
        # Generation evaluation: use question list; penalize missing answers by inserting empty answer strings
        eval_questions = load_eval_questions(questions_file, limit=100)
        if not eval_questions:
            raise SystemExit("Questions file empty or unreadable for generation evaluation.")
        raw_pairs = load_dataset_pairs(str(dataset_path))
        q_to_a: Dict[str, str] = {}
        for q, a in raw_pairs:
            if q not in q_to_a:
                q_to_a[q] = a
        missing = 0
        pairs: List[Tuple[str, str]] = []
        for q in eval_questions:
            if q in q_to_a:
                pairs.append((q, q_to_a[q]))
            else:
                missing += 1
                pairs.append((q, ""))  # Placeholder blank answer -> evaluation engine will penalize
        if missing:
            print(f"[generate-eval] WARNING: {missing} missing answers inserted as empty strings (will be penalized).")
        print(f"[generate-eval] Prepared {len(pairs)} question/answer pairs for evaluation.")
    else:
        # Standard dataset mode: strict 100-question curated list
        eval_questions = load_eval_questions(questions_file, limit=100)
        eval_set: Set[str] = set(eval_questions)
        if len(eval_questions) != 100:
            raise SystemExit(f"Evaluation questions file must contain 100 questions (got {len(eval_questions)}).")

        raw_pairs = load_dataset_pairs(str(dataset_path))
        q_to_a: Dict[str, str] = {}
        for q, a in raw_pairs:
            if q in eval_set and q not in q_to_a:
                q_to_a[q] = a
        missing = [q for q in eval_questions if q not in q_to_a]
        if missing:
            raise SystemExit(f"Dataset missing {len(missing)} required questions. First missing: {missing[:3]}")

        pairs = [(q, q_to_a[q]) for q in eval_questions]
        print(f"[load] Filtered {len(pairs)} evaluation pairs from dataset (strict 100-question set).")

    # Evaluate
    print('[eval] Running evaluation...')
    results = engine.batch_evaluate(pairs, limit=None, progress=not args.no_progress)
    print('[eval] Done.')

    # Aggregate
    include_arabic_accuracy = args.language == "arabic"
    aggregated = aggregate_scores(results, include_arabic_accuracy)
    print('[summary] Aggregated means:')
    for k in sorted(aggregated):
        print(f"  {k}: {aggregated[k]}")

    # Update comparison CSV
    # Build rows order dynamically for this run (if file already exists we retain its structure via ensure_csv_structure)
    rows_order = BASE_CSV_ROWS + (ARABIC_EXTRA_ROWS if include_arabic_accuracy else [])
    update_comparison_csv(comparison_csv_path, answers_label, aggregated, overwrite=args.overwrite, rows_order=rows_order)

    # Results JSONL placement (mode dependent)
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

    meta = {
        'dataset': str(dataset_path),
        'answers_label': answers_label,
        'judge_model': args.judge_model,
        'gen_model': args.gen_model,
        'provider': args.provider if generation_mode else None,
        'use_system_prompt': args.use_system_prompt if generation_mode else None,
        'questions_file': questions_file,
        'language': args.language,
        'mode': args.mode,
        'extended_sample_size': len(pairs) if args.mode == 'extended' else None,
        'comparison_csv': str(comparison_csv_path),
        'timestamp': dt.now().isoformat(),
    }
    append_results_jsonl(results_jsonl, results, meta)
    print('[done] Completed.')
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
