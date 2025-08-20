#!/usr/bin/env python3
"""
Training Dataset Creation Script

This script creates a training dataset by processing questions and generating
enhanced answers using the ParrotAI chain. It includes resume functionality,
better error handling, and progress tracking.

The script now supports both local models and HuggingFace API:
- Local: python create_dataset_script.py --model "microsoft/DialoGPT-medium"
- API: python create_dataset_script.py --use-api --model "google/gemma-3-27b-it"

For API usage, make sure to set HF_TOKEN in your environment variables or .env file.

Features:
- Automatic retry logic for API server errors (502, 503, 504)
- Unicode-safe logging for Windows console compatibility
- Resume functionality to continue from previous progress
- Configurable retry attempts and batch processing
"""

import json
import os
import sys
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

from parrot_ai import ParrotAI, ParrotAIHF, parrot_chain
import parrot_ai.prompts as parrot_prompts


def setup_logging(log_level='INFO'):
    """Set up logging configuration with UTF-8 encoding for Windows."""
    # Configure logging with UTF-8 encoding to handle Arabic text
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler('dataset_creation.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Create console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_and_format_gotquestions(file_path, lang="ar"):
    """
    Load a gotquestions JSON file (Arabic or English) and format it into conversation pairs.

    Args:
        file_path (str): Path to the JSON file
        lang (str): 'ar' for Arabic structure, 'en' for English structure

    Returns:
        list: List of conversation pairs (each pair is a list of dicts with 'role' and 'content')
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    conversations = []

    if lang == "ar":
        # Arabic structure
        # top-level list of categories each with 'articles' (list of article dicts)
        for category in data:
            articles = category.get('articles', [])
            for article in articles:
                question = article.get('name', '')
                answer = article.get('answer', '')
                answer = answer.strip()
                conversation_pair = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
                conversations.append(conversation_pair)
    elif lang == "en":
        # English structure:
        # category -> themes (list) -> theme.articles (dict) -> key -> list[question dict]
        for category in data:
            themes = category.get('themes', [])
            for theme in themes:
                articles = theme.get('articles', [])  # actually a dict in provided structure
                if isinstance(articles, dict):
                    iterable = articles.items()
                else:
                    # fallback: if it's a list like Arabic just treat similarly
                    iterable = [(None, articles)]
                for _, questions in iterable:
                    for q in questions:
                        question = q.get('name', '')
                        answer = q.get('answer', '')
                        # Clean specific English marker
                        answer = answer.replace("\nAnswer\n", "").strip()
                        conversation_pair = [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer}
                        ]
                        conversations.append(conversation_pair)
    else:
        raise ValueError(f"Unsupported lang value: {lang}. Use 'ar' or 'en'.")

    return conversations

def load_and_format_qa_messages_jsonl(file_path):
    """
    Load the ar_qa_catechism.jsonl file and format it into conversation pairs.

    Args:
        file_path (str): Path to the JSONL file

    Returns:
        list: List of conversation pairs (each pair is a list of dicts with 'role' and 'content')
    """
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            msgs = data.get("messages", [])
            # Only keep user/assistant pairs (ignore system if present)
            pair = []
            for msg in msgs:
                if msg["role"] in ("user", "assistant"):
                    pair.append({"role": msg["role"], "content": msg["content"]})
            if len(pair) == 2:
                conversations.append(pair)
    return conversations


def load_combined_data(gotquestions_path, qa_messages_path, logger):
    """Load and combine both datasets."""
    logger.info("Loading gotquestions dataset...")
    formatted_data = load_and_format_gotquestions(gotquestions_path)
    logger.info(f"Loaded {len(formatted_data)} entries from gotquestions")
    
    logger.info("Loading qa_messages dataset...")
    formatted_data_jsonl = load_and_format_qa_messages_jsonl(qa_messages_path)
    logger.info(f"Loaded {len(formatted_data_jsonl)} entries from qa_messages")
    
    # Combine both datasets
    combined_data = formatted_data + formatted_data_jsonl
    logger.info(f"Total combined entries: {len(combined_data)}")
    
    return combined_data


def count_existing_entries(output_file):
    """Count existing entries in the output file."""
    if not os.path.exists(output_file):
        return 0
    
    with open(output_file, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def create_training_dataset(
    combined_data,
    parrot_instance,
    output_file,
    start_index=0,
    batch_save_interval=10,
    max_retries=3,
    logger=None
):
    """
    Create the training dataset with enhanced error handling and progress tracking.
    
    Args:
        combined_data: List of conversation pairs
        parrot_instance: Initialized ParrotAI or ParrotAIHF instance
        output_file: Path to output JSONL file
        start_index: Index to start processing from (for resume functionality)
        batch_save_interval: How often to flush writes to disk
        max_retries: Maximum number of retry attempts for API errors
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    total_data = len(combined_data)
    logger.info(f"Processing {total_data - start_index} remaining entries...")
    
    processed_count = 0
    error_count = 0
    
    # Open file in append mode for incremental saving
    with open(output_file, 'a', encoding='utf-8') as f:
        for i in tqdm(range(start_index, total_data), desc="Generating training data"):
            try:
                data = combined_data[i]
                
                # Generate response using parrot_chain with retry logic
                def generate_response():
                    return parrot_chain(data, parrot_instance)
                
                response = retry_with_backoff(generate_response, max_retries=max_retries, logger=logger)
                
                # Create training example in the format expected for fine-tuning
                training_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": parrot_prompts.MAIN_SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": data[0]["content"]  # User question
                        },
                        {
                            "role": "assistant",
                            "content": response["final_answer"]  # Final answer from chain
                        }
                    ]
                }
                
                # Write the training example as a JSON line
                f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
                
                processed_count += 1
                
                # Periodic flush to ensure data is saved
                if processed_count % batch_save_interval == 0:
                    f.flush()
                    # logger.info(f"Processed {processed_count} entries, saved batch to disk")
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping gracefully...")
                break
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing item {i}: {e}")
                
                # Log the problematic question for debugging with safe Unicode handling
                try:
                    question_preview = data[0]['content'][:100] if data and len(data) > 0 else "Unknown"
                    # Safely handle Unicode characters in logging
                    safe_preview = question_preview.encode('ascii', errors='replace').decode('ascii')
                    logger.error(f"Question preview (ASCII-safe): {safe_preview}...")
                except Exception as log_error:
                    logger.error(f"Could not extract question preview: {log_error}")
                
                # Continue with the next item instead of stopping
                continue
        
        # Final flush
        f.flush()
    
    logger.info(f"Dataset creation completed!")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Errors encountered: {error_count}")
    logger.info(f"Output saved to: {output_file}")
    
    # Count final entries
    final_count = count_existing_entries(output_file)
    logger.info(f"Total training examples in file: {final_count}")
    
    return processed_count, error_count


def retry_with_backoff(func, max_retries=3, initial_delay=1, backoff_factor=2, logger=None):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Factor to multiply delay by after each retry
        logger: Logger instance
        
    Returns:
        Result of the function call
        
    Raises:
        Exception: The last exception encountered if all retries fail
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    last_exception = None
    delay = initial_delay
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            
            # Check if it's a server error that we should retry
            error_msg = str(e).lower()
            if any(error_code in error_msg for error_code in ['502', '503', '504', 'bad gateway', 'service unavailable', 'gateway timeout']):
                if attempt < max_retries:
                    logger.warning(f"API error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= backoff_factor
                    continue
                else:
                    logger.error(f"API error after {max_retries + 1} attempts: {e}")
            else:
                # Non-retryable error, raise immediately
                raise e
    
    # If we've exhausted all retries, raise the last exception
    if last_exception is not None:
        raise last_exception
    else:
        raise RuntimeError("Function failed with no exception recorded")


def main():
    """Main function to orchestrate the dataset creation process."""
    parser = argparse.ArgumentParser(description="Create training dataset using ParrotAI")
    parser.add_argument("--model", default="google/gemma-3-12b-it", help="Model name to load")
    parser.add_argument("--use-api", action="store_true", help="Use HuggingFace API instead of local model")
    parser.add_argument("--api-provider", default="nebius", help="API provider for HuggingFace (default: nebius)")
    parser.add_argument("--output", default="data/temp_training_dataset.jsonl", help="Output file path")
    parser.add_argument("--gotquestions", default="data/arabic/ar_gotquestions.json", help="GotQuestions JSON file path")
    parser.add_argument("--qa-messages", default="data/arabic/ar_qa_catechism.jsonl", help="QA Messages JSONL file path")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch save interval")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum API retry attempts for server errors")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--resume", action="store_true", help="Resume from existing progress")
    
    args = parser.parse_args()
      # Setup logging
    logger = setup_logging(args.log_level)
    logger.info(f"Starting dataset creation at {datetime.now()}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Validate environment for API usage
    if args.use_api:        
        if not os.environ.get("HF_TOKEN"):
            logger.error("HF_TOKEN not found in environment variables")
            logger.error("Please set HF_TOKEN in your .env file or environment variables")
            logger.error("Get your token from: https://huggingface.co/settings/tokens")
            sys.exit(1)
    
    try:
        # Load combined data
        combined_data = load_combined_data(args.gotquestions, args.qa_messages, logger)
        
        # Initialize ParrotAI or ParrotAIHF based on arguments
        if args.use_api:
            logger.info("Using HuggingFace API")
            logger.info(f"API Provider: {args.api_provider}")
            logger.info(f"Model: {args.model}")
            
            try:
                parrot = ParrotAIHF(provider=args.api_provider)
                parrot.set_model(args.model)
                logger.info("HuggingFace API client initialized successfully")
                logger.info(parrot.get_model_info())
            except ValueError as e:
                logger.error(f"Failed to initialize HuggingFace API: {e}")
                logger.error("Make sure HF_TOKEN is set in environment variables or .env file")
                sys.exit(1)
        else:
            logger.info(f"Loading local model: {args.model}")
            parrot = ParrotAI()
            parrot.load_model(args.model)
            
            # Check if model is loaded
            if parrot.is_loaded():
                logger.info("Model loaded successfully:")
                logger.info(parrot.get_model_info())
            else:
                raise RuntimeError("Failed to load model")
        
        # Determine starting index
        start_index = 0
        if args.resume:
            start_index = count_existing_entries(args.output)
            logger.info(f"Resuming from index {start_index}")
        elif os.path.exists(args.output):
            logger.warning(f"Output file {args.output} exists. Use --resume to continue or delete it to start fresh.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                logger.info("Exiting...")
                return
        
        # Create output directory if it doesn't exist
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
          # Create training dataset
        processed, errors = create_training_dataset(
            combined_data=combined_data,
            parrot_instance=parrot,
            output_file=args.output,
            start_index=start_index,
            batch_save_interval=args.batch_size,
            max_retries=args.max_retries,
            logger=logger
        )
        
        logger.info(f"Dataset creation finished at {datetime.now()}")
        logger.info(f"Final statistics - Processed: {processed}, Errors: {errors}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
