import argparse
import logging
import gc
import torch
import sys

from src.context_builder import ContextBuilder
from src.model_runner import ModelRunner
from src.evaluator import LLMJudge, HallucinationEvaluator
from src.leaderboard import LeaderboardGenerator

def parse_args():
    """
    Parses command-line arguments to configure the Hallucination Evaluation pipeline.
    Allows users to easily swap test models, the judge model, and test varying context lengths.
    Includes memory optimization flags (--use-4bit, --use-8bit) for running on consumer GPUs.
    """
    parser = argparse.ArgumentParser(description="Hallucination Evaluator - Context Window Probing Leaderboard")
    
    # Target Models: Smaller models we want to test to see where their context breaks down
    parser.add_argument("--models", nargs="+", 
                        default=["Qwen/Qwen2.5-1.5B-Instruct", "microsoft/Phi-3.5-mini-instruct", "Qwen/Qwen2.5-3B-Instruct"],
                        help="List of Hugging Face model IDs to evaluate")
                        
    # Judge Model: The highly capable "source of truth" used to evaluate the
    # factual faithfulness of the Target Models' summaries. Needs to be larger and instruct-tuned.
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Hugging Face model ID to use as the fact-checking judge")

    # Revision pinning (CWE-494): pin Hub downloads to immutable commit SHAs (or tags/branches)
    # for reproducible, integrity-checked loads. Left unset, each load resolves to the repo's
    # default branch and emits a warning.
    parser.add_argument("--model-revisions", nargs="+", default=None,
                        help="Hub revisions for --models, in the same order (commit SHA/tag/branch). "
                             "Must match the number of --models if provided.")
    parser.add_argument("--judge-revision", type=str, default=None,
                        help="Hub revision (commit SHA/tag/branch) for the judge model")
    parser.add_argument("--dataset-revision", type=str, default=None,
                        help="Hub revision (commit SHA/tag/branch) for the evaluation dataset")
                        
    # The varying lengths of text (in tokens) we will feed the models. 
    # As this increases, hallucination rates typically spike.
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[1000, 2000, 4000, 8000],
                        help="Context lengths to evaluate")
                        
    # Number of documents to grab from the dataset per context length
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of document samples to evaluate per context length")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save the leaderboard")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Huggingface cache dir for weights downloading")
    
    # Performance / VRAM tuning options
    parser.add_argument("--use-4bit", action="store_true", 
                        help="Load models in 4-bit quantization to save VRAM (requires bitsandbytes)")
    parser.add_argument("--use-8bit", action="store_true", 
                        help="Load models in 8-bit quantization to save VRAM")
    parser.add_argument("--judge-4bit", action="store_true", 
                        help="Load the Judge model in 4-bit even if test models aren't")

    return parser.parse_args()

def setup_logging():
    """Configures fundamental console logging for tracking evaluation progress."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

def release_runner(runner):
    """Frees a ModelRunner's weights and reclaims GPU memory before loading the next model."""
    del runner.model
    del runner.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    """
    Main entry point for the pipeline. Executes the following phases:
    1. Initializes dataset builders and loads government reports.
    2. Phase 1 (Generation): Iterates through each targeted model, generating summaries at
       incrementing context lengths until finished. Models are loaded/unloaded sequentially 
       to prevent CUDA Out-of-Memory (OOM) errors.
    3. Phase 2 (Evaluation): Loads the Judge LLM, scoring every generated summary for unfaithful 
       information (hallucinations) against the original text.
    4. Phase 3 (Reporting): Aggregates hallucination rates per-length, per-model into a Markdown Leaderboard.
    """
    setup_logging()
    args = parse_args()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Hallucination Evaluation Pipeline")
    logger.info(f"Target Models: {args.models}")
    logger.info(f"Judge Model: {args.judge_model}")
    logger.info(f"Context Lengths: {args.context_lengths}")
    logger.info(f"Samples per length: {args.samples}")
    
    # Validate that per-model revisions, if supplied, line up with the model list
    if args.model_revisions is not None and len(args.model_revisions) != len(args.models):
        logger.error(
            f"--model-revisions has {len(args.model_revisions)} entries but --models has "
            f"{len(args.models)}; they must match one-to-one."
        )
        sys.exit(1)

    # 1. Setup Context Builder (Downloads/caches dataset)
    logger.info("Initializing Context Builder (Dataset)")
    context_builder = ContextBuilder(dataset_name="tau/scrolls", subset="gov_report", split="validation", revision=args.dataset_revision)
    
    # 2. Setup Leaderboard generator
    leaderboard_gen = LeaderboardGenerator(output_dir=args.output_dir)
    
    # Stores generated texts: { "model_name": { context_length: [ (context_text, generated_summary) ] } }
    all_generated_summaries = {}
    
    # Phase 1: Iteratively generate summaries for each target model
    for idx, model_name in enumerate(args.models):
        logger.info("\n====================================")
        logger.info(f"PHASE 1: Generating Summaries - Target Model: {model_name}")

        model_revision = args.model_revisions[idx] if args.model_revisions is not None else None

        # Load test model
        try:
            test_runner = ModelRunner(
                model_name=model_name,
                cache_dir=args.cache_dir,
                load_in_4bit=args.use_4bit,
                load_in_8bit=args.use_8bit,
                revision=model_revision
            )
        except Exception as e:
            logger.error(f"Failed to load target model {model_name}: {e}. Skipping.")
            continue
            
        evaluator = HallucinationEvaluator(
            context_builder=context_builder,
            test_model_runner=test_runner
        )
        
        # Run evaluation map (only generation)
        summaries = evaluator.generate_summaries(
            context_lengths=args.context_lengths,
            num_samples=args.samples
        )
        all_generated_summaries[model_name] = summaries
        
        # Clear VRAM for the next model to avoid OOM
        logger.info(f"Releasing resources for {model_name} before proceeding.")
        release_runner(test_runner)
        del test_runner
        del evaluator

    if not all_generated_summaries:
         logger.error("No valid models could generate summaries, skipping evaluation phase.")
         sys.exit(1)
            
    # Phase 2: Load Judge Model and Evaluate
    logger.info("\n====================================")
    logger.info("PHASE 2: Evaluating Summaries")
    logger.info("Initializing Judge Model")
    judge_4bit = args.use_4bit or args.judge_4bit
    try:
        judge_runner = ModelRunner(
            model_name=args.judge_model,
            cache_dir=args.cache_dir,
            load_in_4bit=judge_4bit,
            load_in_8bit=args.use_8bit and not judge_4bit,
            revision=args.judge_revision
        )
        judge = LLMJudge(judge_model_runner=judge_runner)
    except Exception as e:
        logger.error(f"Failed to load judge model {args.judge_model}: {e}. Aborting.")
        sys.exit(1)
        
    all_results = {}
    for model_name, summaries in all_generated_summaries.items():
        logger.info(f"\nEvaluating Results for Target Model: {model_name}")
        evaluator = HallucinationEvaluator(
            context_builder=context_builder,
            judge=judge
        )
        model_results = evaluator.evaluate_summaries(summaries)
        all_results[model_name] = model_results
        
    logger.info("Releasing resources for Judge Model before proceeding.")
    del judge_runner.model
    del judge_runner.tokenizer
    del judge_runner
    del evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
            
    # Phase 3: Generate final leaderboard
    logger.info("\n====================================")
    if not all_results:
        logger.error("No valid models were evaluated, skipping leaderboard generation.")
        sys.exit(1)
        
    logger.info("Generating Final Leaderboard")
    leaderboard_gen.generate_leaderboard(all_results)
    logger.info(f"Pipeline Complete! Check {args.output_dir} for results.")

if __name__ == "__main__":
    main()
