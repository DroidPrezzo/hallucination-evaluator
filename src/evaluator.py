import logging
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

class LLMJudge:
    """
    Evaluates whether a generated summary contains hallucinations
    by fact-checking it against the source document using an LLM.
    """
    def __init__(self, judge_model_runner):
        self.runner = judge_model_runner
        self.logger = logging.getLogger(__name__)
        
    def evaluate(self, source_text: str, summary: str) -> bool:
        """
        Returns True if the summary contains hallucinations (unfaithful facts), 
        False if it is completely faithful to the source text.
        """
        system_prompt = "You are a strict, objective fact-checker. Your only purpose is to output 'YES' or 'NO' based on whether a summary hallucinated."
        
        user_prompt = f"""Task: Determine if the following SUMMARY contains any statements, claims, or facts that are NOT supported by the SOURCE DOCUMENT.
If the SUMMARY contains ANY information not present or logically implied by the SOURCE DOCUMENT, you must respond with exactly 'YES'. 
If all information in the SUMMARY is fully supported by the SOURCE DOCUMENT, respond with exactly 'NO'.
Do not provide any explanations or other words. Only output 'YES' or 'NO'.

SOURCE DOCUMENT:
{source_text}

SUMMARY:
{summary}

ANSWER:
"""
        response = self.runner.generate_response(
            prompt=user_prompt, 
            system_prompt=system_prompt, 
            max_new_tokens=10, 
            temperature=0.1
        )
        response_upper = response.upper().strip()
        
        if 'YES' in response_upper:
            return True
        elif 'NO' in response_upper:
            return False
        else:
            self.logger.warning(f"Judge returned ambiguous response: {response}. Assuming hallucination as a precaution.")
            return True


class HallucinationEvaluator:
    """
    Orchestrates the evaluation loop: probing a model across 
    different context lengths and tracking hallucination rates.
    """
    def __init__(self, context_builder, test_model_runner=None, judge=None):
        self.context_builder = context_builder
        self.model_runner = test_model_runner
        self.judge = judge
        self.logger = logging.getLogger(__name__)
        
    def generate_summaries(self, context_lengths: List[int], num_samples: int = 10) -> Dict[int, List[Tuple[str, str]]]:
        """
        Phase 1: Generates summaries for the provided target model across multiple context lengths.
        
        Args:
            context_lengths: A list of token limits (e.g., [1000, 2000, 4000]).
            num_samples: How many summaries to generate per context length.
            
        Returns:
            A dictionary mapping context length (int) to a list of tuples containing
            the original truncated text and the generated summary.
        """
        if not self.model_runner:
            raise ValueError("test_model_runner is required for generation phase")
            
        results = {}
        dataset_len = self.context_builder.get_document_length()
        actual_samples = min(num_samples, dataset_len)
        
        for length in context_lengths:
            self.logger.info(f"Generating summaries for context length: {length}")
            length_summaries = []
            
            for i in tqdm(range(actual_samples), desc=f"Length {length} Generation"):
                full_doc = self.context_builder.get_document(i)
                # Truncate the document strictly to the exact number of tested tokens
                context = self.context_builder.build_context(
                    text=full_doc, 
                    tokenizer=self.model_runner.tokenizer, 
                    target_length=length
                )
                summary = self.model_runner.generate_summary(context)
                length_summaries.append((context, summary))
                
            results[length] = length_summaries
            
        return results

    def evaluate_summaries(self, generated_summaries: Dict[int, List[Tuple[str, str]]]) -> Dict[int, float]:
        """
        Phase 2: Passes all generated summaries through the LLMJudge to score for hallucinations.
        
        Args:
            generated_summaries: The dictionary output from `generate_summaries`.
            
        Returns:
            A dictionary mapping context length (int) to the calculated hallucination rate (float).
        """
        if not self.judge:
            raise ValueError("judge is required for evaluation phase")
            
        results = {}
        for length, summaries in generated_summaries.items():
            self.logger.info(f"Evaluating context length: {length}")
            hallucinations = 0
            
            for context, summary in tqdm(summaries, desc=f"Length {length} Evaluation"):
                is_hallucination = self.judge.evaluate(context, summary)
                if is_hallucination:
                    self.logger.info(f"Hallucination DETECTED at length {length}:\n{summary[:200]}...")
                    hallucinations += 1
            
            # Calculate the percentage of samples that contained a hallucination at this length
            actual_samples = len(summaries)
            hallucination_rate = hallucinations / actual_samples if actual_samples > 0 else 0
            results[length] = hallucination_rate
            self.logger.info(f"Length {length} Hallucination Rate: {hallucination_rate:.2f}\n")
            
        return results
