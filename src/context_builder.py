import logging
from typing import Dict, Any, Optional

from datasets import load_dataset
from transformers import PreTrainedTokenizer


class ContextBuilder:
    """
    Responsible for loading source documents and truncating them to specific 
    token lengths to build context windows for model probing.
    """
    def __init__(self, dataset_name: str = "tau/scrolls", subset: str = "gov_report", split: str = "validation"):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Loading dataset {dataset_name} ({subset}) split: {split}")
        try:
            self.dataset = load_dataset(dataset_name, subset, split=split)
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

    def get_document(self, index: int) -> str:
        """
        Retrieves the full text of a document at the specified index.
        """
        if index < 0 or index >= len(self.dataset):
            raise IndexError("Dataset index out of range")
        
        # 'input' is the standard column for text in the scrolls dataset
        # fallback to 'text' if 'input' is not found
        if 'input' in self.dataset.column_names:
            return self.dataset[index]['input']
        elif 'text' in self.dataset.column_names:
            return self.dataset[index]['text']
        else:
            raise KeyError(f"Could not find a text column. Available columns: {self.dataset.column_names}")

    def get_document_length(self) -> int:
        """
        Returns the total number of documents in the dataset.
        """
        return len(self.dataset)

    def build_context(self, text: str, tokenizer: PreTrainedTokenizer, target_length: int) -> str:
        """
        Truncates a document so that it tokenizes to exactly `target_length` tokens.
        
        Args:
            text: The full document text.
            tokenizer: The HF tokenizer corresponding to the model being evaluated.
            target_length: The maximum number of tokens for the context window.
            
        Returns:
            The truncated text as a string.
        """
        # Tokenize without adding special tokens (BOS, EOS, etc) as we just want to truncate the content
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= target_length:
            self.logger.debug(f"Document token length ({len(tokens)}) is <= target length ({target_length}). Returning full text.")
            return text
            
        truncated_tokens = tokens[:target_length]
        
        # Decode back to string
        # skip_special_tokens=True guarantees we don't accidentally leak unwanted special tokens into the string
        truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return truncated_text
