import logging
from typing import Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, BitsAndBytesConfig

class ModelRunner:
    """
    Wrapper around Hugging Face `transformers` to load open-weight models locally.
    Handles device mapping, memory quantization (4-bit/8-bit), and generating text 
    by properly applying model-specific chat templates.
    """
    def __init__(self, model_name: str, cache_dir: Optional[str] = None, use_bfloat16: bool = True, load_in_4bit: bool = False, load_in_8bit: bool = False):
        """
        Initializes a Hugging Face model and its tokenizer.
        Automatically maps the model to available GPUs and calculates the most optimal
        torch data type depending on hardware support.
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Determine optimal dtype for GPU evaluation
        dtype = torch.bfloat16 if use_bfloat16 and torch.cuda.is_bf16_supported() else torch.float16
        
        self.logger.info(f"Loading tokenizer for {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
        self.logger.info(f"Loading model {model_name} (4bit={load_in_4bit}, 8bit={load_in_8bit}) in {dtype} with device_map='auto'")
        
        kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "cache_dir": cache_dir,
            "trust_remote_code": False
        }
        
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
            
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.logger.info(f"Successfully loaded {model_name}")

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, max_new_tokens: int = 512, temperature: float = 0.6) -> str:
        """
        Generates a generic response using the provided prompt and optional system prompt.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # We attempt to use chat template, fallback back to standard prompts if not supported
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt", 
                add_generation_prompt=True,
                return_dict=True
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
        except (ValueError, AttributeError, KeyError) as e:
            self.logger.warning(f"Failed to use apply_chat_template ({e}). Falling back to raw prompt string.")
            fallback_prompt = ""
            if system_prompt:
                fallback_prompt += f"System: {system_prompt}\n"
            fallback_prompt += f"User: {prompt}\n\nAssistant:\n"
            inputs = self.tokenizer(fallback_prompt, return_tensors="pt").to(self.model.device)
            input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
        # Slice off the input prompt to get only the output text
        prompt_length = input_ids.shape[1]
        response = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        return response.strip()

    def generate_summary(self, context: str, max_new_tokens: int = 512, temperature: float = 0.6) -> str:
        """
        Generates a summary of the provided text using the loaded model.
        """
        system_prompt = "You are a helpful and accurate assistant that summarizes long documents. Ensure your summary only includes facts present in the source text. Do not hallucinate outside information."
        user_prompt = f"Please read the following document and write a concise, comprehensive summary.\n\nDocument:\n{context}"
        return self.generate_response(user_prompt, system_prompt=system_prompt, max_new_tokens=max_new_tokens, temperature=temperature)
