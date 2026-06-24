# Hallucination Evaluator

A local, lightweight tool for evaluating Large Language Models (LLMs) and finding their context window breaking points.

As context windows grow larger (128k+ tokens), models often lose the ability to accurately recall and summarize information, leading to hallucinations. This tool uses an LLM-as-a-judge system to iteratively probe open-weight models with increasingly large documents and generate a leaderboard of their hallucination rates.

## AI Security & Red Teaming

This project serves as a practical tool for **AI Red Teaming** and **Vulnerability Assessment**. 

In the field of AI security, understanding the operational boundaries of an LLM is critical. When models are fed contexts that exceed their effective processing capacity, they often exhibit unpredictable degradation—leading to severe hallucinations, data leakage, or failure to follow safety guardrails. By iteratively probing these boundaries, security engineers can:
- Map out the exact "breaking point" where a model transitions from reliable to unfaithful.
- Stress-test system prompts and contextual boundaries against adversarial payload lengths.
- Generate quantitative metrics for model comparison before deploying LLMs into production environments that handle large documents.

## Features
- **100% Local Validation:** Uses local Hugging Face `transformers` models. No API keys required.
- **Dynamic Context Sizing:** Automatically downloads the `tau/scrolls` governmental report dataset and strictly truncates it to test exact token boundaries.
- **LLM-as-a-Judge:** Uses a capable instruct model (e.g., `Qwen2.5-7B-Instruct`) to fact-check generated summaries strictly against the original source text.
- **VRAM Optimized:** Built-in support for 4-bit and 8-bit quantization to run large context evaluations on consumer GPUs.
- **Security Hardened:** Aligned with the [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/). The judge and summarizer treat document/summary text as untrusted data and resist embedded prompt-injection (LLM01); the supply chain is pinned via versioned dependencies, optional Hub revision pinning, and `trust_remote_code=False` (LLM03/LLM04).

## Installation

### Prerequisites
You need a CUDA-compatible GPU to run inference efficiently. Python 3.10+ is recommended.

```bash
# Clone the repository
git clone https://github.com/DroidPrezzo/hallucination-evaluator.git
cd hallucination-evaluator

# Install dependencies
pip install -r requirements.txt
```

## Usage

The primary entry point is `main.py`. By default, it runs an evaluation on a few popular small models (`Qwen2.5-1.5B`, `Phi-3.5-mini`, `Qwen2.5-3B`) using `Qwen2.5-7B` as the judge, testing up to 8000 tokens.

```bash
python main.py
```

### Advanced Usage & VRAM Optimization

If you have limited VRAM (e.g., a 12GB or 16GB GPU), you should use the `--use-4bit` flag to quantize the test models, and `--judge-4bit` to quantize the fact-checker.

To find the true breaking point of a model, push the `--context-lengths` higher:

```bash
python main.py \
  --models Qwen/Qwen2.5-7B-Instruct \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --context-lengths 4000 8000 16000 32000 \
  --samples 20 \
  --use-4bit \
  --judge-4bit
```

### Reproducible & Integrity-Checked Runs

By default, models and the dataset load from their Hub default branch (and a warning is logged). For reproducible benchmarks and to defend against a silently-updated or compromised upstream repo, pin every download to an immutable commit SHA:

```bash
python main.py \
  --models Qwen/Qwen2.5-7B-Instruct \
  --model-revisions <commit-sha> \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --judge-revision <commit-sha> \
  --dataset-revision <commit-sha>
```

### CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--models` | Space-separated list of Hugging Face model IDs to test. | `Qwen/Qwen2.5-1.5B-Instruct microsoft/Phi-3.5-mini-instruct Qwen/Qwen2.5-3B-Instruct` |
| `--judge-model` | The Hugging Face model ID used to fact-check summaries. | `Qwen/Qwen2.5-7B-Instruct` |
| `--model-revisions` | Hub revisions (commit SHA/tag/branch) for `--models`, in the same order. Must match the count of `--models` if given. | `None` |
| `--judge-revision` | Hub revision (commit SHA/tag/branch) to pin the judge model. | `None` |
| `--dataset-revision` | Hub revision (commit SHA/tag/branch) to pin the evaluation dataset. | `None` |
| `--context-lengths` | The token limits to test the models at. | `1000 2000 4000 8000` |
| `--samples` | Number of documents to summarize and evaluate per length. | `10` |
| `--output-dir` | Directory to save the final `leaderboard.md`. | `results` |
| `--cache-dir` | Hugging Face cache directory for downloaded weights. | `None` |
| `--use-4bit` | Loads target models in 4-bit precision (saves massive VRAM). | `False` |
| `--use-8bit` | Loads target models in 8-bit precision. | `False` |
| `--judge-4bit` | Forces the judge model to load in 4-bit precision. | `False` |

## Output

Once the pipeline completes both the Generation and Evaluation phases for all models, it aggregates the hallucination rates and outputs a markdown table to `results/leaderboard.md`.

*Note: A 0.00 rate means the model summarized the text perfectly faithfully 100% of the time. Higher rates mean the model is hallucinating facts not present in the source text.*

## Acknowledgments
Co-authored by **Claude Code** (Anthropic's agentic coding assistant), which assisted in refactoring, security and CVE remediation, OWASP LLM Top 10 hardening, and debugging this evaluation pipeline.
