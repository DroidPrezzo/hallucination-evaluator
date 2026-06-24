import pandas as pd
from typing import Dict
import logging
import os


def _sanitize_csv_cell(value):
    """
    Neutralizes spreadsheet formula injection (OWASP LLM05: Improper Output Handling).
    A model name (or any future model-derived field) beginning with =, +, -, or @ is
    treated as a formula by Excel/Sheets; prefixing a single quote forces it to render
    as plain text when the leaderboard CSV is opened.
    """
    if isinstance(value, str) and value and value[0] in ("=", "+", "-", "@"):
        return "'" + value
    return value


class LeaderboardGenerator:
    """
    Aggregates the evaluation results into a formatted Leaderboard CSV and Markdown table.
    """
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_leaderboard(self, results: Dict[str, Dict[int, float]]) -> pd.DataFrame:
        """
        Takes raw result data and generates the leaderboard.
        
        Args:
            results: A dict mapping model names to dictionaries of {context_length: hallucination_rate}
            
        Returns:
            A pandas DataFrame representing the leaderboard.
        """
        self.logger.info("Generating leaderboard from results")
        
        # Determine all unique context lengths across all evaluated models
        all_lengths = set()
        for model_results in results.values():
            all_lengths.update(model_results.keys())
            
        sorted_lengths = sorted(list(all_lengths))
        
        data = []
        for model, model_results in results.items():
            row = {"Model": _sanitize_csv_cell(model)}
            for length in sorted_lengths:
                # Format as percentage entirely for easy reading
                if length in model_results:
                    row[f"{length} Tokens (Hallucination Rate)"] = f"{model_results[length] * 100:.1f}%"
                else:
                    row[f"{length} Tokens (Hallucination Rate)"] = "N/A"
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Output to files
        csv_path = os.path.join(self.output_dir, "leaderboard.csv")
        md_path = os.path.join(self.output_dir, "leaderboard.md")
        
        df.to_csv(csv_path, index=False)
        with open(md_path, "w") as f:
            f.write("# Context Probing Hallucination Leaderboard\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n*Lower is better. Represents the percentage of summaries at that context length containing at least one unfaithful fact.*")
            
        self.logger.info(f"Leaderboard saved to {csv_path} and {md_path}")
        return df
