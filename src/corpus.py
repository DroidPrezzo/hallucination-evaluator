from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union


class Corpus:
    """A local corpus of source documents (or bundles) stored as JSONL.

    Presents the same read interface the pipeline needs from a document source:
    length, text-by-index, plus per-document id and contamination risk. Built by
    scripts/build_corpus.py.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        with open(self.path) as f:
            self.records: list[dict[str, Any]] = [
                json.loads(line) for line in f if line.strip()
            ]
        self.name = self.path.stem

    def get_document_length(self) -> int:
        return len(self.records)

    def get_document(self, index: int) -> str:
        return self.records[index]["text"]

    def get_doc_id(self, index: int) -> str:
        return self.records[index]["doc_id"]

    def get_contamination_risk(self, index: int) -> str:
        return self.records[index].get("contamination_risk", "unknown")


class ScrollsSource:
    """Adapter presenting the tau/scrolls HF dataset as a document source.

    Wraps a ContextBuilder so the pipeline can treat the legacy dataset and a
    local Corpus through one interface. Doc ids keep the historical
    "scrolls:{index}" form so idempotency keys are unchanged.
    """

    name = "scrolls"

    def __init__(self, context_builder: Any):
        self._cb = context_builder

    def get_document_length(self) -> int:
        return self._cb.get_document_length()

    def get_document(self, index: int) -> str:
        return self._cb.get_document(index)

    def get_doc_id(self, index: int) -> str:
        return f"scrolls:{index}"

    def get_contamination_risk(self, index: int) -> str:
        return "public_legacy"
