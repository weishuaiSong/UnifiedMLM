"""Two-step answer extraction.

Step 1: rule-based string match against the candidate set
        (option letters for MCQ, gold string for open-ended).
Step 2: fall back to sentence-transformer cosine similarity if step 1 misses.

This mirrors the extraction pipeline used in the template paper (2412.08307).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ExtractionResult:
    predicted: str           # e.g. "A" for MCQ, or candidate text for open
    method: str              # "rule" | "similarity" | "none"
    raw: str                 # original model text
    score: float | None = None  # similarity score, if used


_LETTER_PATTERNS = [
    re.compile(r"\banswer\s*(?:is|:)\s*\(?([A-E])\)?\b", re.IGNORECASE),
    re.compile(r"\b(?:option|choice)\s*\(?([A-E])\)?\b", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-E])\)?[\.\)\:\s]", re.IGNORECASE),
    re.compile(r"\b([A-E])\b"),  # last-resort: any standalone letter
]


class TwoStepExtractor:
    """Configurable extractor.

    For MCQ: candidates is the dict of {letter: option_text}.
    For open: candidates is a list of acceptable strings (typically just the gold).
    """

    def __init__(
        self,
        sim_threshold: float = 0.5,
        sentence_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        lazy: bool = True,
    ) -> None:
        self.sim_threshold = sim_threshold
        self._sentence_model_name = sentence_model
        self._sentence_model = None  # type: ignore[assignment]
        if not lazy:
            self._ensure_sentence_model()

    def _ensure_sentence_model(self) -> Any:
        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer

            self._sentence_model = SentenceTransformer(self._sentence_model_name)
        return self._sentence_model

    # ---------- MCQ ----------

    def extract_mcq(self, text: str, choices: dict[str, str]) -> ExtractionResult:
        text = (text or "").strip()
        if not text:
            return ExtractionResult(predicted="", method="none", raw=text)

        valid = set(choices.keys())

        # Step 1: regex rule match against letter patterns.
        for pat in _LETTER_PATTERNS:
            m = pat.search(text)
            if m and m.group(1).upper() in valid:
                return ExtractionResult(predicted=m.group(1).upper(), method="rule", raw=text)

        # Also accept "the answer is <option text>" substring matches.
        lowered = text.lower()
        for letter, opt in choices.items():
            if opt and opt.lower() in lowered:
                return ExtractionResult(predicted=letter, method="rule", raw=text)

        # Step 2: sentence-transformer similarity against each option text.
        model = self._ensure_sentence_model()
        option_letters = list(choices.keys())
        option_texts = [choices[l] for l in option_letters]
        embeddings = model.encode([text] + option_texts, convert_to_numpy=True, normalize_embeddings=True)
        pred_vec, opt_vecs = embeddings[0], embeddings[1:]
        sims = opt_vecs @ pred_vec
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score >= self.sim_threshold:
            return ExtractionResult(
                predicted=option_letters[best], method="similarity", raw=text, score=score
            )
        return ExtractionResult(predicted="", method="none", raw=text, score=score)

    # ---------- Open-ended ----------

    def extract_open(self, text: str, gold: str) -> ExtractionResult:
        text = (text or "").strip()
        if not text:
            return ExtractionResult(predicted="", method="none", raw=text)

        if gold and gold.lower() in text.lower():
            return ExtractionResult(predicted=gold, method="rule", raw=text)

        model = self._ensure_sentence_model()
        embeddings = model.encode([text, gold], convert_to_numpy=True, normalize_embeddings=True)
        score = float(embeddings[0] @ embeddings[1])
        if score >= self.sim_threshold:
            return ExtractionResult(predicted=gold, method="similarity", raw=text, score=score)
        return ExtractionResult(predicted=text, method="none", raw=text, score=score)
