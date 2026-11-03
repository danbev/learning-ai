"""Calibration text shared by the GPTQ examples.
"""
from datasets import load_dataset


def load_calibration_texts(n_samples: int = 128, min_chars: int = 512, split: str = "train") -> list[str]:
    """Return up to n_samples paragraphs from wikitext-2 that are at least min_chars long."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    texts = []
    for row in ds:
        text = row["text"].strip()
        # wikitext has many empty lines and short "= Heading =" lines, skip them
        if len(text) >= min_chars and not text.startswith("="):
            texts.append(text)
            if len(texts) == n_samples:
                break
    return texts
