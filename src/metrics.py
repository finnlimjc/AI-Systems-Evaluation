import re

import jiwer
import numpy as np
import pandas as pd
from bert_score import score
from jiwer import wer, cer, process_words

from src.config import DEVICE

_TRANSFORMATION = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])


def normalize_text(text:str) -> str:
    """Normalizes text by converting to lowercase, removing punctuation, and stripping whitespace."""
    if not text:
        return ""
    lowered = text.lower()
    no_punctuation = re.sub(r"[^\w\s]", "", lowered)
    normalized = re.sub(r"\s+", " ", no_punctuation).strip()
    return normalized


def compute_metrics(reference:str, hypothesis:str) -> dict:
    """Calculates WER, CER, Precision, Recall, and F1 Score with alignment details."""
    norm_ref = normalize_text(reference)
    norm_hyp = normalize_text(hypothesis)
    
    calc_wer = wer(norm_ref, norm_hyp)
    calc_cer = cer(norm_ref, norm_hyp)
    word_details = process_words(norm_ref, norm_hyp)
    
    sub = word_details.substitutions
    dele = word_details.deletions
    ins = word_details.insertions
    hits = word_details.hits
    
    n_predicted = hits + sub + ins
    n_reference = hits + sub + dele
    precision = hits / n_predicted if n_predicted > 0 else 0.0
    recall = hits / n_reference if n_reference > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        "wer": round(calc_wer * 100, 2),
        "cer": round(calc_cer * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "hits": hits,
        "substitutions": sub,
        "deletions": dele,
        "insertions": ins
    }
    return metrics


def compute_error_breakdown(reference:str, hypothesis:str) -> dict:
    """Compute WER, SER (substitution error rate), DER, IER and hit rate, all in percent."""
    ref_clean = _TRANSFORMATION(reference)
    hyp_clean = _TRANSFORMATION(hypothesis)
    
    empty_breakdown = {"wer": 0, "ser": 0, "der": 0, "ier": 0, "hit_rate": 0}
    if not ref_clean:
        return empty_breakdown
    
    output = jiwer.process_words(ref_clean, hyp_clean)
    n_ref_words = output.substitutions + output.deletions + output.hits
    if n_ref_words == 0:
        return empty_breakdown
    
    total_errors = output.substitutions + output.deletions + output.insertions
    breakdown = {
        "wer": total_errors / n_ref_words * 100,
        "ser": output.substitutions / n_ref_words * 100,
        "der": output.deletions / n_ref_words * 100,
        "ier": output.insertions / n_ref_words * 100,
        "hit_rate": output.hits / n_ref_words * 100
    }
    return breakdown


def compute_bert_f1(hypotheses:list[str], references:list[str]) -> np.ndarray:
    """Per-sample BERT-Score F1 of hypotheses against references."""
    _, _, f1 = score(hypotheses, references, lang="en", device=DEVICE, batch_size=16)
    f1_scores = f1.numpy()
    return f1_scores


def normalize_metric(values:pd.Series, lower_is_better:bool=True) -> pd.Series:
    """Normalize values to a 0-100 scale where 100 is best."""
    min_val = values.min()
    max_val = values.max()
    scaled = (values - min_val) / (max_val - min_val) * 100
    normalized = 100 - scaled if lower_is_better else scaled
    return normalized
