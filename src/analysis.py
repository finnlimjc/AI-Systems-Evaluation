import zlib
from difflib import SequenceMatcher

import jiwer
import pandas as pd

from src.config import ACCENT_PROMPTS
from src.metrics import normalize_text

FUNCTION_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "of", "to", "for", "with", "by", "from", "as", "is", "was", "were", "are",
    "be", "been", "he", "she", "it", "they", "we", "you", "i", "his", "her", "their", "this", "that", "these", "those", "thee", "thy"
})
ABBREVIATIONS = frozenset({"missus", "mrs", "mister", "mr", "doctor", "dr", "saint", "st"})
NEAR_SPELLING_RATIO = 0.7
LOOP_COMPRESSION_RATIO = 2.4 #gzip threshold Whisper uses to flag repetitive text (Radford et al., 2022, sec. 4.5)
TRUNCATED_MAX_WORDS = 2
MINOR_WER = 20
RUNAWAY_WER = 100
ECHO_WORDS = 4 #leading prompt words that must reappear for an output to count as echoing the prompt

EDIT_CATEGORIES = ["Numeral / abbreviation", "Function-word swap", "Near-spelling variant", "Different word", "Dropped word", "Extra word"]
OUTPUT_TYPES = ["Exact", "Minor errors", "Heavy errors", "Truncated", "Prompt echo", "Repetition loop", "Runaway"]


def word_edits(reference:str, hypothesis:str) -> list[tuple[str, str, str]]:
    """Word-level edits (kind, reference word, hypothesis word) after the same normalisation the metrics use."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    alignment = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words)).alignments[0]
    
    edits = []
    for chunk in alignment:
        ref_span = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
        hyp_span = hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]
        if chunk.type == "substitute":
            edits.extend(("substitute", ref_word, hyp_word) for ref_word, hyp_word in zip(ref_span, hyp_span))
        elif chunk.type == "delete":
            edits.extend(("delete", ref_word, "") for ref_word in ref_span)
        elif chunk.type == "insert":
            edits.extend(("insert", "", hyp_word) for hyp_word in hyp_span)
    return edits


def classify_edit(kind:str, ref_word:str, hyp_word:str) -> str:
    """Assign one rule-based category to a word edit."""
    if kind == "delete":
        return "Dropped word"
    if kind == "insert":
        return "Extra word"
    
    is_numeral = any(char.isdigit() for char in ref_word + hyp_word)
    is_abbreviation = ref_word in ABBREVIATIONS and hyp_word in ABBREVIATIONS
    if is_numeral or is_abbreviation:
        return "Numeral / abbreviation"
    if ref_word in FUNCTION_WORDS and hyp_word in FUNCTION_WORDS:
        return "Function-word swap"
    if SequenceMatcher(None, ref_word, hyp_word).ratio() >= NEAR_SPELLING_RATIO:
        return "Near-spelling variant"
    return "Different word"


def build_edit_table(df:pd.DataFrame, model_col:str, ref_col:str, hyp_col:str) -> pd.DataFrame:
    """One row per word edit: Model, Kind, Reference word, Hypothesis word, Category."""
    rows = [
        (model, kind, ref_word, hyp_word, classify_edit(kind, ref_word, hyp_word))
        for model, reference, hypothesis in zip(df[model_col], df[ref_col].fillna(""), df[hyp_col].fillna(""))
        for kind, ref_word, hyp_word in word_edits(reference, hypothesis)
    ]
    edits = pd.DataFrame(rows, columns=["Model", "Kind", "Reference word", "Hypothesis word", "Category"], dtype="object")
    return edits


def edit_category_counts(edits:pd.DataFrame, model_order:list[str]) -> pd.DataFrame:
    """Edit counts per model (rows) and category (columns, EDIT_CATEGORIES order)."""
    counts = pd.crosstab(edits["Model"], edits["Category"]).reindex(index=model_order, columns=EDIT_CATEGORIES, fill_value=0)
    return counts


def top_confusions(edits:pd.DataFrame, model:str, n_rows:int=8) -> pd.DataFrame:
    """Most frequent (reference word, hypothesis word) edits of one model."""
    model_edits = edits[edits["Model"] == model]
    confusions = model_edits.groupby(["Kind", "Reference word", "Hypothesis word"]).size().rename("Count").reset_index()
    top = confusions.sort_values(["Count", "Reference word"], ascending=[False, True]).head(n_rows).reset_index(drop=True)
    return top


def compression_ratio(text:str) -> float:
    """Length of the UTF-8 text over its zlib-compressed length; high values mean repetitive text."""
    raw = text.encode("utf-8")
    ratio = len(raw) / len(zlib.compress(raw)) if raw else 0.0
    return ratio


def classify_output(reference:str, hypothesis:str) -> str:
    """Label one transcript by how it failed, checked from the most specific failure to the least."""
    norm_ref = normalize_text(reference)
    norm_hyp = normalize_text(hypothesis)
    if norm_hyp == norm_ref:
        return "Exact"
    
    ref_len = len(norm_ref.split())
    hyp_words = norm_hyp.split()
    echoes_prompt = any(" ".join(normalize_text(prompt).split()[:ECHO_WORDS]) in norm_hyp for prompt in ACCENT_PROMPTS.values())
    if echoes_prompt:
        return "Prompt echo"
    if compression_ratio(hypothesis) > LOOP_COMPRESSION_RATIO:
        return "Repetition loop"
    if len(hyp_words) <= TRUNCATED_MAX_WORDS and ref_len > TRUNCATED_MAX_WORDS:
        return "Truncated"
    
    error_rate = jiwer.wer(norm_ref, norm_hyp) * 100
    if error_rate > RUNAWAY_WER:
        return "Runaway"
    label = "Minor errors" if error_rate <= MINOR_WER else "Heavy errors"
    return label


def output_type_counts(df:pd.DataFrame, hyp_col:str, model_order:list[str]) -> pd.DataFrame:
    """Share (%) of transcripts of each output type per model (rows) for one hypothesis column."""
    labels = df.apply(lambda row: classify_output(row["Reference"], row[hyp_col] if isinstance(row[hyp_col], str) else ""), axis=1)
    counts = pd.crosstab(df["Model"], labels).reindex(index=model_order, columns=OUTPUT_TYPES, fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0) * 100
    return shares


def pick_examples(df:pd.DataFrame, model:str, sort_col:str, n_rows:int=3, ascending:bool=False) -> pd.DataFrame:
    """The n_rows most extreme rows of one model by sort_col."""
    model_rows = df[df["Model"] == model]
    examples = model_rows.sort_values(sort_col, ascending=ascending).head(n_rows)
    return examples
