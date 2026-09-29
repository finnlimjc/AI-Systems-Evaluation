import logging
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline, Pipeline, WhisperProcessor, WhisperForConditionalGeneration

from src.config import DEVICE, LATENCY_DTYPE, ACCENT_DTYPE, MODEL_IDS, GENERATE_KWARGS, WARMUP_CLIPS, N_REPEATS
from src.config import ACCENT_PROMPTS
from src.metrics import compute_metrics, compute_error_breakdown

# Warnings that are expected here: clips are timed one at a time on purpose, and the other is raised inside Whisper's own generate()
_SILENCED_WARNINGS = {
    "transformers.pipelines.base": "pipelines sequentially on GPU",
    "transformers.generation.utils": "Passing `generation_config` together with"
}
for logger_name, phrase in _SILENCED_WARNINGS.items():
    logging.getLogger(logger_name).addFilter(lambda record, phrase=phrase: phrase not in record.getMessage())

def build_pipe(model_id:str) -> Pipeline:
    """Build an Automatic Speech Recognition (ASR) pipeline for one Whisper checkpoint."""
    pipe = pipeline(
        "automatic-speech-recognition", # Uses the ASR task specific to Whisper models (chosen by model param)
        model=model_id,
        dtype=LATENCY_DTYPE,
        device=DEVICE,
        model_kwargs={"attn_implementation": "sdpa"} # Recommended as described in (https://huggingface.co/openai/whisper-large-v3)
    )
    pipe.tokenizer.clean_up_tokenization_spaces = False #Whisper's BPE tokenizer ignores it and warns when True
    return pipe

def build_pipes() -> dict[str, Pipeline]:
    """Build one Automatic Speech Recognition (ASR) pipeline per model in config/MODEL_IDS."""
    pipes = {model_name: build_pipe(model_id) for model_name, model_id in MODEL_IDS.items()}
    return pipes

def sync_gpu() -> None:
    """Make sure queued GPU work has finished so timings are accurate."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timed_transcribe(pipe:Pipeline, audio_input:dict) -> tuple[str, float]:
    """Transcribe one in-memory clip and return (text, elapsed_seconds)."""
    sync_gpu() #Flush
    start = time.perf_counter()
    output = pipe(dict(audio_input), generate_kwargs=GENERATE_KWARGS)
    sync_gpu() # Wait until all GPU Kernels have completed
    elapsed = time.perf_counter() - start
    return output["text"], elapsed

def run_latency_benchmark(pipes:dict[str, Pipeline], clips:list[dict]) -> pd.DataFrame:
    """
    Time every clip individually on every model, keeping the median of N_REPEATS runs.
    
    Inputs:
        - pipes: model name -> ASR pipeline
        - clips: clip metadata with pre-loaded "audio_input"
    
    Outputs:
        - one row per (model, clip) with latency, RTF and accuracy metrics
    """
    rows = []
    for model_name, pipe in pipes.items():
        # Warm-up (not recorded): removes CUDA init / kernel compilation from the timings
        for clip in clips[:WARMUP_CLIPS]:
            timed_transcribe(pipe, clip["audio_input"])
        
        for clip in tqdm(clips, desc=model_name):
            timings = np.empty(N_REPEATS, dtype=np.float64)
            for repeat in range(N_REPEATS):
                text, elapsed = timed_transcribe(pipe, clip["audio_input"])
                timings[repeat] = elapsed
                
                # Take the first transcript
                if repeat == 0:
                    hyp_text = text
            
            latency = float(np.median(timings))
            duration = clip["duration_sec"]
            metrics = compute_metrics(clip["reference"], hyp_text)
            
            rows.append({
                "System": model_name,
                "Clip ID": clip["id"],
                "Category": clip["category"],
                "Duration (s)": duration,
                "Latency (s)": round(latency, 4),
                "Latency Std (s)": round(float(np.std(timings)), 4),
                "RTF": round(latency / duration, 4),
                "WER (%)": metrics["wer"],
                "CER (%)": metrics["cer"],
                "Precision (%)": metrics["precision"],
                "Recall (%)": metrics["recall"],
                "F1 Score (%)": metrics["f1_score"],
                "Reference": clip["reference"],
                "Hypothesis": hyp_text
            })
    
    column_dtypes = {
        "System": "object",
        "Clip ID": "object",
        "Category": "object",
        "Duration (s)": "float64",
        "Latency (s)": "float64",
        "Latency Std (s)": "float64",
        "RTF": "float64",
        "WER (%)": "float64",
        "CER (%)": "float64",
        "Precision (%)": "float64",
        "Recall (%)": "float64",
        "F1 Score (%)": "float64",
        "Reference": "object",
        "Hypothesis": "object"
    }
    df_results = pd.DataFrame(rows).astype(column_dtypes)
    
    return df_results

def _generate_text(model, processor, input_features:torch.Tensor, prompt_ids:torch.Tensor|None=None) -> str:
    """Generate and decode one transcript, optionally conditioned on prompt_ids (e.g. an accent hint)."""
    with torch.no_grad():
        generated_ids = model.generate(input_features, language="en", task="transcribe", prompt_ids=prompt_ids)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return text

def run_accent_evaluation(accent_data:dict[str, list[dict]]) -> pd.DataFrame:
    """
    Transcribe every accent sample with and without the accent prompt on every model.
    Source: https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/whisper
    
    Inputs:
        - accent_data: accent name -> list of {"audio", "sentence"} samples
        
    Outputs:
        - one row per (model, accent, sample) with base and prompted hypotheses and error breakdowns
    """
    rows = []
    for model_name, model_id in tqdm(MODEL_IDS.items(), desc="Models"):
        processor = WhisperProcessor.from_pretrained(model_id) #  wraps a feature extractor and a tokenizer into a single processor.
        model = WhisperForConditionalGeneration.from_pretrained(model_id).to(DEVICE, dtype=ACCENT_DTYPE)
        
        for accent_name, samples in accent_data.items():
            if len(samples) == 0:
                continue
            
            prompt = ACCENT_PROMPTS.get(accent_name, "")
            prompt_ids = torch.tensor(processor.tokenizer.encode(prompt)).to(DEVICE) if prompt else None
            
            for sample_idx, sample in enumerate(samples):
                # 2D representation of frequencies (log-Mel filter bank features to mimic how humans perceive sound)
                input_features = processor(
                    sample["audio"]["array"],
                    sampling_rate=sample["audio"]["sampling_rate"],
                    return_tensors="pt" #wrap as tensor
                ).input_features.to(DEVICE).to(ACCENT_DTYPE)
                
                base_text = _generate_text(model, processor, input_features)
                prompted_text = _generate_text(model, processor, input_features, prompt_ids)
                
                reference = sample["sentence"]
                base = compute_error_breakdown(reference, base_text)
                prompted = compute_error_breakdown(reference, prompted_text)
                ser_reduction = (base["ser"] - prompted["ser"]) / base["ser"] * 100 if base["ser"] > 0 else 0
                
                rows.append({
                    "Model": model_name,
                    "Accent": accent_name.capitalize(),
                    "Sample Index": sample_idx + 1,
                    "Reference": reference,
                    "Base Hypothesis": base_text,
                    "Prompted Hypothesis": prompted_text,
                    "Base WER (%)": round(base["wer"], 2),
                    "Prompted WER (%)": round(prompted["wer"], 2),
                    "Base SER (%)": round(base["ser"], 2),
                    "Prompted SER (%)": round(prompted["ser"], 2),
                    "SER Reduction (Δ%)": round(ser_reduction, 2),
                    "Base DER (%)": round(base["der"], 2),
                    "Prompted DER (%)": round(prompted["der"], 2),
                    "Base IER (%)": round(base["ier"], 2),
                    "Prompted IER (%)": round(prompted["ier"], 2)
                })
    
    column_dtypes = {
        "Model": "object",
        "Accent": "object",
        "Sample Index": "int64",
        "Reference": "object",
        "Base Hypothesis": "object",
        "Prompted Hypothesis": "object",
        "Base WER (%)": "float64",
        "Prompted WER (%)": "float64",
        "Base SER (%)": "float64",
        "Prompted SER (%)": "float64",
        "SER Reduction (Δ%)": "float64",
        "Base DER (%)": "float64",
        "Prompted DER (%)": "float64",
        "Base IER (%)": "float64",
        "Prompted IER (%)": "float64"
    }
    df_accent = pd.DataFrame(rows).astype(column_dtypes)
    return df_accent