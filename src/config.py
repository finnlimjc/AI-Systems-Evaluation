from pathlib import Path

import torch
from dotenv import load_dotenv

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "notebooks" / "results"
DATASET_DIR = PROJECT_ROOT / "dataset"

# Populates HF_TOKEN in the environment, which huggingface_hub and datasets read automatically
load_dotenv(PROJECT_ROOT / "secrets.env")

# Result files
LATENCY_RESULTS_FILENAME = "stt_benchmark_results.csv"
COST_EFFICIENCY_FILENAME = "stt_cost_efficiency_summary.csv"
ACCENT_RESULTS_FILENAME = "whisper_part2_accent_results.csv"
BERT_SCORES_FILENAME = "whisper_bert_score_comparison.csv"
FINAL_SUMMARY_FILENAME = "whisper_final_summary_report.csv"

# Hardware
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LATENCY_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
ACCENT_DTYPE = torch.float32

# Models
MODEL_IDS = {
    "Whisper Tiny": "openai/whisper-tiny",
    "Whisper Base": "openai/whisper-base",
    "Whisper Large-v3": "openai/whisper-large-v3"
}
MODEL_ORDER = list(MODEL_IDS)
GENERATE_KWARGS = {"language": "en", "task": "transcribe"} #English-only benchmark, so no time is spent on language detection

# Part 1: latency benchmark
N_LIBRISPEECH_SAMPLES = 100
WARMUP_CLIPS = 2 #untimed clips run before measuring, per model
N_REPEATS = 3 #each clip is timed this many times; the median is recorded
LIBRISPEECH_SPLITS = [
    {
        "split_name": "validation.clean",
        "url": "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/clean/validation/*.parquet",
        "file_prefix": "librispeech_clean",
        "id_prefix": "ls_clean",
        "category": "LibriSpeech (Clean)"
    },
    {
        "split_name": "validation.other",
        "url": "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/other/validation/*.parquet",
        "file_prefix": "librispeech_other",
        "id_prefix": "ls_other",
        "category": "LibriSpeech (Challenging)"
    }
]

# Hourly hosting rates ($/hr): AWS EC2 on-demand, Linux, us-east-1 (https://aws.amazon.com/ec2/pricing/on-demand/)
GPU_HOURLY_COST = {
    "Whisper Tiny": 0.526, #g4dn.xlarge (1x NVIDIA T4), the GPU class the benchmark ran on in Colab
    "Whisper Base": 0.526, #g4dn.xlarge (1x NVIDIA T4), same instance as Tiny
    "Whisper Large-v3": 1.006 #g5.xlarge (1x NVIDIA A10G, 24GB), large enough for the ~10GB VRAM Large-v3 needs
}

# Part 2: accent evaluation
NUM_SAMPLES = 100 #samples per accent
SAMPLE_RATE = 16_000
ACCENT_PROMPTS = {
    "singaporean": "The following is an English conversation spoken with a Singaporean accent.",
    "indian": "The following is an English transcription spoken with an Indian accent.",
    "british": "The following is an English transcription spoken with a British accent."
}
ACCENT_LABEL_MAP = {
    "singaporean": ["Singaporean English"],
    "indian": ["India and South Asia (India, Pakistan, Sri Lanka)"],
    "british": ["England English"]
}
WESTBROOK_BRITISH_ACCENTS = ("English", "Scottish", "Irish", "NorthernIrish")
SVARAH_TEXT_COLUMNS = ("sentence", "transcription", "text", "raw_transcription")
