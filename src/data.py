import io
import logging

import librosa
import soundfile as sf
from datasets import load_dataset, Audio, IterableDataset

from src.config import DATASET_DIR, N_LIBRISPEECH_SAMPLES, LIBRISPEECH_SPLITS, NUM_SAMPLES, SAMPLE_RATE
from src.config import ACCENT_PROMPTS, ACCENT_LABEL_MAP, WESTBROOK_BRITISH_ACCENTS, SVARAH_TEXT_COLUMNS

logger = logging.getLogger(__name__)


def download_librispeech() -> list[dict]:
    """Stream the LibriSpeech splits, write each clip to DATASET_DIR and return the clip metadata."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    clips = []
    for split_config in LIBRISPEECH_SPLITS:
        split_name = split_config["split_name"]
        stream = load_dataset("parquet", data_files={split_name: split_config["url"]}, split=split_name, streaming=True)
        stream = stream.cast_column("audio", Audio(decode=False))
        
        for clip_idx, sample in enumerate(stream):
            if clip_idx >= N_LIBRISPEECH_SAMPLES:
                break
            audio = _decode_audio(sample["audio"], target_rate=None)
            file_path = DATASET_DIR / f"{split_config['file_prefix']}_{clip_idx + 1}.wav"
            sf.write(file_path, audio["array"], audio["sampling_rate"])
            duration = len(audio["array"]) / audio["sampling_rate"]
            
            clips.append({
                "id": f"{split_config['id_prefix']}_{clip_idx + 1}",
                "category": split_config["category"],
                "file_path": str(file_path),
                "duration_sec": round(duration, 2),
                "reference": sample["text"]
            })
    return clips


def preload_audio(clips:list[dict]) -> list[dict]:
    """Read every clip's wav into memory so file I/O and decoding are not part of the latency timings."""
    loaded_clips = []
    for clip in clips:
        audio, sampling_rate = sf.read(clip["file_path"], dtype="float32")
        loaded_clips.append({**clip, "audio_input": {"raw": audio, "sampling_rate": sampling_rate}})
    return loaded_clips


def load_accent_data(n_samples:int=NUM_SAMPLES) -> dict[str, list[dict]]:
    """Collect n_samples per accent, topping up from fallback datasets when common-accent runs short.
    
    Inputs:
        - n_samples: target number of samples per accent
    Outputs:
        - dict mapping accent name to a list of {"audio", "sentence"} samples; a source that fails to load is logged and skipped
    """
    accent_data = {accent: [] for accent in ACCENT_PROMPTS}
    
    fallback_loaders = {
        "british": _fill_british_from_westbrook,
        "indian": _fill_indian_from_svarah,
        "singaporean": _fill_singaporean_from_mnsc
    }
    _fill_from_common_accent(accent_data, n_samples)
    for accent, fill_from_fallback in fallback_loaders.items():
        if len(accent_data[accent]) < n_samples:
            try:
                fill_from_fallback(accent_data[accent], n_samples)
            except Exception as error:
                logger.warning("Fallback dataset for '%s' failed to load: %s", accent, error)
    return accent_data


def _decode_audio(audio:dict, target_rate:int|None=SAMPLE_RATE) -> dict:
    """Decode raw audio bytes with soundfile (no FFmpeg/torchcodec), mixing down to mono and resampling to target_rate unless it is None."""
    array, sampling_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
    mono_array = array.mean(axis=1)
    if target_rate is not None and sampling_rate != target_rate:
        mono_array = librosa.resample(mono_array, orig_sr=sampling_rate, target_sr=target_rate)
        sampling_rate = target_rate
    decoded_audio = {"array": mono_array, "sampling_rate": sampling_rate}
    return decoded_audio


def _stream_audio(path:str, split:str, audio_column:str="audio", **kwargs) -> IterableDataset:
    stream = load_dataset(path, split=split, streaming=True, **kwargs)
    stream = stream.cast_column(audio_column, Audio(decode=False))
    return stream


def _fill_from_common_accent(accent_data:dict[str, list[dict]], n_samples:int) -> None:
    try:
        stream = _stream_audio("DTU54DL/common-accent", "train")
        for sample in stream:
            accent_label = sample.get("accent", "")
            for target, valid_labels in ACCENT_LABEL_MAP.items():
                is_match = any(label in accent_label for label in valid_labels)
                if is_match and len(accent_data[target]) < n_samples:
                    accent_data[target].append({"audio": _decode_audio(sample["audio"]), "sentence": sample["sentence"]})
                    break
            
            if all(len(samples) >= n_samples for samples in accent_data.values()):
                break
    except Exception as error:
        logger.warning("DTU54DL/common-accent failed to load: %s", error)


def _fill_british_from_westbrook(samples:list[dict], n_samples:int) -> None:
    stream = _stream_audio("westbrook/English_Accent_DataSet", "train")
    accent_feature = stream.features["accent"]
    for sample in stream:
        accent_val = sample.get("accent", "")
        if isinstance(accent_val, int):
            accent_str = accent_feature.names[accent_val] if hasattr(accent_feature, "names") else ""
        else:
            accent_str = str(accent_val)
        
        if accent_str in WESTBROOK_BRITISH_ACCENTS:
            samples.append({"audio": _decode_audio(sample["audio"]), "sentence": sample["raw_text"]})
        
        if len(samples) >= n_samples:
            break


def _fill_indian_from_svarah(samples:list[dict], n_samples:int) -> None:
    stream = _stream_audio("ai4bharat/Svarah", "test")
    for sample in stream:
        if len(samples) >= n_samples:
            break
        sentence = next((sample[column] for column in SVARAH_TEXT_COLUMNS if sample.get(column)), "")
        samples.append({"audio": _decode_audio(sample["audio"]), "sentence": sentence})


def _fill_singaporean_from_mnsc(samples:list[dict], n_samples:int) -> None:
    stream = _stream_audio("MERaLiON/Multitask-National-Speech-Corpus-v1", "train", audio_column="context", data_dir="ASR-PART1-Test")
    for sample in stream:
        if len(samples) >= n_samples:
            break
        samples.append({"audio": _decode_audio(sample["context"]), "sentence": sample.get("answer", "")})
