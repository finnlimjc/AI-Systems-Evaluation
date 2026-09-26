# AI Systems Evaluation: Whisper Speech-to-Text (STT) Benchmark

A Speech-to-Text benchmark of OpenAI Whisper (Tiny, Base, Large-v3) across three dimensions: **latency and cost**, **accent robustness**, and **semantic preservation**. It also tests whether an accent-specific decoder prompt (`prompt_ids`) improves transcription of Singaporean, Indian and British English.

## Key Findings

✅ **Model size matters:** on accented speech Whisper Large-v3 reaches ~10% WER against ~30-35% for Tiny and Base, at roughly 6-7x the latency and 11-14x the cost per audio hour of the smaller models.

✅ **Transcripts stay semantically close to the reference:** base BERT-F1 is 0.92-0.96 for all models despite word-level errors.

⚠️ **The accent prompt hurts small models:** with the prompt, Tiny and Base often produce repeated or hallucinated text. Median WER rises from 22 to 98 (Tiny) and from 15 to 83 (Base), and the mean reaches 437-1184% because insertions are unbounded. Large-v3 is barely affected (median 5.6 to 7.7). The cause has not been isolated: see [Limitations](#known-limitations--future-work).

## Evaluation Structure

The benchmark notebook has four parts. Its cells are thin drivers that call the code in `src/`.

| Part | What it does | Dataset | Main metrics |
|---|---|---|---|
| 1. Latency & efficiency | Times each clip on each model (untimed warm-up, median of 3 runs) | LibriSpeech validation, clean + other, 100 clips each | WER, CER, latency p50/p95/p99, RTF, cost per 1,000 audio hours |
| 2. Accent robustness | Transcribes each sample with and without an accent prompt | `DTU54DL/common-accent`, topped up from Westbrook (British), Svarah (Indian) or MNSC (Singaporean); 100 per accent | WER, SER, DER, IER, SER reduction |
| 3. Semantic evaluation | BERT-Score of the Part 2 hypotheses against references (no re-inference) | Part 2 outputs | Base / prompted BERT-F1 |
| 4. Comparison | Combines all parts into plots and a summary table | Results CSVs | Accuracy, speed and semantic overviews, prompt-effect charts, base vs prompted radar, table |

Downloads are deterministic: streams are read in file order with no shuffling, so the same samples are selected each run unless the source datasets change.

## Repository Layout

```
notebooks/   STT_Comparison-wPrompt wBERT.ipynb (runs the benchmark), results_analysis.ipynb (figures),
             results/ (CSVs and old PNGs), new_results/ (current PNGs)
src/
  config.py      paths, model ids, sample counts, prompts, GPU hourly cost
  data.py        dataset download, audio decoding (soundfile + librosa)
  inference.py   Whisper pipelines, latency benchmark, accent evaluation
  metrics.py     WER/CER/SER/DER/IER, BERT-F1, normalisation
  aggregate.py   summary tables from raw results
  plots.py       plot configuration and all figures
  io_utils.py    saving and loading results
dataset/     downloaded LibriSpeech wavs (git-ignored)
```

`src/` contains no printing; the notebook does all display and plotting.

## Setup

**Requirements:** Python 3.10+. An NVIDIA GPU is recommended; CPU works but the accent evaluation is slow.

```bash
pip install -r requirements.txt
```

- **GPU:** on Windows, `pip install torch` installs a CPU-only build. Install the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) and check with `torch.cuda.is_available()`. Whisper Large-v3 needs about 10GB of VRAM in fp16, so it will not fit a 6GB card.
- **Hugging Face token:** create `secrets.env` in the project root containing `HF_TOKEN=<your token>`. It is loaded automatically and git-ignored.
- **Audio decoding** uses `soundfile` and `librosa`, so FFmpeg is not required.

## Usage

Open `notebooks/STT_Comparison-wPrompt wBERT.ipynb` and run the cells in order. Sample counts, models, repeats and prompts are set in `src/config.py`. CSVs are written to `notebooks/results/`.

To redraw the figures without re-running any model, run `notebooks/results_analysis.ipynb`. It reads the saved CSVs and writes PNGs to `notebooks/new_results/`.

> **Note:** `STT_Comparison-wPrompt wBERT.ipynb` was run with the earlier plotting code. Its embedded figures, and the PNGs in `notebooks/results/`, are outdated; use `results_analysis.ipynb` and `notebooks/new_results/` for the current charts. Re-running its plotting cells will fail because those plot functions have been replaced.

## Evaluation Metrics

**Accuracy** (text is lower-cased and stripped of punctuation before scoring)
- **WER:** (substitutions + deletions + insertions) / reference words. Not capped at 100%, because insertions are unbounded.
- **SER / DER / IER:** the substitution, deletion and insertion parts of WER.
- **CER:** the same at character level.
- **BERT-F1:** semantic similarity to the reference on a 0-1 scale (higher is better).

**Efficiency** (measured sequentially, one clip at a time)
- **Latency:** time per clip, reported as p50/p95/p99.
- **RTF:** processing time / audio duration.
- **Cost per 1,000 audio hours:** compute time × an hourly GPU rate. Rates are AWS on-demand reference prices set per model in `config.py` (g4dn.xlarge for Tiny and Base, g5.xlarge for Large-v3). Check them against current pricing before relying on them.

## Results

Part 1: LibriSpeech (200 clips per model)

| Model | WER (%) | p50 latency (ms) | RTF | Cost / 1,000 audio hrs ($) |
|---|---|---|---|---|
| Whisper Tiny | 10.2 | 197 | 0.041 | 20.7 |
| Whisper Base | 7.8 | 245 | 0.052 | 26.4 |
| Whisper Large-v3 | 3.8 | 1461 | 0.303 | 286.0 |

Parts 2 and 3: accented speech (300 samples per model)

| Model | Base WER | Prompted WER | Median WER (base → prompted) | Base BERT-F1 | Prompted BERT-F1 |
|---|---|---|---|---|---|
| Whisper Tiny | 29.6 | 1184.1 | 22.2 → 98.0 | 0.921 | 0.836 |
| Whisper Base | 34.8 | 437.2 | 15.4 → 83.3 | 0.934 | 0.862 |
| Whisper Large-v3 | 10.0 | 12.0 | 5.6 → 7.7 | 0.956 | 0.954 |

The mean prompted WER of Tiny and Base is dominated by a minority of looping outputs (WER in the thousands on short sentences), so read it together with the median. Latency figures depend on the hardware the benchmark ran on.

## Output Files

`notebooks/results/` (CSVs; the PNGs here come from the old plotting code)

- CSVs: `stt_benchmark_results.csv` (Part 1), `stt_cost_efficiency_summary.csv`, `whisper_part2_accent_results.csv` (Part 2, the filename predates the rename to "accent"), `whisper_bert_score_comparison.csv` (Part 3), `whisper_final_summary_report.csv`
- Figures (`notebooks/new_results/`, from `results_analysis.ipynb`): `part1_accuracy_latency.png`, `speed_overview.png`, `wer_by_accent.png`, `accuracy_overview.png`, `prompt_shift.png`, `semantic_overview.png`, `radar_profiles.png`, `summary_table.png`. Each has at most four charts, BERT-F1 is kept on its own 0-1 chart, and prompted WER is shown as a median.

## Known Limitations & Future Work

### Limitations
1. **Prompt result is not fully explained.** Prompting degrades Tiny and Base, but it has not been tested whether the `prompt_ids` usage is handled as intended by transformers 5.17, or whether a different prompt wording, a repetition limit, or a longer clip would avoid the loops. Treat it as "this prompt, on these models, does not help", not as a general statement about prompting.
2. **Mixed accent sources.** An accent's 100 samples may come from more than one dataset, with different recording conditions and transcript styles. The Westbrook "British" set also includes Scottish and Irish speakers.
3. **Sample size.** 100 samples per accent, on short sentences (about 11 words), gives noisy per-accent averages.
4. **Sequential benchmark.** Latency and cost describe one request at a time. A batched or concurrent server would show higher throughput and lower cost, which this benchmark does not measure.
5. **Medians in figures.** Prompted WER means for Tiny and Base are dominated by looping outputs, so the figures show medians plus the share of samples above 100% WER. Means are in the tables above. The radar's axis scales are fixed constants (`RADAR_SCALES` in `src/plots.py`), so its shapes compare models, not absolute quality.
6. **Inference only.** No fine-tuning; results depend on the software and hardware used.

### Future Work
- Isolate the prompt failure: verify `prompt_ids` handling, try repetition limits and other prompt wordings
- Add per-request cost, a batched throughput benchmark and a human preference study
- More accents, more samples, and noisy or long-form audio

## References

- Whisper: https://arxiv.org/abs/2212.04356 · Transformers: https://huggingface.co/docs/transformers
- jiwer: https://github.com/jitsi/jiwer · BERT-Score: https://arxiv.org/abs/1904.09675
- Datasets: [LibriSpeech](http://www.openslr.org/12) · [DTU54DL/common-accent](https://huggingface.co/datasets/DTU54DL/common-accent) · [westbrook/English_Accent_DataSet](https://huggingface.co/datasets/westbrook/English_Accent_DataSet) · [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) · [MERaLiON MNSC](https://huggingface.co/datasets/MERaLiON/Multitask-National-Speech-Corpus-v1)
