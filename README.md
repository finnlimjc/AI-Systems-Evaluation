# AI Systems Evaluation: Whisper Speech-to-Text (STT) Benchmark

A comprehensive Speech-to-Text benchmark evaluating OpenAI Whisper model variants (Tiny, Base, Large-v3) across three dimensions: **latency/efficiency**, **accent robustness**, and **semantic preservation**. This project investigates how well Whisper handles non-native English accents and evaluates prompt engineering approaches for accent-specific guidance.

## Key Findings

✅ **Base Whisper is Semantically Robust:** BERT-F1 scores of 0.92-0.96 across all models, indicating strong semantic preservation despite word-level errors.

✅ **Model Size Matters:** Whisper Large-v3 significantly outperforms Tiny/Base on accented speech (10% WER vs. 30-34%), justifying computational cost.

⚠️ **Prompt Engineering Challenge:** Token prefix constraints degrade output quality and accuracy. Current approach not suitable for accent guidance; alternative methods needed - future exploration.

## Project Overview

This project comprehensively evaluates Whisper's performance on accented English speech and documents findings on prompt engineering effectiveness. The evaluation combines traditional ASR metrics (WER, SER, DER, IER) with semantic similarity metrics (BERT-F1) to provide multi-dimensional analysis.

### Evaluation Goals

1. Benchmark Whisper variants on speed, accuracy, and cost trade-offs
2. Assess accent robustness across Singaporean, Indian, British English
3. Investigate prompt engineering limitations and document findings
4. Measure semantic preservation using BERT-Score
5. Provide comprehensive analysis of model capabilities and limitations

---

## Project Structure

The evaluation is organized in 4 Parts:

### Part 1: Latency & Efficiency Benchmarking - Deterministic Evaluation
- Dataset: LibriSpeech (clean + challenging, 200 samples)
- Metrics: WER, CER, Latency (ms), RTF, Cost/1000 audio hours
- Goal: Accuracy vs. Speed trade-off analysis

### Part 2: Accent Robustness & Prompt Engineering Analysis
- Dataset: DTU54DL/common-accent + fallback datasets (100 samples per accent)
- Metrics: WER, SER, DER, IER, prompt impact analysis
- **Finding:** Token prefix prompts degrade accuracy
- Goal: Quantify prompt engineering limitations and document findings

### Part 3: Semantic Evaluation
- Method: BERT-Score (semantic similarity)
- Metrics: Base BERT-F1, Prompted BERT-F1, BERT-F1 Improvement
- Goal: Prove semantic meaning preservation despite word-level errors

### Part 4: Comprehensive Comparison
- Visualizations: Dashboard, heatmap, radar charts, waterfall, table
- Goal: Unified view across all methods and metrics

---

## Installation & Setup

### Requirements
- Python 3.8+
- CUDA 11.0+ (GPU recommended, CPU mode supported)

### Dependencies
```bash
pip install jiwer transformers torch librosa pandas matplotlib seaborn soundfile datasets accelerate bert-score
```

### GPU Setup
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## Usage

### Running the Notebook
```bash
jupyter notebook WIP\ STT_Comparison-wPrompt wBERT.ipynb
```
Run cells sequentially.

## Evaluation Metrics

### Accuracy Metrics
- WER: Word Error Rate (%) - Lower is better
- SER: Substitution Error Rate (%) - Lower is better
- DER: Deletion Error Rate (%) - Lower is better
- IER: Insertion Error Rate (%) - Lower is better
- BERT-F1: Semantic similarity (0-100%) - Higher is better

### Efficiency Metrics
- Latency: Time per clip (ms) - Lower is better
- RTF: Real-Time Factor - Lower is better
- Cost/1K hrs: Compute cost per 1,000 audio hours ($) - Lower is better

---

## Output Files

### CSV Exports
- stt_benchmark_results.csv - Part 1 results
- whisper_part2_accent_results.csv - Part 2 results
- whisper_bert_score_comparison.csv - Part 3 BERT-Score results
- whisper_final_summary_report.csv - Final comprehensive report

### Visualizations (PNG)
- whisper_comprehensive_dashboard.png - 9-panel overview
- whisper_metrics_heatmap.png - Normalized metrics matrix
- whisper_radar_profiles.png - 6-D capability profiles
- whisper_impact_waterfall.png - Method impact timeline
- whisper_summary_table.png - Visual summary table
- whisper_wer_comparison.png - WER base vs. prompted
- whisper_bert_f1_comparison.png - BERT-F1 comparison
- whisper_wer_vs_bertf1_scatter.png - Correlation analysis

---

## Key Features

✅ Multi-Model Comparison: Tiny, Base, Large-v3
✅ Prompt Engineering: Accent-specific decoder guidance
✅ Multiple Metrics: WER, SER, DER, IER, BERT-F1, Latency, Cost
✅ Accent Coverage: Singaporean, Indian, British English
✅ Semantic Evaluation: BERT-Score for meaning preservation
✅ Configurable: Adjust samples, models, evaluation methods
✅ Production-Ready: CSV exports for further analysis

---

## Methodology

### Approach Tested: Prompt Engineering via Token Prefix
Attempted using `prompt_ids` parameter to bias language decoder with accent context:
- **Implementation:** Tokenize accent-specific text and pass as prompt
- **Result:** Output constrained to prefix tokens, degrading transcription quality
- **Limitations:** Current transformers 5.17.0 implementation unsuitable for this use case
- **Lesson Learned:** Token prefix constraints fundamentally different from semantic guidance

### BERT-Score Semantic Evaluation
Measures semantic similarity using RoBERTa contextual embeddings:
- Proves meaning is preserved despite word-level errors
- Complements WER by evaluating comprehension-level accuracy
- BERT-F1 0.92+ indicates transcriptions remain meaningful even with 30%+ WER

### Error Analysis Metrics
- **WER:** Total word errors (substitutions + deletions + insertions)
- **SER:** Substitution Error Rate (words replaced with incorrect words)
- **DER:** Deletion Error Rate (words missing from transcription)
- **IER:** Insertion Error Rate (extra words added to transcription)
- **BERT-F1:** Semantic similarity (0-1 scale, higher is better)

---

## Actual Results & Analysis

### Baseline Performance (Base Whisper Inference)
- **WER:** Whisper Tiny 30%, Base 34.24%, Large-v3 10%
- **BERT-F1:** Tiny 0.9204, Base 0.9347, Large-v3 0.9558
- Larger models significantly more robust to accented speech

### Prompt Engineering Results
- **Approach:** Token prefix constraints via `prompt_ids` parameter
- **Outcome:** Degrades accuracy - WER increases to 417-1163%, output truncated to 1-3 words
- **BERT-F1 Impact:** -0.07 to -0.08 degradation on smaller models, minimal on Large-v3
- **Conclusion:** Not viable for accent guidance without alternative implementation

### Key Insights
1. **Semantic Robustness:** Base inference maintains 92-95% semantic fidelity
2. **Model Scaling Effect:** Large-v3 performance degrades minimally with prompts (0.9558→0.9537)
3. **Accent Handling:** Larger models naturally handle accents well without guidance

---

## Known Limitations & Future Work

### Limitations
1. **Dataset Size:** 100 samples per accent (sufficient for proof-of-concept)
2. **GPU Memory:** Large-v3 requires ~10GB VRAM for inference
3. **Prompt Engineering:** Current `prompt_ids` implementation unsuitable; requires alternative approach
4. **Accent Coverage:** Limited to 3 English accents; needs expansion
5. **Inference-Only:** No model fine-tuning; testing existing capabilities

### Future Work
- Explore language model rescoring for accent guidance
- Fine-tuning on accent-specific datasets
- Evaluate alternative prompt injection methods
- Test on additional accents (Australian, South African, etc.)
- Extended evaluation with 500+ samples per accent
- Noisy audio robustness testing

---

## Contributing

Extend the project:
1. Add new accents (modify ACCENT_LABEL_MAP in Cell 8)
2. Test additional datasets (CommonVoice, Mozilla STT)
3. Evaluate other STT models (Google Cloud, Azure, Deepgram)
4. Add language-specific prompting strategies
5. Implement real-time inference evaluation

---

## References

### Whisper Model
- OpenAI Whisper Paper: https://arxiv.org/abs/2212.04356
- Hugging Face Transformers: https://huggingface.co/docs/transformers

### Evaluation Metrics
- jiwer: https://github.com/jitsi/jiwer
- BERT-Score: https://arxiv.org/abs/1904.09675

### Datasets
- LibriSpeech: http://www.openslr.org/12
- DTU54DL/common-accent: https://huggingface.co/datasets/DTU54DL/common-accent
- ai4bharat/Svarah: https://huggingface.co/datasets/ai4bharat/Svarah
- MERaLiON: https://huggingface.co/datasets/MERaLiON

---

## Acknowledgments

- OpenAI for Whisper
- Hugging Face for Transformers library
- Dataset providers: LibriSpeech, DTU54DL, ai4bharat, MERaLiON

---

## Project Status

**Version:** 1.0 (Complete)
**Last Updated:** September, 2026
**Status:** Ready for Publication - Comprehensive Evaluation Complete
**Python:** 3.10+
**PyTorch:** 2.14.0+
**Transformers:** 5.17.0+
**GPU:** CUDA 11.0+ (optional; CPU mode supported)

### Outputs Generated
- ✅ `whisper_part2_accent_results.csv` - 900 samples across 3 models × 3 accents
- ✅ `whisper_bert_score_comparison.csv` - BERT-F1 semantic evaluation
- ✅ Comprehensive visualizations and analysis charts
- ✅ GitHub-ready documentation and git configuration

---

**Note:** This benchmark documents an experimental evaluation. Token prefix prompting approach is documented as ineffective for this use case, providing valuable learning for future work.
