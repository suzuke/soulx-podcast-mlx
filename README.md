# SoulX-Podcast-MLX

A faithful **Apple Silicon (MLX) native port** of [SoulX-Podcast](https://github.com/Soul-AILab/SoulX-Podcast) — long-form, multi-speaker, multi-turn podcast TTS — running **faster than real-time** on a Mac, with quality on par with the original PyTorch implementation.

> Built by porting the SoulX-Podcast weights onto [mlx-audio](https://github.com/Blaizzy/mlx-audio)'s CosyVoice/Chatterbox-S3Gen MLX modules, plus an MLX `Qwen3-1.7B` LLM stage. Every component was verified bit-exact (or within bf16 noise) against the PyTorch reference.

## Why

The official SoulX-Podcast hardcodes CUDA. On Apple Silicon the only path is PyTorch-MPS with CPU fallback (~RTF 2.7 — much slower than real-time). This port is **MLX-native end-to-end**.

| | PyTorch-MPS (official) | **SoulX-Podcast-MLX (this)** |
|---|---|---|
| Backend | PyTorch + MPS + CPU fallback | MLX (Metal) native |
| RTF (M3 Max) | ~2.7 (slower than real-time) | **~0.55 (≈2× faster than real-time)** |
| Quality | reference | on par (verified bit-exact per component) |

## Architecture

SoulX-Podcast = `LLM (semantic tokens) → Flow Matching (mel) → HiFi-GAN/iSTFT (waveform)` (CosyVoice2 lineage).

| Stage | Module | MLX source | Verification vs PyTorch |
|---|---|---|---|
| Prompt audio → speech tokens | `s3_tokenizer_stage.py` | mlx-audio `S3TokenizerV2` | model bit-exact (251/251) |
| Text → speech tokens | `llm_stage.py` | `mlx-lm` Qwen3-1.7B + RAS sampling | logits cos 1.0 / 0.999 (bf16) |
| Speech tokens → mel | `flow_vocoder.py` | mlx-audio `CausalMaskedDiffWithXvec` | mu corr 1.0, mel corr 0.999 |
| Mel → waveform | `flow_vocoder.py` | mlx-audio `HiFTGenerator` (+0.01 slope fix) | bit-exact (corr 1.0) |
| Script + prompt prep | `input_prep.py` | CAMPPlus (onnx) + MLX mel front-ends | spk-emb bit-exact, prompt-mel corr 1.0 |
| Orchestration | `pipeline.py` | multi-turn `forward_longform` loop | — |

## Requirements

- Apple Silicon Mac (M-series), macOS
- Python 3.11
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) (tested at commit `4ee9539`, v0.4.4) — provides the CosyVoice/Chatterbox S3Gen MLX modules this port builds on
- SoulX-Podcast-1.7B weights from [Hugging Face](https://huggingface.co/Soul-AILab/SoulX-Podcast-1.7B)

## Setup

```bash
# 1. clone this repo + mlx-audio (this port subclasses its s3gen modules)
git clone https://github.com/suzuke/soulx-podcast-mlx.git
cd soulx-podcast-mlx
git clone https://github.com/Blaizzy/mlx-audio.git        # → ./mlx-audio
#   (pin to the tested commit for reproducibility:)
#   (cd mlx-audio && git checkout 4ee9539)

# 2. runtime deps
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. download SoulX-Podcast-1.7B weights
huggingface-cli download Soul-AILab/SoulX-Podcast-1.7B \
    --local-dir pretrained_models/SoulX-Podcast-1.7B

# 4. one-time weight conversion (PyTorch → MLX). needs torch too:
pip install -r requirements-convert.txt
python convert_weights.py pretrained_models/SoulX-Podcast-1.7B weights
#   → produces weights/hift_mlx.safetensors + weights/flow_mlx.safetensors
```

`mlx-audio` is found via `MLX_AUDIO_PATH` (defaults to `./mlx-audio`); converted weights via `SOULX_MLX_WEIGHTS` (defaults to `./weights`).

## Usage

```bash
python soulx_mlx/pipeline.py <script.json> pretrained_models/SoulX-Podcast-1.7B out.wav
```

Script format (multi-speaker, multi-turn; paralinguistic tags supported):

```json
{
  "speakers": {
    "S1": {"prompt_audio": "ref/spk1.wav", "prompt_text": "transcript of spk1.wav"},
    "S2": {"prompt_audio": "ref/spk2.wav", "prompt_text": "transcript of spk2.wav"}
  },
  "text": [
    ["S1", "哈喽,欢迎收听本期节目 <|laughter|>"],
    ["S2", "对呀,今天我们来聊聊..."]
  ]
}
```

`prompt_audio` = a clean ~10s reference clip per speaker (zero-shot voice cloning); `prompt_text` must be its exact transcript. Paralinguistic tags: `<|laughter|>`, `<|sigh|>`, `<|breathing|>`, `<|coughing|>`, `<|throat_clearing|>`. Mandarin/English + Chinese dialects (with the `-dialect` model).

## Development journey

This port was non-trivial: 5 components ported, **7 bugs** found and fixed via layer-by-layer adversarial A/B against the PyTorch reference (each stage's intermediate tensors compared until divergence was localized). The full faithful write-up — every problem and how it was solved — is in **[DEVLOG.md](DEVLOG.md)**.

TL;DR of the 7 bugs:
1. flow ran in training mode (`.eval()` missing) → dropout garbled output
2. converted conv weights saved from non-contiguous MLX views → wrong strides (`np.ascontiguousarray` fix)
3. LLM applied `temperature/top_k/top_p` that SoulX's RAS path does *not* → over-generation + degeneration
4. flow prompt-mel front-end: power-vs-magnitude, `norm=None` vs slaney, missing reflect padding
5. `volume_normalize` used peak→0.9 instead of SoulX's percentile→0.1 → loud hiss
6. CFM ODE steps defaulted to 10 instead of SoulX's 15
7. CFM reused a fixed noise buffer across turns instead of fresh per-call noise

## Credits & License

- **[SoulX-Podcast](https://github.com/Soul-AILab/SoulX-Podcast)** (Soul AI Lab) — original model & weights (Apache 2.0)
- **[mlx-audio](https://github.com/Blaizzy/mlx-audio)** (Prince Canuma / Blaizzy) — the MLX CosyVoice/Chatterbox S3Gen modules this port builds on
- **[MLX](https://github.com/ml-explore/mlx)** (Apple) — the array framework

Licensed under **Apache 2.0** (see [LICENSE](LICENSE)). This is an unofficial community port; not affiliated with Soul AI Lab.

> ⚠️ Do not use for unauthorized voice cloning, impersonation, fraud, or deepfakes (per the upstream SoulX-Podcast disclaimer).
