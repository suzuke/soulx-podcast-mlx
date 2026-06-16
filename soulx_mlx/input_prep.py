"""M5a — Input Preparation Pipeline for SoulX-MLX."""
import json
import re
import sys
import numpy as np
import mlx.core as mx

import os as _os
sys.path.insert(0, _os.environ.get("MLX_AUDIO_PATH", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "mlx-audio")))
from mlx_audio.codec.models.s3 import S3TokenizerV2
from mlx_audio.dsp import stft, hanning
from mlx_lm.utils import load as load_llm

_MEL_FILTERS_PATH_S3 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "assets", "mel_filters.npz")


# ── mel spectrogram (16kHz log-mel, 128 bins) ────────────────────────
def _load_mel_filters(n_mels: int = 128) -> mx.array:
    return mx.array(np.load(_MEL_FILTERS_PATH_S3)[f"mel_{n_mels}"])


def log_mel_16k(audio: np.ndarray) -> mx.array:
    audio_mx = mx.array(audio.astype(np.float32))
    window = hanning(401)[:-1]
    freqs = stft(audio_mx, n_fft=400, hop_length=160, window=window,
                 center=True).swapaxes(0, 1)
    freqs = freqs[:, :-1]
    mags = mx.abs(freqs) ** 2
    mel = _load_mel_filters(128) @ mags
    log_spec = mx.maximum(mel, 1e-10).log10()
    log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


# ── mel spectrogram (24kHz mel, 80 bins, for flow) ───────────────────
def mel_24k(audio: np.ndarray, sample_rate: int = 24000) -> mx.array:
    n_fft, hop, n_mels = 1920, 480, 80
    # FIX: 對齊 SoulX mel_spectrogram —— (1) STFT 前 reflect-pad (n_fft-hop)/2=720 每側(center=False),
    # (2) 用「幅度」非功率(去掉 **2),(3) mel basis 用 slaney 正規化(librosa 預設)。
    pad = (n_fft - hop) // 2
    audio = np.pad(audio.astype(np.float32), pad, mode="reflect")
    audio_mx = mx.array(audio)
    window = hanning(n_fft + 1)[:-1]
    freqs = stft(audio_mx, n_fft=n_fft, hop_length=hop, window=window,
                 center=False).swapaxes(0, 1)
    mags = mx.abs(freqs)  # 幅度 (SoulX: sqrt(real^2+imag^2)),非功率
    from mlx_audio.dsp import mel_filters
    filters = mel_filters(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels,
                          f_min=0, f_max=8000, norm="slaney", mel_scale="slaney")
    mel = filters @ mags
    return mx.log(mx.maximum(mel, 1e-5))


# ── audio loading ─────────────────────────────────────────────────────
def load_audio(path: str, sr: int) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(path, sr=sr, mono=True, dtype=np.float32)
    return audio


def volume_normalize(audio: np.ndarray, coeff: float = 0.1) -> np.ndarray:
    # FIX: 精確移植 SoulX audio_volume_normalize(percentile 縮放到 coeff=0.1),
    # 取代原本的 peak→0.9(那會讓音量過大 → prompt mel offset → 沙沙聲)。
    audio = audio.astype(np.float32)
    temp = np.sort(np.abs(audio))
    if temp[-1] < 0.1:
        scaling_factor = max(temp[-1], 1e-3)
        audio = audio / scaling_factor * 0.1
    temp = temp[temp > 0.01]
    L = temp.shape[0]
    if L <= 10:
        return audio
    volume = np.mean(temp[int(0.9 * L): int(0.99 * L)])
    audio = audio * np.clip(coeff / volume, a_min=0.1, a_max=10)
    max_value = np.max(np.abs(audio))
    if max_value > 1:
        audio = audio / max_value
    return audio


# ── CAMPPlus speaker embedding (onnxruntime) ──────────────────────────
class CAMPPlusONNX:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, fbank: np.ndarray) -> np.ndarray:
        """fbank: (T, 80) — output: (192,)"""
        inp = fbank.astype(np.float32)[np.newaxis, ...]  # (1, T, 80)
        out = self.session.run(None, {self.input_name: inp})[0]
        return out.flatten().astype(np.float32)


# ── kaldi fbank (MLX native) ──────────────────────────────────────────
def kaldi_fbank_np(audio: np.ndarray, sample_rate: int = 16000,
                   num_mel_bins: int = 80) -> np.ndarray:
    from mlx_audio.tts.models.chatterbox.s3gen.xvector import kaldi_fbank
    fbank_mx = kaldi_fbank(mx.array(audio.astype(np.float32)),
                           sample_rate=sample_rate, num_mel_bins=num_mel_bins)
    return np.array(fbank_mx)


# ── podcast JSON parser ────────────────────────────────────────────────
SPK_DICT = ["<|SPEAKER_0|>", "<|SPEAKER_1|>", "<|SPEAKER_2|>", "<|SPEAKER_3|>"]
TEXT_START = "<|text_start|>"
TEXT_END = "<|text_end|>"
AUDIO_START = "<|semantic_token_start|>"
TASK_PODCAST = "<|task_podcast|>"


def parse_podcast_script(json_path: str) -> dict:
    with open(json_path) as f:
        data = json.load(f)

    speakers = data["speakers"]
    speaker_names = list(speakers.keys())
    turns = data["text"]

    prompt_wavs = [speakers[sn]["prompt_audio"] for sn in speaker_names]
    prompt_texts = [speakers[sn]["prompt_text"] for sn in speaker_names]
    dialect_prompts = [speakers[sn].get("dialect_prompt", "") for sn in speaker_names]
    use_dialect = any(d for d in dialect_prompts)

    texts, spks = [], []
    for spk_name, text in turns:
        if not text.startswith(f"[{spk_name}]"):
            text = f"[{spk_name}]{text}"
        texts.append(text)
        spks.append(speaker_names.index(spk_name))

    return {
        "prompt_wavs": prompt_wavs,
        "prompt_texts": prompt_texts,
        "texts": texts,
        "spks": spks,
        "use_dialect": use_dialect,
        "dialect_prompts": dialect_prompts,
        "speaker_names": speaker_names,
    }


def normalize_text(text: str) -> str:
    text = re.sub(r"\[S[1-9]\]", "", text).strip()
    text = re.sub(r"<\|.*?\|>", "", text).strip()
    return text


# ── tokenizer helpers ──────────────────────────────────────────────────
class TextTokenizer:
    def __init__(self, model_path: str):
        _, self.tok = load_llm(model_path)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text)


# ── main input preparation ────────────────────────────────────────────
def prepare_inputs(script_json_path: str, model_path: str,
                   project_root: str | None = None,
                   output_dir: str | None = None) -> dict:
    import os as _os
    if project_root is None:
        project_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(
            _os.path.abspath(script_json_path))))
    if not _os.path.isdir(_os.path.join(project_root, "example")):
        project_root = _os.path.dirname(model_path)

    parsed = parse_podcast_script(script_json_path)
    tok = TextTokenizer(model_path)
    spk_model = CAMPPlusONNX(f"{model_path}/campplus.onnx")
    s3_model = S3TokenizerV2.from_pretrained("speech_tokenizer_v2_25hz")

    N = len(parsed["prompt_wavs"])
    prompt_log_mels = []
    prompt_mels_flow = []
    prompt_mel_lens_flow = []
    spk_embs = []
    prompt_text_tokens = []
    spk_ids = []

    for i in range(N):
        wav_path = parsed["prompt_wavs"][i]
        if not wav_path.startswith("/"):
            wav_path = _os.path.join(project_root, wav_path)

        audio_16k = volume_normalize(load_audio(wav_path, 16000))
        audio_24k = volume_normalize(load_audio(wav_path, 24000))

        # 16k log-mel for s3tokenizer → prompt speech tokens
        log_mel = log_mel_16k(audio_16k)
        prompt_log_mels.append(log_mel)

        # 24k mel for flow — shape (T, 80) to match SoulX
        mel_f = mel_24k(audio_24k)
        mel_f_np = np.array(mel_f)  # (80, T) → transpose to (T, 80)
        mel_f_np = mel_f_np.T
        if mel_f_np.shape[0] % 2 != 0:
            mel_f_np = mel_f_np[:-1]
        prompt_mels_flow.append(mel_f_np)
        prompt_mel_lens_flow.append(mel_f_np.shape[0])

        # Speaker embedding via CAMPPlus
        fbank = kaldi_fbank_np(audio_16k)
        fbank = fbank - fbank.mean(axis=0, keepdims=True)
        spk_emb = spk_model(fbank)
        spk_embs.append(spk_emb)

        # Prompt text tokens
        ptext = normalize_text(parsed["prompt_texts"][i])
        ptext = f"{SPK_DICT[i]}{TEXT_START}{ptext}{TEXT_END}{AUDIO_START}"
        if i == 0:
            ptext = f"{TASK_PODCAST}{ptext}"
        prompt_text_tokens.append(tok.encode(ptext))

    # Pad prompt log-mels
    max_log_mel_len = max(int(m.shape[1]) for m in prompt_log_mels)
    log_mel_padded = mx.zeros((N, 128, max_log_mel_len))
    log_mel_lens = []
    for i, m in enumerate(prompt_log_mels):
        L = int(m.shape[1])
        log_mel_padded[i, :, :L] = m
        log_mel_lens.append(L)

    # Stack prompt mels for flow — variable length, store as list
    # (forward_longform handles variable-length via prompt_mels_lens)

    # Dialog text tokens
    text_tokens = []
    for text, spk in zip(parsed["texts"], parsed["spks"]):
        t = normalize_text(text)
        t = f"{SPK_DICT[spk]}{TEXT_START}{t}{TEXT_END}{AUDIO_START}"
        text_tokens.append(tok.encode(t))
        spk_ids.append(spk)

    result = {
        "prompt_mels_for_llm": np.array(log_mel_padded),
        "prompt_mels_lens_for_llm": np.array(log_mel_lens, dtype=np.int32),
        "prompt_text_tokens_for_llm": prompt_text_tokens,
        "text_tokens_for_llm": text_tokens,
        "prompt_mels_for_flow": [np.array(m) for m in prompt_mels_flow],
        "prompt_mels_lens_for_flow": np.array(prompt_mel_lens_flow, dtype=np.int32),
        "spk_emb_for_flow": np.array(spk_embs, dtype=np.float32),
        "spk_ids": spk_ids,
        "use_dialect_prompt": parsed["use_dialect"],
        "dialect_prompts": parsed["dialect_prompts"],
    }

    if output_dir:
        for key, val in result.items():
            if isinstance(val, np.ndarray):
                np.savez_compressed(f"{output_dir}/{key}.npz", data=val)
            elif isinstance(val, list) and val and isinstance(val[0], np.ndarray):
                np.savez_compressed(f"{output_dir}/{key}.npz",
                                    **{str(i): v for i, v in enumerate(val)})
        print(f"Saved inputs to {output_dir}/")

    return result


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 3:
        print("Usage: python input_prep.py <script.json> <model_path> [output_dir]")
        _sys.exit(1)

    result = prepare_inputs(
        _sys.argv[1], _sys.argv[2],
        project_root=_sys.argv[3] if len(_sys.argv) > 3 else None,
        output_dir=_sys.argv[4] if len(_sys.argv) > 4 else None)
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape} {v.dtype}")
        elif isinstance(v, list):
            print(f"  {k}: list[{len(v)}]")
    print("Done.")
