"""M3 — S3 Tokenizer: prompt audio → speech tokens (MLX port)."""
import sys
import numpy as np
import mlx.core as mx

import os as _os
sys.path.insert(0, _os.environ.get("MLX_AUDIO_PATH", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "mlx-audio")))
from mlx_audio.codec.models.s3 import S3TokenizerV2
from mlx_audio.dsp import stft, hanning

_MEL_FILTERS_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "assets", "mel_filters.npz")


def _load_mel_filters(n_mels: int = 128) -> mx.array:
    return mx.array(np.load(_MEL_FILTERS_PATH)[f"mel_{n_mels}"])


def log_mel_spectrogram(audio: np.ndarray, sample_rate: int = 16000,
                         n_mels: int = 128, n_fft: int = 400,
                         hop_length: int = 160) -> mx.array:
    audio_mx = mx.array(audio)
    window = hanning(n_fft + 1)[:-1]
    freqs = stft(audio_mx, n_fft=n_fft, hop_length=hop_length,
                 window=window, center=True).swapaxes(0, 1)
    freqs = freqs[:, :-1]
    mags = mx.abs(freqs) ** 2
    filters = _load_mel_filters(n_mels)
    mel = filters @ mags
    log_spec = mx.maximum(mel, 1e-10).log10()
    log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class S3TokenizerMLX:
    def __init__(self, model_name: str = "speech_tokenizer_v2_25hz"):
        self.model = S3TokenizerV2.from_pretrained(model_name)

    def load_audio(self, path: str, sr: int = 16000) -> np.ndarray:
        import librosa
        audio, _ = librosa.load(path, sr=sr, mono=True, dtype=np.float32)
        return audio

    def quantize(self, audio: np.ndarray) -> np.ndarray:
        mel = log_mel_spectrogram(audio)
        mel_batch = mel[None, ...]
        mel_len = mx.array([mel.shape[1]], dtype=mx.int32)
        codes, code_lens = self.model.quantize(mel_batch, mel_len)
        codes_np = np.array(codes[0, :int(code_lens[0].item())])
        return codes_np

    def quantize_from_mel(self, mel: np.ndarray) -> np.ndarray:
        mel_batch = mx.array(mel)[None, ...]
        mel_len = mx.array([mel.shape[1]], dtype=mx.int32)
        codes, code_lens = self.model.quantize(mel_batch, mel_len)
        codes_np = np.array(codes[0, :int(code_lens[0].item())])
        return codes_np


def compare_tokens(a: np.ndarray, b: np.ndarray, name: str = "") -> dict:
    min_len = min(len(a), len(b))
    match = int(np.sum(a[:min_len] == b[:min_len]))
    return {
        "name": name,
        "len_a": len(a), "len_b": len(b),
        "match": match, "total": min_len,
        "accuracy": match / min_len * 100 if min_len > 0 else 0,
    }


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage:")
        print("  python s3_tokenizer_stage.py <audio.wav> [--ref ref.npz]")
        print("  python s3_tokenizer_stage.py --mel-ref ref.npz")
        _sys.exit(1)

    tok = S3TokenizerMLX()

    if _sys.argv[1] == "--mel-ref":
        ref = np.load(_sys.argv[2])
        mel_pt = ref["mel"]
        pt_tokens = ref["tokens"]
        codes = tok.quantize_from_mel(mel_pt)
        result = compare_tokens(codes, pt_tokens, "MLX(PyTorch mel) vs PyTorch")
        for k, v in result.items():
            print(f"  {k}: {v}")
        _sys.exit(0)

    audio_path = _sys.argv[1]
    print(f"Loading audio: {audio_path}")
    audio = tok.load_audio(audio_path)
    print(f"  Audio: {audio.shape}, {len(audio)/16000:.1f}s")

    print("Quantizing...")
    codes = tok.quantize(audio)
    print(f"  Tokens: {len(codes)}, range=[{codes.min()},{codes.max()}]")

    ref_path = None
    if len(_sys.argv) >= 4 and _sys.argv[2] == "--ref":
        ref_path = _sys.argv[3]

    if ref_path:
        ref = np.load(ref_path)
        result = compare_tokens(codes, ref["tokens"], "MLX vs PyTorch")
        print()
        for k, v in result.items():
            print(f"  {k}: {v}")

    out_path = audio_path.rsplit(".", 1)[0] + "_s3_tokens.npz"
    np.savez_compressed(out_path, tokens=codes)
    print(f"\nSaved tokens → {out_path}")
