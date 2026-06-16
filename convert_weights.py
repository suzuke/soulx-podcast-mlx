"""Convert SoulX-Podcast PyTorch weights (flow.pt, hift.pt) → MLX safetensors.

One-time setup step. Requires: torch, mlx, mlx-audio (on MLX_AUDIO_PATH / PYTHONPATH), safetensors.

Usage:
    python convert_weights.py <soulx_model_dir> [out_weights_dir]

e.g.  python convert_weights.py pretrained_models/SoulX-Podcast-1.7B weights
"""
import os
import re
import sys
import argparse
import numpy as np

# allow `from mlx_audio...` via MLX_AUDIO_PATH
_mlxa = os.environ.get("MLX_AUDIO_PATH")
if _mlxa:
    sys.path.insert(0, _mlxa)


def convert_hift(model_dir: str, out_dir: str) -> None:
    """Vocoder (HiFTGenerator). torch only. Handles weight_norm merge + conv transpose + condnet reindex."""
    import torch
    from safetensors.numpy import save_file

    sd = torch.load(f"{model_dir}/hift.pt", map_location="cpu", weights_only=True)

    # merge weight_norm (w = g * v / ||v||)
    merged, handled = {}, set()
    for k in sd:
        if k.endswith(".parametrizations.weight.original0"):
            base = k[: -len(".parametrizations.weight.original0")]
            g = sd[base + ".parametrizations.weight.original0"].float().numpy()
            v = sd[base + ".parametrizations.weight.original1"].float().numpy()
            norm = np.sqrt((v ** 2).sum(axis=tuple(range(1, v.ndim)), keepdims=True))
            merged[base + ".weight"] = g * v / norm
            handled.add(base + ".parametrizations.weight.original0")
            handled.add(base + ".parametrizations.weight.original1")
    for k in sd:
        if k not in handled:
            merged[k] = sd[k].float().numpy()

    # condnet index //2 + conv layout transpose
    out = {}
    for k, w in merged.items():
        nk = k
        m = re.match(r"(f0_predictor\.condnet\.)(\d+)(\..*)", nk)
        if m:
            nk = f"{m.group(1)}{int(m.group(2)) // 2}{m.group(3)}"
        if nk.endswith(".weight") and w.ndim == 3:
            if re.match(r"ups\.\d+\.weight$", nk):       # ConvTranspose1d (in,out,k)->(out,k,in)
                w = np.transpose(w, (1, 2, 0))
            else:                                         # Conv1d (out,in,k)->(out,k,in)
                w = np.transpose(w, (0, 2, 1))
        out[nk] = np.ascontiguousarray(w.astype(np.float32))  # contiguous! (non-contig view saves wrong strides)

    save_file(out, f"{out_dir}/hift_mlx.safetensors")
    print(f"  hift_mlx.safetensors: {len(out)} tensors")


def convert_flow(model_dir: str, out_dir: str) -> None:
    """Flow (CausalMaskedDiffWithXvec). torch (read) + mlx-audio sanitize (weight_norm/transpose/rename)."""
    import torch
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from safetensors.numpy import save_file
    from mlx_audio.tts.models.chatterbox.s3gen.s3gen import S3Token2Wav

    sd = torch.load(f"{model_dir}/flow.pt", map_location="cpu", weights_only=True)
    raw = {f"flow.{k}": mx.array(v.float().contiguous().numpy()) for k, v in sd.items()}

    # conformer encoder index rename (sanitize covers estimator, not encoders.N)
    def rn(k):
        k = re.sub(r"\.up_encoders\.(\d+)\.", r".up_encoders_\1.", k)
        k = re.sub(r"(?<!up_)\.encoders\.(\d+)\.", r".encoders_\1.", k)
        return k
    raw = {rn(k): v for k, v in raw.items()}

    model = S3Token2Wav()
    san = model.sanitize(raw)  # merges weight_norm, transposes convs, renames estimator/embed/attn
    exp = {k for k, _ in tree_flatten(model.parameters()) if k.startswith("flow.")}
    got = {k: v for k, v in san.items() if k in exp}
    missing = exp - set(got) - {
        "flow.decoder.rand_noise", "flow.encoder.embed.pos_enc.pe", "flow.encoder.up_embed.pos_enc.pe",
    }
    if missing:
        print(f"  WARNING: {len(missing)} flow params unmatched: {sorted(missing)[:5]} ...")

    out = {k: np.ascontiguousarray(np.array(v)) for k, v in got.items()}  # contiguous!
    save_file(out, f"{out_dir}/flow_mlx.safetensors")
    print(f"  flow_mlx.safetensors: {len(out)} tensors")


def main():
    ap = argparse.ArgumentParser(description="Convert SoulX-Podcast PyTorch weights to MLX.")
    ap.add_argument("model_dir", help="SoulX-Podcast-1.7B directory (with flow.pt, hift.pt)")
    ap.add_argument("out_dir", nargs="?", default="weights", help="output dir (default: weights/)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Converting {args.model_dir} → {args.out_dir}/")
    convert_hift(args.model_dir, args.out_dir)
    convert_flow(args.model_dir, args.out_dir)
    print("Done. (the Qwen3 LLM safetensors are loaded directly from the model dir — no conversion needed)")


if __name__ == "__main__":
    main()
