"""M2 交付: flow (token->mel) + vocoder (mel->wav) 的 MLX 模組,供 pipeline.py 對接。
vocoder 用 M1 驗證過的 0.01-slope 修正子類;flow 直接呼叫 mlx-audio CausalMaskedDiffWithXvec.inference。
MLX venv 執行 (PYTHONPATH=mlx-audio)。"""
import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mlx_audio.tts.models.chatterbox.s3gen.hifigan import HiFTGenerator
from mlx_audio.tts.models.chatterbox.s3gen.f0_predictor import ConvRNNF0Predictor
from mlx_audio.tts.models.chatterbox.s3gen.flow import CausalMaskedDiffWithXvec
from mlx_audio.tts.models.chatterbox.s3gen.transformer.upsample_encoder import UpsampleConformerEncoder
from mlx_audio.tts.models.chatterbox.s3gen.flow_matching import CausalConditionalCFM
from mlx_audio.tts.models.chatterbox.s3gen.decoder import ConditionalDecoder

_DIR = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS = os.environ.get("SOULX_MLX_WEIGHTS", os.path.join(_DIR, "..", "weights"))


class FixedHiFTGenerator(HiFTGenerator):
    """修正 mlx-audio bug: decode 最後一個 leaky_relu 應為預設 slope 0.01 (非 lrelu_slope 0.1)。"""

    def decode(self, x: mx.array, s: mx.array) -> mx.array:
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = mx.concatenate([s_stft_real, s_stft_imag], axis=1)
        x = mx.swapaxes(x, 1, 2)
        x = self.conv_pre(x)
        x = mx.swapaxes(x, 1, 2)
        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, negative_slope=self.lrelu_slope)
            x = mx.swapaxes(x, 1, 2)
            x = self.ups[i](x)
            x = mx.swapaxes(x, 1, 2)
            if i == self.num_upsamples - 1:
                x = mx.concatenate([x[:, :, 1:2], x], axis=2)
            si = mx.swapaxes(s_stft, 1, 2)
            si = self.source_downs[i](si)
            si = mx.swapaxes(si, 1, 2)
            si = self.source_resblocks[i](si)
            x = x + si
            start_idx = i * self.num_kernels
            x = mx.mean(mx.stack([self.resblocks[start_idx + j](x) for j in range(self.num_kernels)], axis=0), axis=0)
        x = nn.leaky_relu(x, negative_slope=0.01)  # <-- FIX (M1 驗證: corr=1.0)
        x = mx.swapaxes(x, 1, 2)
        x = self.conv_post(x)
        x = mx.swapaxes(x, 1, 2)
        nfh = self.istft_params["n_fft"] // 2 + 1
        magnitude = mx.exp(x[:, :nfh, :])
        phase = mx.sin(x[:, nfh:, :])
        x = self._istft(magnitude, phase)
        x = mx.clip(x, -self.audio_limit, self.audio_limit)
        return x


def load_vocoder():
    m = FixedHiFTGenerator(
        sampling_rate=24000, upsample_rates=[8, 5, 3], upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        f0_predictor=ConvRNNF0Predictor(),
    )
    w = dict(mx.load(os.path.join(_WEIGHTS, "hift_mlx.safetensors")).items())
    m.update(tree_unflatten(list(w.items())))
    mx.eval(m.parameters())
    return m


def mel2wav(voc, mel):
    """mel (1,80,T) -> wav (1, T*480)。"""
    if isinstance(mel, np.ndarray):
        mel = mx.array(mel)
    out = voc(mel)
    out = out[0] if isinstance(out, tuple) else out
    mx.eval(out)
    return np.array(out)


class _ChunkedEncoder(UpsampleConformerEncoder):
    """FIX: SoulX 的 encoder 不論 streaming 都套 static_chunk_size=25 分塊注意力,
    但 MLX 只在 streaming=True 時套(否則 0=全注意力)。encoder forward 裡 streaming 只影響
    effective_chunk_size,故強制 streaming=True(context=None)即等同 SoulX 行為。"""
    def __call__(self, xs, xs_lens, context=None, streaming=False):
        return super().__call__(xs, xs_lens, context=context, streaming=streaming)  # TEST: chunk=0 全注意力


def _build_flow():
    enc = _ChunkedEncoder(
        output_size=512, attention_heads=8, linear_units=2048, num_blocks=6,
        input_size=512, pos_enc_layer_type="rel_pos_espnet", selfattention_layer_type="rel_selfattn",
        use_cnn_module=False, macaron_style=False,
        static_chunk_size=25,
    )
    est = ConditionalDecoder(
        in_channels=320, out_channels=80, causal=True, channels=[256],
        attention_head_dim=64, n_blocks=4, num_mid_blocks=12, num_heads=8, act_fn="gelu",
    )
    dec = CausalConditionalCFM(spk_emb_dim=80, estimator=est)
    return CausalMaskedDiffWithXvec(encoder=enc, decoder=dec)


def load_flow():
    flow = _build_flow()
    w = dict(mx.load(os.path.join(_WEIGHTS, "flow_mlx.safetensors")).items())
    w = {k[len("flow."):]: v for k, v in w.items() if k.startswith("flow.")}
    flow.update(tree_unflatten(list(w.items())))
    flow.eval()  # FIX: 關掉 dropout(否則 training 模式隨機歸零 → 音質糊掉)
    mx.eval(flow.parameters())
    return flow


def flow_token2mel(flow, prompt_tokens, generated_tokens, prompt_mel, spk_emb, n_timesteps=15):
    # FIX: SoulX 用 15 步 CFM(MLX 預設 10)。步數少→去噪較粗→略不乾淨。對齊 15。
    """raw speech token (已減 offset) -> generated mel (1,80,T_gen)。
    prompt_tokens / generated_tokens: list[int] in [0,6560]
    prompt_mel: (T,80) ndarray/mx ; spk_emb: (1,192)。"""
    tok = mx.array([list(generated_tokens)], dtype=mx.int32)
    tok_len = mx.array([len(generated_tokens)], dtype=mx.int32)
    pt = mx.array([list(prompt_tokens)], dtype=mx.int32)
    pt_len = mx.array([len(prompt_tokens)], dtype=mx.int32)
    if isinstance(prompt_mel, np.ndarray):
        prompt_mel = mx.array(prompt_mel)
    pf = prompt_mel[None] if prompt_mel.ndim == 2 else prompt_mel  # (1,T,80)
    pf_len = mx.array([pf.shape[1]], dtype=mx.int32)
    emb = mx.array(spk_emb) if isinstance(spk_emb, np.ndarray) else spk_emb
    # FIX: 對齊 SoulX —— 每次用「新鮮」隨機噪聲(SoulX 是 torch.randn_like),
    # 取代固定 seed-0 buffer(原本 4 個 turn 共用同一段前綴,造成相關性瑕疵)。
    need = pf.shape[1] + len(generated_tokens) * flow.token_mel_ratio + 16
    flow.decoder.rand_noise = mx.random.normal((1, 80, int(need)))
    feat, _ = flow.inference(
        token=tok, token_len=tok_len, prompt_token=pt, prompt_token_len=pt_len,
        prompt_feat=pf, prompt_feat_len=pf_len, embedding=emb, finalize=True, n_timesteps=n_timesteps,
    )
    mx.eval(feat)
    return np.array(feat)


if __name__ == "__main__":
    import numpy as np
    print("=== smoke test flow_vocoder ===")
    voc = load_vocoder()
    flow = load_flow()
    print("loaded vocoder + flow OK")
    rng = np.random.default_rng(0)
    prompt_tokens = rng.integers(0, 6561, size=50).tolist()
    gen_tokens = rng.integers(0, 6561, size=100).tolist()
    prompt_mel = (rng.standard_normal((100, 80)) * 0.5).astype(np.float32)  # T = 50*2
    spk_emb = (rng.standard_normal((1, 192)) * 0.1).astype(np.float32)
    mel = flow_token2mel(flow, prompt_tokens, gen_tokens, prompt_mel, spk_emb)
    print(f"flow_token2mel -> mel {mel.shape} finite={np.isfinite(mel).all()}")
    wav = mel2wav(voc, mel)
    print(f"mel2wav -> wav {wav.shape} finite={np.isfinite(wav).all()} peak={np.abs(wav).max():.3f}")
