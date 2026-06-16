"""MLX LLM Stage — Qwen3-1.7B + RAS sampling for SoulX-Podcast M4."""
import mlx.core as mx
import numpy as np
from dataclasses import dataclass
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import apply_top_k, apply_top_p


@dataclass
class SamplingParams:
    temperature: float = 0.6
    repetition_penalty: float = 1.25
    top_k: int = 100
    top_p: float = 0.9
    min_tokens: int = 8
    max_tokens: int = 3000
    use_ras: bool = True
    win_size: int = 25
    tau_r: float = 0.2


SPEECH_TOKEN_OFFSET = 152927
SPEECH_EOS = 151675


class MLXLLMEngine:
    def __init__(self, model_path: str, eos_token_id: int = SPEECH_EOS):
        self.model, self.tokenizer = load(model_path)
        self.eos_token_id = eos_token_id

    def generate_speech_tokens(
        self,
        text_tokens: list[int],
        prompt_speech_tokens: list[int],
        params: SamplingParams | None = None,
    ) -> np.ndarray:
        prompt = list(text_tokens)
        for t in prompt_speech_tokens:
            prompt.append(t + SPEECH_TOKEN_OFFSET)
        prompt.append(SPEECH_EOS)

        result = self.generate(prompt, params)
        generated_ids = result["token_ids"]
        if generated_ids and generated_ids[-1] == SPEECH_EOS:
            generated_ids = generated_ids[:-1]
        return np.array([t - SPEECH_TOKEN_OFFSET for t in generated_ids], dtype=np.int32)

    def get_logits(self, input_ids: list[int]) -> np.ndarray:
        x = mx.array([input_ids])
        logits = self.model(x)
        mx.eval(logits)
        return np.array(logits[0, -1, :].astype(mx.float32))

    def generate(
        self,
        prompt: list[int],
        params: SamplingParams | None = None,
        greedy: bool = False,
    ) -> dict:
        if params is None:
            params = SamplingParams()

        prompt_len = len(prompt)
        ids = list(prompt)
        cache = make_prompt_cache(self.model)

        x = mx.array([ids])
        logits = self.model(x, cache=cache)

        generated = []
        for _step in range(params.max_tokens):
            raw_logits = logits[0, -1, :].astype(mx.float32)

            # FIX: 對齊 SoulX _ras_sample_hf_engine —— HF/RAS 路徑只套 rep_penalty,
            # 不套 temperature/top_k/top_p(那些只在 vLLM 路徑用)。多套會讓分布變尖 → 重複退化 → 噪聲。
            scores = self._rep_penalty(raw_logits, ids, prompt_len, params.repetition_penalty)

            if params.use_ras and not greedy:
                scores = self._ras_check(raw_logits, scores, ids, params.win_size, params.tau_r)

            mx.eval(scores)

            if greedy:
                next_token = mx.argmax(scores).item()
            else:
                next_token = mx.random.categorical(scores).item()  # categorical 內含 softmax

            ids.append(next_token)
            generated.append(next_token)

            if next_token == self.eos_token_id and len(generated) >= params.min_tokens:
                break

            x = mx.array([[next_token]])
            logits = self.model(x, cache=cache)
            mx.eval(logits)

        return {
            "text": self.tokenizer.decode(generated),
            "token_ids": generated,
        }

    def _rep_penalty(self, logits, ids, prompt_len, penalty):
        if penalty == 1.0 or len(ids) <= prompt_len:
            return logits
        arr = np.array(logits)
        for t in set(ids[prompt_len:]):
            if arr[t] < 0:
                arr[t] *= penalty
            else:
                arr[t] /= penalty
        return mx.array(arr)

    def _ras_check(self, raw_logits, penalized_scores, ids, win_size, tau_r):
        # 對齊 SoulX: candidate 從 rep-penalized 分布取樣;若在最後 win_size 視窗重複次數+1 >= win*tau_r,
        # 退回 raw logits(去掉 rep penalty)。categorical 內含 softmax,傳 logits 即可。
        candidate = mx.random.categorical(penalized_scores).item()
        rep = ids[-win_size:].count(candidate) + 1
        if rep >= win_size * tau_r:
            return raw_logits
        return penalized_scores


def save_tensor(path: str, tensor: np.ndarray, name: str = "data") -> None:
    np.savez_compressed(path, **{name: tensor})


def load_tensor(path: str, name: str = "data") -> np.ndarray:
    return np.load(path)[name]


def compare_logits(
    mlx_logits: np.ndarray,
    hf_logits: np.ndarray,
    tol: float = 1e-3,
) -> dict:
    diff = mlx_logits - hf_logits
    mae = float(np.mean(np.abs(diff)))
    max_diff = float(np.max(np.abs(diff)))
    cos_sim = float(np.dot(mlx_logits, hf_logits) / (
        np.linalg.norm(mlx_logits) * np.linalg.norm(hf_logits)
    ))
    match_pct = float(np.mean(np.abs(diff) < tol) * 100)
    return {
        "mae": mae,
        "max_diff": max_diff,
        "cosine_sim": cos_sim,
        "match_%": match_pct,
        "pass": cos_sim > 0.999 and mae < 0.5,
    }


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else (
        "../SoulX-Podcast/pretrained_models/SoulX-Podcast-1.7B/"
    )
    engine = MLXLLMEngine(model_path)

    # Dump mode: save MLX logits to .npz for given input_ids
    if len(sys.argv) >= 4 and sys.argv[2] == "dump":
        input_npz = sys.argv[3]
        output_npz = sys.argv[4] if len(sys.argv) > 4 else "mlx_logits.npz"
        input_ids = load_tensor(input_npz, "input_ids").tolist()
        logits = engine.get_logits(input_ids)
        save_tensor(output_npz, logits, "logits")
        print(f"Saved MLX logits ({logits.shape}) → {output_npz}")

    # Compare mode: compare MLX vs HF logits
    elif len(sys.argv) >= 5 and sys.argv[2] == "compare":
        input_npz = sys.argv[3]
        hf_npz = sys.argv[4]
        input_ids = load_tensor(input_npz, "input_ids").tolist()
        hf_logits = load_tensor(hf_npz, "logits")
        mlx_logits = engine.get_logits(input_ids)
        result = compare_logits(mlx_logits, hf_logits)
        for k, v in result.items():
            print(f"  {k}: {v}")

    # Generate mode: generate from a prompt file or inline
    elif len(sys.argv) >= 3 and sys.argv[2] == "gen":
        if len(sys.argv) >= 4:
            input_ids = load_tensor(sys.argv[3], "input_ids").tolist()
        else:
            input_ids = [151643, 151644, 151667]
        result = engine.generate(input_ids)
        print(f"Generated {len(result['token_ids'])} tokens")
        print(f"IDs: {result['token_ids']}")
        print(f"Text: {result['text']}")

    # Speech mode: text_tokens + prompt_speech_tokens → raw speech tokens (.npz)
    elif len(sys.argv) >= 5 and sys.argv[2] == "speech":
        text_tokens = load_tensor(sys.argv[3], "text_tokens").tolist()
        prompt_speech = load_tensor(sys.argv[4], "prompt_speech").tolist()
        out_path = sys.argv[5] if len(sys.argv) > 5 else "speech_tokens.npz"
        raw_tokens = engine.generate_speech_tokens(text_tokens, prompt_speech)
        save_tensor(out_path, raw_tokens, "speech_tokens")
        print(f"Generated {len(raw_tokens)} raw speech tokens → {out_path}")

    else:
        print("Usage:")
        print("  python llm_stage.py <model_path> dump <input.npz> [output.npz]")
        print("  python llm_stage.py <model_path> compare <input.npz> <hf_logits.npz>")
        print("  python llm_stage.py <model_path> gen [prompt.npz]")
        print("  python llm_stage.py <model_path> speech <text.npz> <prompt_speech.npz> [out.npz]")
