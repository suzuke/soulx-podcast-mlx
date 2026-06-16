"""M5 — forward_longform orchestration pipeline (llm → flow → vocoder)."""
import sys
import time
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("MLX_AUDIO_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mlx-audio")))

from mlx_audio.codec.models.s3 import S3TokenizerV2
from llm_stage import MLXLLMEngine, SamplingParams, SPEECH_TOKEN_OFFSET, SPEECH_EOS
from s3_tokenizer_stage import S3TokenizerMLX

from flow_vocoder import load_flow, load_vocoder, flow_token2mel, mel2wav


# ── helpers ────────────────────────────────────────────────────────────
def _align_speech_to_mel(prompt_speech_tokens, prompt_speech_lens,
                         prompt_mels_flow, prompt_mels_lens_flow):
    """Align speech tokens with flow mel, matching forward_longform logic."""
    prompt_size = len(prompt_speech_tokens)
    aligned_speech = []
    aligned_mels = []
    aligned_mel_lens = []

    for i in range(prompt_size):
        st = prompt_speech_tokens[i]
        st_len = int(prompt_speech_lens[i])
        st = st[:st_len]

        mel = prompt_mels_flow[i]
        mel_len = int(prompt_mels_lens_flow[i])

        if st_len * 2 > mel_len:
            st = st[:int(mel_len / 2)]
            aligned_mel_len = mel_len
        else:
            mel = mel[:st_len * 2]
            aligned_mel_len = st_len * 2

        aligned_speech.append(st)
        aligned_mels.append(mel)
        aligned_mel_lens.append(aligned_mel_len)

    return aligned_speech, aligned_mels, aligned_mel_lens


def _build_prompt_inputs(prompt_speech_aligned, prompt_text_tokens):
    """Build prompt inputs: text_tokens + speech_tokens(offset) + EOS."""
    prompt_inputs = []
    history_inputs = []
    for i in range(len(prompt_speech_aligned)):
        st = prompt_speech_aligned[i]
        speech_with_offset = [int(t) + SPEECH_TOKEN_OFFSET for t in st] + [SPEECH_EOS]
        combined = prompt_text_tokens[i] + speech_with_offset
        prompt_inputs.append(combined)
        history_inputs.append(combined)
    return prompt_inputs, history_inputs


def _reset_history(history_inputs, prompt_inputs, valid_turn_size,
                   prompt_context, history_context, history_text_context):
    """Reset KV cache with truncated history window."""
    prompt_text_bound = max(
        prompt_context,
        len(history_inputs) - history_text_context - history_context,
    )
    inputs = []
    for item in (
        history_inputs[:prompt_context]
        + history_inputs[prompt_text_bound:-history_context]
        + prompt_inputs[-history_context:]
    ):
        inputs.extend(item)
    valid_turn_size = prompt_context + len(history_inputs) - prompt_text_bound
    return inputs, valid_turn_size


# ── main pipeline ──────────────────────────────────────────────────────
def run_pipeline(data: dict, model_path: str, output_path: str,
                 sampling_params: SamplingParams | None = None) -> str:
    if sampling_params is None:
        sampling_params = SamplingParams()

    # Load models
    llm = MLXLLMEngine(model_path)
    s3 = S3TokenizerMLX()
    flow = load_flow()
    voc = load_vocoder()

    prompt_size = len(data["prompt_text_tokens_for_llm"])
    turn_size = len(data["text_tokens_for_llm"])

    # ── 1. Audio tokenization (prompt audio → speech tokens) ──────
    prompt_log_mels = data["prompt_mels_for_llm"]
    prompt_log_mel_lens = data["prompt_mels_lens_for_llm"]

    prompt_speech_tokens = []
    prompt_speech_lens = []
    for i in range(prompt_size):
        mel_np = np.array(prompt_log_mels[i, :, :int(prompt_log_mel_lens[i])])
        import mlx.core as mx
        mel_batch = mx.array(mel_np)[None, ...]
        mel_len = mx.array([mel_batch.shape[2]], dtype=mx.int32)
        codes, code_lens = s3.model.quantize(mel_batch, mel_len)
        st = np.array(codes[0, :int(code_lens[0].item())])
        prompt_speech_tokens.append(st.astype(np.int32))
        prompt_speech_lens.append(len(st))

    # ── 2. Align speech tokens with flow mel ──────────────────────
    prompt_speech_aligned, prompt_mels_aligned, prompt_mel_lens_aligned = \
        _align_speech_to_mel(
            prompt_speech_tokens, prompt_speech_lens,
            data["prompt_mels_for_flow"], data["prompt_mels_lens_for_flow"])

    # ── 3. Build prompt inputs for LLM ────────────────────────────
    prompt_inputs_list, history_inputs = _build_prompt_inputs(
        prompt_speech_aligned, data["prompt_text_tokens_for_llm"])

    # ── 4. Multi-turn generation loop ─────────────────────────────
    inputs = []
    for pi in prompt_inputs_list:
        inputs.extend(pi)

    valid_turn_size = prompt_size
    generated_wavs = []

    for i in range(turn_size):
        # Cache reset check
        if (valid_turn_size > 10 or len(inputs) > 6192):
            inputs, valid_turn_size = _reset_history(
                history_inputs, prompt_inputs_list, valid_turn_size, 2, 2, 2)

        valid_turn_size += 1

        # Append this turn's text tokens
        turn_text = data["text_tokens_for_llm"][i]
        inputs.extend(turn_text)

        # LLM generate
        t0 = time.time()
        llm_out = llm.generate(inputs, params=sampling_params)
        token_ids = llm_out["token_ids"]

        inputs.extend(token_ids)
        prompt_inputs_list.append(turn_text + token_ids)
        history_inputs.append(turn_text[:-1])

        # Extract raw speech tokens (subtract offset, drop eos)
        raw_speech = [t - SPEECH_TOKEN_OFFSET for t in token_ids]
        if raw_speech and raw_speech[-1] == SPEECH_EOS - SPEECH_TOKEN_OFFSET:
            raw_speech = raw_speech[:-1]

        # Flow input: prompt speech (aligned) + generated speech (separate)
        spk = data["spk_ids"][i]
        prompt_st = [int(t) for t in prompt_speech_aligned[spk].tolist()]

        prompt_mel = np.array(prompt_mels_aligned[spk])
        spk_emb = data["spk_emb_for_flow"][spk:spk + 1]

        # Flow → mel (returns generated portion only)
        gen_mel = flow_token2mel(flow, prompt_st, raw_speech,
                                 prompt_mel, spk_emb)

        # Vocoder → wav
        wav = mel2wav(voc, np.array(gen_mel))
        generated_wavs.append(wav)

        elapsed = time.time() - t0
        print(f"  turn {i+1}/{turn_size}: {len(token_ids)} tokens, "
              f"prompt={len(prompt_st)}+gen={len(raw_speech)}, {elapsed:.1f}s")

    # ── 5. Concatenate and save ───────────────────────────────────
    all_wavs = []
    for w in generated_wavs:
        w = np.array(w).flatten()
        all_wavs.append(w)
    full_wav = np.concatenate(all_wavs)

    import soundfile as sf
    sf.write(output_path, full_wav.astype(np.float32), 24000)
    print(f"Saved → {output_path} ({len(full_wav)/24000:.1f}s)")

    return output_path


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 4:
        print("Usage: python pipeline.py <script.json> <model_path> <output.wav>")
        print("  Optional env: PROJECT_ROOT (default=auto-detect)")
        _sys.exit(1)

    import os as _os
    project_root = _os.environ.get("PROJECT_ROOT",
                                    _os.path.dirname(_os.path.dirname(
                                        _os.path.abspath(_sys.argv[1]))))

    from input_prep import prepare_inputs
    print("Preparing inputs...")
    data = prepare_inputs(_sys.argv[1], _sys.argv[2],
                          project_root=project_root)

    print(f"Pipeline: {len(data['text_tokens_for_llm'])} turns, "
          f"{len(data['prompt_text_tokens_for_llm'])} speakers")

    run_pipeline(data, _sys.argv[2], _sys.argv[3])
