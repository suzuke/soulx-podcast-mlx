# 開發紀錄 — SoulX-Podcast 移植到 MLX (Apple Silicon) 全程

本文如實記錄把 SoulX-Podcast 從 PyTorch/CUDA 移植成 Apple Silicon 原生 MLX 的完整過程:策略決定、逐元件移植、遇到的 **7 個 bug** 以及每一個是怎麼被定位與解決的。

---

## 0. 動機與策略

SoulX-Podcast 官方寫死 CUDA。在 Mac 上只能跑 PyTorch-MPS + CPU fallback,RTF 約 2.7(比即時慢)。目標:**全 MLX 原生、比即時快**。

**關鍵策略決定**:不要「從零手刻」。SoulX 的架構是 `LLM + Flow Matching + HiFi-GAN/iSTFT`,屬 **CosyVoice2 血統**;而 [mlx-audio](https://github.com/Blaizzy/mlx-audio) 的 Chatterbox `s3gen` 模組就是同源的 MLX 實作,連 config 都一字不差。所以工作是:**站在 mlx-audio 上做「權重轉換 + glue」**,把原本估計「數天~數週從零」縮成「移植 + 除錯」。

驗證方法論貫穿全程:**逐層 adversarial A/B** —— 把每個階段的中間張量(權重、encoder 輸出、mel、波形)和 PyTorch 參考逐一對比,直到把分歧縮小到單一函式/單一張量。

開發以 pair coding 進行:一個 OpenCode 實例負責 LLM / s3 tokenizer / 輸入準備 / 編排骨架,主實例負責 flow / vocoder 的權重轉換與最終音質除錯。

---

## 1. 元件移植與驗證

| 階段 | 來源 | 驗證結果 |
|---|---|---|
| s3 tokenizer(prompt 音→speech token) | mlx-audio `S3TokenizerV2`(`mlx-community/CosyVoice2-0.5B-S3Tokenizer`) | 模型 bit-exact 251/251 |
| LLM(text→speech token) | `mlx-lm` Qwen3-1.7B + 自訂 RAS 取樣 | 單步 logits cos 1.0 / 0.999(bf16) |
| Flow(speech token→mel) | mlx-audio `CausalMaskedDiffWithXvec` | encoder mu corr 1.0、matched-noise mel corr 0.999 |
| Vocoder(mel→波形) | mlx-audio `HiFTGenerator` | bit-exact corr 1.0 |
| 輸入準備(CAMPPlus、雙 mel) | onnxruntime CAMPPlus + MLX mel 前端 | spk-emb bit-exact、prompt-mel corr 1.0 |

權重轉換的通用手法(weight_norm 合併 `w=g·v/‖v‖`、Conv1d `(out,in,k)→(out,k,in)`、ConvTranspose `(in,out,k)→(out,k,in)`、命名 remap)先在 vocoder 上以 bit-exact 驗證成功,再套用到 flow。

---

## 2. 七個 bug 與解法

下面每個 bug 都是「先出現症狀 → 逐層 A/B 定位 → 找到根因 → 修正 → 重新驗證」。

### Bug 1 — flow 跑在 training 模式,dropout 沒關
- **症狀**:第一版全 MLX 輸出完全糊掉(雖然能跑、RTF 0.76)。
- **定位**:逐層 A/B 把 flow 的 `mu`(encoder 輸出)對 PyTorch → corr 0.09(近隨機);再往內推到 `embed` 輸出,印出實際數值發現**某些位置變 0、其餘 ×1.111**(= 1/(1−0.1))。
- **根因**:MLX `nn.Module` 預設 `training=True`,`EspnetRelPositionalEncoding`/`LinearNoSubsampling` 的 dropout 會隨機歸零;PyTorch 端是 `eval()`。
- **修正**:`load_flow()` 載完權重後呼叫 `flow.eval()`。

### Bug 2 — 非連續 array 存檔損壞 conv 權重
- **症狀**:修了 dropout 後,`mu` 仍 corr ≈ 0(但變成確定性的錯)。
- **定位**:逐層往 encoder 內推 → `pre_lookahead_layer` → 單一 `conv1`。比對發現 conv1 權重和「PyTorch 權重的任何轉置」都不相關(corr ≈ 0)。但**獨立呼叫 sanitize 時 conv1 是對的(corr 1.0)**——矛盾。在存檔前 instrument 印出,發現 `got` 在記憶體中正確、但寫進 safetensors 再讀回就錯。
- **根因**:MLX 的 `mx.swapaxes/transpose` 產生**非連續 view**;`safetensors.save_file(np.array(view))` 會依底層 strides 存出錯誤資料。
- **修正**:存檔前 `np.ascontiguousarray(...)`(vocoder 的轉換器一開始就有用、flow 轉換器漏了)。修後 `mu` corr 1.0、matched-noise 完整 mel corr 0.999。

> 過程中也排除了兩個「以為是、其實不是」的假設:`static_chunk_size`(SoulX 預設 25 但離線路徑等同全注意力,加了沒差)、Snake `alpha_logscale`(兩邊都是 False)。adversarial A/B 的價值就在於用數據否決猜測。

### Bug 3 — LLM 多套了 temperature/top_k/top_p
- **症狀**:flow 修好後音訊「可分辨英文但越後面越吵」;端到端比 PyTorch **多生成 ~65% tokens** 且退化成噪聲。
- **定位**:元件都 bit-exact、SamplingParams 兩邊一致 → 問題在取樣**實作**。比對 SoulX 的 `_ras_sample_hf_engine`。
- **根因**:SoulX 的 HF/RAS 取樣路徑**只套 repetition_penalty**,完全不套 temperature/top_k/top_p(那些只在 vLLM 路徑用,雖然 SamplingParams 有定義)。MLX 多套了 temp 0.6 + top_k + top_p → 分布變尖 → 自回歸退化。
- **修正**:MLX 生成迴圈移除 temp/top_k/top_p,只保留 rep_penalty + RAS,對齊 SoulX。時長 58.8s → 33.3s。

### Bug 4 — flow prompt mel 前端三處錯
- **症狀**:長度修好但 rms 仍高(0.58)、很吵。
- **定位**:flow 的數學是對的(matched-noise mel corr 0.999,但用的是隨機 prompt mel)。改用**真實** prompt mel 做 A/B → corr 0.81、max_diff 13。
- **根因**:`input_prep` 的 24k flow mel 與 SoulX `mel_spectrogram` 有三處不一致:(1) 用**功率** `|X|²` 而非**幅度** `|X|`;(2) mel basis `norm=None` 而非 librosa 預設的 `slaney`;(3) STFT 前缺 `(n_fft-hop)/2=720` 的 reflect padding。
- **修正**:三處對齊 → prompt mel corr 0.81 → 1.0(僅剩常數 offset)。

### Bug 5 — volume_normalize 演算法錯
- **症狀**:修完 mel 後 corr 1.0 但還有 ~0.9 的 log 常數 offset → 音量過大 + 沙沙聲。
- **定位**:filter bank、STFT 幅度逐一比對都 bit-exact → offset 來自**音訊前處理**。
- **根因**:`input_prep.volume_normalize` 用 `audio/peak*0.9`(峰值正規化),但 SoulX `audio_volume_normalize` 是 **percentile 正規化到 coeff=0.1**(排序絕對值、取 top10%~1% 的平均當基準)。兩者音量差約 2.5×。
- **修正**:精確移植 SoulX 的演算法 → prompt mel **bit-exact(corr 1.0、mean 完全對上)**;rms 0.58 → 0.044(對齊 PyTorch)。

### Bug 6 — CFM ODE 步數預設 10(應為 15)
- **症狀**:中文輸出「乾淨多了但 PyTorch 還是略好」。
- **根因**:mlx-audio 的 flow `n_timesteps` 預設 10;SoulX 寫死 15。步數少 → ODE 去噪較粗。
- **修正**:`flow_token2mel` 預設 `n_timesteps=15`。

### Bug 7 — CFM 共用固定噪聲 buffer(應每次 fresh)
- **症狀**:殘餘的「像麥克風噪聲」瑕疵,PyTorch 略乾淨。
- **根因**:mlx-audio CFM 用一個 seed-0 的固定 `rand_noise` buffer,且多輪共用同一段前綴;SoulX 每次用 fresh `torch.randn_like`。共用固定噪聲造成跨 turn 的相關性瑕疵。
- **修正**:`flow_token2mel` 每次呼叫產生新鮮 `mx.random.normal` 噪聲。rms 0.051 ≈ PyTorch 0.052,耳測達到同等品質。

---

## 3. 最終結果

- **品質**:中文 + 英文耳測與 PyTorch SoulX 同級;統計(rms/peak/silence)對齊乾淨參考。
- **速度**:RTF ~0.55(比即時快近 2 倍),對比 PyTorch-MPS 的 ~2.7 約**快 5 倍**。
- **可重現性**:每個元件對 PyTorch bit-exact / bf16 噪聲內。

## 4. 心得

1. **先找同源實作**:把「從零移植」變成「權重轉換 + glue」是最大的省力點。
2. **逐層 A/B 是定位利器**:5 個 bug 散落在 dropout、序列化、取樣、mel 前端、音量正規化——靠把中間張量逐層對比才能一個個揪出,而不是瞎猜。
3. **TTS 的魔鬼在前處理**:flow/vocoder 的權重對了,真正難纏的是 mel 前端與音量正規化這些「不在模型裡」的細節。
4. **序列化陷阱**:框架間搬權重,非連續 view + safetensors 是經典坑(`np.ascontiguousarray`)。
