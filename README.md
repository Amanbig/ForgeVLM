# ForgeVLM

A from-scratch path from attention to modern multimodal foundation models: image understanding, video understanding, and multimodal generation. Each file is one idea. Read them in order, then run the training scripts. They memorize one random or synthetic batch. A falling loss means the wiring is right.

Run everything from the repo root.

```bash
pip install -r requirements.txt
python -m train.train_clip
python -m train.train_vlm
python -m train.train_modern_vlm
python -m train.train_video
python -m train.train_vqvae
python -m train.train_generation
```

CPU is enough. These models are tiny on purpose.

## Map

| Order | File | What it is |
| --- | --- | --- |
| 1 | `attention/self_attention.py` | One attention head, causal mask, projection |
| 2 | `attention/multi_head_attention.py` | Split into heads, supports self & cross-attention masks |
| 3 | `attention/rope.py` | 1D, 2D (spatial), and 3D (spatio-temporal) Rotary Position Embeddings |
| 4 | `transformer/transformer.py` | Pre-norm block: attention, residual, MLP, residual |
| 5 | `embedding/patch_embedding.py` | Image to a sequence of patch vectors |
| 6 | `transformer/vision_transformer.py` | ViT classifier with initialized class and position vectors |
| 7 | `embedding/clip.py` | CLIP image encoder. Same stem as the ViT, returns a vector |
| 8 | `embedding/token_embedding.py` | Token ids to vectors, plus a position |
| 9 | `transformer/text_transformer.py` | CLIP text encoder. Causal. Reads the end-of-text token |
| 10 | `model/model.py` | CLIP. Both towers, a shared space, scaled cosine logits |
| 11 | `loss/contrastive.py` | CLIP loss. Symmetric cross-entropy |
| 12 | `loss/siglip.py` | SigLIP loss. A sigmoid on every pair |
| 13 | `model/projector.py` | MLP, 2x2 SpatialMerge (LLaVA-NeXT), and Perceiver Resampler |
| 14 | `model/dynamic_resolution.py` | AnyRes multi-crop image slicing and spatial newline tokens |
| 15 | `model/language_model.py` | Small causal language model |
| 16 | `model/vlm.py` | Vision tokens prefixed onto the language model (LLaVA-1.0 style) |
| 17 | `model/modern_vlm.py` | Modern VLM with 4x token reduction via 2x2 spatial merging |
| 18 | `embedding/video_embedding.py` | 3D tubelet convolutions for spatio-temporal video encoding |
| 19 | `model/video_vlm.py` | Video VLM for temporal reasoning and video question-answering |
| 20 | `embedding/vq.py` | Vector Quantizer with codebook & straight-through estimator |
| 21 | `model/vqvae.py` | VQ-VAE for discrete visual tokenization and pixel decoding |
| 22 | `loss/vq_loss.py` | VQ-VAE reconstruction MSE and commitment loss |
| 23 | `model/generative_vlm.py` | Unified autoregressive multimodal generator (Chameleon/Show-o style) |
| 24 | `model/flow_matching.py` | Continuous generation head using Flow Matching (Transfusion/DiT style) |
| 25 | `loss/next_token.py` | Predict the next text or visual token |
| 26 | `data/tokenizer.py` | From-scratch character and special token tokenizer |
| 27 | `data/dataset.py` | Synthetic image and video datasets with geometric shapes |
| 28 | `train/train_clip.py` | Memorize one batch with CLIP, then with SigLIP |
| 29 | `train/train_vlm.py` | Memorize one batch with basic VLM |
| 30 | `train/train_modern_vlm.py` | Memorize one batch with Modern VLM (4x token reduction) |
| 31 | `train/train_video.py` | Memorize video temporal dynamics with Video VLM |
| 32 | `train/train_vqvae.py` | Reconstruct images from discrete codebook tokens |
| 33 | `train/train_generation.py` | Text-to-image autoregressive visual generation |

---

## Part 1: Attention & Transformers

### `attention/self_attention.py` & `attention/multi_head_attention.py`
`SelfAttention` and `MultiHeadAttention` project input tokens into queries, keys, and values.
Scores are scaled by $1 / \sqrt{d_k}$, masked, and softened into attention weights before multiplying values.
`MultiHeadAttention` supports:
- Self-attention (queries and keys from the same sequence)
- Cross-attention (queries attend to external context keys/values)
- Arbitrary masks on the correct tensor device.

### `attention/rope.py`
Modern LLMs (Llama 3, Qwen 2) and modern VLMs (Qwen2-VL) replace absolute position embeddings with Rotary Position Embeddings (RoPE).
- **1D RoPE**: Rotates query and key sub-dimensions by complex angles corresponding to sequence index.
- **2D Spatial RoPE**: Splits head dimension into $(h, w)$ halves to preserve 2D grid coordinates.
- **3D Spatio-Temporal RoPE**: Splits head dimension into $(t, h, w)$ thirds to preserve time, height, and width coordinates.

### `transformer/transformer.py`
Pre-norm block: `LayerNorm -> MultiheadAttention -> Residual -> LayerNorm -> MLP -> Residual`.
Takes an optional `attn_mask` (e.g. `causal_mask` for autoregressive language, `prefix_mask` for vision-to-language).

---

## Part 2: Vision Encoders & Contrastive Alignment (CLIP & SigLIP)

### `embedding/patch_embedding.py` & `transformer/vision_transformer.py`
An image $(C, H, W)$ is sliced into non-overlapping patches by a 2D convolution with `kernel_size = stride = patch_size`.
`VisionTransformer` prepends a learned `cls_token`, adds `pos_embedding`, and passes through transformer blocks.

### `embedding/clip.py` & `transformer/text_transformer.py`
- `CLIPEncoder` removes the classifier head and layer-norms the output. `forward_tokens` returns all patches; `forward` returns the class token.
- `TextTransformer` uses causal masking and extracts the vector at the end-of-text position found via `argmax`.

### `model/model.py`, `loss/contrastive.py`, `loss/siglip.py`
- **CLIP**: Projects image and text features into a shared dimension and L2-normalizes them.
  Trained with symmetric cross-entropy: image $i$ picks text $i$, and text $i$ picks image $i$.
- **SigLIP**: Replaces the global softmax with independent sigmoid decisions for every pair. True pair is $+1$, all negative pairs are $-1$. Eliminates cross-device batch coupling.

---

## Part 3: Vision-Language Models (Understanding Images)

### `model/projector.py`
Maps visual features into the language model's embedding width:
1. `Projector`: 2-layer MLP (`Linear -> GELU -> Linear`).
2. `SpatialMergeProjector`: Groups $2 \times 2$ neighboring patches into 1 vector of width $4 \times D$ (pixel unshuffle) before projection. Compresses visual sequence length by $4\times$ while keeping high detail.
3. `PerceiverResampler`: Fixed number of learned query tokens cross-attend to visual tokens, compressing arbitrary visual sequences into a fixed budget.

### `model/dynamic_resolution.py`
AnyRes / Dynamic Resolution (LLaVA-NeXT style):
Slices large images into a grid of standard-sized tiles plus a global overview tile. Appends a learned `<image_newline>` token at the end of each patch row so the language model retains 2D spatial coordinates.

### `model/vlm.py` & `model/modern_vlm.py`
Combines vision and language:
```
Text:   [BOS, USER, <image>, "What is this?", ASSISTANT]
Vision: [patch_1, patch_2, ..., patch_N]
Merged: [BOS, USER, patch_1, ..., patch_N, "What is this?", ASSISTANT]
Target: Predict the assistant's answer autoregressively.
```
`ModernVLM` uses 2x2 spatial merging to drastically reduce prefix token count and accelerate generation.

---

## Part 4: Video Understanding

### `embedding/video_embedding.py`
Videos have shape $(B, C, T, H, W)$.
`VideoPatchEmbedding` applies a 3D convolution with kernel and stride $(T_{\text{tubelet}}, P, P)$ to extract spatio-temporal tubelets spanning both time and space.
`VideoCLIPEncoder` processes tubelets with spatio-temporal positional embeddings.

### `model/video_vlm.py`
Reads video tensors, replaces the `<video>` placeholder token in the prompt with projected tubelet tokens, and performs multi-frame temporal reasoning (action recognition, event ordering, video QA).

---

## Part 5: Visual Generation (Images & Videos)

Modern multimodal models generate visuals using two main paradigms:

### Paradigm A: Discrete Token Generation (Chameleon / Show-o / Parti)
Visuals are represented as discrete tokens from a learned codebook, allowing the language model to generate text and images using the exact same next-token prediction loss!
- `embedding/vq.py`: `VectorQuantizer` maps continuous latents to discrete codebook vectors using Euclidean distance and the Straight-Through Estimator (STE).
- `model/vqvae.py`: `VQVAE` downsamples images into a 2D discrete token grid and decodes discrete tokens back into RGB pixels.
- `model/generative_vlm.py`: Unified vocabulary where text IDs are $[0 \dots V_{\text{text}}-1]$ and visual IDs are $[V_{\text{text}} \dots V_{\text{text}} + K - 1]$.
  When prompted with `[..., ASSISTANT, <boi>]`, the model autoregressively predicts visual tokens until `<eoi>`, which are decoded to an image.

### Paradigm B: Continuous Flow Matching (Transfusion / DiT)
- `model/flow_matching.py`: Given conditioning embeddings from the multimodal model and a noisy latent $x_t$ at time $t \in [0, 1]$, predicts the velocity vector $v_t = x_1 - x_0$.
  Generates samples using an Euler ODE integrator from $t=0$ to $t=1$.

---

## Part 6: Tokenizer & Datasets

- `data/tokenizer.py`: Self-contained character-level tokenizer with multimodal special tokens (`<image>`, `<video>`, `<boi>`, `<eoi>`, `<bos>`, `<eos>`).
- `data/dataset.py`: Synthetic procedural generation of geometric shapes (circles, squares, triangles, crosses) in different colors with natural text captions, plus animated video sequences.

---

## Running the Training Verifications

Each training script verifies a milestone by memorizing a batch:

```bash
# 1. CLIP & SigLIP image-text contrastive alignment
python -m train.train_clip

# 2. Basic Vision-Language Model image QA
python -m train.train_vlm

# 3. Modern VLM with 4x spatial merge compression
python -m train.train_modern_vlm

# 4. Video VLM multi-frame temporal reasoning
python -m train.train_video

# 5. VQ-VAE discrete visual codebook reconstruction
python -m train.train_vqvae

# 6. Unified Generative VLM text-to-image autoregressive generation
python -m train.train_generation
```
