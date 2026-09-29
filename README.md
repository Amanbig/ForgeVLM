# ForgeVLM: From Attention to Modern Multimodal Foundation Models

> **A from-scratch, zero-black-box curriculum for understanding modern Vision-Language Models (VLMs), Video Understanding, and Multimodal Generation.**
>
> Every file is exactly **one concept**. Every tensor transformation is documented step-by-step with shapes. Read the files in order, run the verification scripts, and watch the loss fall to zero.

---

## Quickstart

No heavy GPU or 50GB dataset downloads required. Everything runs locally on **CPU** in seconds.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the 6 core training scripts
python -m train.train_clip          # Stage 2: CLIP & SigLIP image-text alignment
python -m train.train_vlm           # Stage 3: Vision-Language Model image QA
python -m train.train_modern_vlm    # Stage 3: Modern VLM with 4x spatial merge reduction
python -m train.train_video         # Stage 4: Video VLM multi-frame temporal reasoning
python -m train.train_vqvae         # Stage 5: VQ-VAE discrete visual codebook reconstruction
python -m train.train_generation    # Stage 5: Unified text-to-image autoregressive generation
```

---

## The Learning Roadmap

```
+-------------------------------------------------------------------------------------------------+
|                                     STAGE 1: FOUNDATIONS                                        |
|  Self-Attention ---> Multi-Head Attention ---> Rotary Embeddings (RoPE) ---> Transformer Block  |
+-------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+-------------------------------------------------------------------------------------------------+
|                               STAGE 2: CONTRASTIVE ALIGNMENT                                    |
|   Patch Embedding ---> Vision Transformer (ViT) ---> CLIP Dual Towers ---> SigLIP Sigmoid Loss   |
+-------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+-------------------------------------------------------------------------------------------------+
|                             STAGE 3: MODERN IMAGE UNDERSTANDING                                 |
|  2x2 Spatial Merge (LLaVA-NeXT) ---> Dynamic AnyRes Tiling ---> Autoregressive VLM (Image QA)  |
+-------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+-------------------------------------------------------------------------------------------------+
|                                 STAGE 4: VIDEO UNDERSTANDING                                    |
|  3D Tubelet Convolutions ---> Spatio-Temporal RoPE ---> Video-Language Model (Action QA)        |
+-------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+-------------------------------------------------------------------------------------------------+
|                                STAGE 5: MULTIMODAL GENERATION                                   |
|  Vector Quantizer (VQ-VAE) ---> Unified Autoregressive Gen (Chameleon) ---> Flow Matching (DiT) |
+-------------------------------------------------------------------------------------------------+
```

---

## Master File Map

| Order | File | Concept | Input Shape -> Output Shape |
| :---: | :--- | :--- | :--- |
| **1** | [`attention/self_attention.py`](file:///home/amanpreet/Documents/ForgeVLM/attention/self_attention.py) | Single-head scaled dot-product attention | `(B, T, D) -> (B, T, D)` |
| **2** | [`attention/multi_head_attention.py`](file:///home/amanpreet/Documents/ForgeVLM/attention/multi_head_attention.py) | Multi-head self and cross-attention | `(B, T_q, D), (B, T_kv, D) -> (B, T_q, D)` |
| **3** | [`attention/rope.py`](file:///home/amanpreet/Documents/ForgeVLM/attention/rope.py) | 1D, 2D (spatial), and 3D (video) Rotary Position Embeddings | Rotates Q and K heads by position |
| **4** | [`transformer/transformer.py`](file:///home/amanpreet/Documents/ForgeVLM/transformer/transformer.py) | Pre-LN Transformer block & causal/prefix masks | `(B, T, D) -> (B, T, D)` |
| **5** | [`embedding/patch_embedding.py`](file:///home/amanpreet/Documents/ForgeVLM/embedding/patch_embedding.py) | Conv2d patch extractor (ViT stem) | `(B, C, H, W) -> (B, N_patches, D)` |
| **6** | [`transformer/vision_transformer.py`](file:///home/amanpreet/Documents/ForgeVLM/transformer/vision_transformer.py) | Vision Transformer classifier with [CLS] token | `(B, C, H, W) -> (B, num_classes)` |
| **7** | [`embedding/clip.py`](file:///home/amanpreet/Documents/ForgeVLM/embedding/clip.py) | CLIP vision encoder (tokens vs class summary) | `(B, C, H, W) -> (B, 1+N, D) / (B, D)` |
| **8** | [`embedding/token_embedding.py`](file:///home/amanpreet/Documents/ForgeVLM/embedding/token_embedding.py) | Token ID embedding + learned positions | `(B, T) -> (B, T, D)` |
| **9** | [`transformer/text_transformer.py`](file:///home/amanpreet/Documents/ForgeVLM/transformer/text_transformer.py) | Causal text encoder with `<eos>` pooling | `(B, T) -> (B, D)` |
| **10** | [`model/model.py`](file:///home/amanpreet/Documents/ForgeVLM/model/model.py) | Full CLIP dual-tower model | `(B, C, H, W), (B, T) -> (B, B) logits` |
| **11** | [`loss/contrastive.py`](file:///home/amanpreet/Documents/ForgeVLM/loss/contrastive.py) | Symmetric InfoNCE contrastive cross-entropy | Pairwise similarity -> scalar loss |
| **12** | [`loss/siglip.py`](file:///home/amanpreet/Documents/ForgeVLM/loss/siglip.py) | Pairwise sigmoid loss (Zhai et al., 2023) | Pairwise similarity -> scalar loss |
| **13** | [`model/projector.py`](file:///home/amanpreet/Documents/ForgeVLM/model/projector.py) | Linear MLP, 2x2 Spatial Merge, Perceiver Resampler | `(B, N_vis, D_vis) -> (B, N_out, D_lang)` |
| **14** | [`model/dynamic_resolution.py`](file:///home/amanpreet/Documents/ForgeVLM/model/dynamic_resolution.py) | AnyRes multi-crop slicing & spatial newline tokens | High-res image -> tiled patch grid |
| **15** | [`model/language_model.py`](file:///home/amanpreet/Documents/ForgeVLM/model/language_model.py) | Causal LM with weight-tied embedding and head | `(B, T) or (B, T, D) -> (B, T, V)` |
| **16** | [`model/vlm.py`](file:///home/amanpreet/Documents/ForgeVLM/model/vlm.py) | Basic Vision-Language Model (LLaVA-1.0 style) | `(B, C, H, W), (B, T) -> (B, T_merged, V)` |
| **17** | [`model/modern_vlm.py`](file:///home/amanpreet/Documents/ForgeVLM/model/modern_vlm.py) | Modern VLM with 4x spatial merge reduction | Cuts visual tokens from 16 to 4 |
| **18** | [`embedding/video_embedding.py`](file:///home/amanpreet/Documents/ForgeVLM/embedding/video_embedding.py) | 3D tubelet convolutions for spatio-temporal video | `(B, C, T, H, W) -> (B, N_tubelets, D)` |
| **19** | [`model/video_vlm.py`](file:///home/amanpreet/Documents/ForgeVLM/model/video_vlm.py) | Video VLM for action reasoning & video QA | `(B, C, T, H, W), (B, T) -> (B, T_merged, V)` |
| **20** | [`embedding/vq.py`](file:///home/amanpreet/Documents/ForgeVLM/embedding/vq.py) | Vector Quantizer with Straight-Through Estimator (STE) | `(B, D, H, W) -> discrete codebook indices` |
| **21** | [`model/vqvae.py`](file:///home/amanpreet/Documents/ForgeVLM/model/vqvae.py) | VQ-VAE: image to discrete token grid and back | Pixels <---> Discrete codebook tokens |
| **22** | [`loss/vq_loss.py`](file:///home/amanpreet/Documents/ForgeVLM/loss/vq_loss.py) | Reconstruction MSE + codebook commitment loss | Scalar loss |
| **23** | [`model/generative_vlm.py`](file:///home/amanpreet/Documents/ForgeVLM/model/generative_vlm.py) | Unified autoregressive multimodal generator (Chameleon) | Text prompt -> visual tokens -> RGB image |
| **24** | [`model/flow_matching.py`](file:///home/amanpreet/Documents/ForgeVLM/model/flow_matching.py) | Continuous visual generation via Flow Matching (DiT) | Latent noise + text cond -> image latent |
| **25** | [`loss/next_token.py`](file:///home/amanpreet/Documents/ForgeVLM/loss/next_token.py) | Causal next-token prediction loss with `-100` masking | `logits[:, :-1], labels[:, 1:] -> scalar` |
| **26** | [`data/tokens.py`](file:///home/amanpreet/Documents/ForgeVLM/data/tokens.py) | Special tokens (`<image>`, `<video>`, `<boi>`, `<eoi>`) | Token definitions & chat helpers |
| **27** | [`data/tokenizer.py`](file:///home/amanpreet/Documents/ForgeVLM/data/tokenizer.py) | Self-contained character and special token tokenizer | `encode()` and `decode()` |
| **28** | [`data/dataset.py`](file:///home/amanpreet/Documents/ForgeVLM/data/dataset.py) | Procedural synthetic shape & animated video datasets | Clean paired multimodal training batches |
| **29** | [`train/train_clip.py`](file:///home/amanpreet/Documents/ForgeVLM/train/train_clip.py) | CLIP & SigLIP memorization script | Loss falls to near 0 |
| **30** | [`train/train_vlm.py`](file:///home/amanpreet/Documents/ForgeVLM/train/train_vlm.py) | Basic VLM memorization & text generation script | Loss falls to near 0 |
| **31** | [`train/train_modern_vlm.py`](file:///home/amanpreet/Documents/ForgeVLM/train/train_modern_vlm.py) | Modern VLM with 4x spatial merge compression | Verified 4x token length savings |
| **32** | [`train/train_video.py`](file:///home/amanpreet/Documents/ForgeVLM/train/train_video.py) | Video VLM temporal reasoning script | Verifies motion direction QA |
| **33** | [`train/train_vqvae.py`](file:///home/amanpreet/Documents/ForgeVLM/train/train_vqvae.py) | VQ-VAE discrete visual codebook training | Verifies image pixel reconstruction |
| **34** | [`train/train_generation.py`](file:///home/amanpreet/Documents/ForgeVLM/train/train_generation.py) | Text-to-image autoregressive generation script | Text prompt -> visual tokens -> decoded image |

---

## Detailed Step-by-Step Curriculum

### Stage 1: Attention & Transformers from Scratch

#### 1. Self-Attention (`attention/self_attention.py`)
- **The Core Intuition**: Instead of processing tokens in isolation, attention computes how much every token should "look at" every preceding token.
- **The Three Roles**:
  - $Q$ (Query): What am I looking for?
  - $K$ (Key): What do I contain?
  - $V$ (Value): What information do I pass along?
- **The Formula**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
- **Causal Mask ($M$)**: Lower-triangular matrix filled with $0.0$ for past tokens and $-\infty$ for future tokens. This ensures token $t$ cannot cheat by looking at token $t+1$.

#### 2. Multi-Head Attention (`attention/multi_head_attention.py`)
- Why split into heads? A single attention head can only attend to one relationship at a time (e.g. adjacent words). Multiple heads allow the model to attend to syntax, position, color, and object boundaries in parallel.
- Supports **Cross-Attention**: queries come from sequence $A$, keys and values come from sequence $B$.

#### 3. Rotary Position Embeddings - RoPE (`attention/rope.py`)
- Modern LLMs (Llama 3, Qwen 2) and modern VLMs (Qwen2-VL) discard learned absolute position lookup tables.
- Instead, RoPE rotates the Query and Key vectors in complex 2D planes according to their position index:
  $$\mathbf{R}_{\Theta, m} \mathbf{x} = \mathbf{x} \cos(m \theta) + \tilde{\mathbf{x}} \sin(m \theta)$$
- **2D Spatial RoPE**: Splits the head dimension into $(h, w)$ halves to rotate image patches by their 2D grid coordinates.
- **3D Spatio-Temporal RoPE**: Splits the head dimension into $(t, h, w)$ thirds to rotate video tubelets across time, height, and width.

---

### Stage 2: Vision Encoders & Contrastive Alignment

#### 4. Patch Embedding (`embedding/patch_embedding.py`)
- How do we turn a 2D image into 1D tokens?
- A `nn.Conv2d` with `kernel_size = patch_size` and `stride = patch_size` acts as a non-overlapping patch cutter:
  $$\text{Image } (B, 3, 32, 32) \xrightarrow{\text{patch\_size}=8} (B, 64, 4, 4) \xrightarrow{\text{flatten}} (B, 16, 64)$$
- A 32x32 image becomes a sequence of 16 tokens, each representing an 8x8 image patch.

#### 5. Vision Transformer - ViT (`transformer/vision_transformer.py`)
- Prepends a learnable `[CLS]` class token to the 16 patches (sequence length becomes $1 + 16 = 17$).
- Adds positional embeddings so the transformer knows patch layout.
- Processes the sequence through bidirectional transformer blocks.
- The `[CLS]` token at index 0 gathers information from all patches to perform classification.

#### 6. CLIP Dual Towers (`model/model.py`, `loss/contrastive.py`)
- Two encoders: `CLIPEncoder` (image tower) and `TextTransformer` (text tower).
- Both project their output into a shared dimension (e.g., 32) and **L2-normalize** them to length 1.0.
- Dot product = Cosine similarity in $[-1.0, 1.0]$.
- **Symmetric Contrastive Loss**: Image $i$ must pick text $i$ out of the batch, AND text $i$ must pick image $i$.

#### 7. SigLIP: The Modern Contrastive Loss (`loss/siglip.py`)
- Standard CLIP computes a softmax across the entire batch row. This couples all devices and requires huge batch sizes ($>30\text{k}$) to provide enough negatives.
- **SigLIP** (Google DeepMind, 2023) replaces the softmax with independent binary sigmoid decisions for every pair:
  $$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^B \sum_{j=1}^B \log \sigma\left( y_{ij} \cdot (\exp(t) \cdot (\mathbf{u}_i \cdot \mathbf{v}_j) + b) \right)$$
  where $y_{ij} = +1$ for matching pairs and $-1$ for non-matching pairs.

---

### Stage 3: Vision-Language Models (Understanding Images)

#### 8. Projectors: Connecting Vision to Language (`model/projector.py`)
- Vision encoders output vectors of width $D_{\text{vision}}$ (e.g. 64). Language models expect vectors of width $D_{\text{lang}}$ (e.g. 64 or 4096).
- **Linear MLP**: 2-layer MLP (`Linear -> GELU -> Linear`).
- **2x2 SpatialMergeProjector** (LLaVA-NeXT, Qwen2-VL, InternVL):
  - At high resolution, visual tokens become enormous.
  - Groups each $2 \times 2$ neighborhood of patches into 1 vector of width $4 \cdot D$ (pixel unshuffle) before projection.
  - **Reduces token count by $4\times$** (e.g. 16 patches $\rightarrow$ 4 visual tokens) while preserving all sub-pixel detail!
- **PerceiverResampler** (Flamingo, BLIP-2): Fixed number of query tokens cross-attend to visual tokens.

#### 9. Splicing Visual Tokens into Text (`model/vlm.py`)
```
Prompt: [BOS, USER, <image>, "What", "shape?", ASSISTANT]
Vision: [patch_1, patch_2, ..., patch_16]

Spliced Sequence:
[BOS, USER] + [patch_1, ..., patch_16] + ["What", "shape?", ASSISTANT] + [Answer, EOS]
```
- In `expand_labels`, target labels for prompt and image positions are set to `-100`.
- PyTorch's `F.cross_entropy(..., ignore_index=-100)` ignores these positions, ensuring the loss only trains the model to predict the assistant's answer!

---

### Stage 4: Video Understanding

#### 10. 3D Spatio-Temporal Tubelets (`embedding/video_embedding.py`)
- Video tensors have 5 dimensions: $(B, C, T, H, W)$.
- Instead of processing each frame independently with 2D convs, a **3D convolution** with `kernel_size = stride = (tubelet_time, patch_size, patch_size)` extracts tubelets that fuse time and space simultaneously!
  $$\text{Video } (B, 3, 4, 32, 32) \xrightarrow{T_{\text{tubelet}}=2, P=8} (B, 64, 2, 4, 4) \xrightarrow{\text{flatten}} (B, 32, 64)$$
- A 4-frame 32x32 video becomes 32 spatio-temporal tokens.

#### 11. Video-Language Modeling (`model/video_vlm.py`)
- Replaces the `<video>` placeholder token in the prompt with projected tubelet tokens.
- The causal language model attends across the tubelet sequence and performs temporal reasoning (e.g. recognizing direction of motion across time).

---

### Stage 5: Multimodal Generation

#### 12. Discrete Visual Tokenization: VQ-VAE (`embedding/vq.py`, `model/vqvae.py`)
- How can an autoregressive language model generate images?
- An image is continuous pixels, but an LLM predicts discrete token IDs from a vocabulary.
- **Solution (Vector Quantization)**:
  1. An Encoder downsamples the image $4\times$ into a spatial latent grid: $(B, C, 32, 32) \rightarrow (B, D, 8, 8)$.
  2. For each latent vector, the **Codebook** finds the nearest discrete embedding index $k \in \{0, \dots, K-1\}$.
  3. The Straight-Through Estimator (STE) copies gradients from decoder to encoder.
  4. The Decoder upsamples the discrete codebook vectors back into RGB pixels.
- The 2D image is now an $8 \times 8 = 64$ sequence of integer token IDs, just like words in a sentence!

#### 13. Unified Generative VLM (`model/generative_vlm.py`)
- Uses a unified vocabulary:
  - Text tokens: IDs $[0 \dots V_{\text{text}}-1]$
  - Visual codebook tokens: IDs $[V_{\text{text}} \dots V_{\text{text}} + K - 1]$
- When asked to generate an image:
  ```
  Input Prompt:  [BOS, USER, "Generate red square", ASSISTANT, <boi>]
  Model Output:  [img_tok_1, img_tok_2, ..., img_tok_64, <eoi>]
  ```
- The generated visual tokens are fed directly into `vqvae.decode_from_indices(...)` to synthesize the RGB image!

#### 14. Continuous Flow Matching (`model/flow_matching.py`)
- Used by continuous diffusion/flow models (Transfusion, DiT, Stable Diffusion 3).
- Conditions on transformer text representations to predict the linear velocity field $v_t = x_1 - x_0$.
- Uses Euler ODE sampling from $t=0$ (pure noise) to $t=1$ (clean image).

---

## The "Falling Loss" Principle

Every training script in ForgeVLM follows the **memorization principle**:
> *If a model cannot memorize a small, fixed batch of 4 to 8 examples down to near-zero loss, its internal wiring, gradient flow, or shapes are broken.*

Run all verification scripts to test your build:

```bash
# 1. CLIP & SigLIP image-text contrastive alignment
# Expected: CLIP loss falls from ~2.9 -> 0.004; SigLIP falls from ~9.2 -> 0.14
python -m train.train_clip

# 2. Vision-Language Model image understanding
# Expected: VLM loss falls from ~44.9 -> 0.0003; generated text matches ground truth
python -m train.train_vlm

# 3. Modern VLM with 4x spatial merge reduction
# Expected: Modern VLM loss falls from ~42.9 -> 0.0002 with 4x shorter sequence length
python -m train.train_modern_vlm

# 4. Video VLM temporal reasoning
# Expected: Video loss falls from ~37.3 -> 0.22; generates exact motion answers
python -m train.train_video

# 5. VQ-VAE discrete codebook reconstruction
# Expected: VQ-VAE loss falls from ~0.18 -> 0.07; pixel reconstruction matches
python -m train.train_vqvae

# 6. Unified Generative VLM text-to-image generation
# Expected: Gen loss falls from ~18.2 -> 0.0002; generates visual tokens and decodes RGB image
python -m train.train_generation
```
