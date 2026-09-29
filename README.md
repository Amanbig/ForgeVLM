# ForgeVLM

A from-scratch path from attention to a small vision-language model. Each file is one idea. Read them in order, then run the two training scripts. They memorize one random batch. A falling loss means the wiring is right.

Run everything from the repo root.

```bash
pip install -r requirements.txt
python -m train.train_clip
python -m train.train_vlm
```

CPU is enough. These models are tiny on purpose.

## Map

| Order | File | What it is |
| --- | --- | --- |
| 1 | `attention/self_attention.py` | One attention head, causal mask |
| 2 | `attention/multi_head_attention.py` | The same idea, split into heads |
| 3 | `transformer/transformer.py` | Pre-norm block: attention, residual, MLP, residual |
| 4 | `embedding/patch_embedding.py` | Image to a sequence of patch vectors |
| 5 | `transformer/vision_transformer.py` | ViT classifier. Class token goes through a linear head |
| 6 | `embedding/clip.py` | CLIP image encoder. Same stem as the ViT, returns a vector |
| 7 | `embedding/token_embedding.py` | Token ids to vectors, plus a position |
| 8 | `transformer/text_transformer.py` | CLIP text encoder. Causal. Reads the end-of-text token |
| 9 | `model/model.py` | CLIP. Both towers, a shared space, scaled cosine logits |
| 10 | `loss/contrastive.py` | CLIP loss. Symmetric cross-entropy |
| 11 | `loss/siglip.py` | SigLIP loss. A sigmoid on every pair |
| 12 | `model/projector.py` | MLP from vision width to language width |
| 13 | `model/language_model.py` | Small causal language model |
| 14 | `model/vlm.py` | Vision tokens prefixed onto the language model |
| 15 | `loss/next_token.py` | Predict the next text token |
| 16 | `train/train_clip.py` | Memorize one batch with CLIP, then with SigLIP |
| 17 | `train/train_vlm.py` | Memorize one batch with the VLM |

## What you already built

### Attention

`SelfAttention` and `MultiHeadAttention` mix a sequence with a causal mask. Token `i` can see tokens `0..i`. That is the language-model rule.

Three things to finish in those two files before you lean on them:

- `SelfAttention` builds `self.out` and never applies it. The last step of that forward pass is `self.out(attn @ v)`. `MultiHeadAttention` already does this.
- The mask is built with `torch.ones(...)`, so it lives on CPU. Build it on `x.device` when you move the model to a GPU.
- The vision block is `transformer/transformer.py`, and it lets every token see every other token. The class token sits at position 0. Under the causal mask in your attention files, that token can see only itself, so a ViT built on those files would ignore every patch.

### Transformer block

`TransformerBlock` is pre-norm. Normalize, attend, add the residual, normalize, MLP, add the residual. The MLP widens to `4 * embed_dim` and uses GELU.

`forward` takes an optional `attn_mask`. Leave it empty for vision. Pass `causal_mask` for text. Pass `prefix_mask` when image tokens sit in front of text tokens.

### Patch embedding and the ViT

`PatchEmbedding` is a convolution whose kernel and stride both equal the patch size. A `32x32` image with patch size `8` becomes `16` tokens. The forward flattens the grid and swaps it to `(batch, tokens, embed_dim)`.

`VisionTransformer` prepends a class token, adds a position vector, runs the blocks, and classifies from the class token: `self.head(x[:, 0])`.

`cls_token` and `pos_embedding` start at zero there. Every position begins identical, so the block has no location signal until those tables move. `CLIPEncoder` draws them from a normal with std `0.02`. The same init belongs on the ViT when you train it.

## CLIP

CLIP trains an image tower and a text tower so that a matching pair lands in the same direction and the other pairs in the batch do not.

```
image -> patches + class token + positions -> transformer -> class token -> linear -> L2 normalize
text  -> tokens + positions -> causal transformer -> end-of-text token -> linear -> L2 normalize
logits = exp(logit_scale) * image_features @ text_features.T
```

For a batch of 2 the logit matrix is:

```
            text 0    text 1
image 0      s00       s01
image 1      s10       s11
```

Image 0 should score text 0 above text 1. Text 0 should score image 0 above image 1. That is the same matrix, read by rows and then by columns.

### `embedding/clip.py`

`CLIPEncoder` is your ViT with the classification head removed.

- `forward_tokens` returns `(batch, num_patches + 1, embed_dim)` after a final `LayerNorm`.
- `forward` returns the class token only, shape `(batch, embed_dim)`.

The published CLIP also layer-norms the sequence before the blocks. This one follows your ViT and layer-norms at the end. `LayerNorm` runs per token, so taking the class token before or after that norm is the same operation.

### `embedding/token_embedding.py`

The text-side cousin of patch embedding. `nn.Embedding` turns ids into vectors. A second table adds a learned position. Positions are `0 .. length-1` on the same device as the ids.

### `transformer/text_transformer.py`

Same block as the vision tower. The mask is causal, so a word can see the words before it and not the words after it.

CLIP reads one position: the end-of-text token. Give that id the largest number in the vocabulary and put it once in each caption. `argmax` finds it. That token has attended over the whole caption, which is the same job the class token does for an image.

### `model/model.py`

`CLIP` owns both encoders, two bias-free projections, and `logit_scale`.

The towers can have different widths. The projections land them in one `projection_dim`. Features are L2-normalized, so the dot product is cosine similarity.

`logit_scale` is stored as a log. It starts at `log(1 / 0.07)`, so the multiplier starts near `14.3`. The forward clamps `exp(logit_scale)` at `100` so a big update cannot blow up the softmax.

If every similarity in the row were equal, the loss would be `log(batch_size)`, about `2.08` for 8 pairs. The scale starts near `14`, so small random differences become large logit gaps and the first step often lands higher than that. `train/train_clip.py` memorizes 8 fixed pairs and drives the loss from around `3` down near `0`.

### `loss/contrastive.py`

```python
targets = torch.arange(batch_size)  # [0, 1, 2, ...]
loss = (cross_entropy(logits_per_image, targets) + cross_entropy(logits_per_text, targets)) / 2
```

## SigLIP

CLIP's softmax couples every pair in the row. A bigger batch changes the loss, and the model needs a huge batch to see enough negatives.

SigLIP scores every image-text pair on its own. The matching pair is labeled `+1`. Every other pair is labeled `-1`. Each pair is trained with a sigmoid:

```
logits = exp(logit_scale) * cosine + logit_bias
loss   = mean over images of sum over texts of -logsigmoid(label * logits)
```

`SigLIPLoss` owns its own scale and bias. Scale starts at `log(10)`. Bias starts at `-10`, so an unseen pair begins on the "not a match" side of the sigmoid.

`train/train_clip.py` memorizes the same kind of batch a second time with this loss. Swap it in anywhere you have normalized image features and text features.

## The vision-language model

CLIP gives you one vector per image. A VLM keeps the patch tokens, maps them into the language model, and asks that model to predict the next word.

```
image -> CLIPEncoder.forward_tokens -> drop the class token
      -> Projector (linear, GELU, linear)
      -> concatenate in front of the text embeddings
      -> language model
      -> next-token loss on the text positions only
```

### Mask

`prefix_mask` is the attention pattern:

- An image token can see every image token.
- Text stays hidden from the image tokens. The caption is the thing being predicted.
- A text token can see every image token and the text tokens up to itself.

The last image position has seen the whole image. The hidden state there is what predicts the first word.

### `model/language_model.py`

Token embedding, causal blocks, a final norm, a linear head over the vocabulary. The head shares its weight with the token table, so the model has one vector per word and uses it both to read the word and to score it.

`forward` is text only. `forward_embeddings` is what the VLM calls after it has concatenated vision tokens. `embed` is exposed so the VLM can place text embeddings after vision embeddings.

### `loss/next_token.py`

Sequence layout: `[patch tokens | text tokens]`.

The vector at the last patch predicts text token 0. The vector at text token 0 predicts text token 1. Patch positions are not targets. With no prefix, the function falls back to the ordinary shift: position `t` predicts token `t + 1`.

### Training the demo

`train/train_vlm.py` trains every parameter, including the vision tower, so a random batch can be memorized.

A real run usually freezes the vision tower and trains the projector first, then unfreezes more. This demo skips that. Its only job is to show that image tokens and text tokens form one sequence and that the loss is applied to the words.

A language model that is guessing among 64 words sits near `log(64) ≈ 4.16`. This demo starts higher, because the projected image tokens are random and the head is confident about the wrong words. `train/train_vlm.py` brings that loss from the tens down near `0` on 4 fixed pairs.

## What these scripts are checking

`train/train_clip.py`

- Logits have shape `(batch, batch)`.
- CLIP loss on 8 fixed pairs falls.
- SigLIP loss on 8 fixed pairs falls.

`train/train_vlm.py`

- Logits have shape `(batch, num_patches + text_length, vocab)`.
- Next-token loss on 4 fixed pairs falls.

Both scripts exit with an error if the loss stays flat. Weight decay is `0`. The optimizer is AdamW at `3e-3`. The same batch is reused every step. That is memorization, and it is the right first test. A flat loss on 8 pairs means the wiring is wrong.

## What is yours next

The model code stops at tensors of token ids. The next files, when you want real data, are:

1. A tokenizer. Start with characters, or a small word vocabulary. Keep the end-of-text id as the largest id so `TextTransformer` can find it with `argmax`.
2. A dataset that returns `(image, token_ids)` at a fixed image size and a fixed text length. Pad short captions and keep a padding mask for later.
3. A training loop that loads that dataset. Start from `train/train_clip.py` and replace `make_batch`.
4. An eval that holds out pairs and reports recall: how often the true caption is the top score for its image.

Init `cls_token` and `pos_embedding` in `VisionTransformer` the same way `CLIPEncoder` does before you train the classifier on real images.
