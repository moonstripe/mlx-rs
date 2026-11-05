"""Independent f32 NumPy EmbeddingGemma 2 (Gemma 4) vision tower and image embedding.

An image is an already resized `[height, width, 3]` uint8 RGB array whose sides are
multiples of `patch_size * pooling_kernel_size`. Patches are row-major over the
patch grid with `(row, column, channel)` order inside a patch; each patch adds the
column and row position embeddings. After the bidirectional encoder, each
`k x k` block of patches is averaged into one soft token (row-major over blocks)
and scaled by `sqrt(hidden_size)`, then `embed_vision` projects it into the text
width. The text sequence is `<bos> <boi> <image> * tokens <eoi> <eos>`.
"""

import numpy as np

from .common import rms_norm
from .embedding_gemma2 import EmbeddingGemma2Text, gelu_tanh, softmax


def rms_no_scale(x, eps):
    return rms_norm(x, np.float32(1), eps)


def axial_rope(x, positions, theta):
    """x: `[length, heads, head_dim]`; positions: `[length, 2]` as (column, row)."""
    dims = positions.shape[-1]
    channels = 2 * (x.shape[-1] // (2 * dims))
    half = channels // 2
    timescale = np.float32(theta) ** ((np.float32(2) / channels) * np.arange(half, dtype=np.float32))
    parts = []
    for d in range(dims):
        part = x[..., d * channels:(d + 1) * channels]
        angles = positions[:, d : d + 1].astype(np.float32) / timescale
        angles = np.concatenate([angles, angles], axis=-1)[:, None, :]
        rotated = np.concatenate([-part[..., half:], part[..., :half]], axis=-1)
        parts.append(part * np.cos(angles) + rotated * np.sin(angles))
    return np.concatenate(parts, axis=-1)


class VisionTower:
    PREFIX = "vision_tower."

    def __init__(self, config, weights, mutation=None):
        self.config = config["vision_config"]
        self.text_eps = np.float32(config["text_config"]["rms_norm_eps"])
        self.eps = np.float32(self.config["rms_norm_eps"])
        self.weights = {k: v.astype(np.float32) for k, v in weights.items()
                        if k.startswith(self.PREFIX) or k.startswith("embed_vision.")}
        self.mutation = mutation

    def w(self, name):
        return self.weights[name]

    def linear(self, x, name):
        return x @ self.w(name + ".weight").T

    def patches(self, image):
        p = self.config["patch_size"]
        height, width, channels = image.shape
        rows, cols = height // p, width // p
        pixels = image.astype(np.float32) * np.float32(1 / 255)
        patches = pixels.reshape(rows, p, cols, p, channels).transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(rows * cols, p * p * channels)
        if self.mutation == "channel_major_patches":
            patches = pixels.reshape(rows, p, cols, p, channels).transpose(0, 2, 4, 1, 3)
            patches = patches.reshape(rows * cols, p * p * channels)
        grid_rows, grid_cols = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        positions = np.stack([grid_cols.ravel(), grid_rows.ravel()], axis=-1)
        if self.mutation == "swap_axes":
            positions = positions[:, ::-1]
        return patches, positions, rows, cols

    def features(self, image):
        """Soft tokens in text width, `[tokens, text_hidden]`."""
        c = self.config
        patches, positions, rows, cols = self.patches(image)
        pre = self.PREFIX
        x = self.linear(np.float32(2) * (patches - np.float32(0.5)), pre + "patch_embedder.input_proj")
        table = self.w(pre + "patch_embedder.position_embedding_table")
        x = x + table[0][positions[:, 0]] + table[1][positions[:, 1]]
        heads, kv_heads, dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        theta = c["rope_parameters"]["rope_theta"]
        length = x.shape[0]
        for i in range(c["num_hidden_layers"]):
            l = f"{pre}encoder.layers.{i}."
            h = rms_norm(x, self.w(l + "input_layernorm.weight"), self.eps)
            q = self.linear(h, l + "self_attn.q_proj.linear").reshape(length, heads, dim)
            k = self.linear(h, l + "self_attn.k_proj.linear").reshape(length, kv_heads, dim)
            v = self.linear(h, l + "self_attn.v_proj.linear").reshape(length, kv_heads, dim)
            q = rms_norm(q, self.w(l + "self_attn.q_norm.weight"), self.eps)
            k = rms_norm(k, self.w(l + "self_attn.k_norm.weight"), self.eps)
            if self.mutation != "vision_no_v_norm":
                v = rms_no_scale(v, self.eps)
            q, k = axial_rope(q, positions, theta), axial_rope(k, positions, theta)
            k = np.repeat(k, heads // kv_heads, axis=1)
            v = np.repeat(v, heads // kv_heads, axis=1)
            scores = np.einsum("qhd,khd->hqk", q, k)
            out = np.einsum("hqk,khd->qhd", softmax(scores), v).reshape(length, heads * dim)
            out = self.linear(out, l + "self_attn.o_proj.linear")
            x = x + rms_norm(out, self.w(l + "post_attention_layernorm.weight"), self.eps)
            h = rms_norm(x, self.w(l + "pre_feedforward_layernorm.weight"), self.eps)
            h = gelu_tanh(self.linear(h, l + "mlp.gate_proj.linear")) * self.linear(h, l + "mlp.up_proj.linear")
            x = x + rms_norm(self.linear(h, l + "mlp.down_proj.linear"),
                             self.w(l + "post_feedforward_layernorm.weight"), self.eps)
        k = c["pooling_kernel_size"]
        pooled = x.reshape(rows // k, k, cols // k, k, -1).mean(axis=(1, 3)).reshape(-1, x.shape[-1])
        if self.mutation == "column_major_pool":
            pooled = x.reshape(rows // k, k, cols // k, k, -1).mean(axis=(1, 3))
            pooled = pooled.transpose(1, 0, 2).reshape(-1, x.shape[-1])
        pooled = pooled * np.float32(np.sqrt(c["hidden_size"]))
        normed = rms_no_scale(pooled, self.eps)
        return self.linear(normed, "embed_vision.embedding_projection")


def image_sequence(config, tokens):
    text = config["text_config"]
    return ([text["bos_token_id"], config["boi_token_id"]] + [config["image_token_id"]] * tokens
            + [config["eoi_token_id"], text["eos_token_id"]])


class EmbeddingGemma2Images(EmbeddingGemma2Text):
    def __init__(self, config, weights, mutation=None):
        super().__init__(config, weights, mutation)
        self.config = config
        self.vision = VisionTower(config, weights, mutation)

    def embed_images(self, images):
        features = [self.vision.features(image) for image in images]
        sequences = [image_sequence(self.config, len(f)) for f in features]
        length = max(map(len, sequences))
        pad = self.text["pad_token_id"]
        ids = np.array([s + [pad] * (length - len(s)) for s in sequences], np.int64)
        valid = np.array([[True] * len(s) + [False] * (length - len(s)) for s in sequences])
        image = ids == self.config["image_token_id"]
        x = self.w("embed_tokens.weight")[np.where(image, pad, ids)] * np.float32(np.sqrt(self.text["hidden_size"]))
        for row, f in enumerate(features):
            x[row, image[row]] = f
        return self.pool(self.hidden_from_embeddings(x, valid), valid)
