"""Independent f32 NumPy EmbeddingGemma 2 (Gemma 4) audio front-end, encoder and embedding.

The front-end follows the published feature extractor: zero-pad the waveform to a
multiple of 128 samples, prepend `frame_length // 2` zeros, take frames of
`frame_length` samples every `hop_length`, apply a periodic Hann window, take the
magnitude of a `fft_length` real FFT, project onto an HTK triangular mel bank and
add `mel_floor` before the log. A frame is valid only if the sample after it is
real audio; invalid frames are zero.

The Conformer encoder is written in its global form: query `t` attends to key `k`
when `0 <= t - k < attention_context_left - 1` (or `0 < k - t < context_right`)
and the key is valid, with a relative-position term from sinusoids at offsets
`t - k`. The text sequence is `<bos> <boa> <audio> * tokens <eoa> <eos>`.
"""

import math

import numpy as np

from .common import rms_norm
from .embedding_gemma2 import EmbeddingGemma2Text


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_bank(fe):
    bins = fe["fft_length"] // 2 + 1
    mels = np.linspace(hz_to_mel(fe["min_frequency"]), hz_to_mel(fe["max_frequency"]), fe["feature_size"] + 2)
    edges = mel_to_hz(mels)
    freqs = np.linspace(0, fe["sampling_rate"] / 2, bins)
    bank = np.zeros((bins, fe["feature_size"]))
    for m in range(fe["feature_size"]):
        lower, center, upper = edges[m], edges[m + 1], edges[m + 2]
        rising = (freqs - lower) / (center - lower)
        falling = (upper - freqs) / (upper - center)
        bank[:, m] = np.maximum(0.0, np.minimum(rising, falling))
    return bank


def log_mel(wave, fe, mutation=None):
    """Returns `([frames, feature_size] f32, [frames] bool)`."""
    frame, hop = fe["frame_length"], fe["hop_length"]
    length = len(wave)
    padded = -(-length // 128) * 128
    samples = np.zeros(frame // 2 + padded, np.float32)
    samples[frame // 2 : frame // 2 + length] = wave
    real = np.zeros(frame // 2 + padded, bool)
    real[frame // 2 : frame // 2 + length] = True
    count = (len(samples) - (frame + 1)) // hop + 1
    if count <= 0:
        return np.zeros((0, fe["feature_size"]), np.float32), np.zeros(0, bool)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame) / frame)).astype(np.float32)
    frames = np.stack([samples[i * hop : i * hop + frame] for i in range(count)]) * window
    spectrum = np.abs(np.fft.rfft(frames.astype(np.float64), n=fe["fft_length"], axis=-1))
    features = np.log(spectrum @ mel_bank(fe) + fe["mel_floor"]).astype(np.float32)
    end = frame if mutation == "frame_end_inclusive" else frame + 1
    valid = np.array([real[i * hop + end - 1] if i * hop + end - 1 < len(real) else False for i in range(count)])
    return features * valid[:, None], valid


def clip_linear(x, w, name):
    x = np.clip(x, w[name + ".input_min"], w[name + ".input_max"])
    x = x @ w[name + ".linear.weight"].T
    return np.clip(x, w[name + ".output_min"], w[name + ".output_max"])


def layer_norm(x, weight, eps):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + np.float32(eps)) * weight


def silu(x):
    return x / (np.float32(1) + np.exp(-x))


def softplus(x):
    return np.logaddexp(x, np.float32(0))


def conv2d_s2(x, weight):
    """x: `[T, F, Cin]` zero-padded by one on both axes; weight `[Cout, 3, 3, Cin]`."""
    x = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    t_out = (x.shape[0] - 3) // 2 + 1
    f_out = (x.shape[1] - 3) // 2 + 1
    out = np.zeros((t_out, f_out, weight.shape[0]), np.float32)
    for i in range(3):
        for j in range(3):
            patch = x[i : i + 2 * t_out : 2, j : j + 2 * f_out : 2]
            out += patch @ weight[:, i, j, :].T
    return out


class AudioTower:
    def __init__(self, config, weights, mutation=None):
        self.c = config["audio_config"]
        self.w = {k: v.astype(np.float32) for k, v in weights.items()
                  if k.startswith(("audio_tower.", "embed_audio."))}
        self.mutation = mutation

    def clip_linear(self, x, w, name):
        if self.mutation == "no_clip":
            return x @ w[name + ".linear.weight"].T
        return clip_linear(x, w, name)

    def sscp(self, features, valid):
        c, w = self.c, self.w
        x = features[:, :, None]
        for index in range(len(c["subsampling_conv_channels"])):
            p = f"audio_tower.subsample_conv_projection.layer{index}."
            x = x * valid[:, None, None]
            x = conv2d_s2(x, w[p + "conv.weight"])
            valid = valid[::2][: x.shape[0]]
            x = np.maximum(layer_norm(x, w[p + "norm.weight"], c["rms_norm_eps"]), 0)
        x = x.reshape(x.shape[0], -1)
        return x @ w["audio_tower.subsample_conv_projection.input_proj_linear.weight"].T, valid

    def attention(self, x, valid, p):
        c, w = self.c, self.w
        length = x.shape[0]
        heads = c["num_attention_heads"]
        dim = c["hidden_size"] // heads
        past = max(0, c["attention_context_left"] - 1)
        future = c["attention_context_right"]
        if self.mutation == "wide_window":
            past += 1
        q = self.clip_linear(x, w, p + "q_proj").reshape(length, heads, dim)
        k = self.clip_linear(x, w, p + "k_proj").reshape(length, heads, dim)
        v = self.clip_linear(x, w, p + "v_proj").reshape(length, heads, dim)
        q_scale = dim**-0.5 / math.log(2)
        k_scale = math.log(1 + math.e) / math.log(2)
        q = q * (np.float32(q_scale) * softplus(w[p + "per_dim_scale"]))
        k = k * np.float32(k_scale)

        # Sinusoids at offsets past, ..., -future, projected per head.
        offsets = np.arange(past, -future - 1, -1)
        if self.mutation == "wide_window":
            offsets = np.arange(past - 1, -future - 1, -1)
        half = c["hidden_size"] // 2
        inv = np.exp(np.arange(half, dtype=np.float32) * np.float32(-math.log(10000.0) / max(half - 1, 1)))
        angles = offsets.astype(np.float32)[:, None] * inv
        timing = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
        relative = (timing @ w[p + "relative_k_proj.weight"].T).reshape(len(offsets), heads, dim)

        t = np.arange(length)[:, None]
        s = np.arange(length)[None, :]
        distance = t - s
        allowed = ((distance >= 0) & (distance < past)) | ((distance < 0) & (-distance < future))
        allowed &= valid[None, :]
        content = np.einsum("thd,shd->hts", q, k)
        index = np.clip(offsets[0] - distance, 0, len(offsets) - 1)
        position = np.einsum("thd,ohd->hto", q, relative)
        position = np.take_along_axis(position, np.broadcast_to(index, (heads, length, length)), axis=-1)
        logits = content + position
        cap = np.float32(c["attention_logit_cap"])
        if self.mutation != "no_softcap":
            logits = np.tanh(logits / cap) * cap
        logits = np.where(allowed[None], logits, np.float32(c["attention_invalid_logits_value"]))
        logits = logits - logits.max(-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(-1, keepdims=True)
        context = np.einsum("hts,shd->thd", probs, v).reshape(length, -1)
        return self.clip_linear(context, w, p + "post")

    def feed_forward(self, x, p):
        c, w = self.c, self.w
        eps = np.float32(c["rms_norm_eps"])
        g = np.float32(c["gradient_clipping"])
        h = rms_norm(np.clip(x, -g, g), w[p + "pre_layer_norm.weight"], eps)
        h = self.clip_linear(silu(self.clip_linear(h, w, p + "ffw_layer_1")), w, p + "ffw_layer_2")
        h = rms_norm(np.clip(h, -g, g), w[p + "post_layer_norm.weight"], eps)
        return x + h * np.float32(c["residual_weight"])

    def light_conv(self, x, p):
        c, w = self.c, self.w
        eps = np.float32(c["rms_norm_eps"])
        g = np.float32(c["gradient_clipping"])
        h = self.clip_linear(rms_norm(x, w[p + "pre_layer_norm.weight"], eps), w, p + "linear_start")
        a, b = np.split(h, 2, axis=-1)
        h = a / (np.float32(1) + np.exp(-b))
        kernel = w[p + "depthwise_conv1d.weight"][:, :, 0]  # [channels, K]
        size = kernel.shape[1]
        h = np.pad(h, ((size - 1, 0), (0, 0)))
        h = sum(h[j : j + x.shape[0]] * kernel[:, j] for j in range(size))
        h = rms_norm(np.clip(h, -g, g), w[p + "conv_norm.weight"], eps)
        return self.clip_linear(silu(h), w, p + "linear_end") + x

    def features(self, wave, fe):
        """Soft tokens `[valid_frames, text_hidden]` for one waveform."""
        c, w = self.c, self.w
        mel, valid = log_mel(wave, fe, self.mutation)
        x, valid = self.sscp(mel, valid)
        eps = np.float32(c["rms_norm_eps"])
        g = np.float32(c["gradient_clipping"])
        for i in range(c["num_hidden_layers"]):
            p = f"audio_tower.layers.{i}."
            x = self.feed_forward(x, p + "feed_forward1.")
            h = rms_norm(np.clip(x, -g, g), w[p + "norm_pre_attn.weight"], eps)
            h = np.clip(self.attention(h, valid, p + "self_attn."), -g, g)
            x = x + rms_norm(h, w[p + "norm_post_attn.weight"], eps)
            x = x * valid[:, None]
            x = self.light_conv(x, p + "lconv1d.")
            x = self.feed_forward(x, p + "feed_forward2.")
            x = rms_norm(np.clip(x, -g, g), w[p + "norm_out.weight"], eps)
        x = x @ w["audio_tower.output_proj.weight"].T + w["audio_tower.output_proj.bias"]
        x = x[valid]
        x = rms_norm(x, np.float32(1), eps)
        return x @ w["embed_audio.embedding_projection.weight"].T


def audio_sequence(config, tokens):
    text = config["text_config"]
    return ([text["bos_token_id"], config["boa_token_id"]] + [config["audio_token_id"]] * tokens
            + [config["eoa_token_index"], text["eos_token_id"]])


class EmbeddingGemma2Audio(EmbeddingGemma2Text):
    def __init__(self, config, weights, feature_extractor, mutation=None):
        super().__init__(config, weights, mutation)
        self.config = config
        self.fe = feature_extractor
        self.audio = AudioTower(config, weights, mutation)

    def embed_audio(self, waves):
        features = [self.audio.features(wave, self.fe) for wave in waves]
        sequences = [audio_sequence(self.config, len(f)) for f in features]
        length = max(map(len, sequences))
        pad = self.text["pad_token_id"]
        ids = np.array([s + [pad] * (length - len(s)) for s in sequences], np.int64)
        valid = np.array([[True] * len(s) + [False] * (length - len(s)) for s in sequences])
        audio = ids == self.config["audio_token_id"]
        x = self.w("embed_tokens.weight")[np.where(audio, pad, ids)] * np.float32(np.sqrt(self.text["hidden_size"]))
        for row, f in enumerate(features):
            x[row, audio[row]] = f
        return self.pool(self.hidden_from_embeddings(x, valid), valid)
