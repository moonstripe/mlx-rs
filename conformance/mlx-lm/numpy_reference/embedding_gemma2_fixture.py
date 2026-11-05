"""Writes the tiny seeded EmbeddingGemma 2 text fixture and its NumPy expectations.

The fixture exercises sliding and full layers, a `per_layer_config` override,
GQA, per-layer inputs, non-unit layer scalars and right padding. Each named
mutation of the reference must move some expectation beyond the tolerance.
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from numpy_reference.common import read_json, read_safetensors, write_safetensors
from numpy_reference.embedding_gemma2 import EmbeddingGemma2Text

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "embedding-fixtures"
NUMPY_VERSION = "2.2.6"
SEED = 1729
TOLERANCE = {"atol": 2e-4, "rtol": 2e-4}
MUTATIONS = ["no_v_norm", "window_exclusive", "no_layer_scalar", "pool_padding"]

SPECIAL = ["<pad>", "<eos>", "<bos>", "<unk>"]
WORDS = (
    "the a small large red green blue fox dog cat bird runs jumps sleeps over under "
    "near hill river tree stone house road quick slow bright dark old new one two three "
    "four five planet mars star moon sun task query text title none search result"
).split()
MEDIA = ["<|image|>", "<|audio|>", "<|video|>"]
VOCAB = SPECIAL + WORDS + [f"w{i}" for i in range(64 - len(SPECIAL) - len(WORDS) - len(MEDIA))] + MEDIA

CONFIG = {
    "model_type": "embedding_gemma2",
    "image_token_id": VOCAB.index("<|image|>"),
    "audio_token_id": VOCAB.index("<|audio|>"),
    "video_token_id": VOCAB.index("<|video|>"),
    "text_config": {
        "model_type": "embedding_gemma2_text",
        "attention_bias": False,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "embedding_dim": 24,
        "head_dim": 8,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 32,
        "hidden_size_per_layer_input": 8,
        "intermediate_size": 48,
        "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
        "max_position_embeddings": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 3,
        "num_key_value_heads": 2,
        "per_layer_config": {"02": {"head_dim": 16, "num_key_value_heads": 1}},
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
        "sliding_window": 2,
        "vocab_size": len(VOCAB),
    },
}

# Lengths 3..12 with the batch padded to the longest; window 2 leaves most pairs masked.
TEXTS = [
    "task search result query planet mars",
    "the quick red fox jumps over the slow dog near the old stone house",
    "one",
    "title none text a bright star over the dark river",
]


def tokenizer_json():
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": VOCAB.index(t), "content": t, "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": False, "special": True}
            for t in SPECIAL + MEDIA
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<eos>", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": "<eos>", "type_id": 0}},
            ],
            "special_tokens": {
                t: {"id": t, "ids": [VOCAB.index(t)], "tokens": [t]} for t in ("<bos>", "<eos>")
            },
        },
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": {t: i for i, t in enumerate(VOCAB)}, "unk_token": "<unk>"},
    }


def shapes():
    text = CONFIG["text_config"]
    hidden, inter = text["hidden_size"], text["intermediate_size"]
    width, count = text["hidden_size_per_layer_input"], text["num_hidden_layers"]
    p = "language_model."
    out = {
        p + "embed_tokens.weight": (text["vocab_size"], hidden),
        p + "ple.per_layer_model_projection.weight": (count * width, hidden),
        p + "ple.per_layer_projection_norm.weight": (width,),
        p + "norm.weight": (hidden,),
        p + "embedding_projection.weight": (text["embedding_dim"], hidden),
    }
    for i in range(count):
        o = text["per_layer_config"].get(f"{i:02d}", {})
        d = o.get("head_dim", text["head_dim"])
        q = text["num_attention_heads"] * d
        kv = o.get("num_key_value_heads", text["num_key_value_heads"]) * d
        l = f"{p}layers.{i}."
        out.update({
            l + "self_attn.q_proj.weight": (q, hidden),
            l + "self_attn.k_proj.weight": (kv, hidden),
            l + "self_attn.v_proj.weight": (kv, hidden),
            l + "self_attn.o_proj.weight": (hidden, q),
            l + "self_attn.q_norm.weight": (d,),
            l + "self_attn.k_norm.weight": (d,),
            l + "mlp.gate_proj.weight": (inter, hidden),
            l + "mlp.up_proj.weight": (inter, hidden),
            l + "mlp.down_proj.weight": (hidden, inter),
            l + "input_layernorm.weight": (hidden,),
            l + "post_attention_layernorm.weight": (hidden,),
            l + "pre_feedforward_layernorm.weight": (hidden,),
            l + "post_feedforward_layernorm.weight": (hidden,),
            l + "ple_block.per_layer_input_gate.weight": (width, hidden),
            l + "ple_block.per_layer_projection.weight": (hidden, width),
            l + "ple_block.post_per_layer_input_norm.weight": (hidden,),
            l + "layer_scalar": (1,),
        })
    return out


def weights():
    rng = np.random.Generator(np.random.PCG64(SEED))
    tensors = {}
    for name, shape in sorted(shapes().items()):
        if len(shape) == 2:
            value = rng.normal(0, shape[1] ** -0.5, shape)
        elif name.endswith("layer_scalar"):
            value = rng.uniform(0.5, 1.5, shape)
        else:
            value = rng.uniform(0.5, 1.5, shape)
        tensors[name] = value.astype(np.float32)
    # Media towers are present in real checkpoints and must be ignored by the text embedder.
    tensors["vision_tower.patch_embedder.input_proj.weight"] = np.zeros((4, 4), np.float32)
    tensors["embed_audio.embedding_projection.weight"] = np.zeros((4, 4), np.float32)
    return tensors


def encode(text):
    return [2] + [VOCAB.index(word) for word in text.split()] + [1]


def dump(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_common(directory, config, tokenizer, tensors):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(dump(config))
    (directory / "tokenizer.json").write_text(dump(tokenizer))
    (directory / "tokenizer_config.json").write_text(dump({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}))
    write_safetensors(directory / "model.safetensors", tensors)
    return read_safetensors(directory / "model.safetensors")



def write_text(directory):
    directory = Path(directory)
    tensors = write_common(directory, CONFIG, tokenizer_json(), weights())
    ids = [encode(text) for text in TEXTS]
    batch = EmbeddingGemma2Text(CONFIG, tensors).embed(ids)
    singles = np.concatenate([EmbeddingGemma2Text(CONFIG, tensors).embed([s]) for s in ids])
    if not np.isfinite(batch).all():
        raise ValueError("reference embeddings are not finite")
    # Padding must not leak into valid rows; the reference itself proves it before blessing.
    if not np.allclose(batch, singles, atol=1e-6, rtol=0):
        raise ValueError("reference rows depend on batch padding")
    for mutation in MUTATIONS:
        mutated = EmbeddingGemma2Text(CONFIG, tensors, mutation).embed(ids)
        excess = np.abs(mutated - batch) - (TOLERANCE["atol"] + TOLERANCE["rtol"] * np.abs(batch))
        if excess.max() <= 0:
            raise ValueError(f"fixture does not detect mutation {mutation}")
    (directory / "inputs.json").write_text(dump({"texts": TEXTS, "token_ids": ids, "seed": SEED}))
    (directory / "expectations.json").write_text(dump({
        "provenance": f"numpy {NUMPY_VERSION} f32 reference, numpy_reference/embedding_gemma2.py",
        "tolerances": {"embeddings": TOLERANCE},
        "mutations_detected": MUTATIONS,
    }))
    write_safetensors(directory / "expectations.safetensors", {"embeddings": batch})


VISION_SEED = SEED + 1
VISION_MUTATIONS = ["channel_major_patches", "swap_axes", "column_major_pool", "vision_no_v_norm"]


def vision_config():
    config = json.loads(json.dumps(CONFIG))
    config["boi_token_id"] = VOCAB.index("w0")
    config["eoi_token_id"] = VOCAB.index("w1")
    # GQA, an axial head of 8 (two rotated halves of 4), 2x2 patches pooled 3x3.
    config["vision_config"] = {
        "model_type": "gemma4_vision",
        "attention_bias": False,
        "default_output_length": 6,
        "global_head_dim": 8,
        "head_dim": 8,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "patch_size": 2,
        "pooling_kernel_size": 3,
        "position_embedding_size": 16,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 100.0, "rope_type": "axial"},
        "standardize": False,
        "use_clipped_linears": False,
    }
    return config


def vision_tokenizer():
    tokenizer = tokenizer_json()
    vocab = tokenizer["model"]["vocab"]
    for filler, name in (("w0", "<|image>"), ("w1", "<image|>")):
        vocab[name] = vocab.pop(filler)
        tokenizer["added_tokens"].append({"id": vocab[name], "content": name, "single_word": False,
                                          "lstrip": False, "rstrip": False, "normalized": False, "special": True})
    tokenizer["added_tokens"].sort(key=lambda token: token["id"])
    return tokenizer


def vision_weights(config):
    c = config["vision_config"]
    hidden, inter, dim = c["hidden_size"], c["intermediate_size"], c["head_dim"]
    q, kv = c["num_attention_heads"] * dim, c["num_key_value_heads"] * dim
    shapes = {
        "vision_tower.patch_embedder.input_proj.weight": (hidden, 3 * c["patch_size"] ** 2),
        "vision_tower.patch_embedder.position_embedding_table": (2, c["position_embedding_size"], hidden),
        "embed_vision.embedding_projection.weight": (config["text_config"]["hidden_size"], hidden),
    }
    for i in range(c["num_hidden_layers"]):
        l = f"vision_tower.encoder.layers.{i}."
        shapes.update({
            l + "self_attn.q_proj.linear.weight": (q, hidden),
            l + "self_attn.k_proj.linear.weight": (kv, hidden),
            l + "self_attn.v_proj.linear.weight": (kv, hidden),
            l + "self_attn.o_proj.linear.weight": (hidden, q),
            l + "self_attn.q_norm.weight": (dim,),
            l + "self_attn.k_norm.weight": (dim,),
            l + "mlp.gate_proj.linear.weight": (inter, hidden),
            l + "mlp.up_proj.linear.weight": (inter, hidden),
            l + "mlp.down_proj.linear.weight": (hidden, inter),
            l + "input_layernorm.weight": (hidden,),
            l + "post_attention_layernorm.weight": (hidden,),
            l + "pre_feedforward_layernorm.weight": (hidden,),
            l + "post_feedforward_layernorm.weight": (hidden,),
        })
    rng = np.random.Generator(np.random.PCG64(VISION_SEED))
    tensors = {}
    for name, shape in sorted(shapes.items()):
        if name.endswith("position_embedding_table"):
            value = rng.normal(0, 0.5, shape)
        elif len(shape) == 2:
            value = rng.normal(0, shape[1] ** -0.5, shape)
        else:
            value = rng.uniform(0.5, 1.5, shape)
        tensors[name] = value.astype(np.float32)
    return tensors


def vision_images():
    rng = np.random.Generator(np.random.PCG64(VISION_SEED))
    # 6 x 9 patches pool to 2 x 3 tokens; 6 x 6 pixels is one token. Non-square
    # grids distinguish rows from columns.
    return [rng.integers(0, 256, shape, dtype=np.uint8) for shape in ((12, 18, 3), (6, 6, 3))]


def write_vision(directory):
    from numpy_reference.embedding_gemma2_vision import EmbeddingGemma2Images

    directory = Path(directory)
    config = vision_config()
    text = {k: v for k, v in weights().items() if not k.startswith("vision_tower.")}
    tensors = write_common(directory, config, vision_tokenizer(), {**text, **vision_weights(config)})
    images = vision_images()
    batch = EmbeddingGemma2Images(config, tensors).embed_images(images)
    singles = np.concatenate([EmbeddingGemma2Images(config, tensors).embed_images([i]) for i in images])
    if not np.isfinite(batch).all():
        raise ValueError("reference embeddings are not finite")
    if not np.allclose(batch, singles, atol=1e-6, rtol=0):
        raise ValueError("reference rows depend on batch padding")
    for mutation in VISION_MUTATIONS:
        mutated = EmbeddingGemma2Images(config, tensors, mutation).embed_images(images)
        excess = np.abs(mutated - batch) - (TOLERANCE["atol"] + TOLERANCE["rtol"] * np.abs(batch))
        if excess.max() <= 0:
            raise ValueError(f"fixture does not detect mutation {mutation}")
    write_safetensors(directory / "images.safetensors", {f"image{i}": image for i, image in enumerate(images)})
    (directory / "expectations.json").write_text(dump({
        "provenance": f"numpy {NUMPY_VERSION} f32 reference, numpy_reference/embedding_gemma2_vision.py",
        "tolerances": {"embeddings": TOLERANCE},
        "mutations_detected": VISION_MUTATIONS,
    }))
    write_safetensors(directory / "expectations.safetensors", {"embeddings": batch})


AUDIO_SEED = SEED + 2
AUDIO_MUTATIONS = ["wide_window", "frame_end_inclusive", "no_clip", "no_softcap"]
FRONT_END = {
    "feature_extractor_type": "Gemma4AudioFeatureExtractor",
    "dither": 0.0,
    "feature_size": 16,
    "fft_length": 64,
    "fft_overdrive": False,
    "frame_length": 40,
    "hop_length": 20,
    "input_scale_factor": 1.0,
    "max_frequency": 800.0,
    "mel_floor": 0.001,
    "min_frequency": 0.0,
    "per_bin_mean": None,
    "per_bin_stddev": None,
    "preemphasis": 0.0,
    "preemphasis_htk_flavor": True,
    "sampling_rate": 1600,
}


def audio_config():
    config = json.loads(json.dumps(CONFIG))
    config["boa_token_id"] = VOCAB.index("w2")
    config["eoa_token_index"] = VOCAB.index("w3")
    # Two heads of 8, a 3-key past window in chunks of 4, active clipping and soft-capping.
    config["audio_config"] = {
        "model_type": "gemma4_audio",
        "attention_chunk_size": 4,
        "attention_context_left": 4,
        "attention_context_right": 0,
        "attention_invalid_logits_value": -1e9,
        "attention_logit_cap": 2.0,
        "conv_kernel_size": 3,
        "gradient_clipping": 3.0,
        "hidden_act": "silu",
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "output_proj_dims": 24,
        "residual_weight": 0.5,
        "rms_norm_eps": 1e-6,
        "subsampling_conv_channels": [4, 2],
        "use_clipped_linears": True,
    }
    return config


def audio_tokenizer():
    tokenizer = tokenizer_json()
    vocab = tokenizer["model"]["vocab"]
    for filler, name in (("w2", "<|audio>"), ("w3", "<audio|>")):
        vocab[name] = vocab.pop(filler)
        tokenizer["added_tokens"].append({"id": vocab[name], "content": name, "single_word": False,
                                          "lstrip": False, "rstrip": False, "normalized": False, "special": True})
    tokenizer["added_tokens"].sort(key=lambda token: token["id"])
    return tokenizer


def audio_weights(config):
    c = config["audio_config"]
    d, k = c["hidden_size"], c["conv_kernel_size"]
    rng = np.random.Generator(np.random.PCG64(AUDIO_SEED))
    tensors = {}
    channels_in, bins = 1, FRONT_END["feature_size"]
    for i, channels in enumerate(c["subsampling_conv_channels"]):
        p = f"audio_tower.subsample_conv_projection.layer{i}."
        tensors[p + "conv.weight"] = rng.normal(0, (9 * channels_in) ** -0.5, (channels, 3, 3, channels_in))
        tensors[p + "norm.weight"] = rng.uniform(0.5, 1.5, (channels,))
        channels_in, bins = channels, (bins + 2 - 3) // 2 + 1
    tensors["audio_tower.subsample_conv_projection.input_proj_linear.weight"] = rng.normal(
        0, (bins * channels_in) ** -0.5, (d, bins * channels_in))
    for i in range(c["num_hidden_layers"]):
        p = f"audio_tower.layers.{i}."
        linears = {"feed_forward1.ffw_layer_1": (4 * d, d), "feed_forward1.ffw_layer_2": (d, 4 * d),
                   "feed_forward2.ffw_layer_1": (4 * d, d), "feed_forward2.ffw_layer_2": (d, 4 * d),
                   "self_attn.q_proj": (d, d), "self_attn.k_proj": (d, d), "self_attn.v_proj": (d, d),
                   "self_attn.post": (d, d), "lconv1d.linear_start": (2 * d, d), "lconv1d.linear_end": (d, d)}
        for name, shape in linears.items():
            # Scales above 1/sqrt(fan-in) and tight bounds make every clamp bind somewhere.
            tensors[p + name + ".linear.weight"] = rng.normal(0, 2.0 * shape[1] ** -0.5, shape)
            tensors[p + name + ".input_min"] = np.array(-rng.uniform(1.0, 2.0))
            tensors[p + name + ".input_max"] = np.array(rng.uniform(1.0, 2.0))
            tensors[p + name + ".output_min"] = np.array(-rng.uniform(1.0, 2.0))
            tensors[p + name + ".output_max"] = np.array(rng.uniform(1.0, 2.0))
        for norm in ["feed_forward1.pre_layer_norm", "feed_forward1.post_layer_norm",
                     "feed_forward2.pre_layer_norm", "feed_forward2.post_layer_norm",
                     "lconv1d.pre_layer_norm", "lconv1d.conv_norm", "norm_pre_attn",
                     "norm_post_attn", "norm_out"]:
            tensors[p + norm + ".weight"] = rng.uniform(0.5, 1.5, (d,))
        tensors[p + "lconv1d.depthwise_conv1d.weight"] = rng.normal(0, k ** -0.5, (d, k, 1))
        tensors[p + "self_attn.per_dim_scale"] = rng.normal(0, 0.5, (d // c["num_attention_heads"],))
        tensors[p + "self_attn.relative_k_proj.weight"] = rng.normal(0, 2.0 * d ** -0.5, (d, d))
    out = c["output_proj_dims"]
    tensors["audio_tower.output_proj.weight"] = rng.normal(0, d ** -0.5, (out, d))
    tensors["audio_tower.output_proj.bias"] = rng.normal(0, 0.1, (out,))
    tensors["embed_audio.embedding_projection.weight"] = rng.normal(
        0, out ** -0.5, (config["text_config"]["hidden_size"], out))
    return {name: np.asarray(value, np.float32) for name, value in tensors.items()}


def audio_waves():
    rng = np.random.Generator(np.random.PCG64(AUDIO_SEED))
    t = np.arange(900) / FRONT_END["sampling_rate"]
    chirp = 0.5 * np.sin(2 * np.pi * (60 + 300 * t) * t) + 0.05 * rng.standard_normal(t.size)
    # 900 samples give 11 soft tokens and 300 give 4, padded in one batch.
    return [chirp.astype(np.float32), (0.3 * rng.standard_normal(300)).astype(np.float32)]


def write_audio(directory):
    from numpy_reference.embedding_gemma2_audio import EmbeddingGemma2Audio

    directory = Path(directory)
    config = audio_config()
    text = {k: v for k, v in weights().items() if not k.startswith(("vision_tower.", "embed_audio."))}
    tensors = write_common(directory, config, audio_tokenizer(), {**text, **audio_weights(config)})
    (directory / "preprocessor_config.json").write_text(dump(FRONT_END))
    waves = audio_waves()
    batch = EmbeddingGemma2Audio(config, tensors, FRONT_END).embed_audio(waves)
    singles = np.concatenate([EmbeddingGemma2Audio(config, tensors, FRONT_END).embed_audio([w]) for w in waves])
    if not np.isfinite(batch).all():
        raise ValueError("reference embeddings are not finite")
    if not np.allclose(batch, singles, atol=1e-6, rtol=0):
        raise ValueError("reference rows depend on batch padding")
    for mutation in AUDIO_MUTATIONS:
        mutated = EmbeddingGemma2Audio(config, tensors, FRONT_END, mutation).embed_audio(waves)
        excess = np.abs(mutated - batch) - (TOLERANCE["atol"] + TOLERANCE["rtol"] * np.abs(batch))
        if excess.max() <= 0:
            raise ValueError(f"fixture does not detect mutation {mutation}")
    write_safetensors(directory / "waveforms.safetensors", {f"waveform{i}": w for i, w in enumerate(waves)})
    (directory / "expectations.json").write_text(dump({
        "provenance": f"numpy {NUMPY_VERSION} f32 reference, numpy_reference/embedding_gemma2_audio.py",
        "tolerances": {"embeddings": TOLERANCE},
        "mutations_detected": AUDIO_MUTATIONS,
        "soft_tokens": [11, 4],
    }))
    write_safetensors(directory / "expectations.safetensors", {"embeddings": batch})


FIXTURES = {
    "embedding-gemma2-text": write_text,
    "embedding-gemma2-vision": write_vision,
    "embedding-gemma2-audio": write_audio,
}


def digest(directory):
    h = hashlib.sha256()
    for path in sorted(Path(directory).iterdir()):
        h.update(path.name.encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="regenerate and compare without writing")
    args = parser.parse_args()
    if np.__version__ != NUMPY_VERSION:
        parser.error(f"requires numpy {NUMPY_VERSION}, got {np.__version__}")
    for name, write in FIXTURES.items():
        output = OUTPUT / name
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            write(a)
            write(b)
            if digest(a) != digest(b):
                raise SystemExit(f"{name}: generation is not deterministic")
            if args.check:
                if not output.is_dir() or digest(a) != digest(output):
                    raise SystemExit(f"{output} differs from regeneration")
                print(f"{name}: matches regeneration")
                continue
        write(output)
        print(output)


if __name__ == "__main__":
    main()
