use crate::ConfigError;
use serde::Deserialize;
use std::collections::BTreeMap;

/// The only `config.json` model type the embedder admits.
pub(super) const MODEL_TYPE: &str = "embedding_gemma2";

/// Validated text-encoder facts; every dimension fits MLX's `i32` shapes.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct EncoderConfig {
    pub(super) vocabulary_size: usize,
    pub(super) hidden_size: usize,
    pub(super) intermediate_size: usize,
    pub(super) per_layer_width: usize,
    pub(super) embedding_dim: usize,
    pub(super) rms_norm_eps: f32,
    pub(super) sliding_window: usize,
    pub(super) max_positions: usize,
    pub(super) pad_token_id: u32,
    pub(super) media_tokens: Vec<u32>,
    pub(super) layers: Vec<LayerShape>,
    /// Present when the checkpoint declares a vision tower.
    pub(super) vision: Option<VisionConfig>,
    /// Present when the checkpoint declares an audio tower.
    pub(super) audio: Option<AudioConfig>,
}

/// Validated Gemma 4 vision-tower facts and the image placeholder tokens.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct VisionConfig {
    pub(super) hidden_size: usize,
    pub(super) intermediate_size: usize,
    pub(super) layers: usize,
    pub(super) heads: usize,
    pub(super) kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) patch_size: usize,
    pub(super) pooling_kernel_size: usize,
    pub(super) position_embedding_size: usize,
    /// Upper bound on soft tokens per image (`default_output_length`).
    pub(super) max_soft_tokens: usize,
    pub(super) rms_norm_eps: f32,
    pub(super) rope_theta: f32,
    pub(super) image_token_id: u32,
    pub(super) boi_token_id: u32,
    pub(super) eoi_token_id: u32,
}

/// Validated Gemma 4 audio-encoder facts and the audio placeholder tokens.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct AudioConfig {
    pub(super) hidden_size: usize,
    pub(super) layers: usize,
    pub(super) heads: usize,
    /// Output channels of each stride-2 subsampling convolution.
    pub(super) subsampling_channels: Vec<usize>,
    pub(super) conv_kernel_size: usize,
    pub(super) chunk_past: usize,
    pub(super) chunk_future: usize,
    pub(super) invalid_logit: f32,
    pub(super) logit_cap: f32,
    pub(super) gradient_clipping: f32,
    pub(super) residual_weight: f32,
    pub(super) output_dims: usize,
    pub(super) rms_norm_eps: f32,
    pub(super) audio_token_id: u32,
    pub(super) boa_token_id: u32,
    pub(super) eoa_token_id: u32,
}

/// The admitted log-mel front-end from `preprocessor_config.json`.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct FrontEnd {
    pub(super) sampling_rate: usize,
    pub(super) feature_size: usize,
    pub(super) frame_length: usize,
    pub(super) hop_length: usize,
    pub(super) fft_length: usize,
    pub(super) min_frequency: f64,
    pub(super) max_frequency: f64,
    pub(super) mel_floor: f64,
}

/// One encoder layer's attention geometry after `per_layer_config` overrides.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct LayerShape {
    pub(super) heads: usize,
    pub(super) kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) sliding: bool,
    pub(super) rope_theta: f32,
}

#[derive(Deserialize)]
struct RawConfig {
    text_config: RawText,
    image_token_id: Option<u32>,
    audio_token_id: Option<u32>,
    video_token_id: Option<u32>,
    boi_token_id: Option<u32>,
    eoi_token_id: Option<u32>,
    vision_config: Option<RawVision>,
    boa_token_id: Option<u32>,
    eoa_token_index: Option<u32>,
    audio_config: Option<RawAudio>,
}

#[derive(Deserialize)]
struct RawAudio {
    model_type: String,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    subsampling_conv_channels: Vec<usize>,
    conv_kernel_size: usize,
    attention_chunk_size: usize,
    attention_context_left: usize,
    attention_context_right: usize,
    attention_invalid_logits_value: f64,
    attention_logit_cap: f64,
    gradient_clipping: f64,
    hidden_act: String,
    output_proj_dims: Option<usize>,
    residual_weight: f64,
    rms_norm_eps: f64,
    use_clipped_linears: bool,
}

#[derive(Deserialize)]
struct RawFrontEnd {
    feature_extractor_type: String,
    sampling_rate: usize,
    feature_size: usize,
    frame_length: usize,
    hop_length: usize,
    fft_length: usize,
    min_frequency: f64,
    max_frequency: f64,
    mel_floor: f64,
    #[serde(default)]
    dither: f64,
    #[serde(default)]
    preemphasis: f64,
    #[serde(default)]
    fft_overdrive: bool,
    #[serde(default = "unit")]
    input_scale_factor: f64,
    per_bin_mean: Option<serde_json::Value>,
    per_bin_stddev: Option<serde_json::Value>,
}

fn unit() -> f64 {
    1.0
}

#[derive(Deserialize)]
struct RawVision {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    global_head_dim: Option<usize>,
    patch_size: usize,
    pooling_kernel_size: usize,
    position_embedding_size: usize,
    default_output_length: usize,
    rms_norm_eps: f64,
    hidden_activation: String,
    attention_bias: bool,
    #[serde(default)]
    use_clipped_linears: bool,
    #[serde(default)]
    standardize: bool,
    rope_parameters: RawAxialRope,
}

#[derive(Deserialize)]
struct RawAxialRope {
    rope_theta: f64,
    rope_type: String,
}

#[derive(Deserialize)]
struct RawText {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    hidden_size_per_layer_input: usize,
    embedding_dim: usize,
    rms_norm_eps: f64,
    sliding_window: usize,
    max_position_embeddings: usize,
    attention_bias: bool,
    hidden_activation: String,
    pad_token_id: u32,
    layer_types: Vec<String>,
    per_layer_config: BTreeMap<String, Override>,
    rope_parameters: BTreeMap<String, RawRope>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Override {
    head_dim: Option<usize>,
    num_key_value_heads: Option<usize>,
    num_attention_heads: Option<usize>,
}

#[derive(Deserialize)]
struct RawRope {
    rope_theta: f64,
    rope_type: String,
}

fn invalid(field: impl Into<String>, reason: impl Into<String>) -> ConfigError {
    ConfigError::InvalidNumericField {
        field: field.into(),
        reason: reason.into(),
    }
}

fn positive(field: &str, value: usize) -> Result<usize, ConfigError> {
    if value == 0 || i32::try_from(value).is_err() {
        return Err(invalid(field, "must be positive and fit in i32"));
    }
    Ok(value)
}

pub(super) fn parse(bytes: &[u8]) -> Result<EncoderConfig, ConfigError> {
    let value: serde_json::Value = serde_json::from_slice(bytes)?;
    let model_type = value
        .get("model_type")
        .ok_or_else(|| ConfigError::MissingField {
            field: "model_type".into(),
        })?;
    if model_type != MODEL_TYPE {
        return Err(ConfigError::UnsupportedArchitecture(
            model_type
                .as_str()
                .unwrap_or("non-string model_type")
                .into(),
        ));
    }
    let raw: RawConfig = serde_json::from_value(value)?;
    let text = raw.text_config;
    if text.attention_bias {
        return Err(ConfigError::UnsupportedArchitecture(format!(
            "{MODEL_TYPE} with attention_bias"
        )));
    }
    if text.hidden_activation != "gelu_pytorch_tanh" {
        return Err(ConfigError::UnsupportedArchitecture(format!(
            "{MODEL_TYPE} with hidden_activation {}",
            text.hidden_activation
        )));
    }
    let vocabulary_size = positive("vocab_size", text.vocab_size)?;
    let hidden_size = positive("hidden_size", text.hidden_size)?;
    let intermediate_size = positive("intermediate_size", text.intermediate_size)?;
    let layer_count = positive("num_hidden_layers", text.num_hidden_layers)?;
    let per_layer_width = positive(
        "hidden_size_per_layer_input",
        text.hidden_size_per_layer_input,
    )?;
    let embedding_dim = positive("embedding_dim", text.embedding_dim)?;
    let sliding_window = positive("sliding_window", text.sliding_window)?;
    let max_positions = positive("max_position_embeddings", text.max_position_embeddings)?;
    // Packed per-layer inputs are one [hidden, layers * width] projection.
    layer_count
        .checked_mul(per_layer_width)
        .filter(|&n| i32::try_from(n).is_ok())
        .ok_or_else(|| invalid("hidden_size_per_layer_input", "layers * width exceeds i32"))?;
    let rms_norm_eps = text.rms_norm_eps as f32;
    if !rms_norm_eps.is_finite() || rms_norm_eps <= 0.0 {
        return Err(invalid("rms_norm_eps", "must be finite and positive"));
    }
    if text.pad_token_id as usize >= vocabulary_size {
        return Err(invalid("pad_token_id", "outside the vocabulary"));
    }
    if text.layer_types.len() != layer_count {
        return Err(ConfigError::InvalidLayerPattern {
            expected: layer_count,
            actual: text.layer_types.len(),
        });
    }
    // The reference encoder forces the final layer to full attention; reject rather than rewrite.
    if text.layer_types.last().map(String::as_str) != Some("full_attention") {
        return Err(invalid(
            "layer_types",
            "the final layer must be full_attention",
        ));
    }
    let mut overrides = BTreeMap::new();
    for (key, value) in text.per_layer_config {
        let index = key
            .parse::<usize>()
            .ok()
            .filter(|&index| index < layer_count)
            .ok_or_else(|| invalid("per_layer_config", format!("invalid layer index {key:?}")))?;
        if overrides.insert(index, value).is_some() {
            return Err(invalid(
                "per_layer_config",
                format!("duplicate layer index {key:?}"),
            ));
        }
    }
    let mut thetas = BTreeMap::new();
    for (kind, rope) in &text.rope_parameters {
        if rope.rope_type != "default" {
            return Err(ConfigError::UnsupportedRope(format!(
                "{kind}: {}",
                rope.rope_type
            )));
        }
        let theta = rope.rope_theta as f32;
        if !theta.is_finite() || theta <= 0.0 {
            return Err(invalid(
                format!("rope_parameters.{kind}.rope_theta"),
                "must be finite and positive",
            ));
        }
        thetas.insert(kind.as_str(), theta);
    }
    let mut layers = Vec::with_capacity(layer_count);
    for (index, kind) in text.layer_types.iter().enumerate() {
        let sliding = match kind.as_str() {
            "sliding_attention" => true,
            "full_attention" => false,
            other => {
                return Err(ConfigError::UnsupportedArchitecture(format!(
                    "{MODEL_TYPE} layer type {other}"
                )))
            }
        };
        let rope_theta = *thetas
            .get(kind.as_str())
            .ok_or_else(|| ConfigError::MissingField {
                field: format!("rope_parameters.{kind}"),
            })?;
        let layer = overrides.get(&index);
        let field = |name: &str| format!("per_layer_config.{index:02}.{name}");
        let heads = positive(
            &field("num_attention_heads"),
            layer
                .and_then(|o| o.num_attention_heads)
                .unwrap_or(text.num_attention_heads),
        )?;
        let kv_heads = positive(
            &field("num_key_value_heads"),
            layer
                .and_then(|o| o.num_key_value_heads)
                .unwrap_or(text.num_key_value_heads),
        )?;
        let head_dim = positive(
            &field("head_dim"),
            layer.and_then(|o| o.head_dim).unwrap_or(text.head_dim),
        )?;
        if heads % kv_heads != 0 {
            return Err(invalid(
                field("num_key_value_heads"),
                "must divide num_attention_heads",
            ));
        }
        if head_dim % 2 != 0 {
            return Err(invalid(field("head_dim"), "RoPE requires an even head_dim"));
        }
        heads
            .checked_mul(head_dim)
            .filter(|&n| i32::try_from(n).is_ok())
            .ok_or_else(|| invalid(field("head_dim"), "heads * head_dim exceeds i32"))?;
        layers.push(LayerShape {
            heads,
            kv_heads,
            head_dim,
            sliding,
            rope_theta,
        });
    }
    let media_tokens: Vec<u32> = [raw.image_token_id, raw.audio_token_id, raw.video_token_id]
        .into_iter()
        .flatten()
        .collect();
    if let Some(&token) = media_tokens
        .iter()
        .chain(&raw.boi_token_id)
        .chain(&raw.eoi_token_id)
        .chain(&raw.boa_token_id)
        .chain(&raw.eoa_token_index)
        .find(|&&id| id as usize >= vocabulary_size)
    {
        return Err(invalid(
            "media token ids",
            format!("{token} is outside the vocabulary"),
        ));
    }
    let token = |field: &str, id: Option<u32>| {
        id.ok_or_else(|| ConfigError::MissingField {
            field: field.into(),
        })
    };
    let vision = raw
        .vision_config
        .map(|vision| {
            parse_vision(
                vision,
                token("image_token_id", raw.image_token_id)?,
                token("boi_token_id", raw.boi_token_id)?,
                token("eoi_token_id", raw.eoi_token_id)?,
            )
        })
        .transpose()?;
    let audio = raw
        .audio_config
        .map(|audio| {
            parse_audio(
                audio,
                token("audio_token_id", raw.audio_token_id)?,
                token("boa_token_id", raw.boa_token_id)?,
                token("eoa_token_index", raw.eoa_token_index)?,
            )
        })
        .transpose()?;
    Ok(EncoderConfig {
        vocabulary_size,
        hidden_size,
        intermediate_size,
        per_layer_width,
        embedding_dim,
        rms_norm_eps,
        sliding_window,
        max_positions,
        pad_token_id: text.pad_token_id,
        media_tokens,
        layers,
        vision,
        audio,
    })
}

fn parse_audio(
    raw: RawAudio,
    audio_token_id: u32,
    boa_token_id: u32,
    eoa_token_id: u32,
) -> Result<AudioConfig, ConfigError> {
    let unsupported = |what: String| {
        Err(ConfigError::UnsupportedArchitecture(format!(
            "{MODEL_TYPE} audio tower with {what}"
        )))
    };
    if raw.model_type != "gemma4_audio" {
        return unsupported(format!("model_type {}", raw.model_type));
    }
    if raw.hidden_act != "silu" {
        return unsupported(format!("hidden_act {}", raw.hidden_act));
    }
    // Both checkpoint forms carry clip bounds for every audio linear.
    if !raw.use_clipped_linears {
        return unsupported("unclipped linears".into());
    }
    let Some(output_dims) = raw.output_proj_dims else {
        return unsupported("no output projection".into());
    };
    let field = |name: &str| format!("audio_config.{name}");
    let hidden_size = positive(&field("hidden_size"), raw.hidden_size)?;
    let heads = positive(&field("num_attention_heads"), raw.num_attention_heads)?;
    if hidden_size % heads != 0 {
        return Err(invalid(
            field("num_attention_heads"),
            "must divide hidden_size",
        ));
    }
    if hidden_size % 2 != 0 {
        return Err(invalid(
            field("hidden_size"),
            "sinusoids need an even width",
        ));
    }
    // The feature extractor's 128 mel bins, halved by each stride-2 convolution,
    // times the last channel count feed the input projection.
    if raw.subsampling_conv_channels.is_empty() {
        return Err(invalid(
            field("subsampling_conv_channels"),
            "must not be empty",
        ));
    }
    for &channels in &raw.subsampling_conv_channels {
        positive(&field("subsampling_conv_channels"), channels)?;
    }
    if raw.attention_chunk_size == 0 {
        return Err(invalid(field("attention_chunk_size"), "must be positive"));
    }
    // Each frame must see at least itself or one neighbor.
    if raw.attention_context_left < 2 && raw.attention_context_right < 2 {
        return Err(invalid(
            field("attention_context_left"),
            "the attention window admits no key",
        ));
    }
    let finite = |name: &str, value: f64| {
        let value = value as f32;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(invalid(field(name), "must be finite"))
        }
    };
    let rms_norm_eps = finite("rms_norm_eps", raw.rms_norm_eps)?;
    if rms_norm_eps <= 0.0 {
        return Err(invalid(field("rms_norm_eps"), "must be positive"));
    }
    let logit_cap = finite("attention_logit_cap", raw.attention_logit_cap)?;
    if logit_cap <= 0.0 {
        return Err(invalid(field("attention_logit_cap"), "must be positive"));
    }
    Ok(AudioConfig {
        hidden_size,
        layers: positive(&field("num_hidden_layers"), raw.num_hidden_layers)?,
        heads,
        subsampling_channels: raw.subsampling_conv_channels,
        conv_kernel_size: positive(&field("conv_kernel_size"), raw.conv_kernel_size)?,
        chunk_past: raw.attention_context_left.saturating_sub(1),
        chunk_future: raw.attention_context_right,
        invalid_logit: finite(
            "attention_invalid_logits_value",
            raw.attention_invalid_logits_value,
        )?,
        logit_cap,
        gradient_clipping: finite("gradient_clipping", raw.gradient_clipping)?,
        residual_weight: finite("residual_weight", raw.residual_weight)?,
        output_dims: positive(&field("output_proj_dims"), output_dims)?,
        rms_norm_eps,
        audio_token_id,
        boa_token_id,
        eoa_token_id,
    })
}

/// Parses `preprocessor_config.json`, admitting only the published log-mel
/// front-end: no dither, preemphasis, input scaling, overdrive or per-bin
/// normalization, and a frame that fits the FFT. The audio tower's input
/// projection shape checks `feature_size` against the checkpoint.
pub(super) fn parse_front_end(bytes: &[u8]) -> Result<FrontEnd, ConfigError> {
    let raw: RawFrontEnd = serde_json::from_slice(bytes)?;
    let unsupported = |what: &str| {
        Err(ConfigError::UnsupportedArchitecture(format!(
            "{MODEL_TYPE} audio front-end with {what}"
        )))
    };
    if raw.feature_extractor_type != "Gemma4AudioFeatureExtractor" {
        return unsupported(&raw.feature_extractor_type);
    }
    if raw.dither != 0.0 || raw.preemphasis != 0.0 || raw.input_scale_factor != 1.0 {
        return unsupported("dither, preemphasis or input scaling");
    }
    if raw.fft_overdrive || raw.per_bin_mean.is_some() || raw.per_bin_stddev.is_some() {
        return unsupported("FFT overdrive or per-bin normalization");
    }
    let field = |name: &str| format!("preprocessor_config.{name}");
    let feature_size = positive(&field("feature_size"), raw.feature_size)?;
    let frame_length = positive(&field("frame_length"), raw.frame_length)?;
    if raw.fft_length < frame_length || raw.fft_length < 2 {
        return Err(invalid(field("fft_length"), "must cover frame_length"));
    }
    if !(raw.mel_floor > 0.0 && raw.max_frequency > raw.min_frequency && raw.min_frequency >= 0.0) {
        return Err(invalid(field("mel_floor"), "invalid mel range or floor"));
    }
    Ok(FrontEnd {
        sampling_rate: positive(&field("sampling_rate"), raw.sampling_rate)?,
        feature_size,
        frame_length,
        hop_length: positive(&field("hop_length"), raw.hop_length)?,
        fft_length: raw.fft_length,
        min_frequency: raw.min_frequency,
        max_frequency: raw.max_frequency,
        mel_floor: raw.mel_floor,
    })
}

fn parse_vision(
    raw: RawVision,
    image_token_id: u32,
    boi_token_id: u32,
    eoi_token_id: u32,
) -> Result<VisionConfig, ConfigError> {
    let unsupported = |what: String| {
        Err(ConfigError::UnsupportedArchitecture(format!(
            "{MODEL_TYPE} vision tower with {what}"
        )))
    };
    if raw.model_type != "gemma4_vision" {
        return unsupported(format!("model_type {}", raw.model_type));
    }
    if raw.hidden_activation != "gelu_pytorch_tanh" {
        return unsupported(format!("hidden_activation {}", raw.hidden_activation));
    }
    if raw.attention_bias {
        return unsupported("attention_bias".into());
    }
    // Neither checkpoint form carries clip bounds or standardization tensors.
    if raw.use_clipped_linears {
        return unsupported("use_clipped_linears".into());
    }
    if raw.standardize {
        return unsupported("standardize".into());
    }
    if raw.rope_parameters.rope_type != "axial" {
        return Err(ConfigError::UnsupportedRope(format!(
            "vision_config: {}",
            raw.rope_parameters.rope_type
        )));
    }
    let field = |name: &str| format!("vision_config.{name}");
    let hidden_size = positive(&field("hidden_size"), raw.hidden_size)?;
    let heads = positive(&field("num_attention_heads"), raw.num_attention_heads)?;
    let kv_heads = positive(&field("num_key_value_heads"), raw.num_key_value_heads)?;
    let head_dim = positive(&field("head_dim"), raw.head_dim)?;
    if raw.global_head_dim.is_some_and(|global| global != head_dim) {
        return unsupported("global_head_dim different from head_dim".into());
    }
    if heads % kv_heads != 0 {
        return Err(invalid(
            field("num_key_value_heads"),
            "must divide num_attention_heads",
        ));
    }
    // Axial RoPE rotates one even half of the head per image axis.
    if head_dim % 4 != 0 {
        return Err(invalid(
            field("head_dim"),
            "axial RoPE needs a multiple of 4",
        ));
    }
    let patch_size = positive(&field("patch_size"), raw.patch_size)?;
    let pooling_kernel_size = positive(&field("pooling_kernel_size"), raw.pooling_kernel_size)?;
    let patch_width = 3 * patch_size * patch_size;
    if i32::try_from(patch_width).is_err() {
        return Err(invalid(field("patch_size"), "patch width exceeds i32"));
    }
    let rms_norm_eps = raw.rms_norm_eps as f32;
    if !rms_norm_eps.is_finite() || rms_norm_eps <= 0.0 {
        return Err(invalid(
            field("rms_norm_eps"),
            "must be finite and positive",
        ));
    }
    let rope_theta = raw.rope_parameters.rope_theta as f32;
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        return Err(invalid(
            field("rope_parameters.rope_theta"),
            "must be finite and positive",
        ));
    }
    Ok(VisionConfig {
        hidden_size,
        intermediate_size: positive(&field("intermediate_size"), raw.intermediate_size)?,
        layers: positive(&field("num_hidden_layers"), raw.num_hidden_layers)?,
        heads,
        kv_heads,
        head_dim,
        patch_size,
        pooling_kernel_size,
        position_embedding_size: positive(
            &field("position_embedding_size"),
            raw.position_embedding_size,
        )?,
        max_soft_tokens: positive(&field("default_output_length"), raw.default_output_length)?,
        rms_norm_eps,
        rope_theta,
        image_token_id,
        boi_token_id,
        eoi_token_id,
    })
}
