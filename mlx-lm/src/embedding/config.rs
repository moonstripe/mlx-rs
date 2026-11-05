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
        .find(|&&id| id as usize >= vocabulary_size)
    {
        return Err(invalid(
            "media token ids",
            format!("{token} is outside the vocabulary"),
        ));
    }
    let vision = raw
        .vision_config
        .map(|vision| {
            let token = |field: &str, id: Option<u32>| {
                id.ok_or_else(|| ConfigError::MissingField {
                    field: field.into(),
                })
            };
            parse_vision(
                vision,
                token("image_token_id", raw.image_token_id)?,
                token("boi_token_id", raw.boi_token_id)?,
                token("eoi_token_id", raw.eoi_token_id)?,
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
