use super::config::{parse, LayerShape};
use crate::ConfigError;
use serde_json::{json, Value};

fn base() -> Value {
    json!({
        "model_type": "embedding_gemma2",
        "image_token_id": 61,
        "audio_token_id": 62,
        "text_config": {
            "attention_bias": false,
            "pad_token_id": 0,
            "embedding_dim": 24,
            "head_dim": 8,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 32,
            "hidden_size_per_layer_input": 8,
            "intermediate_size": 48,
            "layer_types": ["sliding_attention", "full_attention"],
            "max_position_embeddings": 64,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "per_layer_config": {"01": {"head_dim": 16, "num_key_value_heads": 1}},
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
            },
            "sliding_window": 2,
            "vocab_size": 64
        }
    })
}

fn parse_value(value: &Value) -> Result<super::config::EncoderConfig, ConfigError> {
    parse(&serde_json::to_vec(value).unwrap())
}

#[test]
fn per_layer_overrides_apply_only_to_their_layer() {
    let config = parse_value(&base()).unwrap();
    assert_eq!(
        config.layers,
        [
            LayerShape {
                heads: 4,
                kv_heads: 2,
                head_dim: 8,
                sliding: true,
                rope_theta: 10000.0
            },
            LayerShape {
                heads: 4,
                kv_heads: 1,
                head_dim: 16,
                sliding: false,
                rope_theta: 1e6
            },
        ]
    );
    assert_eq!(config.media_tokens, [61, 62]);
}

type Mutation = fn(&mut Value);

#[test]
fn rejects_configs_the_encoder_would_silently_misread() {
    let cases: [(&str, Mutation); 6] = [
        ("wrong model type", |v| v["model_type"] = json!("gemma4")),
        ("final sliding layer", |v| {
            v["text_config"]["layer_types"] = json!(["full_attention", "sliding_attention"])
        }),
        ("override past the last layer", |v| {
            v["text_config"]["per_layer_config"] = json!({"02": {"head_dim": 16}})
        }),
        ("unknown override field", |v| {
            v["text_config"]["per_layer_config"] = json!({"01": {"sliding_window": 4}})
        }),
        ("scaled rope", |v| {
            v["text_config"]["rope_parameters"]["full_attention"]["rope_type"] =
                json!("proportional")
        }),
        ("kv heads not dividing heads", |v| {
            v["text_config"]["per_layer_config"] = json!({"01": {"num_key_value_heads": 3}})
        }),
    ];
    for (label, mutate) in cases {
        let mut value = base();
        mutate(&mut value);
        assert!(parse_value(&value).is_err(), "{label} was accepted");
    }
}

fn with_vision() -> Value {
    let mut value = base();
    value["boi_token_id"] = json!(59);
    value["eoi_token_id"] = json!(60);
    value["vision_config"] = json!({
        "model_type": "gemma4_vision",
        "attention_bias": false,
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
        "standardize": false,
        "use_clipped_linears": false
    });
    value
}

#[test]
fn vision_tower_takes_its_placeholder_tokens_from_the_top_level() {
    let vision = parse_value(&with_vision()).unwrap().vision.unwrap();
    assert_eq!(
        (
            vision.image_token_id,
            vision.boi_token_id,
            vision.eoi_token_id
        ),
        (61, 59, 60)
    );
    assert!(parse_value(&base()).unwrap().vision.is_none());
}

#[test]
fn rejects_vision_towers_without_matching_tensors_or_semantics() {
    let cases: [(&str, Mutation); 6] = [
        ("clipped linears", |v| {
            v["vision_config"]["use_clipped_linears"] = json!(true)
        }),
        ("standardize", |v| {
            v["vision_config"]["standardize"] = json!(true)
        }),
        ("1-D rope", |v| {
            v["vision_config"]["rope_parameters"]["rope_type"] = json!("default")
        }),
        ("head not split into axial quarters", |v| {
            v["vision_config"]["head_dim"] = json!(6);
            v["vision_config"]["global_head_dim"] = json!(6)
        }),
        ("missing boi", |v| {
            v.as_object_mut().unwrap().remove("boi_token_id");
        }),
        ("eoi outside the vocabulary", |v| {
            v["eoi_token_id"] = json!(64)
        }),
    ];
    for (label, mutate) in cases {
        let mut value = with_vision();
        mutate(&mut value);
        assert!(parse_value(&value).is_err(), "{label} was accepted");
    }
}
