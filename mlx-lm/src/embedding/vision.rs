//! Gemma 4 vision tower and `embed_vision` projection for EmbeddingGemma 2.

use super::{config::VisionConfig, linear, scalar, slot_key};
use crate::{weights::WeightManifest, WeightError};
use mlx_rs::{fast, nn, ops, utils::StateProjection, Array, Dtype};

pub(super) const PREFIXES: [&str; 2] = ["vision_tower.", "embed_vision."];
const TOWER: &str = "vision_tower.";
/// `rescale_factor` of the published image processor, applied to `u8` channels.
const RESCALE: f64 = 0.003_921_568_627_450_98;

struct Layer {
    q_proj: Array,
    k_proj: Array,
    v_proj: Array,
    o_proj: Array,
    q_norm: Array,
    k_norm: Array,
    gate_proj: Array,
    up_proj: Array,
    down_proj: Array,
    input_layernorm: Array,
    post_attention_layernorm: Array,
    pre_feedforward_layernorm: Array,
    post_feedforward_layernorm: Array,
}

impl Layer {
    fn names(index: usize) -> [String; 13] {
        let p = format!("{TOWER}encoder.layers.{index}");
        [
            format!("{p}.self_attn.q_proj.linear.weight"),
            format!("{p}.self_attn.k_proj.linear.weight"),
            format!("{p}.self_attn.v_proj.linear.weight"),
            format!("{p}.self_attn.o_proj.linear.weight"),
            format!("{p}.self_attn.q_norm.weight"),
            format!("{p}.self_attn.k_norm.weight"),
            format!("{p}.mlp.gate_proj.linear.weight"),
            format!("{p}.mlp.up_proj.linear.weight"),
            format!("{p}.mlp.down_proj.linear.weight"),
            format!("{p}.input_layernorm.weight"),
            format!("{p}.post_attention_layernorm.weight"),
            format!("{p}.pre_feedforward_layernorm.weight"),
            format!("{p}.post_feedforward_layernorm.weight"),
        ]
    }

    fn new(c: &VisionConfig, manifest: &WeightManifest, index: usize) -> Result<Self, WeightError> {
        let hidden = c.hidden_size;
        let queries = c.heads * c.head_dim;
        let kv = c.kv_heads * c.head_dim;
        let shapes: [&[usize]; 13] = [
            &[queries, hidden],
            &[kv, hidden],
            &[kv, hidden],
            &[hidden, queries],
            &[c.head_dim],
            &[c.head_dim],
            &[c.intermediate_size, hidden],
            &[c.intermediate_size, hidden],
            &[hidden, c.intermediate_size],
            &[hidden],
            &[hidden],
            &[hidden],
            &[hidden],
        ];
        let names = Self::names(index);
        let mut slots = names
            .iter()
            .zip(shapes)
            .map(|(name, shape)| slot_key(manifest, name, shape));
        let mut next = || slots.next().expect("thirteen vision slots");
        Ok(Self {
            q_proj: next()?,
            k_proj: next()?,
            v_proj: next()?,
            o_proj: next()?,
            q_norm: next()?,
            k_norm: next()?,
            gate_proj: next()?,
            up_proj: next()?,
            down_proj: next()?,
            input_layernorm: next()?,
            post_attention_layernorm: next()?,
            pre_feedforward_layernorm: next()?,
            post_feedforward_layernorm: next()?,
        })
    }

    fn project<'a>(
        &'a mut self,
        index: usize,
        projection: &mut StateProjection<'a>,
    ) -> Result<(), WeightError> {
        let arrays = [
            &mut self.q_proj,
            &mut self.k_proj,
            &mut self.v_proj,
            &mut self.o_proj,
            &mut self.q_norm,
            &mut self.k_norm,
            &mut self.gate_proj,
            &mut self.up_proj,
            &mut self.down_proj,
            &mut self.input_layernorm,
            &mut self.post_attention_layernorm,
            &mut self.pre_feedforward_layernorm,
            &mut self.post_feedforward_layernorm,
        ];
        for (name, array) in Self::names(index).into_iter().zip(arrays) {
            projection.required(name, array)?;
        }
        Ok(())
    }
}

/// The vision tower plus the projection of its soft tokens into the text width.
pub(super) struct VisionTower {
    pub(super) config: VisionConfig,
    input_proj: Array,
    position_table: Array,
    layers: Vec<Layer>,
    projection: Array,
}

impl VisionTower {
    const INPUT_PROJ: &'static str = "vision_tower.patch_embedder.input_proj.weight";
    const POSITIONS: &'static str = "vision_tower.patch_embedder.position_embedding_table";
    const PROJECTION: &'static str = "embed_vision.embedding_projection.weight";

    pub(super) fn new(
        config: VisionConfig,
        text_hidden: usize,
        manifest: &WeightManifest,
    ) -> Result<Self, WeightError> {
        let hidden = config.hidden_size;
        let patch_width = 3 * config.patch_size * config.patch_size;
        Ok(Self {
            input_proj: slot_key(manifest, Self::INPUT_PROJ, &[hidden, patch_width])?,
            position_table: slot_key(
                manifest,
                Self::POSITIONS,
                &[2, config.position_embedding_size, hidden],
            )?,
            layers: (0..config.layers)
                .map(|index| Layer::new(&config, manifest, index))
                .collect::<Result<_, _>>()?,
            projection: slot_key(manifest, Self::PROJECTION, &[text_hidden, hidden])?,
            config,
        })
    }

    pub(super) fn project<'a>(
        &'a mut self,
        projection: &mut StateProjection<'a>,
    ) -> Result<(), WeightError> {
        projection.required(Self::INPUT_PROJ, &mut self.input_proj)?;
        projection.required(Self::POSITIONS, &mut self.position_table)?;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            layer.project(index, projection)?;
        }
        projection.required(Self::PROJECTION, &mut self.projection)?;
        Ok(())
    }

    /// Soft tokens `[rows / k * columns / k, text_hidden]` for one validated
    /// `[height, width, 3]` image; `rows` and `columns` count patches.
    pub(super) fn features(
        &self,
        pixels: &[u8],
        rows: usize,
        columns: usize,
    ) -> Result<Array, mlx_rs::error::Exception> {
        let c = &self.config;
        let p = c.patch_size as i32;
        let (r, w) = (rows as i32, columns as i32);
        let length = r * w;
        let dtype = self.input_proj.dtype();
        // The processor rescales in f64 and stores f32; a lookup table reproduces it.
        let mut levels = [0.0f32; 256];
        for (value, level) in levels.iter_mut().enumerate() {
            *level = (value as f64 * RESCALE) as f32;
        }
        let scaled: Vec<f32> = pixels.iter().map(|&v| levels[v as usize]).collect();
        // [rows, p, columns, p, 3] -> [rows, columns, p, p, 3]: row-major patches.
        let patches = Array::from_slice(&scaled, &[r, p, w, p, 3])
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[1, length, 3 * p * p])?;
        let patches = patches
            .subtract(Array::from_f32(0.5))?
            .multiply(Array::from_f32(2.0))?
            .as_dtype(dtype)?;

        let mut columns_of = Vec::with_capacity(length as usize);
        let mut rows_of = Vec::with_capacity(length as usize);
        for row in 0..r {
            for column in 0..w {
                columns_of.push(column);
                rows_of.push(row);
            }
        }
        let column_ids = Array::from_slice(&columns_of, &[length]);
        let row_ids = Array::from_slice(&rows_of, &[length]);
        // Table rows [0, size) embed columns and [size, 2 * size) embed rows.
        let size = c.position_embedding_size as i32;
        let table = self.position_table.reshape(&[2 * size, -1])?;
        let positions = table
            .take_axis(&column_ids, 0)?
            .add(table.take_axis(row_ids.add(Array::from_int(size))?, 0)?)?;
        let mut x = linear(&patches, &self.input_proj)?.add(positions)?;

        let rope = AxialRope::new(c, &column_ids, &row_ids, dtype)?;
        for layer in &self.layers {
            x = self.layer(layer, x, &rope, length)?;
        }

        // Average each k x k block of patches, row-major over blocks, as an f32
        // matmul with weights 1 / k^2, then round to the activation dtype.
        let k = c.pooling_kernel_size;
        let tokens = (rows / k) * (columns / k);
        let mut weights = vec![0.0f32; tokens * rows * columns];
        let share = 1.0 / (k * k) as f32;
        for row in 0..rows {
            for column in 0..columns {
                let token = (row / k) * (columns / k) + column / k;
                weights[token * rows * columns + row * columns + column] = share;
            }
        }
        let weights = Array::from_slice(&weights, &[tokens as i32, length]);
        let pooled = weights
            .matmul(x.squeeze_axes(&[0])?.as_dtype(Dtype::Float32)?)?
            .as_dtype(dtype)?;
        let pooled = pooled.multiply(scalar((c.hidden_size as f64).sqrt(), dtype)?)?;
        let normed = fast::rms_norm(pooled, None, c.rms_norm_eps)?;
        linear(&normed, &self.projection)
    }

    fn layer(
        &self,
        layer: &Layer,
        x: Array,
        rope: &AxialRope,
        length: i32,
    ) -> Result<Array, mlx_rs::error::Exception> {
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let heads = |array: Array, count: usize| {
            array.reshape(&[1, length, count as i32, c.head_dim as i32])
        };
        let h = fast::rms_norm(&x, Some(&layer.input_layernorm), eps)?;
        let queries = f32_rms_norm(
            &heads(linear(&h, &layer.q_proj)?, c.heads)?,
            Some(&layer.q_norm),
            eps,
        )?;
        let keys = f32_rms_norm(
            &heads(linear(&h, &layer.k_proj)?, c.kv_heads)?,
            Some(&layer.k_norm),
            eps,
        )?;
        let values = f32_rms_norm(&heads(linear(&h, &layer.v_proj)?, c.kv_heads)?, None, eps)?;
        let attended = fast::scaled_dot_product_attention(
            rope.apply(&queries)?.transpose_axes(&[0, 2, 1, 3])?,
            rope.apply(&keys)?.transpose_axes(&[0, 2, 1, 3])?,
            values.transpose_axes(&[0, 2, 1, 3])?,
            1.0,
            None,
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[1, length, -1])?;
        let x = x.add(fast::rms_norm(
            linear(&attended, &layer.o_proj)?,
            Some(&layer.post_attention_layernorm),
            eps,
        )?)?;
        let h = fast::rms_norm(&x, Some(&layer.pre_feedforward_layernorm), eps)?;
        let gated = nn::gelu_approximate(linear(&h, &layer.gate_proj)?)?
            .multiply(linear(&h, &layer.up_proj)?)?;
        x.add(fast::rms_norm(
            linear(&gated, &layer.down_proj)?,
            Some(&layer.post_feedforward_layernorm),
            eps,
        )?)
    }
}

/// RMS norm computed in `f32` and rounded back, as the tower's q/k/v norms are.
fn f32_rms_norm(
    x: &Array,
    weight: Option<&Array>,
    eps: f32,
) -> Result<Array, mlx_rs::error::Exception> {
    let wide = x.as_dtype(Dtype::Float32)?;
    let variance = wide.square()?.mean_axis(-1, true)?;
    let mut normed = wide.multiply(variance.add(Array::from_f32(eps))?.rsqrt()?)?;
    if let Some(weight) = weight {
        normed = normed.multiply(weight.as_dtype(Dtype::Float32)?)?;
    }
    normed.as_dtype(x.dtype())
}

/// Rotate-half RoPE applied separately to the column half and the row half of
/// each head, with `f32` angles rounded to the activation dtype.
struct AxialRope {
    tables: [(Array, Array); 2],
    channels: i32,
}

impl AxialRope {
    fn new(
        c: &VisionConfig,
        column_ids: &Array,
        row_ids: &Array,
        dtype: Dtype,
    ) -> Result<Self, mlx_rs::error::Exception> {
        let channels = (c.head_dim / 2) as i32;
        let half = channels / 2;
        let exponents = ops::arange::<_, f32>(None, half, None)?
            .multiply(Array::from_f32(2.0 / channels as f32))?;
        let timescale = Array::from_f32(c.rope_theta).power(&exponents)?;
        let table = |ids: &Array| -> Result<(Array, Array), mlx_rs::error::Exception> {
            let angles = ids
                .as_dtype(Dtype::Float32)?
                .reshape(&[-1, 1])?
                .divide(timescale.reshape(&[1, -1])?)?;
            let angles = ops::concatenate(&[&angles, &angles], -1)?.expand_dims(1)?;
            Ok((
                angles.cos()?.as_dtype(dtype)?,
                angles.sin()?.as_dtype(dtype)?,
            ))
        };
        Ok(Self {
            tables: [table(column_ids)?, table(row_ids)?],
            channels,
        })
    }

    /// Rotates `[1, length, heads, head_dim]`.
    fn apply(&self, x: &Array) -> Result<Array, mlx_rs::error::Exception> {
        let parts = x.split_equal(2, -1)?;
        let mut rotated = Vec::with_capacity(2);
        for (part, (cos, sin)) in parts.iter().zip(&self.tables) {
            let halves = part.split_equal(2, -1)?;
            let turned = ops::concatenate(&[&halves[1].negative()?, &halves[0]], -1)?;
            rotated.push(part.multiply(cos)?.add(turned.multiply(sin)?)?);
        }
        debug_assert_eq!(parts[0].shape()[3], self.channels);
        ops::concatenate(&[&rotated[0], &rotated[1]], -1)
    }
}
