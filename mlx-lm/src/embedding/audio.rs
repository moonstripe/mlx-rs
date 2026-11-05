//! Gemma 4 audio front-end, Conformer encoder and `embed_audio` projection.

use super::{
    config::{AudioConfig, FrontEnd},
    linear, slot_key,
};
use crate::{weights::WeightManifest, WeightError};
use mlx_rs::{error::Exception, fast, nn, ops, utils::StateProjection, Array, Dtype};

pub(super) const PREFIXES: [&str; 2] = ["audio_tower.", "embed_audio."];
/// Waveforms are zero-padded to a multiple of this many samples before framing.
const PAD_MULTIPLE: usize = 128;

/// Log-mel features and per-frame validity for one waveform.
pub(super) struct LogMel {
    /// `[frames, feature_size]`, zero where the frame is invalid.
    features: Vec<f32>,
    valid: Vec<bool>,
}

impl FrontEnd {
    /// Mel frames a waveform of `samples` yields, and how many are fully real audio.
    pub(super) fn frames(&self, samples: usize) -> (usize, usize) {
        let padded = samples.div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE;
        let total = (self.frame_length / 2 + padded + self.hop_length)
            .saturating_sub(self.frame_length + 1)
            / self.hop_length;
        // Frame i is valid when the sample after its window is real audio.
        let valid = (0..total)
            .take_while(|i| {
                i * self.hop_length + self.frame_length < self.frame_length / 2 + samples
            })
            .count();
        (total, valid)
    }

    /// HTK triangular filters, `[bins, feature_size]`, without area normalization.
    fn mel_bank(&self) -> Vec<f64> {
        let bins = self.fft_length / 2 + 1;
        let mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
        let hz = |mel: f64| 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
        let (low, high) = (mel(self.min_frequency), mel(self.max_frequency));
        let points = self.feature_size + 2;
        let edges: Vec<f64> = (0..points)
            .map(|i| hz(low + (high - low) * i as f64 / (points - 1) as f64))
            .collect();
        let nyquist = self.sampling_rate as f64 / 2.0;
        let mut bank = vec![0.0; bins * self.feature_size];
        for bin in 0..bins {
            let frequency = nyquist * bin as f64 / (bins - 1) as f64;
            for m in 0..self.feature_size {
                let rising = (frequency - edges[m]) / (edges[m + 1] - edges[m]);
                let falling = (edges[m + 2] - frequency) / (edges[m + 2] - edges[m + 1]);
                bank[bin * self.feature_size + m] = rising.min(falling).max(0.0);
            }
        }
        bank
    }

    /// Semicausal framing, periodic Hann window, `fft_length` real DFT magnitude,
    /// mel projection and `log(mel + mel_floor)`, in `f64` from `f32` frames.
    pub(super) fn log_mel(&self, waveform: &[f32]) -> LogMel {
        let (frame, hop, size) = (self.frame_length, self.hop_length, self.feature_size);
        let (total, valid_frames) = self.frames(waveform.len());
        let bins = self.fft_length / 2 + 1;
        let window: Vec<f32> = (0..frame)
            .map(|n| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / frame as f64).cos()) as f32
            })
            .collect();
        let (mut cos, mut sin) = (vec![0.0; bins * frame], vec![0.0; bins * frame]);
        for k in 0..bins {
            for n in 0..frame {
                // Reduce k * n modulo the FFT length before forming the angle.
                let turn = (k * n % self.fft_length) as f64 / self.fft_length as f64;
                let angle = 2.0 * std::f64::consts::PI * turn;
                cos[k * frame + n] = angle.cos();
                sin[k * frame + n] = angle.sin();
            }
        }
        let bank = self.mel_bank();
        let offset = frame / 2;
        let sample = |index: usize| {
            index
                .checked_sub(offset)
                .and_then(|i| waveform.get(i))
                .copied()
                .unwrap_or(0.0)
        };
        let mut features = vec![0.0f32; total * size];
        let mut windowed = vec![0.0f64; frame];
        let mut magnitude = vec![0.0f64; bins];
        for f in 0..valid_frames {
            for (n, value) in windowed.iter_mut().enumerate() {
                *value = f64::from(sample(f * hop + n) * window[n]);
            }
            for (k, out) in magnitude.iter_mut().enumerate() {
                let (c, s) = (&cos[k * frame..][..frame], &sin[k * frame..][..frame]);
                let (re, im) = windowed
                    .iter()
                    .zip(c.iter().zip(s))
                    .fold((0.0, 0.0), |(re, im), (x, (c, s))| (re + x * c, im - x * s));
                *out = re.hypot(im);
            }
            for m in 0..size {
                let mel: f64 = (0..bins).map(|k| magnitude[k] * bank[k * size + m]).sum();
                features[f * size + m] = (mel + self.mel_floor).ln() as f32;
            }
        }
        let mut valid = vec![false; total];
        valid[..valid_frames].fill(true);
        LogMel { features, valid }
    }
}

/// A linear layer with input and output clamps, as stored in the checkpoint.
struct ClippedLinear {
    weight: Array,
    bounds: [Array; 4],
}

impl ClippedLinear {
    const BOUNDS: [&'static str; 4] = ["input_min", "input_max", "output_min", "output_max"];

    fn new(manifest: &WeightManifest, name: &str, shape: [usize; 2]) -> Result<Self, WeightError> {
        let bound = |which: &str| slot_key(manifest, &format!("{name}.{which}"), &[]);
        Ok(Self {
            weight: slot_key(manifest, &format!("{name}.linear.weight"), &shape)?,
            bounds: [
                bound(Self::BOUNDS[0])?,
                bound(Self::BOUNDS[1])?,
                bound(Self::BOUNDS[2])?,
                bound(Self::BOUNDS[3])?,
            ],
        })
    }

    fn project<'a>(
        &'a mut self,
        name: &str,
        projection: &mut StateProjection<'a>,
    ) -> Result<(), WeightError> {
        projection.required(format!("{name}.linear.weight"), &mut self.weight)?;
        for (which, array) in Self::BOUNDS.iter().zip(&mut self.bounds) {
            projection.required(format!("{name}.{which}"), array)?;
        }
        Ok(())
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let [in_min, in_max, out_min, out_max] = &self.bounds;
        let x = ops::minimum(ops::maximum(x, in_min)?, in_max)?;
        let y = linear(&x, &self.weight)?;
        ops::minimum(ops::maximum(y, out_min)?, out_max)
    }
}

/// A list of `(external key, array)` slots owned by one layer.
struct Slots {
    names: Vec<String>,
    arrays: Vec<Array>,
}

impl Slots {
    fn new(
        manifest: &WeightManifest,
        entries: &[(String, Vec<usize>)],
    ) -> Result<Self, WeightError> {
        Ok(Self {
            names: entries.iter().map(|(name, _)| name.clone()).collect(),
            arrays: entries
                .iter()
                .map(|(name, shape)| slot_key(manifest, name, shape))
                .collect::<Result<_, _>>()?,
        })
    }

    fn project<'a>(&'a mut self, projection: &mut StateProjection<'a>) -> Result<(), WeightError> {
        for (name, array) in self.names.iter().zip(&mut self.arrays) {
            projection.required(name.clone(), array)?;
        }
        Ok(())
    }

    fn get(&self, suffix: &str) -> &Array {
        let index = self
            .names
            .iter()
            .position(|name| name.ends_with(suffix))
            .expect("declared slot");
        &self.arrays[index]
    }
}

struct Block {
    prefix: String,
    linears: Vec<(String, ClippedLinear)>,
    other: Slots,
}

impl Block {
    fn new(c: &AudioConfig, manifest: &WeightManifest, index: usize) -> Result<Self, WeightError> {
        let d = c.hidden_size;
        let prefix = format!("audio_tower.layers.{index}");
        let mut linears = Vec::new();
        for ff in ["feed_forward1", "feed_forward2"] {
            linears.push((format!("{ff}.ffw_layer_1"), [4 * d, d]));
            linears.push((format!("{ff}.ffw_layer_2"), [d, 4 * d]));
        }
        for name in ["q_proj", "k_proj", "v_proj", "post"] {
            linears.push((format!("self_attn.{name}"), [d, d]));
        }
        linears.push(("lconv1d.linear_start".into(), [2 * d, d]));
        linears.push(("lconv1d.linear_end".into(), [d, d]));
        let linears = linears
            .into_iter()
            .map(|(name, shape)| {
                ClippedLinear::new(manifest, &format!("{prefix}.{name}"), shape).map(|l| (name, l))
            })
            .collect::<Result<_, _>>()?;
        let mut entries: Vec<(String, Vec<usize>)> = Vec::new();
        for norm in [
            "feed_forward1.pre_layer_norm",
            "feed_forward1.post_layer_norm",
            "feed_forward2.pre_layer_norm",
            "feed_forward2.post_layer_norm",
            "lconv1d.pre_layer_norm",
            "lconv1d.conv_norm",
            "norm_pre_attn",
            "norm_post_attn",
            "norm_out",
        ] {
            entries.push((format!("{prefix}.{norm}.weight"), vec![d]));
        }
        entries.push((
            format!("{prefix}.lconv1d.depthwise_conv1d.weight"),
            vec![d, c.conv_kernel_size, 1],
        ));
        entries.push((
            format!("{prefix}.self_attn.per_dim_scale"),
            vec![d / c.heads],
        ));
        entries.push((
            format!("{prefix}.self_attn.relative_k_proj.weight"),
            vec![d, d],
        ));
        Ok(Self {
            other: Slots::new(manifest, &entries)?,
            linears,
            prefix,
        })
    }

    fn project<'a>(&'a mut self, projection: &mut StateProjection<'a>) -> Result<(), WeightError> {
        for (name, linear) in &mut self.linears {
            linear.project(&format!("{}.{name}", self.prefix), projection)?;
        }
        self.other.project(projection)
    }

    fn linear(&self, name: &str) -> &ClippedLinear {
        &self
            .linears
            .iter()
            .find(|(n, _)| n == name)
            .expect("declared linear")
            .1
    }

    fn norm(&self, name: &str, x: &Array, eps: f32) -> Result<Array, Exception> {
        fast::rms_norm(x, Some(self.other.get(&format!(".{name}.weight"))), eps)
    }
}

/// The audio encoder plus the projection of its frames into the text width.
pub(super) struct AudioTower {
    pub(super) config: AudioConfig,
    pub(super) front_end: FrontEnd,
    convs: Slots,
    input_proj: Array,
    blocks: Vec<Block>,
    output_weight: Array,
    output_bias: Array,
    projection: Array,
}

impl AudioTower {
    const SSCP: &'static str = "audio_tower.subsample_conv_projection";

    pub(super) fn new(
        config: AudioConfig,
        front_end: FrontEnd,
        text_hidden: usize,
        manifest: &WeightManifest,
    ) -> Result<Self, WeightError> {
        let mut entries = Vec::new();
        let mut channels_in = 1;
        let mut bins = front_end.feature_size;
        for (index, &channels) in config.subsampling_channels.iter().enumerate() {
            let p = format!("{}.layer{index}", Self::SSCP);
            entries.push((
                format!("{p}.conv.weight"),
                vec![channels, 3, 3, channels_in],
            ));
            entries.push((format!("{p}.norm.weight"), vec![channels]));
            channels_in = channels;
            bins = (bins + 2 - 3) / 2 + 1;
        }
        let d = config.hidden_size;
        Ok(Self {
            convs: Slots::new(manifest, &entries)?,
            input_proj: slot_key(
                manifest,
                &format!("{}.input_proj_linear.weight", Self::SSCP),
                &[d, bins * channels_in],
            )?,
            blocks: (0..config.layers)
                .map(|index| Block::new(&config, manifest, index))
                .collect::<Result<_, _>>()?,
            output_weight: slot_key(
                manifest,
                "audio_tower.output_proj.weight",
                &[config.output_dims, d],
            )?,
            output_bias: slot_key(
                manifest,
                "audio_tower.output_proj.bias",
                &[config.output_dims],
            )?,
            projection: slot_key(
                manifest,
                "embed_audio.embedding_projection.weight",
                &[text_hidden, config.output_dims],
            )?,
            config,
            front_end,
        })
    }

    pub(super) fn project<'a>(
        &'a mut self,
        projection: &mut StateProjection<'a>,
    ) -> Result<(), WeightError> {
        self.convs.project(projection)?;
        projection.required(
            format!("{}.input_proj_linear.weight", Self::SSCP),
            &mut self.input_proj,
        )?;
        for block in &mut self.blocks {
            block.project(projection)?;
        }
        projection.required("audio_tower.output_proj.weight", &mut self.output_weight)?;
        projection.required("audio_tower.output_proj.bias", &mut self.output_bias)?;
        projection.required(
            "embed_audio.embedding_projection.weight",
            &mut self.projection,
        )?;
        Ok(())
    }

    /// Soft tokens the encoder keeps for `frames` mel frames of which `valid` are real.
    pub(super) fn tokens(&self, frames: usize, valid: usize) -> usize {
        let (mut frames, mut valid) = (frames, valid);
        for _ in &self.config.subsampling_channels {
            let out = (frames + 2 - 3) / 2 + 1;
            // Every second mask entry, truncated to the output length.
            valid = valid.div_ceil(2).min(out);
            frames = out;
        }
        valid
    }

    /// Soft tokens `[tokens, text_hidden]` for one waveform with at least one token.
    pub(super) fn features(&self, waveform: &[f32]) -> Result<Array, Exception> {
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let mel = self.front_end.log_mel(waveform);
        let frames = mel.valid.len() as i32;
        let bins = self.front_end.feature_size as i32;
        let mut x = Array::from_slice(&mel.features, &[1, frames, bins, 1]);
        let mut valid = mel.valid;
        for index in 0..c.subsampling_channels.len() {
            let p = format!("{}.layer{index}", Self::SSCP);
            let keep = Array::from_slice(&valid, &[1, valid.len() as i32, 1, 1]);
            x = x.multiply(keep.as_dtype(Dtype::Float32)?)?;
            x = ops::pad(&x, &[(0, 0), (1, 1), (1, 1), (0, 0)], None, None)?;
            x = ops::conv2d(
                &x,
                self.convs.get(&format!("{p}.conv.weight")),
                (2, 2),
                (0, 0),
                None,
                None,
            )?;
            let out = x.shape()[1] as usize;
            valid = valid.iter().step_by(2).take(out).copied().collect();
            x = fast::layer_norm(
                &x,
                Some(self.convs.get(&format!("{p}.norm.weight"))),
                None,
                eps,
            )?;
            x = nn::relu(&x)?;
        }
        let length = x.shape()[1];
        let mut x = linear(&x.reshape(&[1, length, -1])?, &self.input_proj)?;
        let keys = Array::from_slice(&valid, &[1, length]);
        let keep = keys.as_dtype(Dtype::Float32)?.reshape(&[1, length, 1])?;
        let attention = Attention::new(c, length as usize, &valid)?;
        let clip = |x: &Array| -> Result<Array, Exception> {
            let g = Array::from_f32(c.gradient_clipping);
            ops::minimum(ops::maximum(x, g.negative()?)?, &g)
        };
        let feed_forward = |block: &Block, ff: &str, x: &Array| -> Result<Array, Exception> {
            let h = block.norm(&format!("{ff}.pre_layer_norm"), &clip(x)?, eps)?;
            let h = block.linear(&format!("{ff}.ffw_layer_1")).forward(&h)?;
            let h = block
                .linear(&format!("{ff}.ffw_layer_2"))
                .forward(&nn::silu(&h)?)?;
            let h = block.norm(&format!("{ff}.post_layer_norm"), &clip(&h)?, eps)?;
            x.add(h.multiply(Array::from_f32(c.residual_weight))?)
        };
        for block in &self.blocks {
            x = feed_forward(block, "feed_forward1", &x)?;
            let h = block.norm("norm_pre_attn", &clip(&x)?, eps)?;
            let h = clip(&attention.forward(block, &h)?)?;
            x = x.add(block.norm("norm_post_attn", &h, eps)?)?;
            x = x.multiply(&keep)?;
            x = self.light_conv(block, &x, &clip)?;
            x = feed_forward(block, "feed_forward2", &x)?;
            x = block.norm("norm_out", &clip(&x)?, eps)?;
        }
        let x = linear(&x, &self.output_weight)?.add(&self.output_bias)?;
        let tokens = valid.iter().take_while(|&&v| v).count() as i32;
        let x = x.reshape(&[length, -1])?;
        let x = x.take_axis(ops::arange::<_, i32>(None, tokens, None)?, 0)?;
        linear(&fast::rms_norm(x, None, eps)?, &self.projection)
    }

    fn light_conv(
        &self,
        block: &Block,
        x: &Array,
        clip: &dyn Fn(&Array) -> Result<Array, Exception>,
    ) -> Result<Array, Exception> {
        let eps = self.config.rms_norm_eps;
        let h = block.norm("lconv1d.pre_layer_norm", x, eps)?;
        let h = block.linear("lconv1d.linear_start").forward(&h)?;
        let halves = h.split_equal(2, -1)?;
        let h = halves[0].multiply(ops::sigmoid(&halves[1])?)?;
        let kernel = self.config.conv_kernel_size as i32;
        let h = ops::pad(&h, &[(0, 0), (kernel - 1, 0), (0, 0)], None, None)?;
        let weight = block.other.get(".lconv1d.depthwise_conv1d.weight");
        let groups = self.config.hidden_size as i32;
        let h = ops::conv1d(&h, weight, None, None, None, groups)?;
        let h = block.norm("lconv1d.conv_norm", &clip(&h)?, eps)?;
        block
            .linear("lconv1d.linear_end")
            .forward(&nn::silu(&h)?)?
            .add(x)
    }
}

/// Banded attention with relative positions: query `t` sees key `s` when
/// `0 <= t - s < past` or `0 < s - t < future`, and the key is valid.
///
/// Each query gathers only its `past + future - 1` window slots, so memory grows
/// linearly with the frame count; masked slots contribute exactly zero, as the
/// reference's out-of-band keys do.
struct Attention {
    /// `[1, length, window]`: the slot's key exists and is valid.
    allowed: Array,
    /// `[length * window]`: each slot's key index, clamped into range.
    keys: Array,
    /// `[window]`: each slot's row in the relative table, `past - distance`.
    rows: Array,
    timing: Array,
    window: i32,
    heads: i32,
    head_dim: i32,
    q_scale: f64,
    k_scale: f32,
    cap: f32,
    invalid: f32,
}

impl Attention {
    fn new(c: &AudioConfig, length: usize, valid: &[bool]) -> Result<Self, Exception> {
        let (past, future) = (c.chunk_past as i64, c.chunk_future as i64);
        // Distances past - 1, ..., 0 and then -1, ..., -(future - 1).
        let distances: Vec<i64> = (0..past).rev().chain((1..future).map(|d| -d)).collect();
        let window = distances.len();
        let mut allowed = Vec::with_capacity(length * window);
        let mut keys = Vec::with_capacity(length * window);
        for t in 0..length as i64 {
            for &d in &distances {
                let s = t - d;
                let inside = (0..length as i64).contains(&s);
                allowed.push(inside && valid[s as usize]);
                keys.push(s.clamp(0, length as i64 - 1) as i32);
            }
        }
        let rows: Vec<i32> = distances.iter().map(|&d| (past - d) as i32).collect();
        // Sinusoids for offsets past, ..., -future, `f32` as in the encoder.
        let span = (past + future + 1) as usize;
        let half = (c.hidden_size / 2) as i32;
        let increment = -(10_000f64.ln()) / f64::from((half - 1).max(1));
        let inverse = ops::arange::<_, f32>(None, half, None)?
            .multiply(Array::from_f32(increment as f32))?
            .exp()?;
        let positions: Vec<f32> = (0..span as i64).map(|i| (past - i) as f32).collect();
        let angles = Array::from_slice(&positions, &[span as i32, 1])
            .multiply(inverse.reshape(&[1, -1])?)?;
        let timing = ops::concatenate(&[angles.sin()?, angles.cos()?], -1)?;
        let head_dim = (c.hidden_size / c.heads) as i32;
        let window = window as i32;
        Ok(Self {
            allowed: Array::from_slice(&allowed, &[1, length as i32, window]),
            keys: Array::from_slice(&keys, &[keys.len() as i32]),
            rows: Array::from_slice(&rows, &[window]),
            timing,
            window,
            heads: c.heads as i32,
            head_dim,
            q_scale: f64::from(head_dim).powf(-0.5) / std::f64::consts::LN_2,
            k_scale: ((1.0 + std::f64::consts::E).ln() / std::f64::consts::LN_2) as f32,
            cap: c.logit_cap,
            invalid: c.invalid_logit,
        })
    }

    fn forward(&self, block: &Block, x: &Array) -> Result<Array, Exception> {
        let length = x.shape()[1];
        let (n, h, w) = (self.heads, self.head_dim, self.window);
        let heads = |y: Array| -> Result<Array, Exception> {
            y.as_dtype(Dtype::Float32)?
                .reshape(&[length, n, h])?
                .transpose_axes(&[1, 0, 2])
        };
        // [heads, length, head_dim] -> [heads, length, window, head_dim].
        let gather = |y: Array| -> Result<Array, Exception> {
            y.take_axis(&self.keys, 1)?.reshape(&[n, length, w, h])
        };
        let scale = block.other.get(".self_attn.per_dim_scale");
        // softplus in the parameter dtype, times the Python-float query scale.
        let factor = ops::logaddexp(scale, ops::zeros_dtype(&[1], scale.dtype())?)?
            .multiply(super::scalar(self.q_scale, scale.dtype())?)?;
        let q = heads(block.linear("self_attn.q_proj").forward(x)?)?.multiply(&factor)?;
        let k = heads(block.linear("self_attn.k_proj").forward(x)?)?
            .multiply(Array::from_f32(self.k_scale))?;
        let v = gather(heads(block.linear("self_attn.v_proj").forward(x)?)?)?;

        let relative_weight = block.other.get(".self_attn.relative_k_proj.weight");
        // [heads, head_dim, window]: the relative embedding of each slot's distance.
        let relative = linear(
            &self.timing.as_dtype(relative_weight.dtype())?,
            relative_weight,
        )?
        .as_dtype(Dtype::Float32)?
        .take_axis(&self.rows, 0)?
        .reshape(&[w, n, h])?
        .transpose_axes(&[1, 2, 0])?;
        let content = q
            .expand_dims(2)?
            .matmul(gather(k)?.transpose_axes(&[0, 1, 3, 2])?)?
            .squeeze_axes(&[2])?;
        let position = q.matmul(&relative)?;
        let cap = Array::from_f32(self.cap);
        let logits = ops::tanh(content.add(position)?.divide(&cap)?)?.multiply(&cap)?;
        let logits = ops::select(&self.allowed, &logits, Array::from_f32(self.invalid))?;
        let probs = ops::softmax_axis(&logits, -1, None)?;
        let context = probs
            .expand_dims(2)?
            .matmul(&v)?
            .squeeze_axes(&[2])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[1, length, -1])?;
        block.linear("self_attn.post").forward(&context)
    }
}
