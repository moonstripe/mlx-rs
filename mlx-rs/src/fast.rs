//! Fast implementations of commonly used multi-op functions.

use std::ffi::CStr;
#[cfg(feature = "metal")]
use std::ffi::CString;

#[cfg(feature = "metal")]
use crate::error::Exception;
use crate::error::Result;
use crate::utils::guard::Guarded;
use crate::utils::IntoOption;
#[cfg(feature = "metal")]
use crate::utils::SUCCESS;
#[cfg(feature = "metal")]
use crate::Dtype;
use crate::{Array, Stream};
use mlx_internal_macros::generate_macro;

/// Optimized implementation of `NN.RoPE`.
#[allow(clippy::too_many_arguments)]
pub fn rope<'a>(
    array: impl AsRef<Array>,
    dimensions: i32,
    traditional: bool,
    base: impl Into<Option<f32>>,
    scale: f32,
    offset: i32,
    freqs: impl Into<Option<&'a Array>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rope(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset,
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`rope`].
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fast"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `rope`"
)]
pub fn rope_device<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: i32,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || {
        rope(array, dimensions, traditional, base, scale, offset, freqs)
    })
}

/// Optimized implementation of `NN.RoPE` with dynamic (array) offset.
///
/// This variant allows specifying the offset as an array, enabling different
/// offsets for different positions in the input.
///
/// # Params
///
/// - `array`: Input array
/// - `dimensions`: The feature dimensions to apply rope to
/// - `traditional`: If true, uses the traditional rope implementation
/// - `base`: The base used to compute angular frequency for each dimension
/// - `scale`: The scale to apply to the positions
/// - `offset`: An array of position offsets
/// - `freqs`: Optional precomputed frequencies
/// - `stream`: Stream to evaluate on
#[allow(clippy::too_many_arguments)]
pub fn rope_dynamic<'a>(
    array: impl AsRef<Array>,
    dimensions: i32,
    traditional: bool,
    base: impl Into<Option<f32>>,
    scale: f32,
    offset: impl AsRef<Array>,
    freqs: impl Into<Option<&'a Array>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rope_dynamic(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset.as_ref().as_ptr(),
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`rope_dynamic`].
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fast"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `rope_dynamic`"
)]
pub fn rope_dynamic_device<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: impl AsRef<Array>,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || {
        rope_dynamic(array, dimensions, traditional, base, scale, offset, freqs)
    })
}

const DEFAULT_MASK_MODE: &CStr = c"";
const CAUSAL_MASK_MODE: &CStr = c"causal";

/// Mask modes for scaled dot product attention.
#[derive(Debug)]
pub enum ScaledDotProductAttentionMask<'a> {
    /// A single mask array
    Array(&'a Array),

    /// Causal masking (no explicit mask array needed)
    Causal,
}

impl<'a> From<&'a Array> for ScaledDotProductAttentionMask<'a> {
    fn from(mask: &'a Array) -> Self {
        ScaledDotProductAttentionMask::Array(mask)
    }
}

impl<'a> IntoOption<ScaledDotProductAttentionMask<'a>> for &'a Array {
    fn into_option(self) -> Option<ScaledDotProductAttentionMask<'a>> {
        Some(ScaledDotProductAttentionMask::Array(self))
    }
}

impl ScaledDotProductAttentionMask<'_> {
    fn as_mode_and_mask(&self) -> (&'static CStr, mlx_sys::mlx_array) {
        match self {
            ScaledDotProductAttentionMask::Array(mask) => (DEFAULT_MASK_MODE, mask.as_ptr()),
            ScaledDotProductAttentionMask::Causal => {
                (CAUSAL_MASK_MODE, unsafe { mlx_sys::mlx_array_new() })
            }
        }
    }
}

/// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
///
/// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
///
/// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
///
/// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
///
/// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
pub fn scaled_dot_product_attention<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    mask: impl IntoOption<ScaledDotProductAttentionMask<'a>>,
    sinks: impl Into<Option<&'a Array>>,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let (mask_mode, mask_arr) = mask.into_option().map_or_else(
        || (DEFAULT_MASK_MODE, unsafe { mlx_sys::mlx_array_new() }),
        |m| m.as_mode_and_mask(),
    );

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_scaled_dot_product_attention(
            res,
            queries.as_ref().as_ptr(),
            keys.as_ref().as_ptr(),
            values.as_ref().as_ptr(),
            scale,
            mask_mode.as_ptr(),
            mask_arr,
            sinks
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            false,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`scaled_dot_product_attention`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fast"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `scaled_dot_product_attention`"
)]
pub fn scaled_dot_product_attention_device<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    #[optional] mask: impl IntoOption<ScaledDotProductAttentionMask<'a>>,
    #[optional] sinks: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || {
        scaled_dot_product_attention(queries, keys, values, scale, mask, sinks)
    })
}

/// Root Mean Square normalization (RMS norm).
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: An optional multiplicative weight. When present, it must be one-dimensional with the
///   same size as the last axis of `x`. When absent, only normalization is applied.
/// - eps: A small additive constant for numerical stability
pub fn rms_norm(x: impl AsRef<Array>, weight: Option<&Array>, eps: f32) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    let empty_weight = weight
        .is_none()
        .then(|| unsafe { Array::from_ptr(mlx_sys::mlx_array_new()) });
    let weight = weight
        .map(Array::as_ptr)
        .unwrap_or_else(|| empty_weight.as_ref().unwrap().as_ptr());
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rms_norm(
            res,
            x.as_ref().as_ptr(),
            weight,
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`rms_norm`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fast"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `rms_norm`"
)]
pub fn rms_norm_device(
    x: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || rms_norm(x, Some(weight.as_ref()), eps))
}

/// Layer normalization.
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///   with the same size as the last axis of `x`.  If not given no scaling will occur.
/// - bias: An additive offset to be added to the result. The `bias` should be one-dimensional
///   with the same size as the last axis of `x`.  It not given no offset will occur.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
pub fn layer_norm<'a>(
    x: impl AsRef<Array>,
    weight: impl Into<Option<&'a Array>>,
    bias: impl Into<Option<&'a Array>>,
    eps: f32,
) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_layer_norm(
            res,
            x.as_ref().as_ptr(),
            weight
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            bias.into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Compatibility shim for [`layer_norm`].
#[generate_macro(customize(forwarding_shim = true, root = "$crate::fast"))]
#[deprecated(
    since = "0.26.0",
    note = "use `with_stream` or `with_device` around `layer_norm`"
)]
pub fn layer_norm_device<'a>(
    #[named] x: impl AsRef<Array>,
    #[optional] weight: impl Into<Option<&'a Array>>,
    #[optional] bias: impl Into<Option<&'a Array>>,
    #[named] eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    crate::with_stream(stream.as_ref(), || layer_norm(x, weight, bias, eps))
}

/// A compile-time template parameter for a [`MetalKernel`].
///
/// Each argument is emitted as a template parameter of the generated `[[kernel]]` function, so
/// the kernel source can refer to it by name (for example `T`, `N` or `USE_BIAS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "metal")]
pub enum TemplateArg {
    /// A type parameter, emitted as the matching Metal type (for example `float`).
    Dtype(Dtype),
    /// An `int` parameter.
    Int(i32),
    /// A `bool` parameter.
    Bool(bool),
}

/// Construction options for [`MetalKernel::new`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg(feature = "metal")]
pub struct MetalKernelOptions {
    /// Metal source placed before the generated kernel, such as helper functions or includes.
    pub header: String,
    /// Copy inputs to row-contiguous layout only when `true`. Defaults to `false`; Python
    /// defaults to `true`. When `false`, the source must handle the inputs' actual layout
    /// (for example through `_strides` and `elem_to_loc`).
    pub ensure_row_contiguous: bool,
    /// Declare outputs as `device atomic<T>*` so the kernel can use Metal atomics on them.
    pub atomic_outputs: bool,
}

/// The shape and dtype of one [`MetalKernel`] output.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(feature = "metal")]
pub struct MetalKernelOutput {
    /// Output shape.
    pub shape: Vec<i32>,
    /// Output element type.
    pub dtype: Dtype,
}

/// Independent per-call options for [`MetalKernel::apply`].
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg(feature = "metal")]
pub struct MetalKernelApplyOptions {
    /// Template parameters, in declaration order.
    pub template_args: Vec<(String, TemplateArg)>,
    /// Fill every output with this value before the kernel runs. When absent, outputs start
    /// uninitialized and the kernel must write every element.
    pub init_value: Option<f32>,
    /// Print the generated Metal source on every call to `apply`.
    pub verbose: bool,
}

/// A custom Metal kernel, compiled from source on first use and cached by MLX.
///
/// The kernel body is written against the names given at construction: each input `a` is a
/// `const device T* a`, or `const constant T* a` when it has fewer than eight elements.
/// Non-scalar inputs also expose `a_shape`, `a_strides` and `a_ndim` when referenced in the
/// source. A 0-d input is a `const constant T& a` and has no shape, strides or ndim arguments.
/// Each output is a `device T*` (or an atomic pointer with `atomic_outputs`), and Metal
/// attributes such as `thread_position_in_grid` are declared automatically when referenced.
///
/// Kernels only run on the GPU. [`MetalKernel::apply`] uses the stream selected by
/// [`crate::with_stream`] or [`crate::with_device`], falling back to the GPU outside a scope,
/// even when the default device is the CPU. Selecting a CPU scope returns an error.
///
/// ```rust
/// use mlx_rs::{fast::{MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput}, Array, Dtype};
///
/// let kernel = MetalKernel::new(
///     "double",
///     &["a"],
///     &["out"],
///     "uint i = thread_position_in_grid.x; out[i] = 2 * a[i];",
///     MetalKernelOptions::default(),
/// )?;
/// let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
/// // SAFETY: Four threads read four f32 inputs and initialize distinct f32 outputs.
/// let out = unsafe {
///     kernel.apply(
///         &[&a],
///         &[MetalKernelOutput { shape: vec![4], dtype: Dtype::Float32 }],
///         (4, 1, 1),
///         (4, 1, 1),
///         MetalKernelApplyOptions::default(),
///     )?
/// };
/// assert_eq!(out[0].as_slice::<f32>(), &[2.0, 4.0, 6.0, 8.0]);
/// # Ok::<(), mlx_rs::error::Exception>(())
/// ```
#[cfg(feature = "metal")]
pub struct MetalKernel {
    raw: mlx_sys::mlx_fast_metal_kernel,
}

#[cfg(feature = "metal")]
impl MetalKernel {
    /// Builds a kernel from its name, input and output names, and Metal body `source`.
    ///
    /// At least one output name is required. Compilation is deferred until the outputs are
    /// evaluated, so source errors surface from [`Array::eval`], not [`apply`](Self::apply).
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        options: MetalKernelOptions,
    ) -> Result<Self> {
        crate::error::INIT_ERR_HANDLER.call_once(crate::error::setup_mlx_error_handler);
        let name = c_string(name)?;
        let source = c_string(source)?;
        let header = c_string(&options.header)?;
        let input_names = StringVector::new(input_names)?;
        let output_names = StringVector::new(output_names)?;
        let raw = unsafe {
            mlx_sys::mlx_fast_metal_kernel_new(
                name.as_ptr(),
                input_names.raw,
                output_names.raw,
                source.as_ptr(),
                header.as_ptr(),
                options.ensure_row_contiguous,
                options.atomic_outputs,
            )
        };
        if raw.ctx.is_null() {
            return Err(crate::error::exception_from_status(
                SUCCESS + 1,
                "creating a Metal kernel",
            ));
        }
        Ok(Self { raw })
    }

    /// Runs the kernel over `grid` threads in groups of `thread_group`, returning one array per
    /// entry of `outputs`.
    ///
    /// `inputs` and `outputs` must match the counts given at construction. Like other
    /// operations the result is lazy; the kernel compiles and runs when the outputs are
    /// evaluated. Source and threadgroup errors surface from [`Array::eval`]. The kernel and
    /// option values may be dropped before evaluating the outputs.
    ///
    /// # Safety
    ///
    /// For the given inputs, outputs, template arguments and launch configuration, the kernel
    /// must only access memory inside the input and output buffers and must never write an input.
    /// It must write every output element before it is read unless `init_value` is set, only
    /// write values valid for the output dtype, and have no unsynchronized conflicting accesses.
    /// These requirements apply throughout deferred execution.
    pub unsafe fn apply(
        &self,
        inputs: &[impl AsRef<Array>],
        outputs: &[MetalKernelOutput],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        options: MetalKernelApplyOptions,
    ) -> Result<Vec<Array>> {
        crate::error::INIT_ERR_HANDLER.call_once(crate::error::setup_mlx_error_handler);
        let config = KernelConfig::new()?;
        for output in outputs {
            config.check(
                unsafe {
                    mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                        config.raw,
                        output.shape.as_ptr(),
                        output.shape.len(),
                        output.dtype.into(),
                    )
                },
                "adding a Metal kernel output",
            )?;
        }
        config.check(
            unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_set_grid(config.raw, grid.0, grid.1, grid.2)
            },
            "setting the Metal kernel grid",
        )?;
        config.check(
            unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(
                    config.raw,
                    thread_group.0,
                    thread_group.1,
                    thread_group.2,
                )
            },
            "setting the Metal kernel thread group",
        )?;
        for (name, arg) in &options.template_args {
            let name = c_string(name)?;
            let status = unsafe {
                match *arg {
                    TemplateArg::Dtype(dtype) => {
                        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                            config.raw,
                            name.as_ptr(),
                            dtype.into(),
                        )
                    }
                    TemplateArg::Int(value) => {
                        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                            config.raw,
                            name.as_ptr(),
                            value,
                        )
                    }
                    TemplateArg::Bool(value) => {
                        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                            config.raw,
                            name.as_ptr(),
                            value,
                        )
                    }
                }
            };
            config.check(status, "adding a Metal kernel template argument")?;
        }
        if let Some(value) = options.init_value {
            config.check(
                unsafe { mlx_sys::mlx_fast_metal_kernel_config_set_init_value(config.raw, value) },
                "setting the Metal kernel init value",
            )?;
        }
        config.check(
            unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_set_verbose(config.raw, options.verbose)
            },
            "setting Metal kernel verbosity",
        )?;

        let inputs = crate::utils::VectorArray::try_from_iter(inputs.iter())?;
        let stream = Stream::thread_local_or_gpu();
        Vec::<Array>::try_from_op(|res| unsafe {
            mlx_sys::mlx_fast_metal_kernel_apply(
                res,
                self.raw,
                inputs.as_ptr(),
                config.raw,
                stream.as_ref().as_ptr(),
            )
        })
    }
}

#[cfg(feature = "metal")]
impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_fast_metal_kernel_free(self.raw) };
    }
}

#[cfg(feature = "metal")]
impl std::fmt::Debug for MetalKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalKernel").finish_non_exhaustive()
    }
}

#[cfg(feature = "metal")]
struct KernelConfig {
    raw: mlx_sys::mlx_fast_metal_kernel_config,
}

#[cfg(feature = "metal")]
impl KernelConfig {
    fn new() -> Result<Self> {
        let raw = unsafe { mlx_sys::mlx_fast_metal_kernel_config_new() };
        if raw.ctx.is_null() {
            return Err(crate::error::exception_from_status(
                SUCCESS + 1,
                "creating a Metal kernel config",
            ));
        }
        Ok(Self { raw })
    }

    fn check(&self, status: i32, operation: &str) -> Result<()> {
        if status == SUCCESS {
            Ok(())
        } else {
            Err(crate::error::exception_from_status(status, operation))
        }
    }
}

#[cfg(feature = "metal")]
impl Drop for KernelConfig {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_fast_metal_kernel_config_free(self.raw) };
    }
}

#[cfg(feature = "metal")]
struct StringVector {
    raw: mlx_sys::mlx_vector_string,
}

#[cfg(feature = "metal")]
impl StringVector {
    fn new(values: &[&str]) -> Result<Self> {
        let owned = values
            .iter()
            .map(|value| c_string(value))
            .collect::<Result<Vec<_>>>()?;
        let mut pointers: Vec<_> = owned.iter().map(|value| value.as_ptr()).collect();
        // mlx-c copies each string into the vector, so `owned` only needs to outlive this call.
        let raw =
            unsafe { mlx_sys::mlx_vector_string_new_data(pointers.as_mut_ptr(), pointers.len()) };
        if raw.ctx.is_null() {
            return Err(crate::error::exception_from_status(
                SUCCESS + 1,
                "creating an MLX string vector",
            ));
        }
        Ok(Self { raw })
    }
}

#[cfg(feature = "metal")]
impl Drop for StringVector {
    fn drop(&mut self) {
        let _ = unsafe { mlx_sys::mlx_vector_string_free(self.raw) };
    }
}

#[cfg(feature = "metal")]
fn c_string(value: &str) -> Result<CString> {
    CString::new(value).map_err(|error| Exception::custom(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ops::indexing::{ArrayIndexOp, IndexOp},
        random::normal,
    };
    use float_eq::assert_float_eq;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_rope() {
        crate::random::seed(71).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let result = rope(a, 8, false, 10000., 1.0, 0, None).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.456_253_77,
            abs <= 0.009_125_075
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            116.800_964,
            abs <= 2.336_019_3
        );
    }

    // Test adapted from Python test_fast.py/test_rope - the Python test accepts both
    // int offset and array offset, which in C/Rust are separate functions
    #[test]
    fn test_rope_dynamic() {
        crate::random::seed(71).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        // Test with array offset - should produce similar results to int offset of 3
        let offset = crate::Array::from_int(3);
        let result = rope_dynamic(&a, 8, false, 10000., 1.0, &offset, None).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);

        // Compare with regular rope using int offset=3
        let result_int_offset = rope(&a, 8, false, 10000., 1.0, 3, None).unwrap();
        assert_eq!(result_int_offset.shape(), [2, 8, 16]);

        // The results should be close
        let diff = &result - &result_int_offset;
        let max_diff = diff.abs().unwrap().max(None).unwrap().item_exact::<f32>();
        assert!(max_diff < 1e-5, "Max difference was {}", max_diff);
    }

    #[test]
    fn test_rms_norm() {
        crate::random::seed(103).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let result = rms_norm(a, Some(&weight), 1e-5).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.872_938_75,
            abs <= 0.017_458_774
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            223.472_32,
            abs <= 4.469_446
        );

        let ones = Array::ones::<f32>(&[2, 4]).unwrap();
        let normalized = rms_norm(&ones, None, 0.0).unwrap();
        assert_eq!(normalized.as_slice::<f32>(), &[1.0; 8]);
    }

    #[test]
    pub fn test_layer_norm_affine() {
        crate::random::seed(635).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let bias = Array::zeros::<f32>(&[16]).unwrap();
        let result = layer_norm(a, &weight, &bias, 1e-5).unwrap();
        let result = result.index((ArrayIndexOp::Ellipsis, 0));
        assert_eq!(result.shape(), [2, 8]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item_exact::<f32>(),
            0.290_990_38,
            abs <= 0.005_819_807_8
        );
        assert_float_eq!(
            result.sum(None).unwrap().item_exact::<f32>(),
            4.655_846,
            abs <= 0.093_116_924
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_fast_sdpa() {
        // This test just makes sure that `scaled_dot_product_attention` is callable
        // in the various cases, based on the Python test `test_fast_sdpa`.

        let Dk = 64;
        let scale = 1.0 / (Dk as f32).sqrt();
        for seq_len in [63, 129, 400] {
            for dtype in [crate::Dtype::Float32, crate::Dtype::Float16] {
                let B = 2;
                let H = 24;
                let q = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                let k = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                let v = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();

                let result = scaled_dot_product_attention(q, k, v, scale, None, None).unwrap();
                assert_eq!(result.shape(), [B, H, seq_len, Dk]);
                assert_eq!(result.dtype(), dtype);
                result.eval().unwrap();
            }
        }
    }

    // Test adapted from Python test `test_fast_sdpa.py/test_sdpa_attention_sinks`
    #[test]
    fn test_fast_sdpa_with_sinks() {
        let b = 2;
        let n_q = 8;
        let t_q = 128;
        let t_kv = 128;
        let d = 64;

        let q = normal::<f32>(&[b, n_q, t_q, d], None, None, None).unwrap();
        let k = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let v = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let scale = (d as f32).powf(-0.5);

        // Test with sinks parameter
        let sinks = normal::<f32>(&[n_q], None, None, None).unwrap() * 10.0;

        let result = scaled_dot_product_attention(&q, &k, &v, scale, None, &sinks).unwrap();
        assert_eq!(result.shape(), &[b, n_q, t_q, d]);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_kernel_template_args_and_shapes() {
        let kernel = MetalKernel::new(
            "scale_shift",
            &["a", "b"],
            &["out1", "out2"],
            "uint elem = thread_position_in_grid.x;
             T v = a[elem] * T(SCALE);
             out1[elem] = SHIFT ? v + b[0] : v;
             out2[elem] = a_shape[0];",
            MetalKernelOptions::default(),
        )
        .unwrap();
        let a = &crate::ops::arange::<_, f32>(None, 8, None).unwrap() / 4.0;
        let b = Array::from_slice(&[0.5_f32], &[1]);
        // SAFETY: Eight threads read valid inputs and initialize distinct outputs of matching dtypes.
        let outputs = crate::with_device(crate::Device::gpu(), || unsafe {
            kernel.apply(
                &[&a, &b],
                &[
                    MetalKernelOutput {
                        shape: vec![8],
                        dtype: Dtype::Float32,
                    },
                    MetalKernelOutput {
                        shape: vec![8],
                        dtype: Dtype::Int32,
                    },
                ],
                (8, 1, 1),
                (4, 1, 1),
                MetalKernelApplyOptions {
                    template_args: vec![
                        ("T".into(), TemplateArg::Dtype(Dtype::Float32)),
                        ("SCALE".into(), TemplateArg::Int(3)),
                        ("SHIFT".into(), TemplateArg::Bool(true)),
                    ],
                    ..Default::default()
                },
            )
        })
        .unwrap();

        assert_eq!(
            outputs[0].to_vec_exact::<f32>().unwrap(),
            [0.5, 1.25, 2.0, 2.75, 3.5, 4.25, 5.0, 5.75]
        );
        assert_eq!(outputs[1].to_vec_exact::<i32>().unwrap(), [8; 8]);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_kernel_init_value_fills_unwritten_outputs() {
        let kernel = MetalKernel::new(
            "touch_half",
            &["a"],
            &["out"],
            "uint e = thread_position_in_grid.x; if (e < 2) out[e] = a[e];",
            MetalKernelOptions::default(),
        )
        .unwrap();
        let a = Array::from_slice(&[0.0_f32, 0.25, 0.5, 0.75], &[4]);
        // SAFETY: Threads 0 and 1 copy valid f32 elements; init_value initializes the rest.
        let out = crate::with_device(crate::Device::gpu(), || unsafe {
            kernel.apply(
                &[&a],
                &[MetalKernelOutput {
                    shape: vec![4],
                    dtype: Dtype::Float32,
                }],
                (4, 1, 1),
                (4, 1, 1),
                MetalKernelApplyOptions {
                    init_value: Some(-1.0),
                    ..Default::default()
                },
            )
        })
        .unwrap();
        assert_eq!(
            out[0].to_vec_exact::<f32>().unwrap(),
            [0.0, 0.25, -1.0, -1.0]
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_kernel_rejects_invalid_arguments() {
        let error = MetalKernel::new("none", &["a"], &[], "", MetalKernelOptions::default())
            .expect_err("a kernel needs at least one output");
        assert!(error.what().contains("at least one output"));
        assert!(MetalKernel::new("nul", &["a\0"], &["out"], "", Default::default()).is_err());

        let kernel = MetalKernel::new(
            "copy",
            &["a"],
            &["out"],
            "out[0] = a[0];",
            Default::default(),
        )
        .unwrap();
        let output = [MetalKernelOutput {
            shape: vec![1],
            dtype: Dtype::Float32,
        }];
        let a = Array::from_slice(&[1.0_f32], &[1]);
        let no_inputs: [&Array; 0] = [];
        // SAFETY: The input count mismatch is rejected before any kernel can execute.
        let error = unsafe {
            kernel.apply(
                &no_inputs,
                &output,
                (1, 1, 1),
                (1, 1, 1),
                Default::default(),
            )
        }
        .expect_err("input count must match construction");
        assert!(error.what().contains("`inputs`"));

        let error = crate::with_device(crate::Device::cpu(), || {
            // SAFETY: One thread copies one f32 between single-element buffers.
            unsafe { kernel.apply(&[&a], &output, (1, 1, 1), (1, 1, 1), Default::default()) }
        })
        .expect_err("custom Metal kernels only run on the GPU");
        assert!(error.what().contains("[metal_kernel]"));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_kernel_gpu_fallback_and_scopes() {
        let kernel = MetalKernel::new(
            "stream_copy",
            &["a"],
            &["out"],
            "out[0] = a[0];",
            Default::default(),
        )
        .unwrap();
        let a = Array::from_slice(&[3.0_f32], &[1]);
        let output = [MetalKernelOutput {
            shape: vec![1],
            dtype: Dtype::Float32,
        }];
        let apply = || {
            // SAFETY: One thread copies one f32 between single-element buffers.
            unsafe { kernel.apply(&[&a], &output, (1, 1, 1), (1, 1, 1), Default::default()) }
        };
        assert_eq!(apply().unwrap()[0].to_vec_exact::<f32>().unwrap(), [3.0]);
        let previous = crate::Device::default();
        crate::Device::set_default(&crate::Device::cpu());
        let fallback = apply();
        let gpu_scope = crate::with_device(crate::Device::gpu(), apply);
        let cpu_scope = crate::with_device(crate::Device::cpu(), apply);
        let gpu_stream = crate::with_stream(&Stream::gpu(), apply);
        let cpu_stream = crate::with_stream(&Stream::cpu(), apply);
        crate::Device::set_default(&previous);
        for outputs in [fallback, gpu_scope, gpu_stream] {
            assert_eq!(outputs.unwrap()[0].to_vec_exact::<f32>().unwrap(), [3.0]);
        }
        assert!(cpu_scope.is_err());
        assert!(cpu_stream.is_err());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_kernel_layout_copies_are_opt_in() {
        let a = Array::from_slice(&[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
        let transposed = a.transpose().unwrap();
        for (options, expected) in [
            (
                MetalKernelOptions::default(),
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            ),
            (
                MetalKernelOptions {
                    ensure_row_contiguous: true,
                    ..Default::default()
                },
                [0.0, 3.0, 1.0, 4.0, 2.0, 5.0],
            ),
        ] {
            let kernel = MetalKernel::new(
                "physical_copy",
                &["a"],
                &["out"],
                "uint i = thread_position_in_grid.x; out[i] = a[i];",
                options,
            )
            .unwrap();
            // SAFETY: Six threads copy distinct f32 elements within both six-element buffers.
            let outputs = unsafe {
                kernel.apply(
                    &[&transposed],
                    &[MetalKernelOutput {
                        shape: vec![6],
                        dtype: Dtype::Float32,
                    }],
                    (6, 1, 1),
                    (6, 1, 1),
                    Default::default(),
                )
            }
            .unwrap();
            assert_eq!(outputs[0].to_vec_exact::<f32>().unwrap(), expected);
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_kernel_errors_are_deferred() {
        for (name, source, size, group) in [
            ("bad_source", "out[0] = a[0] +;", 1, 1),
            (
                "bad_group",
                "uint i = thread_position_in_grid.x; out[i] = a[i];",
                4096,
                4096,
            ),
        ] {
            let kernel =
                MetalKernel::new(name, &["a"], &["out"], source, Default::default()).unwrap();
            let input = Array::from_slice(&vec![1.0_f32; size as usize], &[size]);
            // SAFETY: Invalid source cannot execute; the copy case has one f32 per grid thread.
            let outputs = unsafe {
                kernel.apply(
                    &[&input],
                    &[MetalKernelOutput {
                        shape: vec![size],
                        dtype: Dtype::Float32,
                    }],
                    (size, 1, 1),
                    (group, 1, 1),
                    Default::default(),
                )
            }
            .expect("source and threadgroup errors are deferred");
            assert!(outputs[0].eval().is_err());
        }
        let kernel = MetalKernel::new(
            "after_error",
            &["a"],
            &["out"],
            "out[0] = a;",
            Default::default(),
        )
        .unwrap();
        let input = Array::from_f32(2.0);
        // SAFETY: One thread reads a scalar and initializes the single f32 output.
        let outputs = unsafe {
            kernel.apply(
                &[&input],
                &[MetalKernelOutput {
                    shape: vec![1],
                    dtype: Dtype::Float32,
                }],
                (1, 1, 1),
                (1, 1, 1),
                Default::default(),
            )
        }
        .unwrap();
        assert_eq!(outputs[0].to_vec_exact::<f32>().unwrap(), [2.0]);
    }
}
