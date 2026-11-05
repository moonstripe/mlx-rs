//! EmbeddingGemma 2 text and image embedders against the NumPy f32 reference fixtures.

use anyhow::{ensure, Context, Result};
use mlx_lm::{Embedder, EmbeddingError, RgbImage};
use mlx_rs::{with_device, Array, Device};
use serde_json::Value;
use std::{fs, path::PathBuf};

fn root(name: &str) -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../conformance/mlx-lm/embedding-fixtures")
        .join(name)
}

pub fn fixture() -> PathBuf {
    root("embedding-gemma2-text")
}

fn json(name: &str) -> Result<Value> {
    read_json(&fixture(), name)
}

fn read_json(dir: &std::path::Path, name: &str) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(dir.join(name))?)?)
}

fn tolerance(expectations: &Value) -> Result<(f32, f32)> {
    let tolerance = &expectations["tolerances"]["embeddings"];
    Ok((
        tolerance["atol"].as_f64().context("atol")? as f32,
        tolerance["rtol"].as_f64().context("rtol")? as f32,
    ))
}

fn rows(array: &Array) -> Result<Vec<f32>> {
    ensure!(
        array.dtype() == mlx_rs::Dtype::Float32,
        "embeddings must be f32"
    );
    Ok(array.try_as_slice::<f32>()?.to_vec())
}

fn within(label: &str, expected: &[f32], actual: &[f32], atol: f32, rtol: f32) -> Result<()> {
    ensure!(expected.len() == actual.len(), "{label}: length differs");
    for (index, (&e, &a)) in expected.iter().zip(actual).enumerate() {
        ensure!(
            a.is_finite() && (e - a).abs() <= atol + rtol * e.abs(),
            "{label}: element {index} expected {e}, got {a}"
        );
    }
    Ok(())
}

pub fn run(device: Device) -> Result<()> {
    let inputs = json("inputs.json")?;
    let expectations = json("expectations.json")?;
    let (atol, rtol) = tolerance(&expectations)?;
    let texts: Vec<&str> = inputs["texts"]
        .as_array()
        .context("texts")?
        .iter()
        .map(|t| t.as_str().context("text"))
        .collect::<Result<_>>()?;
    let ids: Vec<Vec<u32>> = serde_json::from_value(inputs["token_ids"].clone())?;
    let golden = Array::load_safetensors(fixture().join("expectations.safetensors"))?;
    let golden = golden.get("embeddings").context("embeddings golden")?;
    let golden_shape = golden.shape().to_vec();
    let golden = rows(golden)?;

    with_device(device, || {
        let embedder = Embedder::from_dir(fixture())?;
        for (text, expected) in texts.iter().zip(&ids) {
            let encoded = embedder
                .tokenizer()
                .encode_with_special_tokens(text, true)?;
            let encoded: Vec<u32> = encoded.into_iter().map(u32::from).collect();
            ensure!(&encoded == expected, "token ids differ for {text:?}");
        }
        let batch = embedder.embed(&texts)?;
        ensure!(
            batch.shape() == golden_shape.as_slice(),
            "batch shape differs"
        );
        ensure!(
            embedder.dimension() as i32 == golden_shape[1],
            "dimension differs"
        );
        within("batch", &golden, &rows(&batch)?, atol, rtol)?;

        // Each text alone must reproduce its padded-batch row.
        let width = embedder.dimension();
        for (index, text) in texts.iter().enumerate() {
            let single = embedder.embed(&[text])?;
            let expected = &golden[index * width..(index + 1) * width];
            within(
                &format!("single {index}"),
                expected,
                &rows(&single)?,
                atol,
                rtol,
            )?;
        }

        ensure!(matches!(
            embedder.embed(&[]),
            Err(EmbeddingError::EmptyBatch)
        ));
        ensure!(matches!(
            embedder.embed(&["one", "a <|image|> fox"]),
            Err(EmbeddingError::MediaToken { input: 1, .. })
        ));
        let long = vec!["one"; 64].join(" ");
        ensure!(matches!(
            embedder.embed(&[long.as_str()]),
            Err(EmbeddingError::SequenceTooLong {
                input: 0,
                length: 66,
                limit: 64
            })
        ));
        let pixels = [0u8; 48 * 48 * 3];
        let image = RgbImage::new(&pixels, 48, 48).context("image")?;
        ensure!(matches!(
            embedder.embed_images(&[image]),
            Err(EmbeddingError::NoVisionTower)
        ));
        Ok(())
    })
}

pub fn run_vision(device: Device) -> Result<()> {
    let dir = root("embedding-gemma2-vision");
    let (atol, rtol) = tolerance(&read_json(&dir, "expectations.json")?)?;
    let golden = Array::load_safetensors(dir.join("expectations.safetensors"))?;
    let golden = rows(golden.get("embeddings").context("embeddings golden")?)?;
    let stored = Array::load_safetensors(dir.join("images.safetensors"))?;
    let mut images = Vec::new();
    for index in 0.. {
        let Some(image) = stored.get(&format!("image{index}")) else {
            break;
        };
        let shape = image.shape();
        ensure!(shape.len() == 3 && shape[2] == 3, "images are [h, w, 3]");
        let (height, width) = (shape[0] as usize, shape[1] as usize);
        images.push((image.try_as_slice::<u8>()?.to_vec(), height, width));
    }
    ensure!(images.len() == 2, "fixture holds two images");

    with_device(device, || {
        let embedder = Embedder::from_dir(&dir)?;
        let width = embedder.dimension();
        let wrapped = images
            .iter()
            .map(|(pixels, h, w)| RgbImage::new(pixels, *h, *w).context("image size"))
            .collect::<Result<Vec<_>>>()?;
        let batch = embedder.embed_images(&wrapped)?;
        ensure!(batch.shape() == [2, width as i32], "batch shape differs");
        within("image batch", &golden, &rows(&batch)?, atol, rtol)?;
        for (index, image) in wrapped.iter().enumerate() {
            let single = embedder.embed_images(&[*image])?;
            let expected = &golden[index * width..(index + 1) * width];
            within(
                &format!("image {index}"),
                expected,
                &rows(&single)?,
                atol,
                rtol,
            )?;
        }

        // Sides must be multiples of 6 here (2-pixel patches pooled 3x3), with at
        // most 6 soft tokens and 16 patches per side.
        let blank = vec![0u8; 18 * 18 * 3];
        for (input, (height, width)) in [(12, 9), (0, 6), (18, 18), (6, 36)].into_iter().enumerate()
        {
            let pixels = &blank[..height * width * 3];
            let image = RgbImage::new(pixels, height, width).context("blank")?;
            let result = embedder.embed_images(&[wrapped[0], image]);
            ensure!(
                matches!(result, Err(EmbeddingError::ImageSize { input: 1, .. })),
                "case {input}: {height}x{width} was accepted"
            );
        }
        ensure!(RgbImage::new(&blank[..10], 2, 2).is_none());
        ensure!(matches!(
            embedder.embed_images(&[]),
            Err(EmbeddingError::EmptyBatch)
        ));
        Ok(())
    })
}
