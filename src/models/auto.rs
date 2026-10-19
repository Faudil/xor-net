use crate::generation::Generator;
use crate::loader::{SafeTensorLoader, SafeTensorRepo, sparse_loader::SparseFile};
use crate::models::llama::{Config, Llama, LlamaConfig};
use crate::models::traits::CausalLM;
use crate::nn::QuantizationConfig;
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::path::Path;

pub struct AutoModelForCausalLM;

impl AutoModelForCausalLM {
    /// Load a 1-bit Bonsai model from a `.gguf` file (dispatching on
    /// `general.architecture`: `qwen3` → dense, `qwen35` → hybrid) or from a
    /// converted `.xq1` file (see [`crate::models::q1::format`]).
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Box<dyn CausalLM>> {
        crate::models::q1::load_causal_lm(path)
    }

    /// Like [`Self::from_file`], wrapped in a ready-to-use [`Generator`]
    /// with the model's tokenizer, chat template and EOS ids. The tokenizer
    /// is taken from `tokenizer_json` if given, else from the tokenizer
    /// embedded in the model file (`tokenizer.ggml.*` metadata).
    pub fn generator_from_file(
        path: impl AsRef<Path>,
        tokenizer_json: Option<&Path>,
    ) -> anyhow::Result<Generator> {
        crate::models::q1::load_generator(path, tokenizer_json)
    }

    pub fn from_local(
        model_dir: &Path,
        quantization_config: QuantizationConfig,
    ) -> anyhow::Result<(Llama, Config)> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let mut config: Config =
            serde_json::from_str::<LlamaConfig>(&config_str)?.into_config(false);
        config.quantization_config = quantization_config;

        let index_path = model_dir.join("model.safetensors.index.json");
        let mut filenames = vec![];
        if index_path.exists() {
            let index_str = std::fs::read_to_string(&index_path)?;
            let index: serde_json::Value = serde_json::from_str(&index_str)?;
            if let Some(weight_map) = index.get("weight_map").and_then(|w| w.as_object()) {
                let mut unique_files: Vec<String> = weight_map
                    .values()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                unique_files.sort();
                unique_files.dedup();
                for file in unique_files {
                    filenames.push(model_dir.join(file));
                }
            }
        } else {
            let single = model_dir.join("model.safetensors");
            if single.exists() {
                filenames.push(single);
            } else {
                // Fallback: glob any *.safetensors files
                let mut entries: Vec<_> = std::fs::read_dir(model_dir)?
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        e.path()
                            .extension()
                            .map_or(false, |ext| ext == "safetensors")
                    })
                    .map(|e| e.path())
                    .collect();
                entries.sort();
                filenames = entries;
            }
        }

        let repo = SafeTensorRepo::load(&filenames)?;
        let loader = SafeTensorLoader::new(&repo);

        // XorSparse: if requested and a pre-built `model.sparse` exists, load it
        // so the engine streams the smaller sparse blobs. Otherwise the engine
        // ternarizes+re-encodes on the fly (also triggered by XORNET_WEIGHT_FMT).
        let sparse_file = if crate::nn::dynamic_linear::sparse_requested() {
            model_dir
                .join("model.sparse")
                .exists()
                .then(|| SparseFile::open(&model_dir.join("model.sparse")).ok())
                .flatten()
        } else {
            None
        };
        let model = Llama::load(&loader, &config, sparse_file.as_ref())?;
        Ok((model, config))
    }

    pub fn from_pretrained(
        repo_id: &str,
        quantization_config: QuantizationConfig,
    ) -> anyhow::Result<(Llama, Config)> {
        let api = Api::new().map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

        let config_filename = repo
            .get("config.json")
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let config_str = std::fs::read_to_string(config_filename)?;
        let mut config: Config = serde_json::from_str::<LlamaConfig>(&config_str)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .into_config(false);

        config.quantization_config = quantization_config;

        let mut filenames = vec![];
        if let Ok(index_file) = repo.get("model.safetensors.index.json") {
            let index_str = std::fs::read_to_string(index_file)?;
            let index: serde_json::Value =
                serde_json::from_str(&index_str).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            if let Some(weight_map) = index.get("weight_map").and_then(|w| w.as_object()) {
                let mut unique_files: Vec<String> = weight_map
                    .values()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                unique_files.sort();
                unique_files.dedup();
                for file in unique_files.into_iter() {
                    let path = repo
                        .get(&file)
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    filenames.push(path);
                }
            }
        } else {
            let path = repo
                .get("model.safetensors")
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            filenames.push(path);
        }

        let repo = SafeTensorRepo::load(&filenames)?;
        let loader = SafeTensorLoader::new(&repo);

        let model = Llama::load(&loader, &config, None)?;
        Ok((model, config))
    }
}
