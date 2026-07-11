use std::path::Path;
use hf_hub::{api::sync::Api, Repo, RepoType};
use crate::models::llama::{Config, Llama, LlamaConfig};
use crate::nn::QuantizationConfig;
use crate::loader::{SafeTensorRepo, SafeTensorLoader};

pub struct AutoModelForCausalLM;

impl AutoModelForCausalLM {
    pub fn from_local(model_dir: &Path, quantization_config: QuantizationConfig) -> anyhow::Result<(Llama, Config)> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let mut config: Config = serde_json::from_str::<LlamaConfig>(&config_str)?
            .into_config(false);
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
                    .filter(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
                    .map(|e| e.path())
                    .collect();
                entries.sort();
                filenames = entries;
            }
        }

        let repo = SafeTensorRepo::load(&filenames)?;
        let loader = SafeTensorLoader::new(&repo);
        let model = Llama::load(&loader, &config)?;
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
            let index: serde_json::Value = serde_json::from_str(&index_str)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
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

        let model = Llama::load(&loader, &config)?;
        Ok((model, config))
    }
}
