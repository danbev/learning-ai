use anyhow::{Context, Result};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, LlamaModel, AddBos, Special},
    sampling::LlamaSampler,
};
use llama_cpp_2::ggml_time_us;
use std::num::NonZeroU32;
use crate::tool::ToolManager;
use std::time::Duration;
use std::io::Write;

pub struct Agent {
    model: LlamaModel,
    backend: LlamaBackend,
    tool_manager: ToolManager,
}

impl Agent {
    pub fn new(model_path: std::path::PathBuf, components: Vec<std::path::PathBuf>) -> Result<Self> {
        let backend = LlamaBackend::init()?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .context("unable to load model")?;
        let tool_manager = ToolManager::new(components)?;

        Ok(Self {
            model,
            backend,
            tool_manager,
        })
    }

    pub fn run(&self, input: &str) -> Result<String> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(2048).unwrap()));
        
        let mut ctx = self.model
            .new_context(&self.backend, ctx_params)
            .context("unable to create context")?;

        let metadata = self.tool_manager.get_metadata();

        let available_tools: String = metadata.iter() .map(|md| {
            let params = md.params.iter()
                .map(|p| format!("{}=<{}>", p.name, p.type_))
                .collect::<Vec<_>>()
                .join(", ");

            let examples = md.usage.iter()
                .map(|ex| format!(
                    "User: {}\nAssistant: {}\n",
                    ex.user,
                    ex.assistent,
                ))
                .collect::<Vec<_>>()
                .join("\n");

            format!(
                "{} - {}\nUsage: USE_TOOL: {}, {}\n\nExamples:\n{}\n",
                md.name,
                md.description,
                md.name,
                params,
                examples
            )
        })
        .collect();

        let prompt = format!("<|user|>
You are a helpful AI assistant. You have access to several tools. When using a tool,
respond with the exact tool command format as shown in the examples below.

Available tools and their usage patterns:
{available_tools}

{input}
<|end|>
<|assistant|>"
);
        println!("Prompt: {}", prompt);

        let tokens_list = self.model
            .str_to_token(&prompt, AddBos::Never)
            .context("failed to tokenize prompt")?;

        let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(512, 1);
        let last_index: i32 = (tokens_list.len() - 1) as i32;
        
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            batch.add(token, i, &[0], i == last_index)?;
        }

        ctx.decode(&mut batch).context("failed to decode")?;

        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;
        let n_len = 200;

        let t_main_start = ggml_time_us();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::dist(1234),
            LlamaSampler::greedy(),
        ]);

        let mut response = String::new();
        while n_cur <= n_len {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if !self.model.is_eog_token(token) {
                let output_bytes = self.model.token_to_bytes(token, Special::Tokenize)?;
                let mut output_string = String::with_capacity(32);
                let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);
                response.push_str(&output_string);
            }

            //print!("{output_string}");
            //std::io::stdout().flush()?;
            
            if self.model.is_eog_token(token) {
                if response.contains("USE_TOOL:") {
                    let metadata = self.tool_manager.get_metadata();

                    for md in &metadata {
                        let tool_prefix = format!("USE_TOOL: {}", md.name);

                        if response.contains(&tool_prefix) {
                            let mut params = Vec::new();

                            if let Some(cmd_part) = response.split(&tool_prefix).nth(1) {
                                let cmd_part = cmd_part.trim_start_matches(", ").trim();
                                for param in &md.params {
                                    let param_prefix = format!("{}=", param.name);

                                    if let Some(value_part) = cmd_part.split(&param_prefix).nth(1) {
                                        let value = value_part.trim();

                                        if !value.is_empty() {
                                            params.push((param.name.clone(), value.to_string()));
                                        }
                                    }
                                }

                                if params.len() == md.params.len() {
                                    return self.tool_manager.execute_tool(&md.name, params);
                                }
                            }
                        }
                    }
                }
            }

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;

            n_cur += 1;
            ctx.decode(&mut batch).with_context(|| "failed to eval")?;
            n_decode += 1;
        }

        eprintln!("\n");

        let t_main_end = ggml_time_us();
        let duration = Duration::from_micros((t_main_end - t_main_start) as u64);
        eprintln!(
            "decoded {} tokens in {:.2} s, speed {:.2} t/s\n",
            n_decode,
            duration.as_secs_f32(),
            n_decode as f32 / duration.as_secs_f32()
        );
        println!("{}", ctx.timings());

        Ok(response)
    }
}
