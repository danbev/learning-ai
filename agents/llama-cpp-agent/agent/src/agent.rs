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
    pub fn new(model_path: std::path::PathBuf) -> Result<Self> {
        let backend = LlamaBackend::init()?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .context("unable to load model")?;
        let tool_manager = ToolManager::new()?;

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

        //let prompt = format!("<|user|>\n You are a helpful AI assistant. You have access to an Echo tool. When asked to echo something, respond ONLY with the exact tool command format'.\n\n Example interaction:\n User: Please echo back 'hello'\n Assistant: USE_TOOL: Echo, value=hello\n\n Available tool:\n Echo - Echoes back the input text\n Usage: USE_TOOL: Echo, value=<text to echo>\n\n {} <|end|>\n<|assistant|>", input);
        let prompt = format!("<|user|>\n You are a helpful AI assistant. You have access to an Echo tool. When asked to echo something, respond ONLY with the exact tool command format and include the complete text to be echoed.\n\n Example interaction:\n User: Please echo back 'hello'\n Assistant: USE_TOOL: Echo, value=hello\n\n Available tool:\n Echo - Echoes back the input text\n Usage: USE_TOOL: Echo, value=<text to echo>\n Note: Make sure to include the complete text after 'value='\n\n {} <|end|>\n<|assistant|>", input);

        /*
        let prompt = format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant. You have access to an Echo tool. When asked to echo something, respond ONLY with the exact tool command format.

Example interaction:
User: Please echo back 'hello'
Assistant: USE_TOOL: Echo, value=hello

Available tool:
Echo - Echoes back the input text
Usage: USE_TOOL: Echo, value=<text to echo><|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>",
            input
        );
        */

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

            if self.model.is_eog_token(token) {
                eprintln!();
                break;
            }

            let output_bytes = self.model.token_to_bytes(token, Special::Tokenize)?;
            let mut output_string = String::with_capacity(32);
            let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);
            response.push_str(&output_string);

            print!("{output_string}");
            std::io::stdout().flush()?;

            if response.contains("USE_TOOL: Echo") && response.contains("value=") {
                if let Some(cmd) = response.split("USE_TOOL: Echo, value=").nth(1) {
                    let value = cmd.trim();
                    if !value.is_empty() {
                        return self.tool_manager.execute_tool("Echo", vec![("value".to_string(), value.to_string())]);
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
