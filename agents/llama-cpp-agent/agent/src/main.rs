use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

mod agent;
mod tool;

#[derive(Parser)]
struct Args {
    /// Path to the model file
    #[arg(short, long)]
    model: PathBuf,
    
    /// The input prompt
    #[arg(short, long)]
    prompt: String,

    /// List of components to load
    #[arg(short, long, required = true, value_parser = validate_wasm_file)]
    components: Vec<PathBuf>,
}

fn validate_wasm_file(s: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(s);
    if path.extension().and_then(|ext| ext.to_str()) == Some("wasm") {
        Ok(path)
    } else {
        Err(format!("File '{}' must have a .wasm extension", s))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let components = args.components;
    let agent = agent::Agent::new(args.model, components)?;
    
    let response = agent.run(&args.prompt)?;
    println!("Agent response: {}", response);

    Ok(())
}
