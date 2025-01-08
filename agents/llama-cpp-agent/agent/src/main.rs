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
}

fn main() -> Result<()> {
    let args = Args::parse();
    let agent = agent::Agent::new(args.model)?;
    
    let response = agent.run(&args.prompt)?;
    println!("Agent response: {}", response);

    Ok(())
}
