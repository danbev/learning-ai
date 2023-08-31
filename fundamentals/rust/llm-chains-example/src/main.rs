use llm_chain::{executor, parameters, prompt};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new ChatGPT executor
    let exec = executor!()?;
    let res = prompt!("", "Explain positional encoding",)
        .run(&parameters!(), &exec)
        .await?;
    println!("{res}");
    Ok(())
}
