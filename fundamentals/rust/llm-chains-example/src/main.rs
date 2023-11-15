use llm_chain::{executor, parameters, prompt};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OpenAI ChatGPT example");
    // Create a new ChatGPT executor
    let exec = executor!()?;
    let query = "Explain positional encoding";
    println!("Query: {query}");
    let res = prompt!("", query,).run(&parameters!(), &exec).await?;
    println!("{res}");
    Ok(())
}
