use llm_chain::options;
use llm_chain::options::ModelRef;
use llm_chain::{executor, parameters, prompt};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = options!(
         Model: ModelRef::from_model_name("pplx-70b-online")
    );
    let exec = executor!(chatgpt, opts.clone())?;
    let query = "Can you give me a summary of RHSA-2020:5566?";
    println!("Query: {query}\n");
    let res = prompt!("", query,).run(&parameters!(), &exec).await?;
    println!("Perplixity AI:\n{res}");
    Ok(())
}
