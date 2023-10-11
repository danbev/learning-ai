use llm_chain::options::*;
use llm_chain::prompt::Conversation;
use llm_chain::{
    chains::conversation::Chain, executor, options, output::StreamExt, parameters, prompt,
    step::Step,
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = options!(
        Model: ModelRef::from_path("./models/llama-2-7b-chat.ggmlv3.q4_0.bin"),
        ModelType: "llama",
        MaxContextSize: 2000_usize,
        MaxTokens: 50_usize,
        Temperature: 0.1,
        RepeatPenalty: 1.1,
        RepeatPenaltyLastN: 64_usize,
        FrequencyPenalty: 0.0,
        PresencePenalty: 0.0,
        Mirostat: 0_i32,
        MirostatTau: 5.0,
        MirostatEta: 0.1,
        PenalizeNl: true,
        NThreads: 4_usize
        //StopSequence: vec!["\n".to_string()]
    );
    let exec = executor!(llama, opts.clone())?;
    let step = Step::for_prompt_with_streaming(prompt!(
        "You are a helpful assistant that answers quesetions. The question is:\n--\n{{query}}\n--\n"));
    let query = "What is the meaning of life?";
    println!("Query: {}", query);
    //let res = Step::for_prompt_with_streaming(chat_prompt)
    let res = step.run(&parameters!().with("query", query), &exec).await?;
    //println!("Result: {}", res.to_immediate().await?);
    let mut stream = res.as_stream().await?;
    while let Some(v) = stream.next().await {
        print!("{}", v);
    }

    Ok(())
}
