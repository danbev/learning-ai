use llm_chain::options::*;
use llm_chain::{executor, options, output::StreamExt, parameters, prompt, step::Step};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = options!(
        Model: ModelRef::from_path("./models/llama-2-7b-chat.ggmlv3.q4_0.bin"),
        ModelType: "llama",
        MaxContextSize: 3000_usize,
        MaxTokens: 200_usize,
        Temperature: 0.1,
        TopP: 0.1,
        TopK: 0,
        RepeatPenalty: 1.0,
        RepeatPenaltyLastN: 0_usize,
        FrequencyPenalty: 1.0,
        PresencePenalty: 1.0,
        Mirostat: 1_i32,
        MirostatTau: 1.0,
        MirostatEta: 0.1,
        PenalizeNl: true,
        NThreads: 4_usize,
        Stream: true,
        TypicalP: 2.0,
        TfsZ: 1.0
    );
    let exec = executor!(llama, opts.clone())?;
    let prompt_str = r#"
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{{ query }} [/INST]
    "#;
    let step = Step::for_prompt_with_streaming(prompt!(prompt_str));
    let query = "What is LoRa?";
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
