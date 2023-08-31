use llm::ModelArchitecture;
use std::{convert::Infallible, io::Write, path::PathBuf};

fn main() {
    let model_architecture = ModelArchitecture::GptNeoX;
    let tokenizer_source = llm::TokenizerSource::Embedded;
    let model_path = PathBuf::from("models/RedPajama-INCITE-Base-3B-v1-q4_0.bin");
    let prompt = "Rust is a cool programming language because";

    let now = std::time::Instant::now();

    let model = llm::load_dynamic(
        Some(model_architecture),
        &model_path,
        tokenizer_source,
        Default::default(),
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| {
        panic!("Failed to load {model_architecture} model from {model_path:?}: {err}")
    });

    println!(
        "Model fully loaded! Elapsed: {}ms",
        now.elapsed().as_millis()
    );

    let mut session = model.start_session(Default::default());

    let res = session.infer::<Infallible>(
        model.as_ref(),
        &mut rand::thread_rng(),
        &llm::InferenceRequest {
            prompt: prompt.into(),
            parameters: &llm::InferenceParameters::default(),
            play_back_previous_tokens: false,
            maximum_token_count: Some(10),
        },
        // OutputRequest
        &mut Default::default(),
        |r| match r {
            llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
                print!("{t}");
                std::io::stdout().flush().unwrap();
                Ok(llm::InferenceFeedback::Continue)
            }
            llm::InferenceResponse::EotToken => Ok(llm::InferenceFeedback::Continue),
            _ => Ok(llm::InferenceFeedback::Halt),
        },
    );

    match res {
        Ok(result) => println!("\n\nInference stats:\n{result}"),
        Err(err) => println!("\n{err}"),
    }
}
