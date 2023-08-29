use reqwest;
use serde::{Deserialize, Serialize};
use std::env;
use std::io;
use std::io::Write;

const ENDPOINT: &str = "https://api.openai.com/v1/completions";

#[derive(Serialize, Debug)]
#[allow(dead_code)]
struct Payload {
    model: String,
    prompt: String,
    max_tokens: usize,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct ResponseData {
    created: i64,
    id: String,
    object: String,
    model: String,
    choices: Vec<Choice>,
    // logprobs: Logprobs,
    //finish_reason: String,
    //usage: Usage,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct Choice {
    finish_reason: String,
    index: usize,
    text: String,
}

fn get_response(api_key: &str, message: &str) -> Result<String, reqwest::Error> {
    let client = reqwest::blocking::Client::new();

    let payload = Payload {
        model: "text-davinci-003".to_string(),
        // When gpt-3.5-turbo-instruct is available, use it instead.
        //model: "gpt-3.5-turbo-instruct".to_string(),
        prompt: message.to_string(),
        max_tokens: 150,
    };

    println!("Payload {:?}\n", payload);
    let response = client
        .post(ENDPOINT)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()?
        .text()?;
    let response: ResponseData = serde_json::from_str(&response).unwrap();
    Ok(response.choices[0].text.trim().to_string())
}

fn main() {
    let api_key = if let Ok(value) = env::var("OPENAI_API_KEY") {
        value
    } else {
        println!("Please set OPENAI_API_KEY environment variable");
        return;
    };

    loop {
        let mut input = String::new();
        print!("You: ");
        io::stdout().flush().unwrap();
        io::stdin().read_line(&mut input).unwrap();

        let trimmed = input.trim();
        if trimmed == "quit" || trimmed == "exit" {
            break;
        }

        match get_response(&api_key, trimmed) {
            Ok(response) => {
                println!("ChatGPT: {}", response)
            }
            Err(err) => println!("Error: {}", err),
        }
    }
}
