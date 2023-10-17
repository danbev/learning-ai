use llm_chain::options::*;
use std::fs;
use text_splitter::TextSplitter;

use async_trait::async_trait;
use llm_chain::executor;
use llm_chain::options;
use llm_chain::prompt::{ChatMessageCollection, StringTemplate};
use llm_chain::schema::{Document, EmptyMetadata};
use llm_chain::step::Step;
use llm_chain::tools::tools::{
    BashTool, VectorStoreTool, VectorStoreToolError, VectorStoreToolInput, VectorStoreToolOutput,
};
use llm_chain::tools::tools::{BashToolError, BashToolInput, BashToolOutput};
use llm_chain::tools::{Tool, ToolCollection, ToolDescription, ToolError};
use llm_chain::traits::VectorStore;
use llm_chain::{multitool, parameters};
use llm_chain_qdrant::{Qdrant, QdrantError};
use std::sync::Arc;

use llm_chain_openai::embeddings::{Embeddings, OpenAIEmbeddingsError};
use qdrant_client::prelude::{QdrantClient, QdrantClientConfig};
use qdrant_client::qdrant::{CreateCollection, Distance, VectorParams, VectorsConfig};
use serde::{Deserialize, Serialize};
use thiserror::Error;
// A simple example generating a prompt with some tools.

// `multitool!` macro cannot handle generic annotations as of now; for now you will need to pass concrete arguments and alias your types
type QdrantTool = VectorStoreTool<Embeddings, EmptyMetadata, Qdrant<Embeddings, EmptyMetadata>>;
type QdrantToolInput = VectorStoreToolInput;
type QdrantToolOutput = VectorStoreToolOutput;
type QdrantToolError =
    VectorStoreToolError<QdrantError<OpenAIEmbeddingsError>, OpenAIEmbeddingsError>;

multitool!(
    Multitool,
    MultitoolInput,
    MultitoolOutput,
    MultitoolError,
    QdrantTool,
    QdrantToolInput,
    QdrantToolOutput,
    QdrantToolError,
    BashTool,
    BashToolInput,
    BashToolOutput,
    BashToolError
);

async fn build_local_qdrant(add_doc: bool) -> Qdrant<Embeddings, EmptyMetadata> {
    let config = QdrantClientConfig::from_url("http://localhost:6334");
    let client = Arc::new(QdrantClient::new(Some(config)).unwrap());
    let collection_name = "my-collection".to_string();
    let embedding_size = 1536;

    if !client
        .has_collection(collection_name.clone())
        .await
        .unwrap()
    {
        client
            .create_collection(&CreateCollection {
                collection_name: collection_name.clone(),
                vectors_config: Some(VectorsConfig {
                    config: Some(qdrant_client::qdrant::vectors_config::Config::Params(
                        VectorParams {
                            size: embedding_size,
                            distance: Distance::Cosine.into(),
                            hnsw_config: None,
                            quantization_config: None,
                            on_disk: None,
                        },
                    )),
                }),
                ..Default::default()
            })
            .await
            .unwrap();
    }

    let embeddings = llm_chain_openai::embeddings::Embeddings::default();

    let qdrant = Qdrant::<llm_chain_openai::embeddings::Embeddings, EmptyMetadata>::new(
        client,
        collection_name,
        embeddings,
        None,
        None,
    );

    if add_doc {
        // Add a single VEX document
        let file_path = "data/vex-stripped.json".to_owned();
        let vex = fs::read_to_string(file_path).expect("Couldn't find or load vex file.");
        // Default implementation uses character count for chunk size
        let max_characters = 1000;
        let splitter = TextSplitter::default().with_trim_chunks(true);
        let chunks = splitter.chunks(&vex, max_characters);
        let chs = chunks.into_iter().map(String::from).collect::<Vec<_>>();

        let doc_ids = qdrant.add_texts(chs).await.unwrap();
        println!("VEX documents stored under IDs: {:?}", doc_ids);
    }
    qdrant
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::var("OPENAI_API_KEY").expect(
        "You need an OPENAI_API_KEY env var with a valid OpenAI API key to run this example",
    );
    // Only need to perform this when we have new documents to add.
    let qdrant = build_local_qdrant(false).await;

    let mut tool_collection = ToolCollection::<Multitool>::new();
    let _opts = options!(
        Model: ModelRef::from_path("./models/llama-2-7b-chat.ggmlv3.q4_0.bin"),
        ModelType: "llama",
        MaxContextSize: 3000_usize,
        MaxTokens: 200_usize,
        Temperature: 1.0, // disabled
        TopP: 1.0, // 1.0 is the default and means no top-p sampling
        TopK: 0,  //
        RepeatPenalty: 1.0, // disbled
        RepeatPenaltyLastN: 0_usize, // disabled
        FrequencyPenalty: 0.0, // disabled
        PresencePenalty: 0.0, // disabled
        Mirostat: 0_i32, // disabled
        MirostatTau: 1.0,
        MirostatEta: 0.1,
        PenalizeNl: true,
        NThreads: 4_usize,
        Stream: false,
        TypicalP: 1.0, // disabled
        TfsZ: 1.0 // disabled
    );
    //let exec = executor!(llama, opts.clone())?;
    let openai_opts = options!(
        Stream: false
    );
    let exec = executor!(chatgpt, openai_opts).unwrap();

    tool_collection.add_tool(BashTool::new().into());
    tool_collection.add_tool(
        QdrantTool::new(
            qdrant,
            "factual information and trivia",
            "facts, news, knowledge or trivia",
        )
        .into(),
    );
    //println!("Tool collection: {:?}", tool_collection.describe());

    /*
    let sys_prompt = r#"
    <s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    Your output must always be YAML.
    <</SYS>>
        "#;

    let prompt = ChatMessageCollection::new()
        .with_system(StringTemplate::tera(sys_prompt))
        .with_user(StringTemplate::combine(vec![
            tool_collection.to_prompt_template().unwrap(),
            StringTemplate::tera("Please perform the following task: {{query}} [/INST]."),
        ]));
    */
    let prompt = ChatMessageCollection::new()
        .with_system(StringTemplate::tera(
            "You are an automated agent for performing tasks. Your output must always be YAML.",
        ))
        .with_user(StringTemplate::combine(vec![
            tool_collection.to_prompt_template().unwrap(),
            StringTemplate::tera("Please perform the following task: {{task}}."),
        ]));

    let query = "Can you show me a summary of RHSA-2020:5566?";
    // Notice that we are passing in the executor to run here.
    let result = Step::for_prompt_template(prompt.into())
        .run(&parameters!("task" => query), &exec)
        .await
        .unwrap();

    // This result is the result of the VectorStoreTool, which is a YAML.
    println!("Result: {}", result);

    match tool_collection
        .process_chat_input(
            &result
                .to_immediate()
                .await?
                .primary_textual_output()
                .unwrap(),
        )
        .await
    {
        Ok(output) => {
            println!("Tool output: {}", output);
            // This only provided the output of the tool I think, and we would
            // now want the LLM to generate a response, which is this case it
            // the summary.
        }
        Err(e) => println!("Error: {}", e),
    }

    Ok(())
}
