use llm_chain::options::*;
use llm_chain::prompt::ChatMessage;
use std::fs;
use text_splitter::TextSplitter;

use async_trait::async_trait;
use llm_chain::executor;
use llm_chain::options;
use llm_chain::prompt::ChatRole;
use llm_chain::prompt::{ChatMessageCollection, StringTemplate};
use llm_chain::schema::{Document, EmptyMetadata};
use llm_chain::step::Step;
use llm_chain::tools::tools::{
    VectorStoreTool, VectorStoreToolError, VectorStoreToolInput, VectorStoreToolOutput,
};
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

// `multitool!` macro cannot handle generic annotations as of now; for now you
// will need to pass concrete arguments and alias your types
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
    QdrantToolError
);

// This would normally be in a separate file, or perhaps done by a separate
// process all together.
async fn build_local_qdrant(add_doc: bool) -> Qdrant<Embeddings, EmptyMetadata> {
    let config = QdrantClientConfig::from_url("http://localhost:6334");
    let client = Arc::new(QdrantClient::new(Some(config)).unwrap());
    let collection_name = "vec-documents".to_string();
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
    let qdrant = build_local_qdrant(false).await;

    let openai_opts = options!(
        Stream: false
    );
    let exec = executor!(chatgpt, openai_opts).unwrap();

    let mut tool_collection = ToolCollection::<Multitool>::new();
    let quadrant_tool = QdrantTool::new(
        qdrant,
        "VEX documents, Red Hat Security Advisories (RHSA) information which use id's in the format RHSA-xxxx-xxxx",
        "vex documents, RHSA information",
    );
    tool_collection.add_tool(quadrant_tool.into());

    let tool_prompt = tool_collection.to_prompt_template().unwrap();
    let template = StringTemplate::combine(vec![
        tool_prompt,
        StringTemplate::tera(
            "You may ONLY use one tool at a time. Please perform the following task: {{task}}.",
        ),
    ]);

    let mut prompt = ChatMessageCollection::new()
        .with_system(StringTemplate::tera(
            "You are an automated agent for performing tasks. Your output must always be YAML.",
        ))
        .with_user(StringTemplate::combine(vec![
            tool_collection.to_prompt_template().unwrap(),
            StringTemplate::tera("Please perform the following task: {{task}}."),
        ]));

    let query = "Can you show me a summary of RHSA-2020:5566?";

    for _ in 1..3 {
        let result = Step::for_prompt_template(prompt.clone().into())
            .run(&parameters!("task" => query), &exec)
            .await
            .unwrap();
        let assistent_text = result
            .to_immediate()
            .await?
            .primary_textual_output()
            .unwrap();
        println!("{}", assistent_text);

        let step = match tool_collection.process_chat_input(&assistent_text).await {
            Ok(tool_output) => {
                println!("Tool output: {}", tool_output);
                StringTemplate::static_string(format!(
                    "```yaml
                    {}
                    ```
                    Proceed with your next command.",
                    tool_output
                ))
            }
            Err(e) => StringTemplate::static_string(format!(
                "Correct your output and perform the task - {}. Your task was: {}",
                e, query
            )),
        };
        println!("User: {}", step);
        prompt.add_message(ChatMessage::system(StringTemplate::static_string(
            assistent_text,
        )));
        prompt.add_message(ChatMessage::user(step));
    }

    Ok(())
}
