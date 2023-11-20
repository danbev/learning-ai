use std::fs;
use text_splitter::TextSplitter;

use async_trait::async_trait;
use llm_chain::executor;
use llm_chain::options;
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
        None,
    );

    if add_doc {
        // Add a single VEX document
        let file_path = "data/vex-stripped.json".to_owned();
        let vex = fs::read_to_string(file_path).expect("Couldn't find or load vex file.");
        // Default implementation uses character count for chunk size
        let max_characters = 1536;
        let splitter = TextSplitter::default().with_trim_chunks(true);
        let chunks = splitter
            .chunks(&vex, max_characters)
            .map(String::from)
            .collect::<Vec<_>>();
        let chs: Vec<Document> = chunks.into_iter().map(Document::new).collect();

        //let doc_ids = qdrant.add_texts(chs).await.unwrap();
        let doc_ids = qdrant.add_documents(chs).await.unwrap();
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

    let quadrant_tool = QdrantTool::new(
        qdrant,
        "VEX documents, Red Hat Security Advisories (RHSA) information which use id's in the format RHSA-xxxx-xxxx",
        "VEX documents, RHSA information",
    );

    let mut tool_collection = ToolCollection::<Multitool>::new();
    tool_collection.add_tool(quadrant_tool.into());

    let tool_prompt = tool_collection.to_prompt_template().unwrap();
    let template = StringTemplate::static_string(tool_prompt.to_string());

    let prompt = ChatMessageCollection::new()
        .with_system(template)
        .with_user(StringTemplate::combine(vec![
            tool_collection.to_prompt_template().unwrap(),
            StringTemplate::tera("Please perform the following task: {{task}}."),
        ]));

    let query = "Can you show me a summary of RHSA-2020:5566?";
    println!("Query: {}", query);

    for _ in 0..2 {
        let result = Step::for_prompt_template(prompt.clone().into())
            .run(&parameters!("task" => query), &exec)
            .await
            .unwrap();

        let assistent_text = result
            .to_immediate()
            .await?
            .primary_textual_output()
            .unwrap();
        match tool_collection.process_chat_input(&assistent_text).await {
            Ok(tool_output) => {
                let yaml = serde_yaml::from_str::<serde_yaml::Value>(&tool_output).unwrap();
                let texts = yaml.get("texts").unwrap();
                let mut joined_text = String::new();
                if let Some(sequence) = texts.as_sequence() {
                    for (i, text) in sequence.iter().enumerate() {
                        if let Some(str_value) = text.as_str() {
                            joined_text.push_str(str_value);
                            if i < sequence.len() - 1 {
                                joined_text.push_str(" ");
                            }
                        }
                    }
                }
                let prompt = ChatMessageCollection::new()
                    .with_system(StringTemplate::tera(
                       "You are a friendly assistent and help answer questions. Use the following as additional context: {{texts}}.",
                    ))
                    .with_user(StringTemplate::combine(vec![
                        StringTemplate::static_string(query),
                    ]));
                let result = Step::for_prompt_template(prompt.clone().into())
                    .run(&parameters!("texts" => joined_text), &exec)
                    .await
                    .unwrap();
                println!("Result: {}", result);
                break;
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }

    Ok(())
}
