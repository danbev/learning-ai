use llm_chain::options::*;
use std::fs;
use text_splitter::TextSplitter;

use async_trait::async_trait;
use llm_chain::executor;
use llm_chain::options;
use llm_chain::prompt::{ChatMessageCollection, StringTemplate};
use llm_chain::schema::EmptyMetadata;
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
    let qdrant = build_local_qdrant(false).await;

    let llama_opts = options!(
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
    let exec = executor!(llama, llama_opts.clone())?;
    let quadrant_tool = QdrantTool::new(
        qdrant,
        "VEX documents, Red Hat Security Advisories (RHSA) information which use id's in the format RHSA-xxxx-xxxx",
        "VEX documents, RHSA information",
    );

    let mut tool_collection = ToolCollection::<Multitool>::new();
    tool_collection.add_tool(quadrant_tool.into());

    let query = "Can you show me a summary of RHSA-2020:5566?";
    println!("Query: {}", query);

    let sys_prompt = r#"<s>[INST] <<SYS>>
Assistant is designed to assist with a wide range of tasks.

{{ system_prompt }}

Here are some previous interactions between the Assistant and a User:

User: Can you show me a summary of the security advisory RHSA-2020:5566?
Assistant:
```yaml
command: VectorStoreTool
input:
  query: "RHSA-2020:5566"
  limit: 1
```

Observation: RHSA-2020:5566 is a security advisory related to openssl and has...
Assistant:
```yaml
command: "Final Answer"
input: "RHSA-2020:5566 is a security advisory related to openssl and has..."}}
```

All responses must be in YAML.

<</SYS>>

{{ user_message }} [/INST] </s>
        "#;

    let prompt = ChatMessageCollection::new().with_system(StringTemplate::tera(sys_prompt));
    let sys_message = tool_collection.to_prompt_template().unwrap().to_string();
    for _ in 0..1 {
        let result = Step::for_prompt_template(prompt.clone().into())
            .run(&parameters!("task" => query, "system_prompt" => sys_message.clone(), "user_message" => query), &exec)
            .await
            .unwrap();

        let assistent_text = result
            .to_immediate()
            .await?
            .primary_textual_output()
            .unwrap();
        println!("Assistent text: {}", assistent_text);
        match tool_collection.process_chat_input(&assistent_text).await {
            Ok(tool_output) => {
                let yaml = serde_yaml::from_str::<serde_yaml::Value>(&tool_output).unwrap();
                println!("YAML: {:?}", yaml);
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
                //println!("Joined text: {}", joined_text);
                let prompt = ChatMessageCollection::new().with_system(StringTemplate::tera(
                    "<s>[INST] <<SYS>> You are an assistent and help answer questions. Only reply with the answer to the question and nothing else.
                       Use the following as additional context: {{texts}} <</SYS>>

                       {{ user_message }} [/INST]",
                ));
                let result = Step::for_prompt_template(prompt.clone().into())
                    .run(
                        &parameters!("texts" => joined_text, "user_message" => query),
                        &exec,
                    )
                    .await
                    .unwrap();
                let output = result.to_immediate().await?;
                println!("output: {}", output);
                break;
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }

    Ok(())
}
