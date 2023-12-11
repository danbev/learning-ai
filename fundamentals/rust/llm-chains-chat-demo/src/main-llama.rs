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

//use llm_chain_openai::embeddings::{Embeddings, OpenAIEmbeddingsError};
use llm_chain_llama::embeddings::{Embeddings, LlamaEmbeddingsError};
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
    VectorStoreToolError<QdrantError<LlamaEmbeddingsError>, LlamaEmbeddingsError>;

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

async fn build_local_qdrant(add_doc: bool, opts: Options) -> Qdrant<Embeddings, EmptyMetadata> {
    let config = QdrantClientConfig::from_url("http://localhost:6334");
    let client = Arc::new(QdrantClient::new(Some(config)).unwrap());
    let collection_name = "vec-documents".to_string();
    //let embedding_size = 1536;
    let embedding_size = 4096;

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

    let embeddings = llm_chain_llama::embeddings::Embeddings::new_with_options(opts).unwrap();

    let qdrant = Qdrant::<llm_chain_llama::embeddings::Embeddings, EmptyMetadata>::new(
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
    let llama_opts = options!(
        Model: ModelRef::from_path("./models/llama-2-7b-chat.Q4_0.gguf"),
        ModelType: "llama",
        MaxContextSize: 4096usize,
        MaxTokens: 4096_usize,
        MaxBatchSize: 4096_usize,
        Temperature: 0.3, // disabled
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
        TfsZ: 1.0, // disabled
        StopSequence: vec!["\n \n".to_string()]
    );
    let qdrant = build_local_qdrant(false, llama_opts.clone()).await;

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

    let sys_prompt = r#"
[INST] <<SYS>>

{{ system_prompt }}

Only respond with the YAML followed by </s> and nothing else.

Here are some previous interactions between the Assistant and a User:

User: What is RHSA-1820:1234?
Assistant:
```yaml
command: VectorStoreTool
input:
  query: "RHSA-1820:1234"
  limit: 3
```

User: Can you show me the details about advisory RHSA-1721:4231?
Assistant:
```yaml
command: VectorStoreTool
input:
  query: "RHSA-1721:4231"
  limit: 3
```

User: What is RHSA-1721:4231?
Assistant:
```yaml
command: VectorStoreTool
input:
  query: "RHSA-1721:4231"
  limit: 3
```

Do not output anything apart from valid YAML. Do not output any text or other information.

<</SYS>>

{{ user_message }} [/INST]"#;

    let prompt = ChatMessageCollection::new().with_system(StringTemplate::tera(sys_prompt));
    /*
    .with_user(StringTemplate::tera(
        "Please perform the following task: {{task}}  [/INST]",
    ));
    */
    let tools_prompt = tool_collection.to_prompt_template().unwrap().to_string();
    for _ in 0..1 {
        let result = Step::for_prompt_template(prompt.clone().into())
            .run(
                &parameters!("user_message" => query, "system_prompt" => tools_prompt.clone()),
                &exec,
            )
            .await
            .unwrap();

        let assistant_text = result
            .to_immediate()
            .await?
            .primary_textual_output()
            .unwrap();
        println!("Assistant text: {}", assistant_text);
        match tool_collection.process_chat_input(&assistant_text).await {
            Ok(tool_output) => {
                let yaml = serde_yaml::from_str::<serde_yaml::Value>(&tool_output).unwrap();
                println!("-------> YAML: {:?}", yaml);
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
                println!("Joined text: {}", joined_text);
                let prompt = ChatMessageCollection::new().with_system(StringTemplate::tera(
                    "[INST] <<SYS>>\nYou are an assistant and help answer questions. Only reply with the answer to the question and nothing else. Use the following as additional context: {{texts}} \n<<SYS>>\n\n ",
                )).with_user(StringTemplate::tera("{{ task  }} [/INST]"));

                let result = Step::for_prompt_template(prompt.clone().into())
                    .run(&parameters!("texts" => joined_text, "task" => query), &exec)
                    .await
                    .unwrap();
                let output = result.to_immediate().await?;
                println!("output: {}", output);
                break;
            }
            Err(e) => {
                eprintln!("process_chat_input error: {:?}", e);
            }
        }
    }

    Ok(())
}
