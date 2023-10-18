## llm-chain notes

### Multitool
The following can be found in the examples that use mulitple tools and it was
not clear to me what the macro was doing.
```rust
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
```
This macro take four fixed inputs which are the first four, followed by a
list of tools and their inputs, outputs and errors. So in this case we have
a single tool called QdrantTool and it has a single input, output and error.

The expanded macro will contain:
```rust
enum Multitool {
    QdrantTool(QdrantTool),
}

impl From<QdrantTool> for Multitool {
    fn from(tool: QdrantTool) -> Self {
        Multitool::QdrantTool(tool)
    }
}

enum MultitoolInput {
    QdrantToolInput(QdrantToolInput),
}

enum MultitoolOutput {
    QdrantToolOutput(QdrantToolOutput),
}

enum MultitoolError {
    #[error("Could not convert")]
    BadVariant,

    #[error(transparent)]
    YamlError(#[from] serde_yaml::Error),

    #[error(transparent)]
    QdrantToolError(#[from] QdrantToolError),
}
```
Now, to understand this we can take a look a the declaration of ToolsCollection:
```rust
#[derive(Default)]
pub struct ToolCollection<T> {
    tools: Vec<T>,
}
```
Notice that it is generic of T and we create a new ToolCollection with:
```rust
    let mut tool_collection = ToolCollection::<Multitool>::new();
```
And we can add tools to this collection using:
```rust
    let quadrant_tool = QdrantTool::new(
        qdrant,
        "VEX documents, Red Hat Security Advisories (RHSA) information which use id's in the format RHSA-xxxx-xxxx",
        "vex documents, RHSA information",
    );
    tool_collection.add_tool(quadrant_tool.into());
```
Notice the call to `.into()` which will call the From implementation for
QdrantTool and convert it into a Multitool. 


Lets start with the VectorStoreToolInput:
```rust
pub struct VectorStoreToolInput {
    query: String,
    limit: u32,
}
```
So we have query string and a limit for the number of results to return?

The output of the tool is:
```rust
#[derive(Serialize, Deserialize)]
pub struct VectorStoreToolOutput {
    texts: Vec<String>,
}
```
So this is just a vector of strings.


Documents are represented as:
```rust
#[derive(Debug)]
pub struct Document<M = EmptyMetadata>
where M: serde::Serialize + serde::de::DeserializeOwned, {
    pub page_content: String,
    pub metadata: Option<M>,
}
```
So Documents are pretty simple struces with a page_content string and optional
metadata. Note that metadata might be very interesting as it allows for adding
information about the origin the content I think. So it would allow us to be
able to provide references to the sources of information which can be important
for content related to security.
