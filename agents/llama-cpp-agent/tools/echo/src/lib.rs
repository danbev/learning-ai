use crate::llama_cpp::tool::types::ToolParams;

wit_bindgen::generate!({
    world: "tool-world",
    path: "../../wit/tool.wit",
});

struct EchoTool;

impl Guest for EchoTool {
    fn run() -> Result<(),()> {
        Ok(())
    }

    fn get_metadata() -> ToolMetadata {
        let params = ToolParams{
            name: "value".to_string(),
            description: "Value to be echoed".to_string(),
            type_: "string".to_string(),
            required: true,
        };

        ToolMetadata {
            name: "Echo".to_string(),
            description: "Echos the passed in value".to_string(),
            version: "0.1.0".to_string(),
            params: vec![params],
        }
    }

    fn execute(params: Vec<(String, String)>) -> ToolResult {
        let value = &params.get(0).unwrap().1;
        ToolResult {
            success: true,
            data: value.to_string(),
            error: None,
        }
    }
}

export!(EchoTool);
