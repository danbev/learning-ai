use crate::llama_cpp::tool::types::ToolParams;
use crate::llama_cpp::tool::types::ToolUsage;
use std::io::Write;

wit_bindgen::generate!({
    world: "tool-world",
    path: "../../wit/tool.wit",
});

struct PrintTool;

impl Guest for PrintTool {
    fn run() -> Result<(),()> {
        Ok(())
    }

    fn get_metadata() -> ToolMetadata {
        let params = ToolParams {
            name: "message".to_string(),
            description: "Message to be printed to stdout".to_string(),
            type_: "string".to_string(),
            required: true,
        };

        let usage = ToolUsage {
            user: "Please print this string 'something'".to_string(),
            assistent: "USE_TOOL: Print, message='something'".to_string(),
        };

        ToolMetadata {
            name: "Print".to_string(),
            description: "Prints the provided message to stdout".to_string(),
            version: "0.1.0".to_string(),
            params: vec![params],
            usage: vec![usage],
        }
    }

    fn execute(params: Vec<(String, String)>) -> ToolResult {
        let message = &params.get(0).unwrap().1;

        println!("{}", message);

        ToolResult {
            success: true,
            data: format!("Successfully printed: {}", message),
            error: None,
        }
        
    }
}

export!(PrintTool);
