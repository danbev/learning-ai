use wasmtime::component::ResourceTable;
use wasmtime::{
    component::{Component, Linker as ComponentLinker},
    Config, Engine, Store,
};
use wasmtime_wasi::WasiCtxBuilder;
use wasmtime_wasi::WasiView;
use wasmtime_wasi::WasiCtx;
use anyhow::{Context, Result};
use clap::{Arg, Command, ArgMatches};
use std::path::PathBuf;

wasmtime::component::bindgen!({
    path: "../wit/tool.wit",
    async: false,
    world: "tool-world"
});

struct ToolCtx {
    table: ResourceTable,
    wasi: WasiCtx,
}

impl WasiView for ToolCtx {
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.table
    }
    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.wasi
    }
}

fn get_component_path() -> Result<PathBuf> {
    let matches = Command::new("run-tool")
        .version("0.1.0")
        .arg(
            Arg::new("component_path")
                .short('c')
                .long("component-path")
                .value_name("COMPONENT_PATH")
                .help("Path to the component wasm")
                .required(true),
        )
        .ignore_errors(true)  // Ignore unknown arguments in the first pass
        .get_matches();

    Ok(PathBuf::from(matches.get_one::<String>("component_path").unwrap()))
}

fn build_dynamic_args(metadata: &ToolMetadata) -> Command {
    let mut cmd = Command::new("run-tool")
        .version("0.1.0")
        .arg(
            Arg::new("component_path")
                .short('c')
                .long("component-path")
                .value_name("COMPONENT_PATH")
                .help("Path to the component wasm")
                .required(true),
        );

    // Add arguments dynamically based on tool metadata
    for param in &metadata.params {
        // Create 'static str (not mut) using Box::leak
        let name: &'static str = Box::leak(param.name.clone().into_boxed_str());
        let description: &'static str = Box::leak(param.description.clone().into_boxed_str());
        let type_: &'static str = Box::leak(param.type_.clone().into_boxed_str());
        
        let arg = Arg::new(name)
            .long(name)
            .help(description)
            .value_name(type_)
            .required(param.required);
        cmd = cmd.arg(arg);
    }

    cmd
}

fn collect_tool_params(matches: &ArgMatches, metadata: &ToolMetadata) -> Vec<(String, String)> {
    metadata
        .params
        .iter()
        .filter_map(|param| {
            matches.get_one::<String>(&param.name).map(|value| {
                (param.name.clone(), value.clone())
            })
        })
        .collect()
}

//#[tokio::main]
//async fn main() -> Result<()> {
fn main() -> Result<()> {
    let component_path = get_component_path()?;
    println!("Component path: {:?}", component_path);

    let mut config = Config::new();
    config.wasm_component_model(true);
    config.async_support(false);
    
    let engine = Engine::new(&config).context("Failed to create Engine")?;
    let component = Component::from_file(&engine, &component_path).context("Failed to create Component")?;
    let wasi_ctx = WasiCtxBuilder::new()
        .inherit_stdio()
        .inherit_args()
        .inherit_env()
        .build();
    let mut store = Store::new(
        &engine,
        ToolCtx {
            wasi: wasi_ctx,
            table: ResourceTable::new(),
        },
    );
    
    let mut component_linker = ComponentLinker::new(&engine);
    wasmtime_wasi::add_to_linker_sync(&mut component_linker)
        .context("Failed to add wasi to linker")?;
    
    let tool = ToolWorld::instantiate(&mut store, &component, &component_linker)
        .context("Failed to instantiate Tool")?;

    // Get metadata
    let metadata = tool.call_get_metadata(&mut store).unwrap();
    println!("Tool metadata:");
    println!("  Name: {}", metadata.name);
    println!("  Description: {}", metadata.description);
    println!("  Version: {}", metadata.version);
    println!("  Parameters:");
    for param in &metadata.params {
        println!("    - {}: {} ({})", param.name, param.description, param.type_);
    }

    // Build the full command with tool parameters and get matches
    let cmd = build_dynamic_args(&metadata);
    
    // Re-parse all arguments with the complete command
    let args: Vec<String> = std::env::args().collect();
    let matches = cmd.try_get_matches_from(args)?;

    println!("\nExecuting tool...");
    let params = collect_tool_params(&matches, &metadata);
    
    let result = tool.call_execute(&mut store, &params).unwrap();
    
    if result.success {
        println!("[Success] Tool output: {}", result.data);
    } else {
        println!("[Error] Tool error: {:?}", result.error);
    }

    Ok(())
}
