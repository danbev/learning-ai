use anyhow::{Context, Result};
use wasmtime::{
    component::{Component, Linker as ComponentLinker},
    Config, Engine, Store,
};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiView};

use std::collections::HashMap;

wasmtime::component::bindgen!({
    path: "../wit/tool.wit",
    async: false,
    world: "tool-world"
});

struct ToolContext {
    table: wasmtime::component::ResourceTable,
    wasi: WasiCtx,
}

impl WasiView for ToolContext {
    fn table(&mut self) -> &mut wasmtime::component::ResourceTable {
        &mut self.table
    }
    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.wasi
    }
}

pub struct ToolManager {
    engine: Engine,
    linker: ComponentLinker<ToolContext>,
    components: HashMap<String, Component>,
    metadata: Vec<ToolMetadata>,
}

impl ToolManager {
    pub fn new(component_files: Vec<std::path::PathBuf>) -> Result<Self> {
        println!("Creating tool manager");
        let mut config = Config::new();
        config.wasm_component_model(true);
        config.async_support(false);
        
        let engine = Engine::new(&config)?;
        let mut linker = ComponentLinker::new(&engine);
        wasmtime_wasi::add_to_linker_sync(&mut linker)?;

        let wasi = WasiCtxBuilder::new()
            .inherit_stdio()
            .inherit_args()
            .inherit_env()
            .build();

        let mut store = Store::new(
            &engine,
            ToolContext {
                wasi,
                table: wasmtime::component::ResourceTable::new(),
            },
        );

        let mut components = HashMap::new();
        let mut metadata = Vec::new();
        for component_path in component_files {
            let component = Component::from_file(&engine, &component_path)?;
            let tool = ToolWorld::instantiate(&mut store, &component, &linker)
                .context("Failed to instantiate Tool")?;
            let md = tool.call_get_metadata(&mut store).unwrap();
            metadata.push(md.clone());
            println!("Tool metadata:");
            println!("  Name: {}", md.name);
            println!("  Description: {}", md.description);
            println!("  Version: {}", md.version);
            println!("  Parameters:");
            for param in &md.params {
                println!("    - {}: {} ({})", param.name, param.description, param.type_);
            }
            components.insert(md.name.clone(), component);
        }

        Ok(Self {
            engine,
            linker,
            components,
            metadata,
        })
    }

    pub fn execute_tool(&self, tool_name: &str, params: Vec<(String, String)>) -> Result<String> {
        let component = self.components.get(tool_name).unwrap();

        let wasi = WasiCtxBuilder::new()
            .inherit_stdio()
            .inherit_args()
            .inherit_env()
            .build();

        let mut store = Store::new(
            &self.engine,
            ToolContext {
                wasi,
                table: wasmtime::component::ResourceTable::new(),
            },
        );

        let tool = ToolWorld::instantiate(&mut store, &component, &self.linker)?;
        
        // Execute the tool
        println!("\nExecuting tool: {}, params: {}", tool_name, params.len());
        for (name, value) in &params {
            println!("  - {}: {}", name, value);
        }
        let result = tool.call_execute(&mut store, &params)?;
        
        if result.success {
            Ok(result.data)
        } else {
            Err(anyhow::anyhow!(
                result.error.unwrap_or_else(|| "Unknown error".to_string())
            ))
        }
    }

    pub fn get_metadata(&self) -> Vec<ToolMetadata> {
        self.metadata.clone()
    }
}
