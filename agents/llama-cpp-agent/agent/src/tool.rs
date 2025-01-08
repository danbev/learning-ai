use anyhow::Result;
use wasmtime::{
    component::{Component, Linker as ComponentLinker},
    Config, Engine, Store,
};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiView};

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
}

impl ToolManager {
    pub fn new() -> Result<Self> {
        let mut config = Config::new();
        config.wasm_component_model(true);
        config.async_support(false);
        
        let engine = Engine::new(&config)?;
        let mut linker = ComponentLinker::new(&engine);
        wasmtime_wasi::add_to_linker_sync(&mut linker)?;

        Ok(Self {
            engine,
            linker,
        })
    }

    pub fn execute_tool(&self, tool_name: &str, params: Vec<(String, String)>) -> Result<String> {
        //TODO(danbev) Fix this hard coded module. This is just for initial testing.
        let component = Component::from_file(
            &self.engine,
            "../components/echo-tool-component.wasm",
        )?;

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
        println!("Execution result: {}", result.data);
        
        if result.success {
            Ok(result.data)
        } else {
            Err(anyhow::anyhow!(
                result.error.unwrap_or_else(|| "Unknown error".to_string())
            ))
        }
    }
}
