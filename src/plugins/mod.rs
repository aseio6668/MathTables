use std::collections::HashMap;
use std::any::Any;

pub type PluginFunction = Box<dyn Fn(&[&dyn Any]) -> Box<dyn Any> + Send + Sync>;

#[derive(Default)]
pub struct PluginRegistry {
    plugins: HashMap<String, Plugin>,
}

pub struct Plugin {
    pub name: String,
    pub version: String,
    pub description: String,
    pub functions: HashMap<String, PluginFunction>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }
    
    pub fn register_plugin(&mut self, plugin: Plugin) {
        self.plugins.insert(plugin.name.clone(), plugin);
    }
    
    pub fn get_plugin(&self, name: &str) -> Option<&Plugin> {
        self.plugins.get(name)
    }
    
    pub fn list_plugins(&self) -> Vec<&str> {
        self.plugins.keys().map(|s| s.as_str()).collect()
    }
    
    pub fn call_plugin_function(
        &self, 
        plugin_name: &str, 
        function_name: &str, 
        args: &[&dyn Any]
    ) -> Option<Box<dyn Any>> {
        self.plugins.get(plugin_name)
            .and_then(|plugin| plugin.functions.get(function_name))
            .map(|func| func(args))
    }
}

impl Plugin {
    pub fn new(name: String, version: String, description: String) -> Self {
        Self {
            name,
            version,
            description,
            functions: HashMap::new(),
        }
    }
    
    pub fn add_function(&mut self, name: String, function: PluginFunction) {
        self.functions.insert(name, function);
    }
}