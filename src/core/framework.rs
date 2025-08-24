use crate::domains::*;
use crate::plugins::PluginRegistry;
use std::collections::HashMap;

#[derive(Default)]
pub struct MathTables {
    domains: HashMap<String, Box<dyn MathDomain>>,
    plugin_registry: PluginRegistry,
    initialized: bool,
}

impl MathTables {
    pub fn new() -> Self {
        let mut framework = Self {
            domains: HashMap::new(),
            plugin_registry: PluginRegistry::new(),
            initialized: false,
        };
        
        framework.register_core_domains();
        framework.initialized = true;
        framework
    }
    
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
    
    fn register_core_domains(&mut self) {
        self.register_domain("number_theory", Box::new(number_theory::NumberTheoryDomain::new()));
        self.register_domain("algebra", Box::new(algebra::AlgebraDomain::new()));
        self.register_domain("geometry", Box::new(geometry::GeometryDomain::new()));
        self.register_domain("calculus", Box::new(calculus::CalculusDomain::new()));
        self.register_domain("discrete", Box::new(discrete::DiscreteDomain::new()));
    }
    
    pub fn register_domain(&mut self, name: &str, domain: Box<dyn MathDomain>) {
        self.domains.insert(name.to_string(), domain);
    }
    
    pub fn get_domain(&self, name: &str) -> Option<&dyn MathDomain> {
        self.domains.get(name).map(|d| d.as_ref())
    }
    
    pub fn list_domains(&self) -> Vec<String> {
        self.domains.keys().cloned().collect()
    }
    
    pub fn plugin_registry(&mut self) -> &mut PluginRegistry {
        &mut self.plugin_registry
    }
}