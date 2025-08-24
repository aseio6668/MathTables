use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AbstractObject {
    pub name: String,
    pub properties: HashMap<String, String>,
    pub category: String,
}

#[derive(Debug, Clone)]
pub struct QuinePutnamArgument {
    pub premise_1: String,
    pub premise_2: String,
    pub conclusion: String,
    pub examples: Vec<String>,
}

pub struct PhilosophyDomain;

impl PhilosophyDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn create_abstract_object(name: &str, category: &str) -> AbstractObject {
        AbstractObject {
            name: name.to_string(),
            properties: HashMap::new(),
            category: category.to_string(),
        }
    }
    
    pub fn add_property_to_abstract_object(obj: &mut AbstractObject, key: &str, value: &str) {
        obj.properties.insert(key.to_string(), value.to_string());
    }
    
    pub fn get_quine_putnam_argument() -> QuinePutnamArgument {
        QuinePutnamArgument {
            premise_1: "We should have ontological commitment to entities indispensable to scientific theories".to_string(),
            premise_2: "Mathematical entities are indispensable to scientific theories".to_string(),
            conclusion: "Therefore, we should commit to the existence of mathematical entities".to_string(),
            examples: vec![
                "Periodic cicadas' prime number life cycles".to_string(),
                "Hexagonal structure of bee honeycombs".to_string(),
                "Königsberg bridge problem impossibility".to_string(),
            ],
        }
    }
    
    pub fn evaluate_indispensability(theory: &str, math_component: &str) -> MathResult<bool> {
        let known_indispensable = vec![
            ("quantum_mechanics", "complex_numbers"),
            ("general_relativity", "differential_geometry"),
            ("biology", "prime_numbers"),
            ("crystallography", "group_theory"),
        ];
        
        for (t, m) in known_indispensable {
            if theory.contains(t) && math_component.contains(m) {
                return Ok(true);
            }
        }
        
        Ok(false)
    }
}

impl MathDomain for PhilosophyDomain {
    fn name(&self) -> &str { "Philosophy of Mathematics" }
    fn description(&self) -> &str { "Mathematical philosophy including abstract objects and indispensability arguments" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "create_abstract_object" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("create_abstract_object requires 2 arguments".to_string()));
                }
                let name = args[0].downcast_ref::<String>().ok_or_else(|| MathError::InvalidArgument("First argument must be String".to_string()))?;
                let category = args[1].downcast_ref::<String>().ok_or_else(|| MathError::InvalidArgument("Second argument must be String".to_string()))?;
                Ok(Box::new(Self::create_abstract_object(name, category)))
            },
            "get_quine_putnam_argument" => {
                if !args.is_empty() {
                    return Err(MathError::InvalidArgument("get_quine_putnam_argument requires no arguments".to_string()));
                }
                Ok(Box::new(Self::get_quine_putnam_argument()))
            },
            "evaluate_indispensability" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("evaluate_indispensability requires 2 arguments".to_string()));
                }
                let theory = args[0].downcast_ref::<String>().ok_or_else(|| MathError::InvalidArgument("First argument must be String".to_string()))?;
                let math_component = args[1].downcast_ref::<String>().ok_or_else(|| MathError::InvalidArgument("Second argument must be String".to_string()))?;
                Ok(Box::new(Self::evaluate_indispensability(theory, math_component)?))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "create_abstract_object".to_string(),
            "get_quine_putnam_argument".to_string(),
            "evaluate_indispensability".to_string(),
        ]
    }
}