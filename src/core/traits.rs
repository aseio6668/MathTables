use crate::core::types::MathResult;
use std::any::Any;

pub trait MathDomain: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn version(&self) -> &str;
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>>;
    
    fn list_operations(&self) -> Vec<String>;
    
    fn supports_operation(&self, operation: &str) -> bool {
        self.list_operations().contains(&operation.to_string())
    }
}

pub trait MathObject: Send + Sync + Any {
    fn as_any(&self) -> &dyn Any;
    fn type_name(&self) -> &'static str;
    fn display(&self) -> String;
}

pub trait Computable {
    type Output;
    
    fn compute(&self) -> Self::Output;
}

pub trait Transformable<T> {
    fn transform(&self, transformation: &T) -> Self;
}

pub trait Validatable {
    fn is_valid(&self) -> bool;
    fn validate(&self) -> Result<(), String>;
}