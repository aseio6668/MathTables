use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;

pub struct CalculusDomain;

impl CalculusDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn numerical_derivative(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }
    
    pub fn numerical_integral_simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> MathResult<f64> {
        if n % 2 != 0 {
            return Err(MathError::InvalidArgument("Number of intervals must be even for Simpson's rule".to_string()));
        }
        
        let h = (b - a) / n as f64;
        let mut sum = f(a) + f(b);
        
        for i in 1..n {
            let x = a + i as f64 * h;
            if i % 2 == 0 {
                sum += 2.0 * f(x);
            } else {
                sum += 4.0 * f(x);
            }
        }
        
        Ok(h / 3.0 * sum)
    }
    
    pub fn numerical_integral_trapezoidal(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = (b - a) / n as f64;
        let mut sum = 0.5 * (f(a) + f(b));
        
        for i in 1..n {
            let x = a + i as f64 * h;
            sum += f(x);
        }
        
        h * sum
    }
    
    pub fn limit_numerical(f: impl Fn(f64) -> f64, x: f64, direction: LimitDirection) -> f64 {
        let epsilon = 1e-10;
        match direction {
            LimitDirection::Left => f(x - epsilon),
            LimitDirection::Right => f(x + epsilon),
            LimitDirection::Both => {
                let left = f(x - epsilon);
                let right = f(x + epsilon);
                if (left - right).abs() < 1e-8 {
                    (left + right) / 2.0
                } else {
                    f64::NAN
                }
            }
        }
    }
    
    pub fn taylor_series_sin(x: f64, terms: usize) -> f64 {
        let mut result = 0.0;
        let mut term = x;
        let mut sign = 1.0;
        
        for n in 0..terms {
            result += sign * term;
            term *= x * x / ((2 * n + 2) * (2 * n + 3)) as f64;
            sign *= -1.0;
        }
        
        result
    }
    
    pub fn taylor_series_cos(x: f64, terms: usize) -> f64 {
        let mut result = 1.0;
        let mut term = 1.0;
        let mut sign = -1.0;
        
        for n in 1..terms {
            term *= x * x / ((2 * n - 1) * (2 * n)) as f64;
            result += sign * term;
            sign *= -1.0;
        }
        
        result
    }
    
    pub fn taylor_series_exp(x: f64, terms: usize) -> f64 {
        let mut result = 1.0;
        let mut term = 1.0;
        
        for n in 1..terms {
            term *= x / n as f64;
            result += term;
        }
        
        result
    }
    
    pub fn newton_raphson(f: impl Fn(f64) -> f64, df: impl Fn(f64) -> f64, initial_guess: f64, tolerance: f64, max_iterations: usize) -> MathResult<f64> {
        let mut x = initial_guess;
        
        for _ in 0..max_iterations {
            let fx = f(x);
            let dfx = df(x);
            
            if dfx.abs() < tolerance {
                return Err(MathError::ComputationError("Derivative too close to zero".to_string()));
            }
            
            let new_x = x - fx / dfx;
            
            if (new_x - x).abs() < tolerance {
                return Ok(new_x);
            }
            
            x = new_x;
        }
        
        Err(MathError::ComputationError("Newton-Raphson method did not converge".to_string()))
    }
    
    pub fn partial_derivative_numerical(f: impl Fn(f64, f64) -> f64, x: f64, y: f64, variable: PartialVariable, h: f64) -> f64 {
        match variable {
            PartialVariable::X => (f(x + h, y) - f(x - h, y)) / (2.0 * h),
            PartialVariable::Y => (f(x, y + h) - f(x, y - h)) / (2.0 * h),
        }
    }
}

pub enum LimitDirection {
    Left,
    Right,
    Both,
}

pub enum PartialVariable {
    X,
    Y,
}

impl MathDomain for CalculusDomain {
    fn name(&self) -> &str { "Calculus" }
    fn description(&self) -> &str { "Mathematical domain for differential and integral calculus" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "taylor_series_sin" => {
                if args.len() != 2 { 
                    return Err(MathError::InvalidArgument("taylor_series_sin requires 2 arguments".to_string())); 
                }
                let x = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let terms = args[1].downcast_ref::<usize>().ok_or_else(|| MathError::InvalidArgument("Second argument must be usize".to_string()))?;
                Ok(Box::new(Self::taylor_series_sin(*x, *terms)))
            },
            "taylor_series_cos" => {
                if args.len() != 2 { 
                    return Err(MathError::InvalidArgument("taylor_series_cos requires 2 arguments".to_string())); 
                }
                let x = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let terms = args[1].downcast_ref::<usize>().ok_or_else(|| MathError::InvalidArgument("Second argument must be usize".to_string()))?;
                Ok(Box::new(Self::taylor_series_cos(*x, *terms)))
            },
            "taylor_series_exp" => {
                if args.len() != 2 { 
                    return Err(MathError::InvalidArgument("taylor_series_exp requires 2 arguments".to_string())); 
                }
                let x = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let terms = args[1].downcast_ref::<usize>().ok_or_else(|| MathError::InvalidArgument("Second argument must be usize".to_string()))?;
                Ok(Box::new(Self::taylor_series_exp(*x, *terms)))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "numerical_derivative".to_string(),
            "numerical_integral_simpson".to_string(),
            "numerical_integral_trapezoidal".to_string(),
            "limit_numerical".to_string(),
            "taylor_series_sin".to_string(),
            "taylor_series_cos".to_string(),
            "taylor_series_exp".to_string(),
            "newton_raphson".to_string(),
            "partial_derivative_numerical".to_string(),
        ]
    }
}