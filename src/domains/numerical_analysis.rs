use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;

#[derive(Debug, Clone)]
pub struct InterpolationResult {
    pub interpolated_values: Vec<f64>,
    pub method: InterpolationMethod,
}

#[derive(Debug, Clone)]
pub enum InterpolationMethod {
    Linear,
    Polynomial,
    Spline,
    Lagrange,
}

#[derive(Debug, Clone)]
pub struct IntegrationResult {
    pub value: f64,
    pub error_estimate: Option<f64>,
    pub method: IntegrationMethod,
}

#[derive(Debug, Clone)]
pub enum IntegrationMethod {
    Trapezoidal,
    Simpson,
    AdaptiveQuadrature,
    RombergIntegration,
}

#[derive(Debug, Clone)]
pub struct ODESolution {
    pub t_values: Vec<f64>,
    pub y_values: Vec<Vec<f64>>,
    pub method: ODEMethod,
}

#[derive(Debug, Clone)]
pub enum ODEMethod {
    Euler,
    RungeKutta4,
    AdamsBashforth,
}

pub type RealFunction = fn(f64) -> f64;
pub type ODEFunction = fn(f64, &[f64]) -> Vec<f64>;

pub struct NumericalAnalysisDomain;

impl NumericalAnalysisDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn linear_interpolation(x_data: &[f64], y_data: &[f64], x_query: &[f64]) -> MathResult<InterpolationResult> {
        if x_data.len() != y_data.len() || x_data.len() < 2 {
            return Err(MathError::InvalidArgument("Invalid interpolation data".to_string()));
        }
        
        let mut interpolated_values = Vec::new();
        
        for &x in x_query {
            if x < x_data[0] || x > x_data[x_data.len() - 1] {
                return Err(MathError::DomainError("Query point outside interpolation range".to_string()));
            }
            
            let mut i = 0;
            while i < x_data.len() - 1 && x_data[i + 1] < x {
                i += 1;
            }
            
            let x0 = x_data[i];
            let x1 = x_data[i + 1];
            let y0 = y_data[i];
            let y1 = y_data[i + 1];
            
            let y = y0 + (y1 - y0) * (x - x0) / (x1 - x0);
            interpolated_values.push(y);
        }
        
        Ok(InterpolationResult {
            interpolated_values,
            method: InterpolationMethod::Linear,
        })
    }
    
    pub fn lagrange_interpolation(x_data: &[f64], y_data: &[f64], x_query: &[f64]) -> MathResult<InterpolationResult> {
        if x_data.len() != y_data.len() || x_data.is_empty() {
            return Err(MathError::InvalidArgument("Invalid interpolation data".to_string()));
        }
        
        let mut interpolated_values = Vec::new();
        let n = x_data.len();
        
        for &x in x_query {
            let mut result = 0.0;
            
            for i in 0..n {
                let mut basis = 1.0;
                
                for j in 0..n {
                    if i != j {
                        basis *= (x - x_data[j]) / (x_data[i] - x_data[j]);
                    }
                }
                
                result += y_data[i] * basis;
            }
            
            interpolated_values.push(result);
        }
        
        Ok(InterpolationResult {
            interpolated_values,
            method: InterpolationMethod::Lagrange,
        })
    }
    
    pub fn trapezoidal_integration(f: RealFunction, a: f64, b: f64, n: usize) -> MathResult<IntegrationResult> {
        if n == 0 || a >= b {
            return Err(MathError::InvalidArgument("Invalid integration parameters".to_string()));
        }
        
        let h = (b - a) / n as f64;
        let mut sum = 0.5 * (f(a) + f(b));
        
        for i in 1..n {
            let x = a + i as f64 * h;
            sum += f(x);
        }
        
        let result = h * sum;
        
        Ok(IntegrationResult {
            value: result,
            error_estimate: None,
            method: IntegrationMethod::Trapezoidal,
        })
    }
    
    pub fn simpson_integration(f: RealFunction, a: f64, b: f64, n: usize) -> MathResult<IntegrationResult> {
        if n % 2 != 0 || n == 0 || a >= b {
            return Err(MathError::InvalidArgument("Simpson's rule requires even number of intervals".to_string()));
        }
        
        let h = (b - a) / n as f64;
        let mut sum = f(a) + f(b);
        
        for i in 1..n {
            let x = a + i as f64 * h;
            let coeff = if i % 2 == 0 { 2.0 } else { 4.0 };
            sum += coeff * f(x);
        }
        
        let result = h * sum / 3.0;
        
        Ok(IntegrationResult {
            value: result,
            error_estimate: None,
            method: IntegrationMethod::Simpson,
        })
    }
    
    pub fn adaptive_quadrature(f: RealFunction, a: f64, b: f64, tolerance: f64) -> MathResult<IntegrationResult> {
        let result = Self::adaptive_simpson(f, a, b, tolerance, f(a), f((a + b) / 2.0), f(b), 15)?;
        
        Ok(IntegrationResult {
            value: result,
            error_estimate: Some(tolerance),
            method: IntegrationMethod::AdaptiveQuadrature,
        })
    }
    
    fn adaptive_simpson(
        f: RealFunction,
        a: f64,
        b: f64,
        tolerance: f64,
        fa: f64,
        fc: f64,
        fb: f64,
        depth: usize,
    ) -> MathResult<f64> {
        if depth == 0 {
            return Err(MathError::ComputationError("Maximum recursion depth reached in adaptive quadrature".to_string()));
        }
        
        let c = (a + b) / 2.0;
        let h = b - a;
        let d = (a + c) / 2.0;
        let e = (c + b) / 2.0;
        let fd = f(d);
        let fe = f(e);
        
        let s1 = (h / 6.0) * (fa + 4.0 * fc + fb);
        let s2 = (h / 12.0) * (fa + 4.0 * fd + 2.0 * fc + 4.0 * fe + fb);
        
        if (s2 - s1).abs() <= 15.0 * tolerance {
            Ok(s2 + (s2 - s1) / 15.0)
        } else {
            let left = Self::adaptive_simpson(f, a, c, tolerance / 2.0, fa, fd, fc, depth - 1)?;
            let right = Self::adaptive_simpson(f, c, b, tolerance / 2.0, fc, fe, fb, depth - 1)?;
            Ok(left + right)
        }
    }
    
    pub fn euler_method(
        f: ODEFunction,
        t0: f64,
        y0: &[f64],
        t_end: f64,
        step_size: f64,
    ) -> MathResult<ODESolution> {
        if step_size <= 0.0 || t_end <= t0 {
            return Err(MathError::InvalidArgument("Invalid ODE parameters".to_string()));
        }
        
        let n_steps = ((t_end - t0) / step_size).ceil() as usize;
        let mut t_values = Vec::with_capacity(n_steps + 1);
        let mut y_values = Vec::with_capacity(n_steps + 1);
        
        t_values.push(t0);
        y_values.push(y0.to_vec());
        
        let mut t = t0;
        let mut y = y0.to_vec();
        
        for _ in 0..n_steps {
            let dy = f(t, &y);
            
            for (yi, dyi) in y.iter_mut().zip(dy.iter()) {
                *yi += step_size * dyi;
            }
            
            t += step_size;
            t_values.push(t);
            y_values.push(y.clone());
        }
        
        Ok(ODESolution {
            t_values,
            y_values,
            method: ODEMethod::Euler,
        })
    }
    
    pub fn runge_kutta_4(
        f: ODEFunction,
        t0: f64,
        y0: &[f64],
        t_end: f64,
        step_size: f64,
    ) -> MathResult<ODESolution> {
        if step_size <= 0.0 || t_end <= t0 {
            return Err(MathError::InvalidArgument("Invalid ODE parameters".to_string()));
        }
        
        let n_steps = ((t_end - t0) / step_size).ceil() as usize;
        let mut t_values = Vec::with_capacity(n_steps + 1);
        let mut y_values = Vec::with_capacity(n_steps + 1);
        
        t_values.push(t0);
        y_values.push(y0.to_vec());
        
        let mut t = t0;
        let mut y = y0.to_vec();
        let n = y.len();
        
        for _ in 0..n_steps {
            let k1 = f(t, &y);
            
            let mut y2 = vec![0.0; n];
            for i in 0..n {
                y2[i] = y[i] + step_size * k1[i] / 2.0;
            }
            let k2 = f(t + step_size / 2.0, &y2);
            
            let mut y3 = vec![0.0; n];
            for i in 0..n {
                y3[i] = y[i] + step_size * k2[i] / 2.0;
            }
            let k3 = f(t + step_size / 2.0, &y3);
            
            let mut y4 = vec![0.0; n];
            for i in 0..n {
                y4[i] = y[i] + step_size * k3[i];
            }
            let k4 = f(t + step_size, &y4);
            
            for i in 0..n {
                y[i] += step_size * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
            }
            
            t += step_size;
            t_values.push(t);
            y_values.push(y.clone());
        }
        
        Ok(ODESolution {
            t_values,
            y_values,
            method: ODEMethod::RungeKutta4,
        })
    }
    
    pub fn bisection_method(f: RealFunction, a: f64, b: f64, tolerance: f64, max_iterations: usize) -> MathResult<f64> {
        if f(a) * f(b) >= 0.0 {
            return Err(MathError::InvalidArgument("Function must have opposite signs at endpoints".to_string()));
        }
        
        let mut a_curr = a;
        let mut b_curr = b;
        
        for _ in 0..max_iterations {
            let c = (a_curr + b_curr) / 2.0;
            
            if (b_curr - a_curr).abs() < tolerance {
                return Ok(c);
            }
            
            if f(c) == 0.0 {
                return Ok(c);
            }
            
            if f(a_curr) * f(c) < 0.0 {
                b_curr = c;
            } else {
                a_curr = c;
            }
        }
        
        Ok((a_curr + b_curr) / 2.0)
    }
    
    pub fn newton_raphson_root(f: RealFunction, df: RealFunction, x0: f64, tolerance: f64, max_iterations: usize) -> MathResult<f64> {
        let mut x = x0;
        
        for _ in 0..max_iterations {
            let fx = f(x);
            let dfx = df(x);
            
            if dfx.abs() < 1e-12 {
                return Err(MathError::ComputationError("Derivative too small in Newton-Raphson method".to_string()));
            }
            
            let x_new = x - fx / dfx;
            
            if (x_new - x).abs() < tolerance {
                return Ok(x_new);
            }
            
            x = x_new;
        }
        
        Err(MathError::ComputationError("Newton-Raphson method did not converge".to_string()))
    }
    
    pub fn secant_method(f: RealFunction, x0: f64, x1: f64, tolerance: f64, max_iterations: usize) -> MathResult<f64> {
        let mut x_prev = x0;
        let mut x_curr = x1;
        
        for _ in 0..max_iterations {
            let f_prev = f(x_prev);
            let f_curr = f(x_curr);
            
            if (f_curr - f_prev).abs() < 1e-12 {
                return Err(MathError::ComputationError("Denominator too small in secant method".to_string()));
            }
            
            let x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev);
            
            if (x_new - x_curr).abs() < tolerance {
                return Ok(x_new);
            }
            
            x_prev = x_curr;
            x_curr = x_new;
        }
        
        Err(MathError::ComputationError("Secant method did not converge".to_string()))
    }
    
    pub fn finite_difference_derivative(f: RealFunction, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }
    
    pub fn finite_difference_second_derivative(f: RealFunction, x: f64, h: f64) -> f64 {
        (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
    }
}

impl MathDomain for NumericalAnalysisDomain {
    fn name(&self) -> &str { "Numerical Analysis" }
    fn description(&self) -> &str { "Numerical methods for integration, differentiation, interpolation, and ODE solving" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, _args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            _ => Err(MathError::InvalidOperation(format!("Operation {} not implemented in compute interface", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "linear_interpolation".to_string(),
            "lagrange_interpolation".to_string(),
            "trapezoidal_integration".to_string(),
            "simpson_integration".to_string(),
            "adaptive_quadrature".to_string(),
            "euler_method".to_string(),
            "runge_kutta_4".to_string(),
            "bisection_method".to_string(),
            "newton_raphson_root".to_string(),
            "secant_method".to_string(),
            "finite_difference_derivative".to_string(),
            "finite_difference_second_derivative".to_string(),
        ]
    }
}