use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;
use std::ops::{Add, Sub, Mul, Div, Neg};

#[derive(Debug, Clone, PartialEq)]
pub struct DualNumber {
    pub value: f64,      // Function value
    pub derivative: f64, // Derivative value
}

#[derive(Debug, Clone)]
pub struct ForwardMode {
    pub variables: Vec<DualNumber>,
    pub gradient: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ReverseMode {
    pub tape: Vec<Operation>,
    pub values: Vec<f64>,
    pub gradients: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Operation {
    pub op_type: OpType,
    pub inputs: Vec<usize>,
    pub output: usize,
    pub local_gradients: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum OpType {
    Input,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Sin,
    Cos,
    Exp,
    Ln,
    Pow,
    Sqrt,
}

#[derive(Debug, Clone)]
pub struct HessianMatrix {
    pub data: Vec<Vec<f64>>,
    pub size: usize,
}

#[derive(Debug, Clone)]
pub struct JacobianMatrix {
    pub data: Vec<Vec<f64>>,
    pub rows: usize,
    pub cols: usize,
}

pub struct AutoDiffDomain;

impl DualNumber {
    pub fn new(value: f64, derivative: f64) -> Self {
        DualNumber { value, derivative }
    }
    
    pub fn variable(value: f64) -> Self {
        DualNumber { value, derivative: 1.0 }
    }
    
    pub fn constant(value: f64) -> Self {
        DualNumber { value, derivative: 0.0 }
    }
    
    pub fn sin(&self) -> Self {
        DualNumber {
            value: self.value.sin(),
            derivative: self.derivative * self.value.cos(),
        }
    }
    
    pub fn cos(&self) -> Self {
        DualNumber {
            value: self.value.cos(),
            derivative: -self.derivative * self.value.sin(),
        }
    }
    
    pub fn exp(&self) -> Self {
        let exp_val = self.value.exp();
        DualNumber {
            value: exp_val,
            derivative: self.derivative * exp_val,
        }
    }
    
    pub fn ln(&self) -> MathResult<Self> {
        if self.value <= 0.0 {
            return Err(MathError::DomainError("Logarithm of non-positive number".to_string()));
        }
        Ok(DualNumber {
            value: self.value.ln(),
            derivative: self.derivative / self.value,
        })
    }
    
    pub fn sqrt(&self) -> MathResult<Self> {
        if self.value < 0.0 {
            return Err(MathError::DomainError("Square root of negative number".to_string()));
        }
        let sqrt_val = self.value.sqrt();
        Ok(DualNumber {
            value: sqrt_val,
            derivative: self.derivative / (2.0 * sqrt_val),
        })
    }
    
    pub fn powi(&self, n: i32) -> Self {
        let value = self.value.powi(n);
        let derivative = if n == 0 {
            0.0
        } else {
            self.derivative * n as f64 * self.value.powi(n - 1)
        };
        DualNumber { value, derivative }
    }
    
    pub fn powf(&self, other: &Self) -> MathResult<Self> {
        if self.value <= 0.0 && other.value != other.value.floor() {
            return Err(MathError::DomainError("Complex result in power operation".to_string()));
        }
        
        let value = self.value.powf(other.value);
        let derivative = if self.value > 0.0 {
            value * (other.derivative * self.value.ln() + other.value * self.derivative / self.value)
        } else if self.value == 0.0 && other.value > 0.0 {
            0.0
        } else {
            return Err(MathError::DomainError("Invalid power operation".to_string()));
        };
        
        Ok(DualNumber { value, derivative })
    }
}

impl Add for DualNumber {
    type Output = DualNumber;
    
    fn add(self, other: DualNumber) -> DualNumber {
        DualNumber {
            value: self.value + other.value,
            derivative: self.derivative + other.derivative,
        }
    }
}

impl Sub for DualNumber {
    type Output = DualNumber;
    
    fn sub(self, other: DualNumber) -> DualNumber {
        DualNumber {
            value: self.value - other.value,
            derivative: self.derivative - other.derivative,
        }
    }
}

impl Mul for DualNumber {
    type Output = DualNumber;
    
    fn mul(self, other: DualNumber) -> DualNumber {
        DualNumber {
            value: self.value * other.value,
            derivative: self.derivative * other.value + self.value * other.derivative,
        }
    }
}

impl Div for DualNumber {
    type Output = MathResult<DualNumber>;
    
    fn div(self, other: DualNumber) -> MathResult<DualNumber> {
        if other.value == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        
        Ok(DualNumber {
            value: self.value / other.value,
            derivative: (self.derivative * other.value - self.value * other.derivative) / (other.value * other.value),
        })
    }
}

impl Neg for DualNumber {
    type Output = DualNumber;
    
    fn neg(self) -> DualNumber {
        DualNumber {
            value: -self.value,
            derivative: -self.derivative,
        }
    }
}

impl ReverseMode {
    pub fn new() -> Self {
        ReverseMode {
            tape: Vec::new(),
            values: Vec::new(),
            gradients: Vec::new(),
        }
    }
    
    pub fn add_input(&mut self, value: f64) -> usize {
        let index = self.values.len();
        self.values.push(value);
        self.gradients.push(0.0);
        self.tape.push(Operation {
            op_type: OpType::Input,
            inputs: vec![],
            output: index,
            local_gradients: vec![],
        });
        index
    }
    
    pub fn add_operation(&mut self, op_type: OpType, inputs: Vec<usize>) -> MathResult<usize> {
        let output_index = self.values.len();
        
        let (value, local_gradients) = match op_type {
            OpType::Add => {
                if inputs.len() != 2 {
                    return Err(MathError::InvalidArgument("Add requires 2 inputs".to_string()));
                }
                let val = self.values[inputs[0]] + self.values[inputs[1]];
                (val, vec![1.0, 1.0])
            },
            OpType::Sub => {
                if inputs.len() != 2 {
                    return Err(MathError::InvalidArgument("Sub requires 2 inputs".to_string()));
                }
                let val = self.values[inputs[0]] - self.values[inputs[1]];
                (val, vec![1.0, -1.0])
            },
            OpType::Mul => {
                if inputs.len() != 2 {
                    return Err(MathError::InvalidArgument("Mul requires 2 inputs".to_string()));
                }
                let a = self.values[inputs[0]];
                let b = self.values[inputs[1]];
                (a * b, vec![b, a])
            },
            OpType::Div => {
                if inputs.len() != 2 {
                    return Err(MathError::InvalidArgument("Div requires 2 inputs".to_string()));
                }
                let a = self.values[inputs[0]];
                let b = self.values[inputs[1]];
                if b == 0.0 {
                    return Err(MathError::DivisionByZero);
                }
                (a / b, vec![1.0 / b, -a / (b * b)])
            },
            OpType::Sin => {
                if inputs.len() != 1 {
                    return Err(MathError::InvalidArgument("Sin requires 1 input".to_string()));
                }
                let x = self.values[inputs[0]];
                (x.sin(), vec![x.cos()])
            },
            OpType::Cos => {
                if inputs.len() != 1 {
                    return Err(MathError::InvalidArgument("Cos requires 1 input".to_string()));
                }
                let x = self.values[inputs[0]];
                (x.cos(), vec![-x.sin()])
            },
            OpType::Exp => {
                if inputs.len() != 1 {
                    return Err(MathError::InvalidArgument("Exp requires 1 input".to_string()));
                }
                let x = self.values[inputs[0]];
                let exp_x = x.exp();
                (exp_x, vec![exp_x])
            },
            OpType::Ln => {
                if inputs.len() != 1 {
                    return Err(MathError::InvalidArgument("Ln requires 1 input".to_string()));
                }
                let x = self.values[inputs[0]];
                if x <= 0.0 {
                    return Err(MathError::DomainError("Logarithm of non-positive number".to_string()));
                }
                (x.ln(), vec![1.0 / x])
            },
            OpType::Sqrt => {
                if inputs.len() != 1 {
                    return Err(MathError::InvalidArgument("Sqrt requires 1 input".to_string()));
                }
                let x = self.values[inputs[0]];
                if x < 0.0 {
                    return Err(MathError::DomainError("Square root of negative number".to_string()));
                }
                let sqrt_x = x.sqrt();
                (sqrt_x, vec![1.0 / (2.0 * sqrt_x)])
            },
            _ => return Err(MathError::InvalidOperation("Unsupported operation type".to_string())),
        };
        
        self.values.push(value);
        self.gradients.push(0.0);
        self.tape.push(Operation {
            op_type,
            inputs: inputs.clone(),
            output: output_index,
            local_gradients,
        });
        
        Ok(output_index)
    }
    
    pub fn backward(&mut self, output_gradient: f64) {
        // Reset gradients
        for grad in &mut self.gradients {
            *grad = 0.0;
        }
        
        // Set output gradient
        if let Some(last_op) = self.tape.last() {
            self.gradients[last_op.output] = output_gradient;
        }
        
        // Reverse pass
        for op in self.tape.iter().rev() {
            let output_grad = self.gradients[op.output];
            
            for (i, &input_idx) in op.inputs.iter().enumerate() {
                if i < op.local_gradients.len() {
                    self.gradients[input_idx] += output_grad * op.local_gradients[i];
                }
            }
        }
    }
    
    pub fn get_gradient(&self, variable_index: usize) -> f64 {
        if variable_index < self.gradients.len() {
            self.gradients[variable_index]
        } else {
            0.0
        }
    }
}

impl AutoDiffDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn forward_mode_gradient<F>(f: F, point: &[f64]) -> MathResult<Vec<f64>>
    where
        F: Fn(&[DualNumber]) -> MathResult<DualNumber>,
    {
        let n = point.len();
        let mut gradient = vec![0.0; n];
        
        for i in 0..n {
            let mut dual_point = Vec::new();
            for j in 0..n {
                if i == j {
                    dual_point.push(DualNumber::variable(point[j]));
                } else {
                    dual_point.push(DualNumber::constant(point[j]));
                }
            }
            
            let result = f(&dual_point)?;
            gradient[i] = result.derivative;
        }
        
        Ok(gradient)
    }
    
    pub fn forward_mode_jacobian<F>(f: F, point: &[f64], output_dim: usize) -> MathResult<JacobianMatrix>
    where
        F: Fn(&[DualNumber]) -> MathResult<Vec<DualNumber>>,
    {
        let input_dim = point.len();
        let mut jacobian = vec![vec![0.0; input_dim]; output_dim];
        
        for i in 0..input_dim {
            let mut dual_point = Vec::new();
            for j in 0..input_dim {
                if i == j {
                    dual_point.push(DualNumber::variable(point[j]));
                } else {
                    dual_point.push(DualNumber::constant(point[j]));
                }
            }
            
            let result = f(&dual_point)?;
            if result.len() != output_dim {
                return Err(MathError::InvalidArgument("Function output dimension mismatch".to_string()));
            }
            
            for j in 0..output_dim {
                jacobian[j][i] = result[j].derivative;
            }
        }
        
        Ok(JacobianMatrix {
            data: jacobian,
            rows: output_dim,
            cols: input_dim,
        })
    }
    
    pub fn finite_difference_hessian<F>(f: F, point: &[f64], h: f64) -> MathResult<HessianMatrix>
    where
        F: Fn(&[f64]) -> MathResult<f64>,
    {
        let n = point.len();
        let mut hessian = vec![vec![0.0; n]; n];
        
        // Compute diagonal elements
        for i in 0..n {
            let mut point_plus = point.to_vec();
            let mut point_minus = point.to_vec();
            point_plus[i] += h;
            point_minus[i] -= h;
            
            let f_plus = f(&point_plus)?;
            let f_center = f(point)?;
            let f_minus = f(&point_minus)?;
            
            hessian[i][i] = (f_plus - 2.0 * f_center + f_minus) / (h * h);
        }
        
        // Compute off-diagonal elements
        for i in 0..n {
            for j in (i + 1)..n {
                let mut point_pp = point.to_vec();
                let mut point_pm = point.to_vec();
                let mut point_mp = point.to_vec();
                let mut point_mm = point.to_vec();
                
                point_pp[i] += h; point_pp[j] += h;
                point_pm[i] += h; point_pm[j] -= h;
                point_mp[i] -= h; point_mp[j] += h;
                point_mm[i] -= h; point_mm[j] -= h;
                
                let f_pp = f(&point_pp)?;
                let f_pm = f(&point_pm)?;
                let f_mp = f(&point_mp)?;
                let f_mm = f(&point_mm)?;
                
                let mixed_partial = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
                hessian[i][j] = mixed_partial;
                hessian[j][i] = mixed_partial;
            }
        }
        
        Ok(HessianMatrix {
            data: hessian,
            size: n,
        })
    }
    
    pub fn newton_optimization<F, G>(
        f: F,
        grad_f: G,
        initial_point: &[f64],
        tolerance: f64,
        max_iterations: usize,
        line_search: bool,
    ) -> MathResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> MathResult<f64>,
        G: Fn(&[f64]) -> MathResult<Vec<f64>>,
    {
        let mut x = initial_point.to_vec();
        let n = x.len();
        
        for iteration in 0..max_iterations {
            let gradient = grad_f(&x)?;
            let grad_norm: f64 = gradient.iter().map(|&g| g * g).sum::<f64>().sqrt();
            
            if grad_norm < tolerance {
                return Ok(x);
            }
            
            // Compute Hessian using finite differences
            let hessian = Self::finite_difference_hessian(&f, &x, 1e-8)?;
            
            // Solve Hessian * direction = -gradient
            let direction = Self::solve_linear_system(&hessian.data, &gradient)?;
            
            // Line search or simple step
            let step_size = if line_search {
                Self::backtracking_line_search(&f, &x, &direction, &gradient, 1.0, 0.5, 1e-4)?
            } else {
                1.0
            };
            
            // Update position
            for i in 0..n {
                x[i] -= step_size * direction[i];
            }
            
            if iteration > 0 && step_size * grad_norm < tolerance {
                break;
            }
        }
        
        Ok(x)
    }
    
    fn solve_linear_system(matrix: &[Vec<f64>], rhs: &[f64]) -> MathResult<Vec<f64>> {
        let n = matrix.len();
        if n == 0 || matrix[0].len() != n || rhs.len() != n {
            return Err(MathError::InvalidArgument("Invalid system dimensions".to_string()));
        }
        
        let mut a = matrix.iter().map(|row| row.clone()).collect::<Vec<_>>();
        let mut b = rhs.to_vec();
        
        // Gaussian elimination with partial pivoting
        for i in 0..n {
            let mut max_row = i;
            for k in (i + 1)..n {
                if a[k][i].abs() > a[max_row][i].abs() {
                    max_row = k;
                }
            }
            a.swap(i, max_row);
            b.swap(i, max_row);
            
            if a[i][i].abs() < 1e-12 {
                return Err(MathError::ComputationError("Singular matrix".to_string()));
            }
            
            for k in (i + 1)..n {
                let factor = a[k][i] / a[i][i];
                for j in i..n {
                    a[k][j] -= factor * a[i][j];
                }
                b[k] -= factor * b[i];
            }
        }
        
        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = b[i];
            for j in (i + 1)..n {
                x[i] -= a[i][j] * x[j];
            }
            x[i] /= a[i][i];
        }
        
        Ok(x)
    }
    
    fn backtracking_line_search<F>(
        f: &F,
        x: &[f64],
        direction: &[f64],
        gradient: &[f64],
        initial_step: f64,
        rho: f64,
        c: f64,
    ) -> MathResult<f64>
    where
        F: Fn(&[f64]) -> MathResult<f64>,
    {
        let mut alpha = initial_step;
        let f_x = f(x)?;
        let grad_dot_dir: f64 = gradient.iter().zip(direction.iter()).map(|(&g, &d)| g * d).sum();
        
        for _ in 0..50 { // Maximum line search iterations
            let mut x_new = x.to_vec();
            for i in 0..x.len() {
                x_new[i] -= alpha * direction[i];
            }
            
            let f_new = f(&x_new)?;
            
            // Armijo condition
            if f_new <= f_x - c * alpha * grad_dot_dir.abs() {
                return Ok(alpha);
            }
            
            alpha *= rho;
        }
        
        Ok(alpha)
    }
    
    pub fn compute_directional_derivative<F>(
        f: F,
        point: &[f64],
        direction: &[f64],
    ) -> MathResult<f64>
    where
        F: Fn(&[DualNumber]) -> MathResult<DualNumber>,
    {
        if point.len() != direction.len() {
            return Err(MathError::InvalidArgument("Point and direction must have same dimension".to_string()));
        }
        
        let dual_point: Vec<DualNumber> = point.iter().zip(direction.iter())
            .map(|(&p, &d)| DualNumber::new(p, d))
            .collect();
        
        let result = f(&dual_point)?;
        Ok(result.derivative)
    }
    
    pub fn higher_order_derivatives<F>(
        f: F,
        point: f64,
        order: usize,
    ) -> MathResult<Vec<f64>>
    where
        F: Fn(f64) -> f64,
    {
        if order == 0 {
            return Ok(vec![f(point)]);
        }
        
        let h = 1e-5;
        let mut derivatives = vec![0.0; order + 1];
        derivatives[0] = f(point);
        
        // Use finite differences for higher-order derivatives
        for n in 1..=order {
            let mut points = Vec::new();
            let mut coeffs = Vec::new();
            
            // Generate finite difference coefficients for nth derivative
            for k in 0..=n {
                let x = point + (k as f64 - n as f64 / 2.0) * h;
                points.push(f(x));
                
                // Binomial coefficient with alternating signs
                let mut coeff = 1.0;
                for i in 0..n {
                    coeff *= (n - i) as f64 / (i + 1) as f64;
                }
                if (n - k) % 2 == 1 {
                    coeff = -coeff;
                }
                coeffs.push(coeff);
            }
            
            let mut derivative = 0.0;
            for (point_val, coeff) in points.iter().zip(coeffs.iter()) {
                derivative += coeff * point_val;
            }
            derivative /= h.powi(n as i32);
            
            derivatives[n] = derivative;
        }
        
        Ok(derivatives)
    }
}

impl MathDomain for AutoDiffDomain {
    fn name(&self) -> &str { "Automatic Differentiation" }
    fn description(&self) -> &str { "Forward and reverse mode automatic differentiation for gradients, Jacobians, and Hessians" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, _args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            _ => Err(MathError::InvalidOperation(format!("Operation {} not implemented in compute interface", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "forward_mode_gradient".to_string(),
            "forward_mode_jacobian".to_string(),
            "reverse_mode_gradient".to_string(),
            "finite_difference_hessian".to_string(),
            "newton_optimization".to_string(),
            "directional_derivative".to_string(),
            "higher_order_derivatives".to_string(),
            "dual_number_arithmetic".to_string(),
            "computational_graph".to_string(),
        ]
    }
}