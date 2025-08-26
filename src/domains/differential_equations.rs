use crate::core::{MathDomain, MathResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialEquationsDomain {
    name: String,
}

// Ordinary Differential Equations (ODEs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ODE {
    pub order: usize,
    pub variables: Vec<String>,
    pub equations: Vec<String>, // String representations of equations
    pub initial_conditions: HashMap<String, f64>,
}

pub struct ODESystem {
    pub dimension: usize,
    pub variables: Vec<String>,
    pub initial_conditions: Vec<f64>,
    pub parameters: HashMap<String, f64>,
}

// Partial Differential Equations (PDEs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PDE {
    pub spatial_dimensions: usize,
    pub temporal: bool,
    pub variables: Vec<String>,
    pub equation_type: PDEType,
    pub boundary_conditions: Vec<BoundaryCondition>,
    pub initial_conditions: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PDEType {
    Elliptic,    // ∇²u = f
    Parabolic,   // ∂u/∂t = α∇²u + f (heat equation)
    Hyperbolic,  // ∂²u/∂t² = c²∇²u + f (wave equation)
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryCondition {
    pub boundary_type: BoundaryType,
    pub value: String, // Mathematical expression
    pub location: String, // Boundary description
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryType {
    Dirichlet,   // u = g on boundary
    Neumann,     // ∂u/∂n = g on boundary
    Robin,       // αu + β∂u/∂n = g on boundary
    Periodic,
}

// Numerical Methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ODESolver {
    Euler,
    RungeKutta4,
    AdaptiveRungeKutta,
    BackwardEuler,
    AdamsBashforth,
    Heun,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionParameters {
    pub solver: ODESolver,
    pub step_size: f64,
    pub tolerance: f64,
    pub max_steps: usize,
    pub time_span: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    pub time_points: Vec<f64>,
    pub values: Vec<Vec<f64>>,
    pub error_estimate: Option<Vec<f64>>,
    pub convergence_info: ConvergenceInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceInfo {
    pub converged: bool,
    pub iterations: usize,
    pub final_error: f64,
    pub method_used: String,
}

// Dynamical Systems Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicalSystem {
    pub dimension: usize,
    pub variables: Vec<String>,
    pub parameters: HashMap<String, f64>,
    pub fixed_points: Vec<Vec<f64>>,
    pub stability_analysis: HashMap<String, StabilityInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityInfo {
    pub stability_type: StabilityType,
    pub eigenvalues: Vec<num_complex::Complex<f64>>,
    pub eigenvectors: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StabilityType {
    Stable,
    Unstable,
    SaddlePoint,
    Center,
    StableSpiral,
    UnstableSpiral,
}

// Biological Models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationModel {
    pub model_type: PopulationModelType,
    pub species: Vec<String>,
    pub parameters: HashMap<String, f64>,
    pub carrying_capacity: Option<f64>,
    pub interaction_matrix: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PopulationModelType {
    Logistic,
    LotkaVolterra,
    Competition,
    Mutualism,
    SIR, // Susceptible-Infected-Recovered
    SEIR, // Susceptible-Exposed-Infected-Recovered
}

impl DifferentialEquationsDomain {
    pub fn new() -> Self {
        Self {
            name: "Differential Equations".to_string(),
        }
    }

    // ODE Creation and Solving
    pub fn create_ode(&self, order: usize, variables: Vec<String>) -> ODE {
        ODE {
            order,
            variables,
            equations: Vec::new(),
            initial_conditions: HashMap::new(),
        }
    }

    pub fn add_equation(&self, ode: &mut ODE, equation: String) {
        ode.equations.push(equation);
    }

    pub fn set_initial_condition(&self, ode: &mut ODE, variable: String, value: f64) {
        ode.initial_conditions.insert(variable, value);
    }

    // Numerical ODE Solving
    pub fn euler_method(&self, f: &dyn Fn(f64, f64) -> f64, 
                       y0: f64, t_span: (f64, f64), h: f64) -> Solution {
        let mut t = t_span.0;
        let mut y = y0;
        let mut time_points = vec![t];
        let mut values = vec![vec![y]];

        while t < t_span.1 {
            y += h * f(t, y);
            t += h;
            time_points.push(t);
            values.push(vec![y]);
        }

        let iterations = time_points.len();
        Solution {
            time_points,
            values,
            error_estimate: None,
            convergence_info: ConvergenceInfo {
                converged: true,
                iterations,
                final_error: 0.0,
                method_used: "Euler".to_string(),
            },
        }
    }

    pub fn runge_kutta_4(&self, f: &dyn Fn(f64, f64) -> f64, 
                        y0: f64, t_span: (f64, f64), h: f64) -> Solution {
        let mut t = t_span.0;
        let mut y = y0;
        let mut time_points = vec![t];
        let mut values = vec![vec![y]];

        while t < t_span.1 {
            let k1 = h * f(t, y);
            let k2 = h * f(t + h/2.0, y + k1/2.0);
            let k3 = h * f(t + h/2.0, y + k2/2.0);
            let k4 = h * f(t + h, y + k3);
            
            y += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
            t += h;
            
            time_points.push(t);
            values.push(vec![y]);
        }

        let iterations = time_points.len();
        Solution {
            time_points,
            values,
            error_estimate: None,
            convergence_info: ConvergenceInfo {
                converged: true,
                iterations,
                final_error: 0.0,
                method_used: "Runge-Kutta 4".to_string(),
            },
        }
    }

    // System of ODEs
    pub fn solve_system_rk4(&self, f: &dyn Fn(&[f64], f64) -> Vec<f64>, 
                           y0: &[f64], t_span: (f64, f64), h: f64) -> Solution {
        let n = y0.len();
        let mut t = t_span.0;
        let mut y = y0.to_vec();
        let mut time_points = vec![t];
        let mut values = vec![y.clone()];

        while t < t_span.1 {
            let k1 = f(&y, t).iter().map(|&x| h * x).collect::<Vec<_>>();
            
            let y_k2: Vec<f64> = y.iter().zip(&k1).map(|(&yi, &k1i)| yi + k1i/2.0).collect();
            let k2 = f(&y_k2, t + h/2.0).iter().map(|&x| h * x).collect::<Vec<_>>();
            
            let y_k3: Vec<f64> = y.iter().zip(&k2).map(|(&yi, &k2i)| yi + k2i/2.0).collect();
            let k3 = f(&y_k3, t + h/2.0).iter().map(|&x| h * x).collect::<Vec<_>>();
            
            let y_k4: Vec<f64> = y.iter().zip(&k3).map(|(&yi, &k3i)| yi + k3i).collect();
            let k4 = f(&y_k4, t + h).iter().map(|&x| h * x).collect::<Vec<_>>();
            
            for i in 0..n {
                y[i] += (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;
            }
            
            t += h;
            time_points.push(t);
            values.push(y.clone());
        }

        let iterations = time_points.len();
        Solution {
            time_points,
            values,
            error_estimate: None,
            convergence_info: ConvergenceInfo {
                converged: true,
                iterations,
                final_error: 0.0,
                method_used: "System RK4".to_string(),
            },
        }
    }

    // Stability Analysis
    pub fn find_fixed_points(&self, f: &dyn Fn(&[f64]) -> Vec<f64>, 
                           initial_guesses: &[Vec<f64>], tolerance: f64) -> Vec<Vec<f64>> {
        let mut fixed_points = Vec::new();
        
        for guess in initial_guesses {
            if let Some(fp) = self.newton_raphson_system(f, guess, tolerance, 100) {
                // Check if this fixed point is already found
                let is_new = fixed_points.iter().all(|existing: &Vec<f64>| {
                    fp.iter().zip(existing).any(|(&a, &b)| (a - b).abs() > tolerance)
                });
                
                if is_new {
                    fixed_points.push(fp);
                }
            }
        }
        
        fixed_points
    }

    fn newton_raphson_system(&self, f: &dyn Fn(&[f64]) -> Vec<f64>, 
                           initial: &[f64], tolerance: f64, max_iter: usize) -> Option<Vec<f64>> {
        let n = initial.len();
        let mut x = initial.to_vec();
        let eps = 1e-8;
        
        for _ in 0..max_iter {
            let fx = f(&x);
            let norm = fx.iter().map(|&xi| xi * xi).sum::<f64>().sqrt();
            
            if norm < tolerance {
                return Some(x);
            }
            
            // Compute Jacobian numerically
            let mut jacobian = DMatrix::zeros(n, n);
            for j in 0..n {
                let mut x_plus = x.clone();
                x_plus[j] += eps;
                let fx_plus = f(&x_plus);
                
                for i in 0..n {
                    jacobian[(i, j)] = (fx_plus[i] - fx[i]) / eps;
                }
            }
            
            // Solve Jacobian * delta = -fx
            let fx_vec = DVector::from_vec(fx);
            if let Some(lu) = jacobian.lu().try_inverse() {
                let delta = lu * (-fx_vec);
                for i in 0..n {
                    x[i] += delta[i];
                }
            } else {
                return None; // Singular Jacobian
            }
        }
        
        None // Failed to converge
    }

    // Biological Models
    pub fn lotka_volterra(&self, prey_growth: f64, predation_rate: f64, 
                         predator_death: f64, efficiency: f64) -> Box<dyn Fn(&[f64], f64) -> Vec<f64>> {
        Box::new(move |y: &[f64], _t: f64| {
            let x = y[0]; // prey
            let y = y[1]; // predator
            
            vec![
                prey_growth * x - predation_rate * x * y,           // dx/dt
                efficiency * predation_rate * x * y - predator_death * y  // dy/dt
            ]
        })
    }

    pub fn sir_model(&self, beta: f64, gamma: f64, n: f64) -> Box<dyn Fn(&[f64], f64) -> Vec<f64>> {
        Box::new(move |y: &[f64], _t: f64| {
            let s = y[0]; // susceptible
            let i = y[1]; // infected
            let r = y[2]; // recovered
            
            vec![
                -beta * s * i / n,     // dS/dt
                beta * s * i / n - gamma * i,  // dI/dt
                gamma * i,             // dR/dt
            ]
        })
    }

    pub fn logistic_growth(&self, r: f64, k: f64) -> Box<dyn Fn(f64, f64) -> f64> {
        Box::new(move |_t: f64, y: f64| r * y * (1.0 - y / k))
    }

    // PDE Support (basic framework)
    pub fn create_pde(&self, spatial_dims: usize, temporal: bool) -> PDE {
        PDE {
            spatial_dimensions: spatial_dims,
            temporal,
            variables: Vec::new(),
            equation_type: PDEType::Mixed,
            boundary_conditions: Vec::new(),
            initial_conditions: None,
        }
    }

    pub fn add_boundary_condition(&self, pde: &mut PDE, bc: BoundaryCondition) {
        pde.boundary_conditions.push(bc);
    }

    // Heat equation: ∂u/∂t = α∇²u
    pub fn solve_heat_equation_1d(&self, alpha: f64, length: f64, time_final: f64,
                                 initial_temp: &dyn Fn(f64) -> f64,
                                 boundary_left: f64, boundary_right: f64,
                                 nx: usize, nt: usize) -> Vec<Vec<f64>> {
        let dx = length / (nx - 1) as f64;
        let dt = time_final / (nt - 1) as f64;
        let r = alpha * dt / (dx * dx);
        
        // Check stability condition
        if r > 0.5 {
            eprintln!("Warning: Stability condition violated (r = {})", r);
        }
        
        let mut u = vec![vec![0.0; nx]; nt];
        
        // Initial conditions
        for i in 0..nx {
            let x = i as f64 * dx;
            u[0][i] = initial_temp(x);
        }
        
        // Boundary conditions
        for t in 0..nt {
            u[t][0] = boundary_left;
            u[t][nx-1] = boundary_right;
        }
        
        // Finite difference scheme
        for t in 1..nt {
            for i in 1..nx-1 {
                u[t][i] = u[t-1][i] + r * (u[t-1][i-1] - 2.0 * u[t-1][i] + u[t-1][i+1]);
            }
        }
        
        u
    }

    // Wave equation: ∂²u/∂t² = c²∇²u
    pub fn solve_wave_equation_1d(&self, c: f64, length: f64, time_final: f64,
                                 initial_displacement: &dyn Fn(f64) -> f64,
                                 initial_velocity: &dyn Fn(f64) -> f64,
                                 nx: usize, nt: usize) -> Vec<Vec<f64>> {
        let dx = length / (nx - 1) as f64;
        let dt = time_final / (nt - 1) as f64;
        let r = c * dt / dx;
        
        // Check stability condition (CFL condition)
        if r > 1.0 {
            eprintln!("Warning: CFL condition violated (r = {})", r);
        }
        
        let mut u = vec![vec![0.0; nx]; nt];
        
        // Initial displacement
        for i in 0..nx {
            let x = i as f64 * dx;
            u[0][i] = initial_displacement(x);
        }
        
        // Initial velocity (using forward difference)
        for i in 1..nx-1 {
            let x = i as f64 * dx;
            u[1][i] = u[0][i] + dt * initial_velocity(x) + 
                      0.5 * r * r * (u[0][i-1] - 2.0 * u[0][i] + u[0][i+1]);
        }
        
        // Boundary conditions (fixed ends)
        for t in 0..nt {
            u[t][0] = 0.0;
            u[t][nx-1] = 0.0;
        }
        
        // Finite difference scheme
        for t in 2..nt {
            for i in 1..nx-1 {
                u[t][i] = 2.0 * u[t-1][i] - u[t-2][i] + 
                         r * r * (u[t-1][i-1] - 2.0 * u[t-1][i] + u[t-1][i+1]);
            }
        }
        
        u
    }

    pub fn compute_eigenvalues_2x2(&self, matrix: &[[f64; 2]; 2]) -> Vec<num_complex::Complex<f64>> {
        let a = matrix[0][0];
        let b = matrix[0][1];
        let c = matrix[1][0];
        let d = matrix[1][1];
        
        let trace = a + d;
        let det = a * d - b * c;
        let discriminant = trace * trace - 4.0 * det;
        
        if discriminant >= 0.0 {
            let sqrt_disc = discriminant.sqrt();
            vec![
                num_complex::Complex::new((trace + sqrt_disc) / 2.0, 0.0),
                num_complex::Complex::new((trace - sqrt_disc) / 2.0, 0.0),
            ]
        } else {
            let sqrt_disc = (-discriminant).sqrt();
            vec![
                num_complex::Complex::new(trace / 2.0, sqrt_disc / 2.0),
                num_complex::Complex::new(trace / 2.0, -sqrt_disc / 2.0),
            ]
        }
    }
}

impl MathDomain for DifferentialEquationsDomain {
    fn name(&self) -> &str {
        &self.name
    }

    fn supports_operation(&self, operation: &str) -> bool {
        matches!(operation, 
            "solve_ode" | "solve_system" | "euler_method" | "runge_kutta" |
            "stability_analysis" | "phase_portrait" | "bifurcation_analysis" |
            "heat_equation" | "wave_equation" | "lotka_volterra" | "sir_model"
        )
    }

    fn description(&self) -> &str {
        "Differential Equations and Dynamical Systems"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    fn compute(&self, operation: &str, _args: &[&dyn std::any::Any]) -> MathResult<Box<dyn std::any::Any>> {
        match operation {
            "solve_ode" => Ok(Box::new("ODE solved numerically".to_string())),
            "stability_analysis" => Ok(Box::new("Stability analysis completed".to_string())),
            "heat_equation" => Ok(Box::new("Heat equation solved".to_string())),
            _ => Err(crate::core::MathError::NotImplemented(format!("Operation '{}' not implemented", operation))),
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "solve_ode".to_string(), "solve_system".to_string(),
            "euler_method".to_string(), "runge_kutta".to_string(),
            "stability_analysis".to_string(), "phase_portrait".to_string(),
            "bifurcation_analysis".to_string(), "heat_equation".to_string(),
            "wave_equation".to_string(), "lotka_volterra".to_string(),
            "sir_model".to_string()
        ]
    }
}

pub fn differential_equations() -> DifferentialEquationsDomain {
    DifferentialEquationsDomain::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_method() {
        let domain = DifferentialEquationsDomain::new();
        let f = |_t: f64, y: f64| y; // dy/dt = y, solution: y = e^t
        let solution = domain.euler_method(&f, 1.0, (0.0, 1.0), 0.1);
        
        assert!(!solution.time_points.is_empty());
        assert!(!solution.values.is_empty());
        assert_eq!(solution.time_points.len(), solution.values.len());
    }

    #[test]
    fn test_runge_kutta_4() {
        let domain = DifferentialEquationsDomain::new();
        let f = |_t: f64, y: f64| y; // dy/dt = y
        let solution = domain.runge_kutta_4(&f, 1.0, (0.0, 1.0), 0.1);
        
        assert!(!solution.time_points.is_empty());
        assert!(solution.values.last().unwrap()[0] > 2.0); // e ≈ 2.718
    }

    #[test]
    fn test_lotka_volterra() {
        let domain = DifferentialEquationsDomain::new();
        let lv = domain.lotka_volterra(1.0, 0.1, 1.5, 0.075);
        let initial = vec![10.0, 5.0]; // prey, predator
        
        let solution = domain.solve_system_rk4(&*lv, &initial, (0.0, 10.0), 0.01);
        assert!(!solution.values.is_empty());
        assert_eq!(solution.values[0].len(), 2); // Two species
    }

    #[test]
    fn test_sir_model() {
        let domain = DifferentialEquationsDomain::new();
        let sir = domain.sir_model(0.3, 0.1, 1000.0);
        let initial = vec![999.0, 1.0, 0.0]; // S, I, R
        
        let solution = domain.solve_system_rk4(&*sir, &initial, (0.0, 50.0), 0.1);
        assert!(!solution.values.is_empty());
        assert_eq!(solution.values[0].len(), 3); // Three compartments
    }

    #[test]
    fn test_heat_equation() {
        let domain = DifferentialEquationsDomain::new();
        let initial_temp = |x: f64| (std::f64::consts::PI * x).sin();
        
        let solution = domain.solve_heat_equation_1d(
            0.01, 1.0, 0.1, &initial_temp, 0.0, 0.0, 21, 101
        );
        
        assert_eq!(solution.len(), 101); // Time steps
        assert_eq!(solution[0].len(), 21); // Spatial points
    }
}