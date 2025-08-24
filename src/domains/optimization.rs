use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub solution: Vec<f64>,
    pub objective_value: f64,
    pub iterations: usize,
    pub converged: bool,
}

#[derive(Debug, Clone)]
pub struct LinearProgramResult {
    pub solution: Vec<f64>,
    pub optimal_value: f64,
    pub status: OptimizationStatus,
}

#[derive(Debug, Clone)]
pub enum OptimizationStatus {
    Optimal,
    Infeasible,
    Unbounded,
    MaxIterationsReached,
}

pub type ObjectiveFunction = fn(&[f64]) -> f64;
pub type GradientFunction = fn(&[f64]) -> Vec<f64>;

pub struct OptimizationDomain;

impl OptimizationDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn golden_section_search(
        f: ObjectiveFunction,
        a: f64,
        b: f64,
        tolerance: f64,
    ) -> MathResult<f64> {
        if a >= b {
            return Err(MathError::InvalidArgument("Invalid interval: a must be less than b".to_string()));
        }
        
        let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        let resphi = 2.0 - phi;
        
        let mut x1 = a + resphi * (b - a);
        let mut x2 = b - resphi * (b - a);
        let mut f1 = f(&[x1]);
        let mut f2 = f(&[x2]);
        
        let mut a_curr = a;
        let mut b_curr = b;
        
        while (b_curr - a_curr).abs() > tolerance {
            if f1 > f2 {
                a_curr = x1;
                x1 = x2;
                f1 = f2;
                x2 = b_curr - resphi * (b_curr - a_curr);
                f2 = f(&[x2]);
            } else {
                b_curr = x2;
                x2 = x1;
                f2 = f1;
                x1 = a_curr + resphi * (b_curr - a_curr);
                f1 = f(&[x1]);
            }
        }
        
        Ok((a_curr + b_curr) / 2.0)
    }
    
    pub fn gradient_descent(
        f: ObjectiveFunction,
        grad_f: GradientFunction,
        initial_point: &[f64],
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> MathResult<OptimizationResult> {
        let mut x = initial_point.to_vec();
        let mut iterations = 0;
        
        for iter in 0..max_iterations {
            let gradient = grad_f(&x);
            let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            
            if grad_norm < tolerance {
                return Ok(OptimizationResult {
                    solution: x.clone(),
                    objective_value: f(&x),
                    iterations: iter,
                    converged: true,
                });
            }
            
            for i in 0..x.len() {
                x[i] -= learning_rate * gradient[i];
            }
            
            iterations = iter + 1;
        }
        
        Ok(OptimizationResult {
            solution: x.clone(),
            objective_value: f(&x),
            iterations,
            converged: false,
        })
    }
    
    pub fn newton_raphson_optimization(
        f: ObjectiveFunction,
        grad_f: GradientFunction,
        initial_point: &[f64],
        max_iterations: usize,
        tolerance: f64,
    ) -> MathResult<OptimizationResult> {
        let mut x = initial_point.to_vec();
        let n = x.len();
        
        for iter in 0..max_iterations {
            let gradient = grad_f(&x);
            let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            
            if grad_norm < tolerance {
                return Ok(OptimizationResult {
                    solution: x.clone(),
                    objective_value: f(&x),
                    iterations: iter,
                    converged: true,
                });
            }
            
            let hessian = Self::finite_difference_hessian(f, &x, 1e-8);
            let delta = Self::solve_linear_system(&hessian, &gradient)?;
            
            for i in 0..n {
                x[i] -= delta[i];
            }
        }
        
        Ok(OptimizationResult {
            solution: x.clone(),
            objective_value: f(&x),
            iterations: max_iterations,
            converged: false,
        })
    }
    
    fn finite_difference_hessian(f: ObjectiveFunction, x: &[f64], h: f64) -> Vec<Vec<f64>> {
        let n = x.len();
        let mut hessian = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    let mut x_plus = x.to_vec();
                    let mut x_minus = x.to_vec();
                    x_plus[i] += h;
                    x_minus[i] -= h;
                    
                    hessian[i][j] = (f(&x_plus) - 2.0 * f(x) + f(&x_minus)) / (h * h);
                } else {
                    let mut x_pp = x.to_vec();
                    let mut x_pm = x.to_vec();
                    let mut x_mp = x.to_vec();
                    let mut x_mm = x.to_vec();
                    
                    x_pp[i] += h; x_pp[j] += h;
                    x_pm[i] += h; x_pm[j] -= h;
                    x_mp[i] -= h; x_mp[j] += h;
                    x_mm[i] -= h; x_mm[j] -= h;
                    
                    hessian[i][j] = (f(&x_pp) - f(&x_pm) - f(&x_mp) + f(&x_mm)) / (4.0 * h * h);
                }
            }
        }
        
        hessian
    }
    
    fn solve_linear_system(matrix: &[Vec<f64>], rhs: &[f64]) -> MathResult<Vec<f64>> {
        let n = matrix.len();
        if n == 0 || matrix[0].len() != n || rhs.len() != n {
            return Err(MathError::InvalidArgument("Invalid matrix dimensions".to_string()));
        }
        
        let mut a = matrix.iter().map(|row| row.clone()).collect::<Vec<_>>();
        let mut b = rhs.to_vec();
        
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
                return Err(MathError::ComputationError("Matrix is singular".to_string()));
            }
            
            for k in (i + 1)..n {
                let factor = a[k][i] / a[i][i];
                for j in i..n {
                    a[k][j] -= factor * a[i][j];
                }
                b[k] -= factor * b[i];
            }
        }
        
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
    
    pub fn simulated_annealing(
        f: ObjectiveFunction,
        initial_solution: &[f64],
        bounds: &[(f64, f64)],
        initial_temp: f64,
        cooling_rate: f64,
        max_iterations: usize,
    ) -> MathResult<OptimizationResult> {
        if initial_solution.len() != bounds.len() {
            return Err(MathError::InvalidArgument("Solution and bounds dimensions mismatch".to_string()));
        }
        
        let mut current_solution = initial_solution.to_vec();
        let mut current_cost = f(&current_solution);
        let mut best_solution = current_solution.clone();
        let mut best_cost = current_cost;
        let mut temperature = initial_temp;
        
        for _iteration in 0..max_iterations {
            let mut new_solution = current_solution.clone();
            
            for i in 0..new_solution.len() {
                let (low, high) = bounds[i];
                let delta = (high - low) * 0.1 * (rand::random::<f64>() - 0.5);
                new_solution[i] = (new_solution[i] + delta).clamp(low, high);
            }
            
            let new_cost = f(&new_solution);
            let cost_diff = new_cost - current_cost;
            
            if cost_diff < 0.0 || rand::random::<f64>() < (-cost_diff / temperature).exp() {
                current_solution = new_solution;
                current_cost = new_cost;
                
                if new_cost < best_cost {
                    best_solution = current_solution.clone();
                    best_cost = new_cost;
                }
            }
            
            temperature *= cooling_rate;
        }
        
        Ok(OptimizationResult {
            solution: best_solution,
            objective_value: best_cost,
            iterations: max_iterations,
            converged: temperature < 1e-6,
        })
    }
    
    pub fn particle_swarm_optimization(
        f: ObjectiveFunction,
        bounds: &[(f64, f64)],
        num_particles: usize,
        max_iterations: usize,
        w: f64, // inertia weight
        c1: f64, // cognitive parameter
        c2: f64, // social parameter
    ) -> MathResult<OptimizationResult> {
        let dimensions = bounds.len();
        let mut particles: Vec<Vec<f64>> = Vec::new();
        let mut velocities: Vec<Vec<f64>> = Vec::new();
        let mut personal_best: Vec<Vec<f64>> = Vec::new();
        let mut personal_best_costs: Vec<f64> = Vec::new();
        
        for _ in 0..num_particles {
            let mut particle = Vec::new();
            let mut velocity = Vec::new();
            
            for &(low, high) in bounds {
                particle.push(low + rand::random::<f64>() * (high - low));
                velocity.push(0.0);
            }
            
            let cost = f(&particle);
            particles.push(particle.clone());
            velocities.push(velocity);
            personal_best.push(particle);
            personal_best_costs.push(cost);
        }
        
        let mut global_best_idx = 0;
        for (i, &cost) in personal_best_costs.iter().enumerate() {
            if cost < personal_best_costs[global_best_idx] {
                global_best_idx = i;
            }
        }
        
        for _iteration in 0..max_iterations {
            for p in 0..num_particles {
                let current_cost = f(&particles[p]);
                
                if current_cost < personal_best_costs[p] {
                    personal_best[p] = particles[p].clone();
                    personal_best_costs[p] = current_cost;
                    
                    if current_cost < personal_best_costs[global_best_idx] {
                        global_best_idx = p;
                    }
                }
                
                for d in 0..dimensions {
                    let r1 = rand::random::<f64>();
                    let r2 = rand::random::<f64>();
                    
                    velocities[p][d] = w * velocities[p][d] 
                        + c1 * r1 * (personal_best[p][d] - particles[p][d])
                        + c2 * r2 * (personal_best[global_best_idx][d] - particles[p][d]);
                    
                    particles[p][d] += velocities[p][d];
                    
                    let (low, high) = bounds[d];
                    particles[p][d] = particles[p][d].clamp(low, high);
                }
            }
        }
        
        Ok(OptimizationResult {
            solution: personal_best[global_best_idx].clone(),
            objective_value: personal_best_costs[global_best_idx],
            iterations: max_iterations,
            converged: true,
        })
    }
}

impl MathDomain for OptimizationDomain {
    fn name(&self) -> &str { "Optimization" }
    fn description(&self) -> &str { "Mathematical optimization algorithms including gradient descent, simulated annealing, and PSO" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, _args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            _ => Err(MathError::InvalidOperation(format!("Operation {} not implemented in compute interface", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "golden_section_search".to_string(),
            "gradient_descent".to_string(),
            "newton_raphson_optimization".to_string(),
            "simulated_annealing".to_string(),
            "particle_swarm_optimization".to_string(),
        ]
    }
}