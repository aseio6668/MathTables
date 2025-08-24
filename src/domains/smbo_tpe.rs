use crate::core::types::*;
use std::f64::consts::{E, PI};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct SMBOTPEDomain;

#[derive(Clone, Debug)]
pub struct TPEOptimizer {
    pub observations: Vec<Observation>,
    pub parameter_bounds: Vec<(f64, f64)>,
    pub gamma: f64,  // quantile for splitting good/bad observations
    pub n_startup_trials: usize,
    pub n_ei_candidates: usize,
}

#[derive(Clone, Debug)]
pub struct Observation {
    pub parameters: Vec<f64>,
    pub objective_value: f64,
}

#[derive(Clone, Debug)]
pub struct ParzenEstimator {
    pub data_points: Vec<f64>,
    pub bandwidth: f64,
    pub bounds: (f64, f64),
}

#[derive(Clone, Debug)]
pub struct TPESuggestion {
    pub parameters: Vec<f64>,
    pub expected_improvement: f64,
    pub acquisition_value: f64,
}

impl SMBOTPEDomain {
    pub fn new() -> Self {
        Self
    }
}

impl TPEOptimizer {
    pub fn new(parameter_bounds: Vec<(f64, f64)>, gamma: f64) -> Self {
        Self {
            observations: Vec::new(),
            parameter_bounds,
            gamma,
            n_startup_trials: 10,
            n_ei_candidates: 100,
        }
    }

    pub fn add_observation(&mut self, parameters: Vec<f64>, objective_value: f64) -> MathResult<()> {
        if parameters.len() != self.parameter_bounds.len() {
            return Err(MathError::InvalidArgument("Parameter dimension mismatch".to_string()));
        }

        // Validate parameter bounds
        for (i, &param) in parameters.iter().enumerate() {
            let (min_bound, max_bound) = self.parameter_bounds[i];
            if param < min_bound || param > max_bound {
                return Err(MathError::InvalidArgument(
                    format!("Parameter {} = {} outside bounds [{}, {}]", i, param, min_bound, max_bound)
                ));
            }
        }

        self.observations.push(Observation {
            parameters,
            objective_value,
        });

        Ok(())
    }

    pub fn suggest_next_parameters(&self) -> MathResult<TPESuggestion> {
        if self.observations.len() < self.n_startup_trials {
            // Random sampling during startup phase
            let mut parameters = Vec::new();
            for &(min_bound, max_bound) in &self.parameter_bounds {
                let param = min_bound + (max_bound - min_bound) * rand_uniform();
                parameters.push(param);
            }

            return Ok(TPESuggestion {
                parameters,
                expected_improvement: 0.0,
                acquisition_value: 1.0,
            });
        }

        // Sort observations by objective value (assume minimization)
        let mut sorted_obs = self.observations.clone();
        sorted_obs.sort_by(|a, b| a.objective_value.partial_cmp(&b.objective_value).unwrap());

        // Split into good (l) and bad (g) observations
        let split_index = ((self.gamma * sorted_obs.len() as f64).ceil() as usize).max(1);
        let good_observations = &sorted_obs[..split_index];
        let bad_observations = &sorted_obs[split_index..];

        // Build parzen estimators for each parameter dimension
        let mut good_estimators = Vec::new();
        let mut bad_estimators = Vec::new();

        for dim in 0..self.parameter_bounds.len() {
            let good_params: Vec<f64> = good_observations.iter()
                .map(|obs| obs.parameters[dim])
                .collect();
            let bad_params: Vec<f64> = bad_observations.iter()
                .map(|obs| obs.parameters[dim])
                .collect();

            let bounds = self.parameter_bounds[dim];
            
            good_estimators.push(ParzenEstimator::new(good_params, bounds)?);
            bad_estimators.push(ParzenEstimator::new(bad_params, bounds)?);
        }

        // Generate candidates and evaluate acquisition function
        let mut best_suggestion = None;
        let mut best_acquisition = f64::NEG_INFINITY;

        for _ in 0..self.n_ei_candidates {
            let mut candidate_params = Vec::new();
            
            // Generate candidate from good observations
            for dim in 0..self.parameter_bounds.len() {
                let param = good_estimators[dim].sample();
                candidate_params.push(param);
            }

            // Calculate acquisition function (l(x) / g(x))
            let l_density = self.calculate_joint_density(&candidate_params, &good_estimators)?;
            let g_density = self.calculate_joint_density(&candidate_params, &bad_estimators)?;
            
            let acquisition_value = if g_density > 1e-10 {
                l_density / g_density
            } else {
                l_density * 1e10  // Large value when g_density is near zero
            };

            if acquisition_value > best_acquisition {
                best_acquisition = acquisition_value;
                
                let expected_improvement = self.calculate_expected_improvement(
                    &candidate_params, &sorted_obs[0].objective_value
                );

                best_suggestion = Some(TPESuggestion {
                    parameters: candidate_params,
                    expected_improvement,
                    acquisition_value,
                });
            }
        }

        best_suggestion.ok_or_else(|| MathError::ComputationError("Failed to generate suggestion".to_string()))
    }

    fn calculate_joint_density(&self, parameters: &[f64], estimators: &[ParzenEstimator]) -> MathResult<f64> {
        let mut joint_density = 1.0;
        
        for (param, estimator) in parameters.iter().zip(estimators) {
            let density = estimator.density(*param);
            joint_density *= density;
        }
        
        Ok(joint_density)
    }

    fn calculate_expected_improvement(&self, _parameters: &[f64], best_value: &f64) -> f64 {
        // Simplified EI calculation - in practice this would involve more complex modeling
        let current_best = *best_value;
        let estimated_mean = current_best * 0.95; // Assume 5% improvement potential
        let estimated_std = current_best.abs() * 0.1;
        
        if estimated_std > 0.0 {
            let z = (current_best - estimated_mean) / estimated_std;
            let phi = standard_normal_pdf(z);
            let capital_phi = standard_normal_cdf(z);
            
            (current_best - estimated_mean) * capital_phi + estimated_std * phi
        } else {
            0.0
        }
    }

    pub fn get_best_parameters(&self) -> MathResult<(Vec<f64>, f64)> {
        if self.observations.is_empty() {
            return Err(MathError::InvalidOperation("No observations available".to_string()));
        }

        let best_obs = self.observations.iter()
            .min_by(|a, b| a.objective_value.partial_cmp(&b.objective_value).unwrap())
            .unwrap();

        Ok((best_obs.parameters.clone(), best_obs.objective_value))
    }

    pub fn optimization_history(&self) -> Vec<f64> {
        let mut best_values = Vec::new();
        let mut current_best = f64::INFINITY;

        for obs in &self.observations {
            if obs.objective_value < current_best {
                current_best = obs.objective_value;
            }
            best_values.push(current_best);
        }

        best_values
    }
}

impl ParzenEstimator {
    pub fn new(data_points: Vec<f64>, bounds: (f64, f64)) -> MathResult<Self> {
        if data_points.is_empty() {
            return Err(MathError::InvalidArgument("Empty data points for Parzen estimator".to_string()));
        }

        // Silverman's rule of thumb for bandwidth selection
        let n = data_points.len() as f64;
        let std_dev = calculate_std_dev(&data_points);
        let bandwidth = 1.06 * std_dev * n.powf(-1.0 / 5.0);

        Ok(Self {
            data_points,
            bandwidth: bandwidth.max(1e-6), // Minimum bandwidth
            bounds,
        })
    }

    pub fn density(&self, x: f64) -> f64 {
        if self.data_points.is_empty() {
            return 0.0;
        }

        let n = self.data_points.len() as f64;
        let mut density = 0.0;

        for &xi in &self.data_points {
            let kernel_value = truncated_normal_kernel(x, xi, self.bandwidth, self.bounds);
            density += kernel_value;
        }

        density / n
    }

    pub fn sample(&self) -> f64 {
        if self.data_points.is_empty() {
            let (min_bound, max_bound) = self.bounds;
            return min_bound + (max_bound - min_bound) * rand_uniform();
        }

        // Sample from mixture of truncated normals
        let idx = (rand_uniform() * self.data_points.len() as f64) as usize;
        let idx = idx.min(self.data_points.len() - 1);
        
        let center = self.data_points[idx];
        let sample = sample_truncated_normal(center, self.bandwidth, self.bounds);
        
        sample
    }
}

// Helper functions for TPE algorithm
fn rand_uniform() -> f64 {
    // Simple linear congruential generator for demo purposes
    static mut SEED: u64 = 1;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        let normalized = ((SEED / 65536) % 32768) as f64 / 32768.0;
        normalized.min(1.0).max(0.0) // Ensure bounds [0, 1]
    }
}

fn calculate_std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 1.0;
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (data.len() - 1) as f64;
    
    variance.sqrt()
}

fn standard_normal_pdf(x: f64) -> f64 {
    (1.0 / (2.0 * PI).sqrt()) * E.powf(-0.5 * x * x)
}

fn standard_normal_cdf(x: f64) -> f64 {
    // Approximation using error function
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

fn erf(x: f64) -> f64 {
    // Approximation of error function
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * E.powf(-x * x);

    sign * y
}

fn truncated_normal_kernel(x: f64, center: f64, bandwidth: f64, bounds: (f64, f64)) -> f64 {
    let (min_bound, max_bound) = bounds;
    
    // Calculate standard normal density
    let z = (x - center) / bandwidth;
    let kernel = standard_normal_pdf(z) / bandwidth;
    
    // Apply truncation bounds
    if x >= min_bound && x <= max_bound {
        kernel
    } else {
        0.0
    }
}

fn sample_truncated_normal(center: f64, bandwidth: f64, bounds: (f64, f64)) -> f64 {
    let (min_bound, max_bound) = bounds;
    
    // Simple rejection sampling for truncated normal
    for _ in 0..100 { // Maximum attempts
        let sample = center + bandwidth * box_muller_transform();
        if sample >= min_bound && sample <= max_bound {
            return sample;
        }
    }
    
    // Fallback to uniform sampling within bounds
    min_bound + (max_bound - min_bound) * rand_uniform()
}

fn box_muller_transform() -> f64 {
    static mut CACHED: Option<f64> = None;
    static mut HAS_CACHED: bool = false;
    
    unsafe {
        if HAS_CACHED {
            HAS_CACHED = false;
            return CACHED.unwrap();
        }
        
        let mut u1 = rand_uniform();
        let u2 = rand_uniform();
        
        // Ensure u1 is not zero to avoid log(0)
        if u1 < 1e-10 {
            u1 = 1e-10;
        }
        
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).sin();
        
        CACHED = Some(z1);
        HAS_CACHED = true;
        
        // Clamp to reasonable range to avoid extreme values
        z0.max(-5.0).min(5.0)
    }
}

// Utility functions for materials optimization
pub fn optimize_tpe_material_properties(
    target_modulus: f64, 
    target_biocompatibility: f64
) -> MathResult<TPEOptimizer> {
    // Define parameter bounds for TPE material optimization
    // [hardness, tensile_strength, crosslink_density, filler_fraction, temperature]
    let bounds = vec![
        (30.0, 90.0),    // Shore A hardness
        (5.0, 50.0),     // Tensile strength (MPa)
        (0.001, 0.1),    // Crosslink density
        (0.0, 0.4),      // Filler fraction
        (293.15, 373.15), // Temperature (K)
    ];
    
    let mut optimizer = TPEOptimizer::new(bounds, 0.25);
    
    // Add some initial observations (simulated experimental data)
    let initial_experiments = vec![
        (vec![60.0, 15.0, 0.01, 0.1, 298.15], 0.75),
        (vec![70.0, 20.0, 0.02, 0.15, 308.15], 0.85),
        (vec![50.0, 10.0, 0.005, 0.05, 298.15], 0.65),
        (vec![80.0, 30.0, 0.05, 0.25, 323.15], 0.95),
        (vec![45.0, 8.0, 0.003, 0.02, 293.15], 0.55),
    ];
    
    for (params, objective) in initial_experiments {
        let fitness = calculate_tpe_fitness(
            &params, target_modulus, target_biocompatibility
        );
        optimizer.add_observation(params, fitness)?;
    }
    
    Ok(optimizer)
}

fn calculate_tpe_fitness(
    parameters: &[f64], 
    target_modulus: f64, 
    target_biocompatibility: f64
) -> f64 {
    let hardness = parameters[0];
    let tensile_strength = parameters[1];
    let crosslink_density = parameters[2];
    let filler_fraction = parameters[3];
    let temperature = parameters[4];
    
    // Simplified TPE property models
    let estimated_modulus = (hardness * 0.1 + tensile_strength * 0.5 + 
                           crosslink_density * 1000.0 + filler_fraction * 200.0) / 4.0;
    
    let biocompat_score = 100.0 - hardness * 0.3 - filler_fraction * 50.0 - 
                         (temperature - 298.15) * 0.1;
    
    // Multi-objective optimization (minimize distance from targets)
    let modulus_error = ((estimated_modulus - target_modulus) / target_modulus).abs();
    let biocompat_error = ((biocompat_score - target_biocompatibility) / target_biocompatibility).abs();
    
    modulus_error + biocompat_error // Minimize total error
}

pub fn optimize_silicone_crosslinking(
    target_modulus_mpa: f64,
    target_elongation: f64
) -> MathResult<TPEOptimizer> {
    // Parameter bounds for silicone optimization
    // [crosslink_density, temperature, catalyst_conc, cure_time]
    let bounds = vec![
        (0.001, 0.02),    // Crosslink density (mol/cm³)
        (323.15, 423.15), // Cure temperature (K)
        (0.001, 0.05),    // Catalyst concentration
        (1800.0, 14400.0), // Cure time (seconds)
    ];
    
    let mut optimizer = TPEOptimizer::new(bounds, 0.3);
    
    // Initial experimental data
    let experiments = vec![
        (vec![0.005, 373.15, 0.01, 3600.0], 0.8),
        (vec![0.01, 393.15, 0.02, 7200.0], 1.2),
        (vec![0.002, 353.15, 0.005, 5400.0], 0.4),
        (vec![0.015, 403.15, 0.03, 10800.0], 1.8),
    ];
    
    for (params, _) in experiments {
        let fitness = calculate_silicone_fitness(
            &params, target_modulus_mpa, target_elongation
        );
        optimizer.add_observation(params, fitness)?;
    }
    
    Ok(optimizer)
}

fn calculate_silicone_fitness(
    parameters: &[f64],
    target_modulus_mpa: f64,
    target_elongation: f64
) -> f64 {
    let crosslink_density = parameters[0];
    let temperature = parameters[1];
    let _catalyst_conc = parameters[2];
    let _cure_time = parameters[3];
    
    // Simplified silicone property models
    let estimated_modulus = 0.05 + crosslink_density * 100.0; // MPa
    let estimated_elongation = 800.0 - crosslink_density * 20000.0; // %
    
    let modulus_error = ((estimated_modulus - target_modulus_mpa) / target_modulus_mpa).abs();
    let elongation_error = ((estimated_elongation - target_elongation) / target_elongation).abs();
    
    // Include temperature penalty for energy efficiency
    let temp_penalty = (temperature - 323.15) / 100.0;
    
    modulus_error + elongation_error + temp_penalty * 0.1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tpe_optimizer_creation() {
        let bounds = vec![(0.0, 1.0), (0.0, 10.0)];
        let optimizer = TPEOptimizer::new(bounds, 0.25);
        
        assert_eq!(optimizer.parameter_bounds.len(), 2);
        assert_eq!(optimizer.gamma, 0.25);
        assert!(optimizer.observations.is_empty());
    }

    #[test]
    fn test_add_observation() {
        let bounds = vec![(0.0, 1.0), (0.0, 10.0)];
        let mut optimizer = TPEOptimizer::new(bounds, 0.25);
        
        let result = optimizer.add_observation(vec![0.5, 5.0], 1.0);
        assert!(result.is_ok());
        assert_eq!(optimizer.observations.len(), 1);
    }

    #[test]
    fn test_add_observation_out_of_bounds() {
        let bounds = vec![(0.0, 1.0), (0.0, 10.0)];
        let mut optimizer = TPEOptimizer::new(bounds, 0.25);
        
        let result = optimizer.add_observation(vec![1.5, 5.0], 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_parzen_estimator() {
        let data = vec![0.1, 0.5, 0.8, 0.3, 0.7];
        let bounds = (0.0, 1.0);
        let estimator = ParzenEstimator::new(data, bounds).unwrap();
        
        let density = estimator.density(0.5);
        assert!(density > 0.0);
        
        let sample = estimator.sample();
        assert!(sample >= 0.0 && sample <= 1.0);
    }

    #[test]
    fn test_suggest_parameters_startup() {
        let bounds = vec![(0.0, 1.0), (0.0, 10.0)];
        let optimizer = TPEOptimizer::new(bounds, 0.25);
        
        let suggestion = optimizer.suggest_next_parameters().unwrap();
        assert_eq!(suggestion.parameters.len(), 2);
        assert!(suggestion.parameters[0] >= 0.0 && suggestion.parameters[0] <= 1.0);
        assert!(suggestion.parameters[1] >= 0.0 && suggestion.parameters[1] <= 10.0);
    }

    #[test]
    fn test_suggest_parameters_with_observations() {
        let bounds = vec![(0.0, 1.0), (0.0, 10.0)];
        let mut optimizer = TPEOptimizer::new(bounds, 0.25);
        optimizer.n_startup_trials = 3;
        
        // Add observations beyond startup
        for i in 0..5 {
            optimizer.add_observation(
                vec![i as f64 * 0.2, i as f64 * 2.0], 
                (5 - i) as f64
            ).unwrap();
        }
        
        let suggestion = optimizer.suggest_next_parameters().unwrap();
        assert_eq!(suggestion.parameters.len(), 2);
        assert!(suggestion.acquisition_value > 0.0);
    }

    #[test]
    fn test_get_best_parameters() {
        let bounds = vec![(0.0, 1.0)];
        let mut optimizer = TPEOptimizer::new(bounds, 0.25);
        
        optimizer.add_observation(vec![0.2], 5.0).unwrap();
        optimizer.add_observation(vec![0.8], 2.0).unwrap();
        optimizer.add_observation(vec![0.5], 3.0).unwrap();
        
        let (best_params, best_value) = optimizer.get_best_parameters().unwrap();
        assert_eq!(best_params, vec![0.8]);
        assert_eq!(best_value, 2.0);
    }

    #[test]
    fn test_optimization_history() {
        let bounds = vec![(0.0, 1.0)];
        let mut optimizer = TPEOptimizer::new(bounds, 0.25);
        
        optimizer.add_observation(vec![0.2], 5.0).unwrap();
        optimizer.add_observation(vec![0.8], 2.0).unwrap();
        optimizer.add_observation(vec![0.5], 3.0).unwrap();
        optimizer.add_observation(vec![0.1], 1.0).unwrap();
        
        let history = optimizer.optimization_history();
        assert_eq!(history, vec![5.0, 2.0, 2.0, 1.0]);
    }

    #[test]
    fn test_tpe_material_optimization() {
        let optimizer = optimize_tpe_material_properties(20.0, 85.0).unwrap();
        assert!(optimizer.observations.len() > 0);
        
        let (best_params, _best_value) = optimizer.get_best_parameters().unwrap();
        assert_eq!(best_params.len(), 5);
    }

    #[test]
    fn test_silicone_optimization() {
        let optimizer = optimize_silicone_crosslinking(1.5, 400.0).unwrap();
        assert!(optimizer.observations.len() > 0);
        
        let suggestion = optimizer.suggest_next_parameters().unwrap();
        assert_eq!(suggestion.parameters.len(), 4);
    }
}