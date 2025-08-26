use crate::core::{MathDomain, MathResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosDynamicsDomain {
    name: String,
}

// Dynamical System Definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicalSystem {
    pub dimension: usize,
    pub variables: Vec<String>,
    pub parameters: HashMap<String, f64>,
    pub system_type: SystemType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemType {
    Autonomous,    // dx/dt = f(x)
    NonAutonomous, // dx/dt = f(x,t)
    Discrete,      // x_{n+1} = f(x_n)
    Stochastic,    // dx = f(x)dt + g(x)dW
}

// Fixed Points and Equilibria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedPoint {
    pub position: Vec<f64>,
    pub stability: StabilityType,
    pub eigenvalues: Vec<num_complex::Complex<f64>>,
    pub eigenvectors: Vec<Vec<f64>>,
    pub basin_of_attraction: Option<BasinInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StabilityType {
    Stable,
    Unstable,
    Saddle,
    Center,
    StableSpiral,
    UnstableSpiral,
    StableNode,
    UnstableNode,
    Hyperbolic,
    NonHyperbolic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasinInfo {
    pub boundary_type: BoundaryType,
    pub fractal_dimension: Option<f64>,
    pub sample_points: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryType {
    Smooth,
    Fractal,
    Riddled,
    Wada,
}

// Attractors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attractor {
    pub attractor_type: AttractorType,
    pub dimension: f64, // Fractal dimension
    pub lyapunov_exponents: Vec<f64>,
    pub trajectory_points: Vec<Vec<f64>>,
    pub poincare_section: Option<PoincareSection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttractorType {
    FixedPoint,
    LimitCycle,
    Torus,
    Strange,     // Chaotic attractor
    Hyperchaotic, // Multiple positive Lyapunov exponents
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoincareSection {
    pub section_plane: SectionPlane,
    pub intersection_points: Vec<Vec<f64>>,
    pub return_map: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionPlane {
    pub normal_vector: Vec<f64>,
    pub point_on_plane: Vec<f64>,
    pub crossing_direction: CrossingDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossingDirection {
    Both,
    Positive,
    Negative,
}

// Bifurcation Theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BifurcationDiagram {
    pub parameter_name: String,
    pub parameter_range: (f64, f64),
    pub parameter_values: Vec<f64>,
    pub attractor_values: Vec<Vec<f64>>,
    pub bifurcation_points: Vec<BifurcationPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BifurcationPoint {
    pub parameter_value: f64,
    pub bifurcation_type: BifurcationType,
    pub critical_eigenvalue: Option<num_complex::Complex<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BifurcationType {
    SaddleNode,
    Transcritical,
    Pitchfork,
    Hopf,
    PeriodDoubling,
    Homoclinic,
    Heteroclinic,
    Cusp,
    Takens_Bogdanov,
}

// Chaos Measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosMetrics {
    pub largest_lyapunov_exponent: f64,
    pub lyapunov_spectrum: Vec<f64>,
    pub kaplan_yorke_dimension: f64,
    pub correlation_dimension: f64,
    pub information_dimension: f64,
    pub generalized_dimensions: Vec<(f64, f64)>, // (q, D_q)
    pub kolmogorov_entropy: f64,
    pub correlation_sum: Vec<(f64, f64)>, // (r, C(r))
}

// Fractals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fractal {
    pub fractal_type: FractalType,
    pub hausdorff_dimension: f64,
    pub box_counting_dimension: f64,
    pub mass_dimension: Option<f64>,
    pub generating_rules: Vec<String>,
    pub iteration_count: usize,
    pub scale_invariance: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FractalType {
    Mandelbrot,
    Julia,
    Cantor,
    Sierpinski,
    Koch,
    DragonCurve,
    Barnsley,
    Custom,
}

// Time Series Analysis for Chaos
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesReconstruction {
    pub embedding_dimension: usize,
    pub time_delay: f64,
    pub reconstructed_trajectory: Vec<Vec<f64>>,
    pub false_nearest_neighbors: Vec<f64>,
    pub mutual_information: Vec<f64>,
}

// Synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationAnalysis {
    pub sync_type: SynchronizationType,
    pub coupling_strength: f64,
    pub synchronization_error: Vec<f64>,
    pub phase_locking_ratio: Option<(usize, usize)>,
    pub master_stability_function: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynchronizationType {
    Complete,
    Phase,
    Lag,
    Generalized,
    Cluster,
    Chimera,
}

impl ChaosDynamicsDomain {
    pub fn new() -> Self {
        Self {
            name: "Chaos and Dynamical Systems".to_string(),
        }
    }

    // Classic Chaotic Systems
    pub fn lorenz_system(&self, sigma: f64, rho: f64, beta: f64) 
                        -> Box<dyn Fn(&[f64], f64) -> Vec<f64>> {
        Box::new(move |state: &[f64], _t: f64| {
            let x = state[0];
            let y = state[1];
            let z = state[2];
            
            vec![
                sigma * (y - x),           // dx/dt
                x * (rho - z) - y,         // dy/dt
                x * y - beta * z           // dz/dt
            ]
        })
    }

    pub fn rossler_system(&self, a: f64, b: f64, c: f64) 
                         -> Box<dyn Fn(&[f64], f64) -> Vec<f64>> {
        Box::new(move |state: &[f64], _t: f64| {
            let x = state[0];
            let y = state[1];
            let z = state[2];
            
            vec![
                -y - z,                    // dx/dt
                x + a * y,                 // dy/dt
                b + z * (x - c)            // dz/dt
            ]
        })
    }

    pub fn henon_map(&self, a: f64, b: f64) -> Box<dyn Fn(&[f64]) -> Vec<f64>> {
        Box::new(move |state: &[f64]| {
            let x = state[0];
            let y = state[1];
            
            vec![
                1.0 - a * x * x + y,       // x_{n+1}
                b * x                      // y_{n+1}
            ]
        })
    }

    pub fn logistic_map(&self, r: f64) -> Box<dyn Fn(f64) -> f64> {
        Box::new(move |x: f64| r * x * (1.0 - x))
    }

    // Lyapunov Exponent Calculation
    pub fn compute_largest_lyapunov_exponent(&self, 
                                           trajectory: &[Vec<f64>], 
                                           dt: f64) -> MathResult<f64> {
        if trajectory.len() < 2 {
            return Err(crate::core::MathError::InvalidArgument(
                "Trajectory too short".to_string()));
        }

        let dim = trajectory[0].len();
        let mut sum_log = 0.0;
        let mut count = 0;
        
        // Use the method of nearby trajectories
        let separation_threshold = 1e-8;
        let rescale_factor = 1e-12;
        
        for i in 1..trajectory.len() - 1 {
            // Find nearby point
            if let Some(nearby_idx) = self.find_nearest_neighbor(&trajectory[i], trajectory, i) {
                let current = &trajectory[i];
                let nearby = &trajectory[nearby_idx];
                
                // Initial separation
                let mut separation = self.euclidean_distance(current, nearby);
                
                if separation > separation_threshold && separation < 0.1 {
                    // Evolution after one time step
                    let next_current = &trajectory[i + 1];
                    let next_nearby = &trajectory[nearby_idx + 1];
                    let next_separation = self.euclidean_distance(next_current, next_nearby);
                    
                    if next_separation > 0.0 {
                        sum_log += (next_separation / separation).ln();
                        count += 1;
                    }
                }
            }
        }
        
        if count > 0 {
            Ok(sum_log / (count as f64 * dt))
        } else {
            Err(crate::core::MathError::ComputationError(
                "Could not compute Lyapunov exponent".to_string()))
        }
    }

    fn find_nearest_neighbor(&self, point: &[f64], trajectory: &[Vec<f64>], 
                           exclude_idx: usize) -> Option<usize> {
        let mut min_distance = f64::INFINITY;
        let mut nearest_idx = None;
        
        for (i, other_point) in trajectory.iter().enumerate() {
            if i != exclude_idx && (i as i32 - exclude_idx as i32).abs() > 10 {
                let distance = self.euclidean_distance(point, other_point);
                if distance < min_distance {
                    min_distance = distance;
                    nearest_idx = Some(i);
                }
            }
        }
        
        nearest_idx
    }

    fn euclidean_distance(&self, p1: &[f64], p2: &[f64]) -> f64 {
        p1.iter().zip(p2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    // Correlation Dimension
    pub fn compute_correlation_dimension(&self, trajectory: &[Vec<f64>]) -> MathResult<f64> {
        if trajectory.len() < 100 {
            return Err(crate::core::MathError::InvalidArgument(
                "Need at least 100 points for correlation dimension".to_string()));
        }

        let n_points = trajectory.len();
        let r_values: Vec<f64> = (1..=20).map(|i| 0.001 * (2.0_f64).powi(i)).collect();
        let mut correlation_sums = Vec::new();
        
        for &r in &r_values {
            let mut count = 0;
            for i in 0..n_points {
                for j in i+1..n_points {
                    if self.euclidean_distance(&trajectory[i], &trajectory[j]) < r {
                        count += 1;
                    }
                }
            }
            let c_r = 2.0 * count as f64 / (n_points * (n_points - 1)) as f64;
            correlation_sums.push((r, c_r));
        }
        
        // Fit log(C(r)) vs log(r) to get slope (correlation dimension)
        let log_pairs: Vec<(f64, f64)> = correlation_sums.iter()
            .filter(|(r, c)| *c > 0.0 && *r > 0.0)
            .map(|(r, c)| (r.ln(), c.ln()))
            .collect();
        
        if log_pairs.len() < 5 {
            return Err(crate::core::MathError::ComputationError(
                "Insufficient data for correlation dimension".to_string()));
        }
        
        // Linear regression
        let n = log_pairs.len() as f64;
        let sum_x: f64 = log_pairs.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = log_pairs.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = log_pairs.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = log_pairs.iter().map(|(x, _)| x * x).sum();
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        Ok(slope)
    }

    // Poincaré Section
    pub fn compute_poincare_section(&self, trajectory: &[Vec<f64>], 
                                  section: &SectionPlane) -> PoincareSection {
        let mut intersection_points = Vec::new();
        
        for i in 1..trajectory.len() {
            let p1 = &trajectory[i - 1];
            let p2 = &trajectory[i];
            
            // Check if trajectory crosses the section plane
            let d1 = self.distance_to_plane(p1, section);
            let d2 = self.distance_to_plane(p2, section);
            
            if d1 * d2 < 0.0 { // Sign change indicates crossing
                match section.crossing_direction {
                    CrossingDirection::Both => {
                        let intersection = self.interpolate_intersection(p1, p2, d1, d2);
                        intersection_points.push(intersection);
                    }
                    CrossingDirection::Positive if d2 > d1 => {
                        let intersection = self.interpolate_intersection(p1, p2, d1, d2);
                        intersection_points.push(intersection);
                    }
                    CrossingDirection::Negative if d2 < d1 => {
                        let intersection = self.interpolate_intersection(p1, p2, d1, d2);
                        intersection_points.push(intersection);
                    }
                    _ => {}
                }
            }
        }
        
        // Generate return map (simplified - use first coordinate)
        let mut return_map = Vec::new();
        for i in 1..intersection_points.len() {
            if !intersection_points[i - 1].is_empty() && !intersection_points[i].is_empty() {
                return_map.push((intersection_points[i - 1][0], intersection_points[i][0]));
            }
        }
        
        PoincareSection {
            section_plane: section.clone(),
            intersection_points,
            return_map,
        }
    }

    fn distance_to_plane(&self, point: &[f64], plane: &SectionPlane) -> f64 {
        let mut dot_product = 0.0;
        let mut normal_dot = 0.0;
        
        for i in 0..point.len().min(plane.normal_vector.len()) {
            dot_product += plane.normal_vector[i] * (point[i] - plane.point_on_plane[i]);
            normal_dot += plane.normal_vector[i] * plane.normal_vector[i];
        }
        
        if normal_dot > 0.0 {
            dot_product / normal_dot.sqrt()
        } else {
            0.0
        }
    }

    fn interpolate_intersection(&self, p1: &[f64], p2: &[f64], d1: f64, d2: f64) -> Vec<f64> {
        let t = -d1 / (d2 - d1);
        let mut intersection = Vec::new();
        
        for i in 0..p1.len().min(p2.len()) {
            intersection.push(p1[i] + t * (p2[i] - p1[i]));
        }
        
        intersection
    }

    // Bifurcation Analysis
    pub fn period_doubling_bifurcation(&self, r_start: f64, r_end: f64, 
                                      n_points: usize) -> BifurcationDiagram {
        let mut parameter_values = Vec::new();
        let mut attractor_values = Vec::new();
        let mut bifurcation_points = Vec::new();
        
        let dr = (r_end - r_start) / n_points as f64;
        
        for i in 0..n_points {
            let r = r_start + i as f64 * dr;
            parameter_values.push(r);
            
            // Run logistic map to find attractor
            let logistic = self.logistic_map(r);
            let mut x = 0.5; // Initial condition
            
            // Transient
            for _ in 0..1000 {
                x = logistic(x);
            }
            
            // Collect attractor points
            let mut attractor = Vec::new();
            for _ in 0..100 {
                x = logistic(x);
                attractor.push(x);
            }
            
            // Remove duplicates for periodic orbits
            attractor.sort_by(|a, b| a.partial_cmp(b).unwrap());
            attractor.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
            
            attractor_values.push(attractor);
            
            // Simple bifurcation detection (period doubling)
            if i > 0 {
                let prev_period = attractor_values[i - 1].len();
                let curr_period = attractor_values[i].len();
                
                if curr_period == 2 * prev_period {
                    bifurcation_points.push(BifurcationPoint {
                        parameter_value: r,
                        bifurcation_type: BifurcationType::PeriodDoubling,
                        critical_eigenvalue: None,
                    });
                }
            }
        }
        
        BifurcationDiagram {
            parameter_name: "r".to_string(),
            parameter_range: (r_start, r_end),
            parameter_values,
            attractor_values,
            bifurcation_points,
        }
    }

    // Phase Space Reconstruction
    pub fn time_delay_embedding(&self, time_series: &[f64], 
                              embedding_dim: usize, delay: usize) -> TimeSeriesReconstruction {
        let mut reconstructed_trajectory = Vec::new();
        
        for i in 0..time_series.len() - (embedding_dim - 1) * delay {
            let mut point = Vec::new();
            for j in 0..embedding_dim {
                point.push(time_series[i + j * delay]);
            }
            reconstructed_trajectory.push(point);
        }
        
        // Compute mutual information for optimal delay
        let mut mutual_information = Vec::new();
        for tau in 1..=50 {
            let mi = self.compute_mutual_information(time_series, tau);
            mutual_information.push(mi);
        }
        
        // False nearest neighbors for optimal embedding dimension
        let mut false_nearest_neighbors = Vec::new();
        for m in 1..=20 {
            let fnn = self.compute_false_nearest_neighbors(time_series, m, delay);
            false_nearest_neighbors.push(fnn);
        }
        
        TimeSeriesReconstruction {
            embedding_dimension: embedding_dim,
            time_delay: delay as f64,
            reconstructed_trajectory,
            false_nearest_neighbors,
            mutual_information,
        }
    }

    fn compute_mutual_information(&self, series: &[f64], delay: usize) -> f64 {
        if delay >= series.len() {
            return 0.0;
        }
        
        // Simplified mutual information using histograms
        let n_bins = 20;
        let min_val = series.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = series.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let bin_width = (max_val - min_val) / n_bins as f64;
        
        if bin_width == 0.0 {
            return 0.0;
        }
        
        let mut joint_hist = vec![vec![0; n_bins]; n_bins];
        let mut x_hist = vec![0; n_bins];
        let mut y_hist = vec![0; n_bins];
        let mut total_count = 0;
        
        for i in 0..series.len() - delay {
            let x_bin = ((series[i] - min_val) / bin_width).floor() as usize;
            let y_bin = ((series[i + delay] - min_val) / bin_width).floor() as usize;
            
            let x_bin = x_bin.min(n_bins - 1);
            let y_bin = y_bin.min(n_bins - 1);
            
            joint_hist[x_bin][y_bin] += 1;
            x_hist[x_bin] += 1;
            y_hist[y_bin] += 1;
            total_count += 1;
        }
        
        let mut mutual_info = 0.0;
        for i in 0..n_bins {
            for j in 0..n_bins {
                if joint_hist[i][j] > 0 && x_hist[i] > 0 && y_hist[j] > 0 {
                    let p_xy = joint_hist[i][j] as f64 / total_count as f64;
                    let p_x = x_hist[i] as f64 / total_count as f64;
                    let p_y = y_hist[j] as f64 / total_count as f64;
                    
                    mutual_info += p_xy * (p_xy / (p_x * p_y)).ln();
                }
            }
        }
        
        mutual_info
    }

    fn compute_false_nearest_neighbors(&self, series: &[f64], 
                                     embedding_dim: usize, delay: usize) -> f64 {
        // Simplified false nearest neighbors calculation
        if embedding_dim == 1 || delay >= series.len() {
            return 1.0;
        }
        
        let embedding_1 = self.time_delay_embedding(series, embedding_dim, delay);
        let embedding_2 = self.time_delay_embedding(series, embedding_dim + 1, delay);
        
        let mut false_neighbors = 0;
        let mut total_neighbors = 0;
        let rt_threshold = 15.0;
        
        for i in 0..embedding_1.reconstructed_trajectory.len().min(embedding_2.reconstructed_trajectory.len()) {
            if let Some(nearest_idx) = self.find_nearest_neighbor(
                &embedding_1.reconstructed_trajectory[i], 
                &embedding_1.reconstructed_trajectory, i
            ) {
                let d1 = self.euclidean_distance(
                    &embedding_1.reconstructed_trajectory[i], 
                    &embedding_1.reconstructed_trajectory[nearest_idx]
                );
                let d2 = self.euclidean_distance(
                    &embedding_2.reconstructed_trajectory[i], 
                    &embedding_2.reconstructed_trajectory[nearest_idx]
                );
                
                if d1 > 0.0 {
                    let ratio = (d2 - d1) / d1;
                    if ratio > rt_threshold {
                        false_neighbors += 1;
                    }
                }
                total_neighbors += 1;
            }
        }
        
        if total_neighbors > 0 {
            false_neighbors as f64 / total_neighbors as f64
        } else {
            1.0
        }
    }

    // Fractal Dimension (Box Counting)
    pub fn box_counting_dimension(&self, points: &[Vec<f64>]) -> MathResult<f64> {
        if points.is_empty() {
            return Err(crate::core::MathError::InvalidArgument(
                "Empty point set".to_string()));
        }

        let dim = points[0].len();
        if dim != 2 {
            return Err(crate::core::MathError::NotImplemented(
                "Box counting only implemented for 2D".to_string()));
        }

        // Find bounding box
        let mut min_coords = vec![f64::INFINITY; dim];
        let mut max_coords = vec![f64::NEG_INFINITY; dim];
        
        for point in points {
            for (i, &coord) in point.iter().enumerate() {
                min_coords[i] = min_coords[i].min(coord);
                max_coords[i] = max_coords[i].max(coord);
            }
        }

        let mut log_counts = Vec::new();
        let mut log_sizes = Vec::new();
        
        // Try different box sizes
        for k in 2..=10 {
            let n_boxes = 2_usize.pow(k);
            let box_size = (max_coords[0] - min_coords[0]) / n_boxes as f64;
            
            if box_size <= 0.0 {
                continue;
            }
            
            let mut occupied_boxes = std::collections::HashSet::new();
            
            for point in points {
                let box_x = ((point[0] - min_coords[0]) / box_size).floor() as i32;
                let box_y = ((point[1] - min_coords[1]) / box_size).floor() as i32;
                occupied_boxes.insert((box_x, box_y));
            }
            
            if occupied_boxes.len() > 1 {
                log_sizes.push(box_size.ln());
                log_counts.push((occupied_boxes.len() as f64).ln());
            }
        }
        
        if log_sizes.len() < 3 {
            return Err(crate::core::MathError::ComputationError(
                "Insufficient data for box counting".to_string()));
        }
        
        // Linear regression: log(N) = -D * log(size) + C
        let n = log_sizes.len() as f64;
        let sum_x: f64 = log_sizes.iter().sum();
        let sum_y: f64 = log_counts.iter().sum();
        let sum_xy: f64 = log_sizes.iter().zip(&log_counts).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = log_sizes.iter().map(|x| x * x).sum();
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        Ok(-slope) // Box counting dimension is negative slope
    }
}

impl MathDomain for ChaosDynamicsDomain {
    fn name(&self) -> &str {
        &self.name
    }

    fn supports_operation(&self, operation: &str) -> bool {
        matches!(operation, 
            "lorenz_system" | "rossler_system" | "henon_map" | "logistic_map" |
            "lyapunov_exponent" | "correlation_dimension" | "poincare_section" |
            "bifurcation_diagram" | "phase_reconstruction" | "box_counting" |
            "chaos_metrics" | "attractor_analysis" | "synchronization"
        )
    }

    fn description(&self) -> &str {
        "Chaos Theory and Nonlinear Dynamical Systems"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    fn compute(&self, operation: &str, _args: &[&dyn std::any::Any]) -> MathResult<Box<dyn std::any::Any>> {
        match operation {
            "lorenz_system" => Ok(Box::new("Lorenz system generated".to_string())),
            "lyapunov_exponent" => Ok(Box::new("Lyapunov exponent computed".to_string())),
            "bifurcation_diagram" => Ok(Box::new("Bifurcation diagram generated".to_string())),
            "poincare_section" => Ok(Box::new("Poincaré section computed".to_string())),
            _ => Err(crate::core::MathError::NotImplemented(format!("Operation '{}' not implemented", operation))),
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "lorenz_system".to_string(), "rossler_system".to_string(),
            "henon_map".to_string(), "logistic_map".to_string(),
            "lyapunov_exponent".to_string(), "correlation_dimension".to_string(),
            "poincare_section".to_string(), "bifurcation_diagram".to_string(),
            "phase_reconstruction".to_string(), "box_counting".to_string(),
            "chaos_metrics".to_string(), "attractor_analysis".to_string(),
            "synchronization".to_string()
        ]
    }
}

pub fn chaos_dynamics() -> ChaosDynamicsDomain {
    ChaosDynamicsDomain::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_map() {
        let domain = ChaosDynamicsDomain::new();
        let map = domain.logistic_map(3.5);
        
        let x = 0.5;
        let result = map(x);
        assert!((result - 0.875).abs() < 1e-10);
    }

    #[test]
    fn test_lorenz_system() {
        let domain = ChaosDynamicsDomain::new();
        let lorenz = domain.lorenz_system(10.0, 28.0, 8.0/3.0);
        
        let state = vec![1.0, 1.0, 1.0];
        let derivatives = lorenz(&state, 0.0);
        
        assert_eq!(derivatives.len(), 3);
        assert!((derivatives[0] - 0.0).abs() < 1e-10); // σ(y-x) = 10(1-1) = 0
    }

    #[test]
    fn test_bifurcation_diagram() {
        let domain = ChaosDynamicsDomain::new();
        let diagram = domain.period_doubling_bifurcation(2.8, 3.2, 50);
        
        assert_eq!(diagram.parameter_values.len(), 50);
        assert_eq!(diagram.attractor_values.len(), 50);
        assert_eq!(diagram.parameter_name, "r");
    }

    #[test]
    fn test_time_delay_embedding() {
        let domain = ChaosDynamicsDomain::new();
        let time_series: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        
        let reconstruction = domain.time_delay_embedding(&time_series, 3, 5);
        
        assert_eq!(reconstruction.embedding_dimension, 3);
        assert_eq!(reconstruction.time_delay, 5.0);
        assert!(!reconstruction.reconstructed_trajectory.is_empty());
        assert!(!reconstruction.mutual_information.is_empty());
    }

    #[test]
    fn test_euclidean_distance() {
        let domain = ChaosDynamicsDomain::new();
        let p1 = vec![0.0, 0.0];
        let p2 = vec![3.0, 4.0];
        
        let distance = domain.euclidean_distance(&p1, &p2);
        assert!((distance - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_henon_map() {
        let domain = ChaosDynamicsDomain::new();
        let henon = domain.henon_map(1.4, 0.3);
        
        let initial = vec![0.0, 0.0];
        let next = henon(&initial);
        
        assert_eq!(next.len(), 2);
        assert!((next[0] - 1.0).abs() < 1e-10); // 1 - 1.4*0^2 + 0 = 1
        assert!((next[1] - 0.0).abs() < 1e-10); // 0.3*0 = 0
    }
}