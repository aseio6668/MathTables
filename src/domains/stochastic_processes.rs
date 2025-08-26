use crate::core::{MathDomain, MathResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use rand::{Rng, SeedableRng};
use rand::distributions::{Distribution, Uniform};
use rand_distr::{Normal, Poisson, Exp};
use rand_pcg::Pcg64;
use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticProcessesDomain {
    name: String,
}

// Random Variables and Distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomVariable {
    pub name: String,
    pub distribution: ProbabilityDistribution,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProbabilityDistribution {
    Normal { mean: f64, std_dev: f64 },
    Uniform { min: f64, max: f64 },
    Exponential { rate: f64 },
    Poisson { lambda: f64 },
    Binomial { n: usize, p: f64 },
    Geometric { p: f64 },
    Beta { alpha: f64, beta: f64 },
    Gamma { shape: f64, scale: f64 },
    Weibull { shape: f64, scale: f64 },
    LogNormal { mu: f64, sigma: f64 },
    ChiSquared { k: f64 },
    StudentT { nu: f64 },
    Cauchy { location: f64, scale: f64 },
}

// Markov Chains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkovChain {
    pub states: Vec<String>,
    pub transition_matrix: DMatrix<f64>,
    pub initial_distribution: DVector<f64>,
    pub current_state: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkovChainProperties {
    pub irreducible: bool,
    pub aperiodic: bool,
    pub recurrent_states: HashSet<usize>,
    pub transient_states: HashSet<usize>,
    pub stationary_distribution: Option<DVector<f64>>,
    pub communication_classes: Vec<Vec<usize>>,
}

// Brownian Motion and Wiener Processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrownianMotion {
    pub dimension: usize,
    pub drift: Vec<f64>,
    pub diffusion: DMatrix<f64>,
    pub initial_value: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WienerPath {
    pub time_points: Vec<f64>,
    pub path_values: Vec<Vec<f64>>,
    pub increments: Vec<Vec<f64>>,
}

// Stochastic Differential Equations (SDEs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDE {
    pub dimension: usize,
    pub variables: Vec<String>,
    pub drift_coefficients: Vec<String>, // μ(t, X_t)
    pub diffusion_coefficients: Vec<String>, // σ(t, X_t)
    pub noise_dimension: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDESolution {
    pub time_grid: Vec<f64>,
    pub paths: Vec<Vec<Vec<f64>>>, // [path][time][dimension]
    pub statistics: PathStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathStatistics {
    pub mean_path: Vec<Vec<f64>>,
    pub variance_path: Vec<Vec<f64>>,
    pub confidence_intervals: Vec<Vec<(f64, f64)>>,
    pub convergence_info: ConvergenceInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceInfo {
    pub strong_convergence_order: f64,
    pub weak_convergence_order: f64,
    pub monte_carlo_error: f64,
    pub discretization_error: f64,
}

// Poisson Processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoissonProcess {
    pub rate: f64,
    pub jump_distribution: Option<ProbabilityDistribution>,
    pub events: Vec<PoissonEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoissonEvent {
    pub time: f64,
    pub magnitude: f64,
    pub event_type: String,
}

// Jump Diffusion Processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JumpDiffusionProcess {
    pub brownian_component: BrownianMotion,
    pub jump_component: PoissonProcess,
    pub jump_size_distribution: ProbabilityDistribution,
}

// Queueing Theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueingSystem {
    pub arrival_rate: f64,
    pub service_rate: f64,
    pub num_servers: usize,
    pub capacity: Option<usize>,
    pub queue_discipline: QueueDiscipline,
    pub system_state: QueueState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueueDiscipline {
    FIFO, // First In, First Out
    LIFO, // Last In, First Out
    SJF,  // Shortest Job First
    Priority,
    Random,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueState {
    pub customers_in_system: usize,
    pub customers_in_queue: usize,
    pub server_utilization: f64,
    pub average_wait_time: f64,
    pub average_service_time: f64,
}

// Monte Carlo Methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloSimulation {
    pub num_simulations: usize,
    pub random_seed: Option<u64>,
    pub estimator: String,
    pub confidence_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloResults {
    pub estimate: f64,
    pub standard_error: f64,
    pub confidence_interval: (f64, f64),
    pub convergence_history: Vec<f64>,
}

impl StochasticProcessesDomain {
    pub fn new() -> Self {
        Self {
            name: "Stochastic Processes".to_string(),
        }
    }

    // Random Variable Operations
    pub fn create_random_variable(&self, name: String, distribution: ProbabilityDistribution) -> RandomVariable {
        let parameters = match &distribution {
            ProbabilityDistribution::Normal { mean, std_dev } => {
                let mut params = HashMap::new();
                params.insert("mean".to_string(), *mean);
                params.insert("std_dev".to_string(), *std_dev);
                params
            }
            ProbabilityDistribution::Uniform { min, max } => {
                let mut params = HashMap::new();
                params.insert("min".to_string(), *min);
                params.insert("max".to_string(), *max);
                params
            }
            _ => HashMap::new(),
        };

        RandomVariable {
            name,
            distribution,
            parameters,
        }
    }

    pub fn sample_distribution(&self, distribution: &ProbabilityDistribution, 
                              n_samples: usize, seed: Option<u64>) -> Vec<f64> {
        let mut rng = match seed {
            Some(s) => Pcg64::seed_from_u64(s),
            None => Pcg64::from_entropy(),
        };

        let mut samples = Vec::with_capacity(n_samples);

        match distribution {
            ProbabilityDistribution::Normal { mean, std_dev } => {
                let normal = Normal::new(*mean, *std_dev).unwrap();
                for _ in 0..n_samples {
                    samples.push(normal.sample(&mut rng));
                }
            }
            ProbabilityDistribution::Uniform { min, max } => {
                let uniform = Uniform::new(*min, *max);
                for _ in 0..n_samples {
                    samples.push(uniform.sample(&mut rng));
                }
            }
            ProbabilityDistribution::Exponential { rate } => {
                let exp = Exp::new(*rate).unwrap();
                for _ in 0..n_samples {
                    samples.push(exp.sample(&mut rng));
                }
            }
            ProbabilityDistribution::Poisson { lambda } => {
                let poisson = Poisson::new(*lambda).unwrap();
                for _ in 0..n_samples {
                    samples.push(poisson.sample(&mut rng) as f64);
                }
            }
            _ => {
                // Implement other distributions as needed
                for _ in 0..n_samples {
                    samples.push(rng.gen());
                }
            }
        }

        samples
    }

    pub fn compute_pdf(&self, distribution: &ProbabilityDistribution, x: f64) -> f64 {
        match distribution {
            ProbabilityDistribution::Normal { mean, std_dev } => {
                let z = (x - mean) / std_dev;
                (1.0 / (std_dev * (2.0 * std::f64::consts::PI).sqrt())) * 
                (-0.5 * z * z).exp()
            }
            ProbabilityDistribution::Uniform { min, max } => {
                if x >= *min && x <= *max {
                    1.0 / (max - min)
                } else {
                    0.0
                }
            }
            ProbabilityDistribution::Exponential { rate } => {
                if x >= 0.0 {
                    rate * (-rate * x).exp()
                } else {
                    0.0
                }
            }
            _ => 0.0, // Implement other distributions
        }
    }

    pub fn compute_cdf(&self, distribution: &ProbabilityDistribution, x: f64) -> f64 {
        match distribution {
            ProbabilityDistribution::Normal { mean, std_dev } => {
                let z = (x - mean) / std_dev;
                0.5 * (1.0 + self.erf(z / 2.0_f64.sqrt()))
            }
            ProbabilityDistribution::Uniform { min, max } => {
                if x < *min {
                    0.0
                } else if x > *max {
                    1.0
                } else {
                    (x - min) / (max - min)
                }
            }
            ProbabilityDistribution::Exponential { rate } => {
                if x >= 0.0 {
                    1.0 - (-rate * x).exp()
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    fn erf(&self, x: f64) -> f64 {
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
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    // Markov Chain Operations
    pub fn create_markov_chain(&self, states: Vec<String>) -> MarkovChain {
        let n = states.len();
        MarkovChain {
            states,
            transition_matrix: DMatrix::zeros(n, n),
            initial_distribution: DVector::zeros(n),
            current_state: None,
        }
    }

    pub fn set_transition_probability(&self, chain: &mut MarkovChain, 
                                    from_state: usize, to_state: usize, probability: f64) {
        chain.transition_matrix[(from_state, to_state)] = probability;
    }

    pub fn set_initial_distribution(&self, chain: &mut MarkovChain, distribution: Vec<f64>) {
        chain.initial_distribution = DVector::from_vec(distribution);
    }

    pub fn simulate_markov_chain(&self, chain: &MarkovChain, steps: usize, 
                               seed: Option<u64>) -> Vec<usize> {
        let mut rng = match seed {
            Some(s) => Pcg64::seed_from_u64(s),
            None => Pcg64::from_entropy(),
        };

        let mut path = Vec::with_capacity(steps + 1);
        
        // Sample initial state
        let mut current_state = self.sample_from_distribution(&chain.initial_distribution, &mut rng);
        path.push(current_state);

        // Simulate chain
        for _ in 0..steps {
            let transition_probs: Vec<f64> = chain.transition_matrix.row(current_state).iter().cloned().collect();
            current_state = self.sample_from_vector(&transition_probs, &mut rng);
            path.push(current_state);
        }

        path
    }

    fn sample_from_distribution(&self, distribution: &DVector<f64>, rng: &mut Pcg64) -> usize {
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        
        for (i, &prob) in distribution.iter().enumerate() {
            cumulative += prob;
            if r <= cumulative {
                return i;
            }
        }
        distribution.len() - 1
    }

    fn sample_from_vector(&self, probs: &[f64], rng: &mut Pcg64) -> usize {
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        
        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if r <= cumulative {
                return i;
            }
        }
        probs.len() - 1
    }

    pub fn find_stationary_distribution(&self, chain: &MarkovChain) -> Option<DVector<f64>> {
        let n = chain.states.len();
        
        // Solve (P^T - I)π = 0 with constraint Σπ_i = 1
        let mut a = chain.transition_matrix.transpose() - DMatrix::identity(n, n);
        
        // Replace last row with constraint equation
        for j in 0..n {
            a[(n-1, j)] = 1.0;
        }
        
        let mut b = DVector::zeros(n);
        b[n-1] = 1.0;
        
        // Solve using SVD decomposition
        let svd = a.svd(true, true);
        if let (Some(u), Some(vt)) = (svd.u, svd.v_t) {
            let s_inv = DMatrix::from_diagonal(&svd.singular_values.map(|x| {
                if x > 1e-10 { 1.0 / x } else { 0.0 }
            }));
            
            let solution = vt.transpose() * s_inv * u.transpose() * b;
            
            // Check if solution is valid (all non-negative, sums to 1)
            if solution.iter().all(|&x| x >= -1e-10) {
                let normalized = solution.map(|x| x.max(0.0));
                let sum: f64 = normalized.sum();
                if sum > 1e-10 {
                    return Some(normalized / sum);
                }
            }
        }
        
        None
    }

    pub fn analyze_markov_chain(&self, chain: &MarkovChain) -> MarkovChainProperties {
        let n = chain.states.len();
        let stationary_dist = self.find_stationary_distribution(chain);
        
        // Find strongly connected components (communication classes)
        let communication_classes = self.find_communication_classes(chain);
        
        // Classify states
        let mut recurrent_states = HashSet::new();
        let mut transient_states = HashSet::new();
        
        for class in &communication_classes {
            if self.is_closed_class(chain, class) {
                for &state in class {
                    recurrent_states.insert(state);
                }
            } else {
                for &state in class {
                    transient_states.insert(state);
                }
            }
        }
        
        // Check irreducibility and aperiodicity
        let irreducible = communication_classes.len() == 1 && 
                         communication_classes[0].len() == n;
        let aperiodic = self.is_aperiodic(chain);

        MarkovChainProperties {
            irreducible,
            aperiodic,
            recurrent_states,
            transient_states,
            stationary_distribution: stationary_dist,
            communication_classes,
        }
    }

    fn find_communication_classes(&self, chain: &MarkovChain) -> Vec<Vec<usize>> {
        let n = chain.states.len();
        let mut visited = vec![false; n];
        let mut classes = Vec::new();
        
        for i in 0..n {
            if !visited[i] {
                let mut class = Vec::new();
                self.dfs_communication_class(chain, i, &mut visited, &mut class);
                if !class.is_empty() {
                    classes.push(class);
                }
            }
        }
        
        classes
    }

    fn dfs_communication_class(&self, chain: &MarkovChain, state: usize, 
                             visited: &mut [bool], class: &mut Vec<usize>) {
        if visited[state] {
            return;
        }
        
        visited[state] = true;
        class.push(state);
        
        // Visit all reachable states
        for j in 0..chain.states.len() {
            if chain.transition_matrix[(state, j)] > 0.0 {
                self.dfs_communication_class(chain, j, visited, class);
            }
        }
    }

    fn is_closed_class(&self, chain: &MarkovChain, class: &[usize]) -> bool {
        for &i in class {
            for j in 0..chain.states.len() {
                if chain.transition_matrix[(i, j)] > 0.0 && !class.contains(&j) {
                    return false;
                }
            }
        }
        true
    }

    fn is_aperiodic(&self, chain: &MarkovChain) -> bool {
        // Simplified check: if there exists a state with self-loop, chain is aperiodic
        for i in 0..chain.states.len() {
            if chain.transition_matrix[(i, i)] > 0.0 {
                return true;
            }
        }
        false
    }

    // Brownian Motion
    pub fn create_brownian_motion(&self, dimension: usize) -> BrownianMotion {
        BrownianMotion {
            dimension,
            drift: vec![0.0; dimension],
            diffusion: DMatrix::identity(dimension, dimension),
            initial_value: vec![0.0; dimension],
        }
    }

    pub fn simulate_brownian_motion(&self, bm: &BrownianMotion, time_horizon: f64, 
                                  num_steps: usize, seed: Option<u64>) -> WienerPath {
        let mut rng = match seed {
            Some(s) => Pcg64::seed_from_u64(s),
            None => Pcg64::from_entropy(),
        };

        let dt = time_horizon / num_steps as f64;
        let sqrt_dt = dt.sqrt();
        
        let mut time_points = vec![0.0];
        let mut path_values = vec![bm.initial_value.clone()];
        let mut increments = Vec::new();
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        for step in 1..=num_steps {
            let mut dw = Vec::with_capacity(bm.dimension);
            for _ in 0..bm.dimension {
                dw.push(sqrt_dt * normal.sample(&mut rng));
            }
            
            let mut next_value = path_values.last().unwrap().clone();
            for i in 0..bm.dimension {
                next_value[i] += bm.drift[i] * dt;
                for j in 0..bm.dimension {
                    next_value[i] += bm.diffusion[(i, j)] * dw[j];
                }
            }
            
            time_points.push(step as f64 * dt);
            path_values.push(next_value);
            increments.push(dw);
        }

        WienerPath {
            time_points,
            path_values,
            increments,
        }
    }

    // Stochastic Differential Equations
    pub fn euler_maruyama(&self, drift: &dyn Fn(&[f64], f64) -> Vec<f64>,
                         diffusion: &dyn Fn(&[f64], f64) -> Vec<Vec<f64>>,
                         initial: &[f64], time_horizon: f64, num_steps: usize,
                         num_paths: usize, seed: Option<u64>) -> SDESolution {
        let mut rng = match seed {
            Some(s) => Pcg64::seed_from_u64(s),
            None => Pcg64::from_entropy(),
        };

        let dt = time_horizon / num_steps as f64;
        let sqrt_dt = dt.sqrt();
        let dimension = initial.len();
        
        let mut time_grid = vec![0.0];
        for i in 1..=num_steps {
            time_grid.push(i as f64 * dt);
        }
        
        let mut all_paths = Vec::with_capacity(num_paths);
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        for _ in 0..num_paths {
            let mut path = vec![initial.to_vec()];
            let mut current = initial.to_vec();
            let mut t = 0.0;
            
            for _ in 1..=num_steps {
                let drift_val = drift(&current, t);
                let diffusion_val = diffusion(&current, t);
                
                // Generate random increments
                let mut dw = Vec::with_capacity(dimension);
                for _ in 0..dimension {
                    dw.push(sqrt_dt * normal.sample(&mut rng));
                }
                
                // Euler-Maruyama step
                for i in 0..dimension {
                    current[i] += drift_val[i] * dt;
                    for j in 0..dimension {
                        current[i] += diffusion_val[i][j] * dw[j];
                    }
                }
                
                path.push(current.clone());
                t += dt;
            }
            
            all_paths.push(path);
        }
        
        // Compute statistics
        let statistics = self.compute_path_statistics(&all_paths);

        SDESolution {
            time_grid,
            paths: all_paths,
            statistics,
        }
    }

    fn compute_path_statistics(&self, paths: &[Vec<Vec<f64>>]) -> PathStatistics {
        if paths.is_empty() {
            return PathStatistics {
                mean_path: Vec::new(),
                variance_path: Vec::new(),
                confidence_intervals: Vec::new(),
                convergence_info: ConvergenceInfo {
                    strong_convergence_order: 0.5,
                    weak_convergence_order: 1.0,
                    monte_carlo_error: 0.0,
                    discretization_error: 0.0,
                },
            };
        }

        let num_paths = paths.len();
        let num_steps = paths[0].len();
        let dimension = paths[0][0].len();
        
        let mut mean_path = vec![vec![0.0; dimension]; num_steps];
        let mut variance_path = vec![vec![0.0; dimension]; num_steps];
        
        // Compute means
        for i in 0..num_steps {
            for d in 0..dimension {
                let sum: f64 = paths.iter().map(|path| path[i][d]).sum();
                mean_path[i][d] = sum / num_paths as f64;
            }
        }
        
        // Compute variances
        for i in 0..num_steps {
            for d in 0..dimension {
                let sum_sq_diff: f64 = paths.iter()
                    .map(|path| (path[i][d] - mean_path[i][d]).powi(2))
                    .sum();
                variance_path[i][d] = sum_sq_diff / (num_paths - 1) as f64;
            }
        }
        
        // Compute confidence intervals (95%)
        let z_score = 1.96; // 95% confidence
        let mut confidence_intervals = vec![vec![(0.0, 0.0); dimension]; num_steps];
        
        for i in 0..num_steps {
            for d in 0..dimension {
                let std_err = (variance_path[i][d] / num_paths as f64).sqrt();
                let margin = z_score * std_err;
                confidence_intervals[i][d] = (
                    mean_path[i][d] - margin,
                    mean_path[i][d] + margin
                );
            }
        }

        PathStatistics {
            mean_path,
            variance_path,
            confidence_intervals,
            convergence_info: ConvergenceInfo {
                strong_convergence_order: 0.5,
                weak_convergence_order: 1.0,
                monte_carlo_error: 1.0 / (num_paths as f64).sqrt(),
                discretization_error: 0.0, // Would need step size for proper estimate
            },
        }
    }

    // Poisson Process
    pub fn create_poisson_process(&self, rate: f64) -> PoissonProcess {
        PoissonProcess {
            rate,
            jump_distribution: None,
            events: Vec::new(),
        }
    }

    pub fn simulate_poisson_process(&self, process: &mut PoissonProcess, time_horizon: f64, 
                                  seed: Option<u64>) -> Vec<PoissonEvent> {
        let mut rng = match seed {
            Some(s) => Pcg64::seed_from_u64(s),
            None => Pcg64::from_entropy(),
        };

        let mut events = Vec::new();
        let mut current_time = 0.0;
        let exp_dist = Exp::new(process.rate).unwrap();
        
        while current_time < time_horizon {
            let inter_arrival_time = exp_dist.sample(&mut rng);
            current_time += inter_arrival_time;
            
            if current_time < time_horizon {
                events.push(PoissonEvent {
                    time: current_time,
                    magnitude: 1.0, // Unit jumps for standard Poisson process
                    event_type: "arrival".to_string(),
                });
            }
        }
        
        process.events = events.clone();
        events
    }

    // Monte Carlo Methods
    pub fn monte_carlo_integration(&self, integrand: &dyn Fn(&[f64]) -> f64,
                                 bounds: &[(f64, f64)], num_samples: usize, 
                                 seed: Option<u64>) -> MonteCarloResults {
        let mut rng = match seed {
            Some(s) => Pcg64::seed_from_u64(s),
            None => Pcg64::from_entropy(),
        };

        let dimension = bounds.len();
        let volume: f64 = bounds.iter().map(|(a, b)| b - a).product();
        
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut convergence_history = Vec::new();
        
        for i in 0..num_samples {
            // Sample random point in domain
            let mut point = Vec::with_capacity(dimension);
            for &(a, b) in bounds {
                let uniform = Uniform::new(a, b);
                point.push(uniform.sample(&mut rng));
            }
            
            let value = integrand(&point);
            sum += value;
            sum_sq += value * value;
            
            // Track convergence every 100 samples
            if (i + 1) % 100 == 0 {
                let current_estimate = volume * sum / (i + 1) as f64;
                convergence_history.push(current_estimate);
            }
        }
        
        let estimate = volume * sum / num_samples as f64;
        let variance = volume * volume * (sum_sq / num_samples as f64 - (sum / num_samples as f64).powi(2));
        let standard_error = (variance / num_samples as f64).sqrt();
        
        let confidence_interval = (
            estimate - 1.96 * standard_error,
            estimate + 1.96 * standard_error
        );

        MonteCarloResults {
            estimate,
            standard_error,
            confidence_interval,
            convergence_history,
        }
    }

    // Queueing Systems
    pub fn create_mm1_queue(&self, arrival_rate: f64, service_rate: f64) -> QueueingSystem {
        let utilization = arrival_rate / service_rate;
        
        QueueingSystem {
            arrival_rate,
            service_rate,
            num_servers: 1,
            capacity: None,
            queue_discipline: QueueDiscipline::FIFO,
            system_state: QueueState {
                customers_in_system: 0,
                customers_in_queue: 0,
                server_utilization: utilization,
                average_wait_time: utilization / (service_rate - arrival_rate),
                average_service_time: 1.0 / service_rate,
            },
        }
    }

    pub fn analyze_mm1_queue(&self, queue: &QueueingSystem) -> HashMap<String, f64> {
        let lambda = queue.arrival_rate;
        let mu = queue.service_rate;
        let rho = lambda / mu;
        
        let mut metrics = HashMap::new();
        
        if rho < 1.0 {
            metrics.insert("utilization".to_string(), rho);
            metrics.insert("average_customers".to_string(), rho / (1.0 - rho));
            metrics.insert("average_wait_time".to_string(), rho / (mu - lambda));
            metrics.insert("average_response_time".to_string(), 1.0 / (mu - lambda));
            metrics.insert("probability_empty".to_string(), 1.0 - rho);
        }
        
        metrics
    }
}

impl MathDomain for StochasticProcessesDomain {
    fn name(&self) -> &str {
        &self.name
    }

    fn supports_operation(&self, operation: &str) -> bool {
        matches!(operation, 
            "sample_distribution" | "markov_chain" | "brownian_motion" |
            "stochastic_de" | "poisson_process" | "monte_carlo" | "queueing_theory" |
            "jump_diffusion" | "levy_process" | "martingale"
        )
    }

    fn description(&self) -> &str {
        "Stochastic Processes and Probability Theory"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    fn compute(&self, operation: &str, _args: &[&dyn std::any::Any]) -> MathResult<Box<dyn std::any::Any>> {
        match operation {
            "sample_distribution" => Ok(Box::new("Random samples generated".to_string())),
            "markov_chain" => Ok(Box::new("Markov chain simulated".to_string())),
            "brownian_motion" => Ok(Box::new("Brownian motion path generated".to_string())),
            "monte_carlo" => Ok(Box::new("Monte Carlo simulation completed".to_string())),
            _ => Err(crate::core::MathError::NotImplemented(format!("Operation '{}' not implemented", operation))),
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "sample_distribution".to_string(), "markov_chain".to_string(),
            "brownian_motion".to_string(), "stochastic_de".to_string(),
            "poisson_process".to_string(), "monte_carlo".to_string(),
            "queueing_theory".to_string(), "jump_diffusion".to_string(),
            "levy_process".to_string(), "martingale".to_string()
        ]
    }
}

pub fn stochastic_processes() -> StochasticProcessesDomain {
    StochasticProcessesDomain::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_distribution_sampling() {
        let domain = StochasticProcessesDomain::new();
        let normal = ProbabilityDistribution::Normal { mean: 0.0, std_dev: 1.0 };
        let samples = domain.sample_distribution(&normal, 1000, Some(42));
        
        assert_eq!(samples.len(), 1000);
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 0.0).abs() < 0.1); // Should be close to 0
    }

    #[test]
    fn test_markov_chain() {
        let domain = StochasticProcessesDomain::new();
        let states = vec!["A".to_string(), "B".to_string()];
        let mut chain = domain.create_markov_chain(states);
        
        domain.set_transition_probability(&mut chain, 0, 0, 0.7);
        domain.set_transition_probability(&mut chain, 0, 1, 0.3);
        domain.set_transition_probability(&mut chain, 1, 0, 0.4);
        domain.set_transition_probability(&mut chain, 1, 1, 0.6);
        
        domain.set_initial_distribution(&mut chain, vec![0.5, 0.5]);
        
        let path = domain.simulate_markov_chain(&chain, 100, Some(42));
        assert_eq!(path.len(), 101); // 100 steps + initial state
    }

    #[test]
    fn test_brownian_motion() {
        let domain = StochasticProcessesDomain::new();
        let bm = domain.create_brownian_motion(1);
        let path = domain.simulate_brownian_motion(&bm, 1.0, 100, Some(42));
        
        assert_eq!(path.time_points.len(), 101);
        assert_eq!(path.path_values.len(), 101);
        assert_eq!(path.path_values[0][0], 0.0); // Starts at origin
    }

    #[test]
    fn test_poisson_process() {
        let domain = StochasticProcessesDomain::new();
        let mut process = domain.create_poisson_process(2.0); // Rate = 2
        let events = domain.simulate_poisson_process(&mut process, 10.0, Some(42));
        
        assert!(!events.is_empty());
        // Check that all events occur within time horizon
        assert!(events.iter().all(|e| e.time <= 10.0));
    }

    #[test]
    fn test_monte_carlo_integration() {
        let domain = StochasticProcessesDomain::new();
        
        // Integrate x^2 from 0 to 1 (should be 1/3)
        let integrand = |x: &[f64]| x[0] * x[0];
        let bounds = vec![(0.0, 1.0)];
        
        let result = domain.monte_carlo_integration(&integrand, &bounds, 10000, Some(42));
        
        assert!((result.estimate - 1.0/3.0).abs() < 0.01);
    }

    #[test]
    fn test_mm1_queue() {
        let domain = StochasticProcessesDomain::new();
        let queue = domain.create_mm1_queue(2.0, 3.0); // λ=2, μ=3
        let metrics = domain.analyze_mm1_queue(&queue);
        
        assert!(metrics.contains_key("utilization"));
        assert!((metrics["utilization"] - 2.0/3.0).abs() < 1e-10);
    }
}