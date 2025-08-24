use crate::core::types::*;
use std::f64::consts::{E, PI};

#[derive(Clone, Debug)]
pub struct BiologyDomain;

impl BiologyDomain {
    pub fn new() -> Self {
        Self
    }

    pub fn calculate_growth_rate(&self, initial_size: f64, final_size: f64, time: f64) -> MathResult<f64> {
        if time <= 0.0 {
            return Err(MathError::InvalidArgument("Time must be positive".to_string()));
        }
        if initial_size <= 0.0 {
            return Err(MathError::InvalidArgument("Initial size must be positive".to_string()));
        }
        
        Ok((final_size / initial_size).ln() / time)
    }

    pub fn exponential_growth(&self, initial_population: f64, growth_rate: f64, time: f64) -> f64 {
        initial_population * E.powf(growth_rate * time)
    }

    pub fn logistic_growth(&self, initial_population: f64, carrying_capacity: f64, 
                          growth_rate: f64, time: f64) -> f64 {
        carrying_capacity / (1.0 + ((carrying_capacity / initial_population - 1.0) * E.powf(-growth_rate * time)))
    }
}

pub fn exponential_growth(initial_population: f64, growth_rate: f64, time: f64) -> f64 {
    initial_population * E.powf(growth_rate * time)
}

pub fn logistic_growth(initial_population: f64, carrying_capacity: f64, 
                      growth_rate: f64, time: f64) -> f64 {
    carrying_capacity / (1.0 + ((carrying_capacity / initial_population - 1.0) * E.powf(-growth_rate * time)))
}

pub fn gompertz_growth(initial_population: f64, carrying_capacity: f64, 
                      growth_rate: f64, time: f64) -> f64 {
    carrying_capacity * E.powf((initial_population / carrying_capacity).ln() * E.powf(-growth_rate * time))
}

pub fn calculate_doubling_time(growth_rate: f64) -> MathResult<f64> {
    if growth_rate <= 0.0 {
        return Err(MathError::InvalidArgument("Growth rate must be positive".to_string()));
    }
    Ok(2.0_f64.ln() / growth_rate)
}

pub fn calculate_half_life(decay_rate: f64) -> MathResult<f64> {
    if decay_rate <= 0.0 {
        return Err(MathError::InvalidArgument("Decay rate must be positive".to_string()));
    }
    Ok(2.0_f64.ln() / decay_rate)
}

pub fn l_system_expand(axiom: &str, rules: &std::collections::HashMap<char, String>, 
                      iterations: usize) -> String {
    let mut current = axiom.to_string();
    
    for _ in 0..iterations {
        let mut next = String::new();
        for c in current.chars() {
            if let Some(replacement) = rules.get(&c) {
                next.push_str(replacement);
            } else {
                next.push(c);
            }
        }
        current = next;
    }
    
    current
}

pub fn turtle_interpret_length(symbol: char, current_length: f64, 
                              length_factor: f64) -> f64 {
    match symbol {
        'F' | 'G' | 'f' => current_length,
        '[' | ']' => current_length,
        '+' | '-' | '&' | '^' | '\\' | '/' | '|' => current_length,
        _ => current_length * length_factor,
    }
}

pub fn turtle_interpret_angle(symbol: char, current_angle: f64, 
                             angle_increment: f64) -> f64 {
    match symbol {
        '+' => current_angle + angle_increment,
        '-' => current_angle - angle_increment,
        '&' => current_angle + angle_increment,  // pitch down
        '^' => current_angle - angle_increment,  // pitch up
        '\\' => current_angle + angle_increment, // roll left
        '/' => current_angle - angle_increment,  // roll right
        '|' => current_angle + PI,               // turn around
        _ => current_angle,
    }
}

pub fn reaction_diffusion_step(u: &[f64], v: &[f64], du: f64, dv: f64, 
                              f: f64, k: f64, dt: f64, dx: f64) -> (Vec<f64>, Vec<f64>) {
    let n = u.len();
    let mut u_new = vec![0.0; n];
    let mut v_new = vec![0.0; n];
    
    let laplacian_factor = dt / (dx * dx);
    
    for i in 1..n-1 {
        let u_laplacian = u[i-1] - 2.0 * u[i] + u[i+1];
        let v_laplacian = v[i-1] - 2.0 * v[i] + v[i+1];
        
        let reaction_u = -u[i] * v[i] * v[i] + f * (1.0 - u[i]);
        let reaction_v = u[i] * v[i] * v[i] - (f + k) * v[i];
        
        u_new[i] = u[i] + du * laplacian_factor * u_laplacian + dt * reaction_u;
        v_new[i] = v[i] + dv * laplacian_factor * v_laplacian + dt * reaction_v;
    }
    
    u_new[0] = u_new[1];
    u_new[n-1] = u_new[n-2];
    v_new[0] = v_new[1];
    v_new[n-1] = v_new[n-2];
    
    (u_new, v_new)
}

pub fn turing_pattern_stability(du: f64, dv: f64, a: f64, b: f64) -> f64 {
    let discriminant = (a + b).powi(2) - 4.0 * (a * b - du * dv);
    if discriminant >= 0.0 {
        let lambda_max = ((a + b) + discriminant.sqrt()) / 2.0;
        lambda_max
    } else {
        0.0
    }
}

pub fn hodgkin_huxley_voltage(v: f64, n: f64, m: f64, h: f64, i_ext: f64, dt: f64) -> f64 {
    let c_m = 1.0;  // membrane capacitance (μF/cm²)
    let g_na = 120.0;  // sodium conductance (mS/cm²)
    let g_k = 36.0;    // potassium conductance (mS/cm²)
    let g_l = 0.3;     // leak conductance (mS/cm²)
    let e_na = 50.0;   // sodium reversal potential (mV)
    let e_k = -77.0;   // potassium reversal potential (mV)
    let e_l = -54.387; // leak reversal potential (mV)
    
    let i_na = g_na * m.powi(3) * h * (v - e_na);
    let i_k = g_k * n.powi(4) * (v - e_k);
    let i_l = g_l * (v - e_l);
    
    v + dt * (i_ext - i_na - i_k - i_l) / c_m
}

pub fn hodgkin_huxley_gate_n(v: f64, n: f64, dt: f64) -> f64 {
    let alpha_n = 0.01 * (v + 55.0) / (1.0 - E.powf(-(v + 55.0) / 10.0));
    let beta_n = 0.125 * E.powf(-(v + 65.0) / 80.0);
    let tau_n = 1.0 / (alpha_n + beta_n);
    let n_inf = alpha_n / (alpha_n + beta_n);
    
    n + dt * (n_inf - n) / tau_n
}

pub fn hodgkin_huxley_gate_m(v: f64, m: f64, dt: f64) -> f64 {
    let alpha_m = 0.1 * (v + 40.0) / (1.0 - E.powf(-(v + 40.0) / 10.0));
    let beta_m = 4.0 * E.powf(-(v + 65.0) / 18.0);
    let tau_m = 1.0 / (alpha_m + beta_m);
    let m_inf = alpha_m / (alpha_m + beta_m);
    
    m + dt * (m_inf - m) / tau_m
}

pub fn hodgkin_huxley_gate_h(v: f64, h: f64, dt: f64) -> f64 {
    let alpha_h = 0.07 * E.powf(-(v + 65.0) / 20.0);
    let beta_h = 1.0 / (1.0 + E.powf(-(v + 35.0) / 10.0));
    let tau_h = 1.0 / (alpha_h + beta_h);
    let h_inf = alpha_h / (alpha_h + beta_h);
    
    h + dt * (h_inf - h) / tau_h
}

pub fn cellular_automaton_step(grid: &[Vec<u8>], rule: fn(u8, Vec<u8>) -> u8) -> Vec<Vec<u8>> {
    let rows = grid.len();
    let cols = grid[0].len();
    let mut new_grid = vec![vec![0; cols]; rows];
    
    for i in 0..rows {
        for j in 0..cols {
            let mut neighbors = Vec::new();
            
            for di in -1..=1 {
                for dj in -1..=1 {
                    if di == 0 && dj == 0 { continue; }
                    
                    let ni = (i as i32 + di) as usize;
                    let nj = (j as i32 + dj) as usize;
                    
                    if ni < rows && nj < cols {
                        neighbors.push(grid[ni][nj]);
                    }
                }
            }
            
            new_grid[i][j] = rule(grid[i][j], neighbors);
        }
    }
    
    new_grid
}

pub fn conway_life_rule(current_state: u8, neighbors: Vec<u8>) -> u8 {
    let alive_neighbors: u8 = neighbors.iter().sum();
    
    match (current_state, alive_neighbors) {
        (1, 2) | (1, 3) => 1,  // survive
        (0, 3) => 1,           // birth
        _ => 0,                // death or stay dead
    }
}

pub fn michaelis_menten_velocity(substrate_conc: f64, vmax: f64, km: f64) -> MathResult<f64> {
    if km < 0.0 {
        return Err(MathError::InvalidArgument("Michaelis constant must be non-negative".to_string()));
    }
    if vmax < 0.0 {
        return Err(MathError::InvalidArgument("Maximum velocity must be non-negative".to_string()));
    }
    
    Ok((vmax * substrate_conc) / (km + substrate_conc))
}

pub fn competitive_inhibition_velocity(substrate_conc: f64, inhibitor_conc: f64, 
                                     vmax: f64, km: f64, ki: f64) -> MathResult<f64> {
    if km < 0.0 || ki < 0.0 {
        return Err(MathError::InvalidArgument("Kinetic constants must be non-negative".to_string()));
    }
    if vmax < 0.0 {
        return Err(MathError::InvalidArgument("Maximum velocity must be non-negative".to_string()));
    }
    
    let apparent_km = km * (1.0 + inhibitor_conc / ki);
    Ok((vmax * substrate_conc) / (apparent_km + substrate_conc))
}

pub fn hill_equation(ligand_conc: f64, dissociation_const: f64, hill_coeff: f64) -> MathResult<f64> {
    if dissociation_const <= 0.0 {
        return Err(MathError::InvalidArgument("Dissociation constant must be positive".to_string()));
    }
    if hill_coeff <= 0.0 {
        return Err(MathError::InvalidArgument("Hill coefficient must be positive".to_string()));
    }
    
    let numerator = ligand_conc.powf(hill_coeff);
    let denominator = dissociation_const.powf(hill_coeff) + numerator;
    
    Ok(numerator / denominator)
}

pub fn lotka_volterra_predator(predator: f64, prey: f64, gamma: f64, delta: f64, dt: f64) -> f64 {
    predator + dt * (gamma * prey * predator - delta * predator)
}

pub fn lotka_volterra_prey(predator: f64, prey: f64, alpha: f64, beta: f64, dt: f64) -> f64 {
    prey + dt * (alpha * prey - beta * prey * predator)
}

pub fn population_carrying_capacity(birth_rate: f64, death_rate: f64, 
                                  environment_resistance: f64) -> MathResult<f64> {
    if death_rate >= birth_rate {
        return Err(MathError::InvalidArgument("Birth rate must exceed death rate for positive carrying capacity".to_string()));
    }
    if environment_resistance <= 0.0 {
        return Err(MathError::InvalidArgument("Environment resistance must be positive".to_string()));
    }
    
    Ok((birth_rate - death_rate) / environment_resistance)
}

pub fn selection_coefficient(fitness_mutant: f64, fitness_wildtype: f64) -> MathResult<f64> {
    if fitness_wildtype <= 0.0 {
        return Err(MathError::InvalidArgument("Wild-type fitness must be positive".to_string()));
    }
    
    Ok((fitness_mutant - fitness_wildtype) / fitness_wildtype)
}

pub fn allele_frequency_change(frequency: f64, selection_coeff: f64, dominance: f64) -> f64 {
    let p = frequency;
    let q = 1.0 - frequency;
    let s = selection_coeff;
    let h = dominance;
    
    let w_bar = 1.0 - s * (2.0 * p * q * h + q * q);
    let delta_p = (p * q * s * (p * (1.0 - 2.0 * h) + q * h)) / w_bar;
    
    delta_p
}

pub fn hardy_weinberg_genotype_freq(allele_freq_p: f64, allele_freq_q: f64) -> (f64, f64, f64) {
    let aa = allele_freq_p * allele_freq_p;
    let ab = 2.0 * allele_freq_p * allele_freq_q;
    let bb = allele_freq_q * allele_freq_q;
    
    (aa, ab, bb)
}

pub fn shannon_diversity_index(species_counts: &[f64]) -> MathResult<f64> {
    let total: f64 = species_counts.iter().sum();
    
    if total <= 0.0 {
        return Err(MathError::InvalidArgument("Total count must be positive".to_string()));
    }
    
    let mut diversity = 0.0;
    for &count in species_counts {
        if count > 0.0 {
            let proportion = count / total;
            diversity -= proportion * proportion.ln();
        }
    }
    
    Ok(diversity)
}

pub fn simpson_diversity_index(species_counts: &[f64]) -> MathResult<f64> {
    let total: f64 = species_counts.iter().sum();
    
    if total <= 0.0 {
        return Err(MathError::InvalidArgument("Total count must be positive".to_string()));
    }
    
    let mut simpson = 0.0;
    for &count in species_counts {
        if count > 0.0 {
            let proportion = count / total;
            simpson += proportion * proportion;
        }
    }
    
    Ok(1.0 - simpson)
}

pub fn calculate_biomass_pyramid(trophic_levels: &[f64], efficiency: f64) -> MathResult<Vec<f64>> {
    if efficiency <= 0.0 || efficiency > 1.0 {
        return Err(MathError::InvalidArgument("Efficiency must be between 0 and 1".to_string()));
    }
    
    let mut pyramid = vec![trophic_levels[0]];
    
    for i in 1..trophic_levels.len() {
        let biomass = pyramid[i-1] * efficiency;
        pyramid.push(biomass);
    }
    
    Ok(pyramid)
}

pub fn metabolic_rate_allometric(body_mass: f64, scaling_constant: f64, 
                                scaling_exponent: f64) -> MathResult<f64> {
    if body_mass <= 0.0 {
        return Err(MathError::InvalidArgument("Body mass must be positive".to_string()));
    }
    if scaling_constant <= 0.0 {
        return Err(MathError::InvalidArgument("Scaling constant must be positive".to_string()));
    }
    
    Ok(scaling_constant * body_mass.powf(scaling_exponent))
}

pub fn cell_division_rate(doubling_time_hours: f64) -> MathResult<f64> {
    if doubling_time_hours <= 0.0 {
        return Err(MathError::InvalidArgument("Doubling time must be positive".to_string()));
    }
    
    Ok(2.0_f64.ln() / doubling_time_hours)
}

pub fn protein_folding_energy(hydrophobic_contacts: u32, hydrogen_bonds: u32, 
                             disulfide_bonds: u32) -> f64 {
    let hydrophobic_energy = -1.5; // kcal/mol per contact
    let hydrogen_energy = -2.0;    // kcal/mol per bond
    let disulfide_energy = -15.0;  // kcal/mol per bond
    
    (hydrophobic_contacts as f64) * hydrophobic_energy +
    (hydrogen_bonds as f64) * hydrogen_energy +
    (disulfide_bonds as f64) * disulfide_energy
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_exponential_growth() {
        let result = exponential_growth(100.0, 0.1, 10.0);
        assert!((result - 271.828).abs() < 0.1);
    }
    
    #[test]
    fn test_logistic_growth() {
        let result = logistic_growth(10.0, 1000.0, 0.1, 20.0);
        assert!(result > 10.0 && result < 1000.0);
    }
    
    #[test]
    fn test_l_system_basic() {
        let mut rules = HashMap::new();
        rules.insert('A', "AB".to_string());
        rules.insert('B', "A".to_string());
        
        let result = l_system_expand("A", &rules, 3);
        assert_eq!(result, "ABAAB");
    }
    
    #[test]
    fn test_michaelis_menten() {
        let result = michaelis_menten_velocity(5.0, 10.0, 2.0).unwrap();
        let expected = (10.0 * 5.0) / (2.0 + 5.0);
        assert!((result - expected).abs() < 1e-10);
    }
    
    #[test]
    fn test_lotka_volterra() {
        let predator = 10.0;
        let prey = 50.0;
        let alpha = 0.1;
        let beta = 0.02;
        let gamma = 0.01;
        let delta = 0.1;
        let dt = 0.1;
        
        let new_prey = lotka_volterra_prey(predator, prey, alpha, beta, dt);
        let new_predator = lotka_volterra_predator(predator, prey, gamma, delta, dt);
        
        assert!(new_prey != prey);
        assert!(new_predator != predator);
    }
    
    #[test]
    fn test_shannon_diversity() {
        let species_counts = vec![30.0, 20.0, 15.0, 10.0, 5.0];
        let result = shannon_diversity_index(&species_counts).unwrap();
        assert!(result > 0.0 && result < 2.0);
    }
    
    #[test]
    fn test_hardy_weinberg() {
        let (aa, ab, bb) = hardy_weinberg_genotype_freq(0.6, 0.4);
        assert!((aa + ab + bb - 1.0).abs() < 1e-10);
        assert!((aa - 0.36).abs() < 1e-10);
        assert!((ab - 0.48).abs() < 1e-10);
        assert!((bb - 0.16).abs() < 1e-10);
    }
    
    #[test]
    fn test_reaction_diffusion() {
        let u = vec![1.0, 0.5, 0.0, 0.5, 1.0];
        let v = vec![0.0, 0.1, 0.5, 0.1, 0.0];
        
        let (u_new, v_new) = reaction_diffusion_step(&u, &v, 2e-5, 1e-5, 0.054, 0.062, 1.0, 1.0);
        
        assert_eq!(u_new.len(), u.len());
        assert_eq!(v_new.len(), v.len());
    }
    
    #[test]
    fn test_hodgkin_huxley() {
        let v = -65.0;
        let n = 0.318;
        let m = 0.053;
        let h = 0.596;
        let i_ext = 0.0;
        let dt = 0.01;
        
        let new_v = hodgkin_huxley_voltage(v, n, m, h, i_ext, dt);
        assert!((new_v - v).abs() < 1.0);
    }
    
    #[test]
    fn test_conway_life() {
        let neighbors_survive = vec![1, 1, 0, 0, 0, 0, 0, 0];
        let neighbors_birth = vec![1, 1, 1, 0, 0, 0, 0, 0];
        let neighbors_death = vec![1, 0, 0, 0, 0, 0, 0, 0];
        
        assert_eq!(conway_life_rule(1, neighbors_survive), 1);
        assert_eq!(conway_life_rule(0, neighbors_birth), 1);
        assert_eq!(conway_life_rule(1, neighbors_death), 0);
    }
    
    #[test]
    fn test_protein_folding() {
        let energy = protein_folding_energy(10, 5, 2);
        let expected = 10.0 * (-1.5) + 5.0 * (-2.0) + 2.0 * (-15.0);
        assert!((energy - expected).abs() < 1e-10);
    }
}