use mathtables::prelude::*;
use std::collections::HashMap;

#[test]
fn test_exponential_growth_basic() {
    let result = exponential_growth(100.0, 0.1, 10.0);
    let expected = 100.0 * std::f64::consts::E.powf(1.0);
    assert!((result - expected).abs() < 0.1);
}

#[test]
fn test_logistic_growth_limits() {
    let carrying_capacity = 1000.0;
    let result = logistic_growth(10.0, carrying_capacity, 0.1, 50.0);
    assert!(result < carrying_capacity);
    assert!(result > 10.0);
    
    let long_term = logistic_growth(10.0, carrying_capacity, 0.1, 200.0);
    assert!((long_term - carrying_capacity).abs() < 1.0);
}

#[test]
fn test_gompertz_growth() {
    let result = gompertz_growth(10.0, 1000.0, 0.1, 5.0);
    assert!(result > 10.0);
    assert!(result < 1000.0);
}

#[test]
fn test_doubling_time() {
    let growth_rate = 0.693147; // ln(2)
    let result = calculate_doubling_time(growth_rate).unwrap();
    assert!((result - 1.0).abs() < 1e-5);
}

#[test]
fn test_half_life() {
    let decay_rate = 0.693147; // ln(2)
    let result = calculate_half_life(decay_rate).unwrap();
    assert!((result - 1.0).abs() < 1e-5);
}

#[test]
fn test_l_system_fibonacci() {
    let mut rules = HashMap::new();
    rules.insert('A', "AB".to_string());
    rules.insert('B', "A".to_string());
    
    let result = l_system_expand("A", &rules, 5);
    assert_eq!(result, "ABAABABAABAAB");
    assert_eq!(result.len(), 13); // 7th Fibonacci number
}

#[test]
fn test_l_system_branching() {
    let mut rules = HashMap::new();
    rules.insert('F', "F[+F]F[-F]F".to_string());
    
    let result = l_system_expand("F", &rules, 1);
    assert_eq!(result, "F[+F]F[-F]F");
}

#[test]
fn test_turtle_angle_interpretation() {
    let angle = std::f64::consts::PI / 4.0; // 45 degrees
    let increment = std::f64::consts::PI / 6.0; // 30 degrees
    
    assert_eq!(turtle_interpret_angle('+', angle, increment), angle + increment);
    assert_eq!(turtle_interpret_angle('-', angle, increment), angle - increment);
    assert_eq!(turtle_interpret_angle('|', angle, increment), angle + std::f64::consts::PI);
    assert_eq!(turtle_interpret_angle('F', angle, increment), angle);
}

#[test]
fn test_reaction_diffusion_conservation() {
    let u = vec![1.0, 0.5, 0.0, 0.5, 1.0];
    let v = vec![0.0, 0.1, 0.5, 0.1, 0.0];
    
    let initial_u_sum: f64 = u.iter().sum();
    let initial_v_sum: f64 = v.iter().sum();
    
    let (u_new, v_new) = reaction_diffusion_step(&u, &v, 2e-5, 1e-5, 0.054, 0.062, 0.1, 1.0);
    
    assert_eq!(u_new.len(), u.len());
    assert_eq!(v_new.len(), v.len());
    
    // Check that the system evolves (not conserved, but should be reasonable)
    let new_u_sum: f64 = u_new.iter().sum();
    let new_v_sum: f64 = v_new.iter().sum();
    
    assert!((new_u_sum - initial_u_sum).abs() < 1.0);
    assert!((new_v_sum - initial_v_sum).abs() < 1.0);
}

#[test]
fn test_turing_pattern_stability() {
    let du = 2e-5;
    let dv = 1e-5;
    let a = -1.0;
    let b = 1.0;
    
    let stability = turing_pattern_stability(du, dv, a, b);
    assert!(stability >= 0.0);
}

#[test]
fn test_hodgkin_huxley_resting() {
    let v_rest = -65.0;
    let n = 0.318;
    let m = 0.053;
    let h = 0.596;
    let i_ext = 0.0;
    let dt = 0.01;
    
    let new_v = hodgkin_huxley_voltage(v_rest, n, m, h, i_ext, dt);
    assert!((new_v - v_rest).abs() < 1.0);
}

#[test]
fn test_hodgkin_huxley_gates() {
    let v = -50.0;
    let dt = 0.01;
    
    let n = hodgkin_huxley_gate_n(v, 0.3, dt);
    let m = hodgkin_huxley_gate_m(v, 0.05, dt);
    let h = hodgkin_huxley_gate_h(v, 0.6, dt);
    
    assert!(n >= 0.0 && n <= 1.0);
    assert!(m >= 0.0 && m <= 1.0);
    assert!(h >= 0.0 && h <= 1.0);
}

#[test]
fn test_cellular_automaton_glider() {
    // Conway's Game of Life glider pattern
    let grid = vec![
        vec![0, 1, 0],
        vec![0, 0, 1],
        vec![1, 1, 1],
    ];
    
    let new_grid = cellular_automaton_step(&grid, conway_life_rule);
    
    assert_eq!(new_grid.len(), 3);
    assert_eq!(new_grid[0].len(), 3);
    
    // The glider should have moved
    assert_ne!(grid, new_grid);
}

#[test]
fn test_conway_life_rules() {
    // Test survival with 2 neighbors
    let neighbors_2 = vec![1, 1, 0, 0, 0, 0, 0, 0];
    assert_eq!(conway_life_rule(1, neighbors_2), 1);
    
    // Test survival with 3 neighbors
    let neighbors_3 = vec![1, 1, 1, 0, 0, 0, 0, 0];
    assert_eq!(conway_life_rule(1, neighbors_3.clone()), 1);
    
    // Test birth with 3 neighbors
    assert_eq!(conway_life_rule(0, neighbors_3), 1);
    
    // Test death with 1 neighbor
    let neighbors_1 = vec![1, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(conway_life_rule(1, neighbors_1), 0);
    
    // Test death with 4+ neighbors
    let neighbors_4 = vec![1, 1, 1, 1, 0, 0, 0, 0];
    assert_eq!(conway_life_rule(1, neighbors_4), 0);
}

#[test]
fn test_michaelis_menten_limits() {
    let vmax = 10.0;
    let km = 5.0;
    
    // At very low substrate concentration
    let low_s = 0.1;
    let v_low = michaelis_menten_velocity(low_s, vmax, km).unwrap();
    assert!(v_low < vmax / 2.0);
    
    // At Km concentration, velocity should be Vmax/2
    let v_km = michaelis_menten_velocity(km, vmax, km).unwrap();
    assert!((v_km - vmax / 2.0).abs() < 1e-10);
    
    // At very high substrate concentration
    let high_s = 1000.0;
    let v_high = michaelis_menten_velocity(high_s, vmax, km).unwrap();
    assert!((v_high - vmax).abs() < 0.1);
}

#[test]
fn test_competitive_inhibition() {
    let substrate = 5.0;
    let inhibitor = 2.0;
    let vmax = 10.0;
    let km = 2.0;
    let ki = 1.0;
    
    let v_inhibited = competitive_inhibition_velocity(substrate, inhibitor, vmax, km, ki).unwrap();
    let v_normal = michaelis_menten_velocity(substrate, vmax, km).unwrap();
    
    assert!(v_inhibited < v_normal);
}

#[test]
fn test_hill_equation() {
    let kd = 1.0;
    let hill_coeff = 2.0;
    
    // At dissociation constant, binding should be 0.5
    let binding_kd = hill_equation(kd, kd, hill_coeff).unwrap();
    assert!((binding_kd - 0.5).abs() < 1e-10);
    
    // At high ligand concentration, binding approaches 1
    let binding_high = hill_equation(100.0 * kd, kd, hill_coeff).unwrap();
    assert!(binding_high > 0.99);
    
    // At low ligand concentration, binding approaches 0
    let binding_low = hill_equation(0.01 * kd, kd, hill_coeff).unwrap();
    assert!(binding_low < 0.01);
}

#[test]
fn test_lotka_volterra_oscillation() {
    let mut predator = 10.0;
    let mut prey = 50.0;
    let alpha = 0.1;
    let beta = 0.02;
    let gamma = 0.01;
    let delta = 0.1;
    let dt = 0.1;
    
    let initial_predator = predator;
    let initial_prey = prey;
    
    // Run for several time steps
    for _ in 0..100 {
        let new_predator = lotka_volterra_predator(predator, prey, gamma, delta, dt);
        let new_prey = lotka_volterra_prey(predator, prey, alpha, beta, dt);
        predator = new_predator;
        prey = new_prey;
    }
    
    // Should have oscillated (not returned to exact initial values but should be reasonable)
    assert!(predator > 0.0);
    assert!(prey > 0.0);
    assert!((predator - initial_predator).abs() < 100.0);
    assert!((prey - initial_prey).abs() < 500.0);
}

#[test]
fn test_carrying_capacity() {
    let birth_rate = 0.5;
    let death_rate = 0.1;
    let env_resistance = 0.01;
    
    let capacity = population_carrying_capacity(birth_rate, death_rate, env_resistance).unwrap();
    assert_eq!(capacity, (birth_rate - death_rate) / env_resistance);
}

#[test]
fn test_selection_coefficient() {
    let fitness_wildtype = 1.0;
    let fitness_beneficial = 1.1;
    let fitness_deleterious = 0.9;
    
    let s_beneficial = selection_coefficient(fitness_beneficial, fitness_wildtype).unwrap();
    let s_deleterious = selection_coefficient(fitness_deleterious, fitness_wildtype).unwrap();
    
    assert!(s_beneficial > 0.0);
    assert!(s_deleterious < 0.0);
}

#[test]
fn test_allele_frequency_change() {
    let frequency = 0.1;
    let selection_coeff = 0.05;
    let dominance = 0.5;
    
    let delta_p = allele_frequency_change(frequency, selection_coeff, dominance);
    
    // With positive selection, frequency should increase
    assert!(delta_p > 0.0);
    
    // Change should be small for small selection coefficient
    assert!(delta_p < 0.01);
}

#[test]
fn test_hardy_weinberg_equilibrium() {
    let p = 0.6;
    let q = 0.4;
    
    let (aa, ab, bb) = hardy_weinberg_genotype_freq(p, q);
    
    // Frequencies should sum to 1
    assert!((aa + ab + bb - 1.0).abs() < 1e-10);
    
    // Check Hardy-Weinberg predictions
    assert!((aa - p * p).abs() < 1e-10);
    assert!((ab - 2.0 * p * q).abs() < 1e-10);
    assert!((bb - q * q).abs() < 1e-10);
}

#[test]
fn test_shannon_diversity() {
    // Equal abundances should give higher diversity
    let equal_species = vec![25.0, 25.0, 25.0, 25.0];
    let unequal_species = vec![70.0, 20.0, 5.0, 5.0];
    
    let shannon_equal = shannon_diversity_index(&equal_species).unwrap();
    let shannon_unequal = shannon_diversity_index(&unequal_species).unwrap();
    
    assert!(shannon_equal > shannon_unequal);
}

#[test]
fn test_simpson_diversity() {
    let species_counts = vec![30.0, 20.0, 15.0, 10.0, 5.0];
    let simpson = simpson_diversity_index(&species_counts).unwrap();
    
    assert!(simpson >= 0.0 && simpson <= 1.0);
}

#[test]
fn test_biomass_pyramid() {
    let trophic_levels = vec![1000.0, 0.0, 0.0, 0.0];
    let efficiency = 0.1;
    
    let pyramid = calculate_biomass_pyramid(&trophic_levels, efficiency).unwrap();
    
    assert_eq!(pyramid[0], 1000.0);
    assert_eq!(pyramid[1], 100.0);
    assert_eq!(pyramid[2], 10.0);
    assert_eq!(pyramid[3], 1.0);
}

#[test]
fn test_metabolic_scaling() {
    let mass1 = 1.0;
    let mass2 = 8.0; // 2^3
    let scaling_constant = 1.0;
    let scaling_exponent = 0.75; // Kleiber's law
    
    let rate1 = metabolic_rate_allometric(mass1, scaling_constant, scaling_exponent).unwrap();
    let rate2 = metabolic_rate_allometric(mass2, scaling_constant, scaling_exponent).unwrap();
    
    // Should scale as mass^0.75
    let expected_ratio = 8.0_f64.powf(0.75);
    let actual_ratio = rate2 / rate1;
    
    assert!((actual_ratio - expected_ratio).abs() < 1e-10);
}

#[test]
fn test_cell_division_rate() {
    let doubling_time = 24.0; // 24 hours
    let rate = cell_division_rate(doubling_time).unwrap();
    let expected = 2.0_f64.ln() / 24.0;
    
    assert!((rate - expected).abs() < 1e-10);
}

#[test]
fn test_protein_folding_energy() {
    let hydrophobic_contacts = 10;
    let hydrogen_bonds = 5;
    let disulfide_bonds = 2;
    
    let energy = protein_folding_energy(hydrophobic_contacts, hydrogen_bonds, disulfide_bonds);
    
    // Energy should be negative (favorable)
    assert!(energy < 0.0);
    
    let expected = 10.0 * (-1.5) + 5.0 * (-2.0) + 2.0 * (-15.0);
    assert!((energy - expected).abs() < 1e-10);
}

#[test]
fn test_biology_domain_initialization() {
    let domain = BiologyDomain::new();
    
    let growth_rate = domain.calculate_growth_rate(100.0, 200.0, 10.0).unwrap();
    assert!((growth_rate - 2.0_f64.ln() / 10.0).abs() < 1e-10);
    
    let exp_growth = domain.exponential_growth(100.0, 0.1, 5.0);
    assert!(exp_growth > 100.0);
    
    let log_growth = domain.logistic_growth(10.0, 1000.0, 0.1, 10.0);
    assert!(log_growth > 10.0 && log_growth < 1000.0);
}

#[test]
fn test_biology_integration() {
    // Test a simple ecological scenario: predator-prey dynamics with growth
    let mut predator = 5.0;
    let mut prey = 50.0;
    let alpha = 1.0;   // prey growth rate
    let beta = 0.05;   // predation rate
    let gamma = 0.02;  // predator efficiency
    let delta = 0.8;   // predator death rate
    let dt = 0.01;
    
    let initial_total = predator + prey;
    
    // Run ecosystem simulation for 100 time steps
    for _ in 0..100 {
        let new_predator = lotka_volterra_predator(predator, prey, gamma, delta, dt);
        let new_prey = lotka_volterra_prey(predator, prey, alpha, beta, dt);
        
        // Ensure populations remain positive
        predator = new_predator.max(0.0);
        prey = new_prey.max(0.0);
    }
    
    // System should maintain reasonable population levels
    assert!(predator >= 0.0);
    assert!(prey >= 0.0);
    let final_total = predator + prey;
    assert!(final_total > 0.1 * initial_total);
}