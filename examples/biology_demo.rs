use mathtables::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    println!("{}", "=".repeat(80));
    println!("                    BIOLOGY DOMAIN COMPREHENSIVE DEMO");
    println!("{}", "=".repeat(80));
    println!();
    
    // Initialize biology domain
    let biology = BiologyDomain::new();
    
    // 1. GROWTH MODELS
    println!("📈 GROWTH MODELS");
    println!("{}", "-".repeat(40));
    
    // Exponential growth (bacterial population)
    let initial_bacteria = 1000.0;
    let growth_rate = 0.693; // doubles every hour
    let time = 3.0; // 3 hours
    let final_bacteria = exponential_growth(initial_bacteria, growth_rate, time);
    println!("Exponential Growth (Bacteria):");
    println!("  Initial: {:.0} cells, Growth rate: {:.3}/hr, Time: {:.1}hr", 
             initial_bacteria, growth_rate, time);
    println!("  Final population: {:.0} cells", final_bacteria);
    
    // Logistic growth (limited environment)
    let carrying_capacity = 1_000_000.0;
    let logistic_final = logistic_growth(initial_bacteria, carrying_capacity, growth_rate, time);
    println!("  Logistic growth (with carrying capacity): {:.0} cells", logistic_final);
    
    // Gompertz growth (tumor growth)
    let tumor_size = gompertz_growth(10.0, 1000.0, 0.1, 30.0);
    println!("  Gompertz growth (tumor): {:.1} mm³", tumor_size);
    
    // Growth rate calculation
    let calculated_rate = biology.calculate_growth_rate(100.0, 800.0, 3.0).unwrap();
    println!("  Calculated growth rate: {:.3}/unit time", calculated_rate);
    
    println!();
    
    // 2. L-SYSTEMS (PLANT GROWTH)
    println!("🌱 L-SYSTEMS AND PLANT MORPHOLOGY");
    println!("{}", "-".repeat(40));
    
    // Algae growth pattern
    let mut algae_rules = HashMap::new();
    algae_rules.insert('A', "AB".to_string());
    algae_rules.insert('B', "A".to_string());
    
    let algae_axiom = "A";
    println!("Algae Growth Pattern (Fibonacci-like):");
    for i in 0..=6 {
        let pattern = l_system_expand(algae_axiom, &algae_rules, i);
        println!("  Generation {}: {} (length: {})", i, pattern, pattern.len());
    }
    
    // Tree branching
    let mut tree_rules = HashMap::new();
    tree_rules.insert('F', "F[+F]F[-F]F".to_string());
    
    let tree_pattern = l_system_expand("F", &tree_rules, 2);
    println!("Tree Branching Pattern:");
    println!("  {}", tree_pattern);
    println!("  Length: {}", tree_pattern.len());
    
    println!();
    
    // 3. NEURAL DYNAMICS
    println!("🧠 NEURAL DYNAMICS (HODGKIN-HUXLEY)");
    println!("{}", "-".repeat(40));
    
    // Simulate action potential
    let mut voltage = -65.0; // resting potential
    let mut n = 0.318;        // potassium gate
    let mut m = 0.053;        // sodium activation gate
    let mut h = 0.596;        // sodium inactivation gate
    let dt = 0.01;
    
    println!("Action Potential Simulation:");
    println!("  Time(ms)  Voltage(mV)  n      m      h");
    
    // Apply current stimulus at t=1ms
    let stimulus_time = 1.0;
    let stimulus_current = 10.0; // μA/cm²
    
    for step in 0..=500 {
        let time = step as f64 * dt;
        let i_ext = if (time - stimulus_time).abs() < 0.5 { stimulus_current } else { 0.0 };
        
        voltage = hodgkin_huxley_voltage(voltage, n, m, h, i_ext, dt);
        n = hodgkin_huxley_gate_n(voltage, n, dt);
        m = hodgkin_huxley_gate_m(voltage, m, dt);
        h = hodgkin_huxley_gate_h(voltage, h, dt);
        
        if step % 50 == 0 {
            println!("  {:6.2}    {:8.2}     {:5.3}  {:5.3}  {:5.3}", 
                     time, voltage, n, m, h);
        }
    }
    
    println!();
    
    // 4. CELLULAR AUTOMATA
    println!("🔬 CELLULAR AUTOMATA (CONWAY'S GAME OF LIFE)");
    println!("{}", "-".repeat(40));
    
    // Glider pattern
    let mut glider = vec![
        vec![0, 1, 0, 0, 0],
        vec![0, 0, 1, 0, 0],
        vec![1, 1, 1, 0, 0],
        vec![0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0],
    ];
    
    println!("Conway's Game of Life - Glider Pattern:");
    for generation in 0..=3 {
        println!("  Generation {}:", generation);
        for row in &glider {
            print!("    ");
            for &cell in row {
                print!("{} ", if cell == 1 { "●" } else { "○" });
            }
            println!();
        }
        if generation < 3 {
            glider = cellular_automaton_step(&glider, conway_life_rule);
        }
        println!();
    }
    
    // 5. BIOCHEMICAL KINETICS
    println!("⚗️ BIOCHEMICAL KINETICS");
    println!("{}", "-".repeat(40));
    
    // Michaelis-Menten kinetics
    let vmax = 100.0; // μmol/min
    let km = 5.0;     // mM
    
    println!("Michaelis-Menten Kinetics:");
    println!("  Vmax: {} μmol/min, Km: {} mM", vmax, km);
    println!("  [S] (mM)  Velocity (μmol/min)  % Vmax");
    
    for &substrate in &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0] {
        let velocity = michaelis_menten_velocity(substrate, vmax, km).unwrap();
        let percent_max = (velocity / vmax) * 100.0;
        println!("  {:8.1}  {:16.1}  {:6.1}%", substrate, velocity, percent_max);
    }
    
    // Hill equation (cooperative binding)
    println!("\nCooperative Binding (Hill Equation):");
    println!("  Hill coefficient: 2.0, Kd: 1.0 mM");
    println!("  [Ligand] (mM)  Binding Fraction");
    
    for &ligand in &[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let binding = hill_equation(ligand, 1.0, 2.0).unwrap();
        println!("  {:12.2}  {:14.3}", ligand, binding);
    }
    
    println!();
    
    // 6. POPULATION DYNAMICS
    println!("🐺🐰 POPULATION DYNAMICS (LOTKA-VOLTERRA)");
    println!("{}", "-".repeat(40));
    
    // Predator-prey model
    let mut predator = 10.0;
    let mut prey = 50.0;
    let alpha = 1.0;   // prey growth rate
    let beta = 0.05;   // predation rate
    let gamma = 0.02;  // predator efficiency
    let delta = 0.8;   // predator death rate
    let dt = 0.01;
    
    println!("Predator-Prey Dynamics:");
    println!("  Time  Predators  Prey");
    
    for step in 0..=1000 {
        if step % 100 == 0 {
            println!("  {:4.1}  {:9.1}  {:4.1}", step as f64 * dt, predator, prey);
        }
        
        let new_predator = lotka_volterra_predator(predator, prey, gamma, delta, dt);
        let new_prey = lotka_volterra_prey(predator, prey, alpha, beta, dt);
        
        predator = new_predator.max(0.0);
        prey = new_prey.max(0.0);
    }
    
    println!();
    
    // 7. EVOLUTIONARY GENETICS
    println!("🧬 EVOLUTIONARY GENETICS");
    println!("{}", "-".repeat(40));
    
    // Hardy-Weinberg equilibrium
    let p = 0.6; // frequency of allele A
    let q = 0.4; // frequency of allele a
    let (aa, ab, bb) = hardy_weinberg_genotype_freq(p, q);
    
    println!("Hardy-Weinberg Equilibrium:");
    println!("  Allele frequencies: A = {:.1}, a = {:.1}", p, q);
    println!("  Genotype frequencies:");
    println!("    AA: {:.3} ({:.1}%)", aa, aa * 100.0);
    println!("    Aa: {:.3} ({:.1}%)", ab, ab * 100.0);
    println!("    aa: {:.3} ({:.1}%)", bb, bb * 100.0);
    
    // Selection coefficient
    let fitness_wildtype = 1.0;
    let fitness_mutant = 1.05; // 5% advantage
    let selection_coeff = selection_coefficient(fitness_mutant, fitness_wildtype).unwrap();
    println!("  Selection coefficient: {:.3}", selection_coeff);
    
    // Allele frequency change
    let initial_freq = 0.1;
    let delta_p = allele_frequency_change(initial_freq, selection_coeff, 0.5);
    println!("  Allele frequency change per generation: {:.6}", delta_p);
    
    println!();
    
    // 8. ECOLOGICAL DIVERSITY
    println!("🌿 ECOLOGICAL DIVERSITY");
    println!("{}", "-".repeat(40));
    
    // Species diversity indices
    let species_counts = vec![120.0, 85.0, 45.0, 30.0, 15.0, 10.0, 5.0];
    let total_individuals: f64 = species_counts.iter().sum();
    
    println!("Species Abundance:");
    for (i, &count) in species_counts.iter().enumerate() {
        let percentage = (count / total_individuals) * 100.0;
        println!("  Species {}: {:.0} individuals ({:.1}%)", i + 1, count, percentage);
    }
    
    let shannon = shannon_diversity_index(&species_counts).unwrap();
    let simpson = simpson_diversity_index(&species_counts).unwrap();
    
    println!("Diversity Indices:");
    println!("  Shannon diversity: {:.3}", shannon);
    println!("  Simpson diversity: {:.3}", simpson);
    
    // Biomass pyramid
    let trophic_levels = vec![10000.0, 0.0, 0.0, 0.0]; // Primary producers
    let efficiency = 0.1; // 10% energy transfer efficiency
    let pyramid = calculate_biomass_pyramid(&trophic_levels, efficiency).unwrap();
    
    println!("\nBiomass Pyramid (kg/m²):");
    let trophic_names = ["Primary Producers", "Primary Consumers", 
                        "Secondary Consumers", "Tertiary Consumers"];
    for (i, &biomass) in pyramid.iter().enumerate() {
        println!("  {}: {:.1} kg/m²", trophic_names[i], biomass);
    }
    
    println!();
    
    // 9. REACTION-DIFFUSION PATTERNS
    println!("🎨 REACTION-DIFFUSION (TURING PATTERNS)");
    println!("{}", "-".repeat(40));
    
    // Initialize concentration arrays
    let size = 20;
    let mut u = vec![1.0; size];
    let mut v = vec![0.0; size];
    
    // Add small perturbation in the middle
    u[size/2] = 0.5;
    v[size/2] = 0.25;
    
    println!("Turing Pattern Formation (1D):");
    println!("Initial state:");
    print!("  U: ");
    for i in (0..size).step_by(4) {
        print!("{:.2} ", u[i]);
    }
    println!();
    
    // Run reaction-diffusion
    let du = 2e-5;
    let dv = 1e-5;
    let f = 0.054;
    let k = 0.062;
    let dt = 1.0;
    let dx = 1.0;
    
    for step in 1..=3 {
        let (u_new, v_new) = reaction_diffusion_step(&u, &v, du, dv, f, k, dt, dx);
        u = u_new;
        v = v_new;
        
        println!("Step {}:", step);
        print!("  U: ");
        for i in (0..size).step_by(4) {
            print!("{:.2} ", u[i]);
        }
        println!();
    }
    
    // Pattern stability analysis
    let stability = turing_pattern_stability(du, dv, -1.0, 1.0);
    println!("  Pattern stability parameter: {:.6}", stability);
    
    println!();
    
    // 10. METABOLIC SCALING
    println!("⚖️ METABOLIC SCALING");
    println!("{}", "-".repeat(40));
    
    println!("Allometric Scaling (Kleiber's Law):");
    println!("  Body Mass (kg)  Metabolic Rate (W)  Mass-Specific Rate (W/kg)");
    
    let masses = vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]; // mouse to elephant
    let scaling_constant = 3.4; // W/kg^0.75
    let scaling_exponent = 0.75;
    
    for &mass in &masses {
        let rate = metabolic_rate_allometric(mass, scaling_constant, scaling_exponent).unwrap();
        let specific_rate = rate / mass;
        println!("  {:11.3}  {:15.1}  {:19.1}", mass, rate, specific_rate);
    }
    
    println!();
    
    // 11. PROTEIN FOLDING ENERGY
    println!("🧬 PROTEIN FOLDING THERMODYNAMICS");
    println!("{}", "-".repeat(40));
    
    let proteins = vec![
        ("Small peptide", 5, 3, 0),
        ("Medium protein", 25, 15, 2),
        ("Large enzyme", 80, 45, 8),
        ("Membrane protein", 120, 30, 12),
    ];
    
    println!("Protein Folding Energies:");
    println!("  Protein Type        Hydrophobic  H-bonds  Disulfide  Energy (kcal/mol)");
    
    for (name, hydrophobic, h_bonds, disulfide) in proteins {
        let energy = protein_folding_energy(hydrophobic, h_bonds, disulfide);
        println!("  {:18}  {:10}   {:7}  {:9}  {:13.1}", 
                 name, hydrophobic, h_bonds, disulfide, energy);
    }
    
    println!();
    
    // 12. PERFORMANCE BENCHMARKS
    println!("⚡ PERFORMANCE BENCHMARKS");
    println!("{}", "-".repeat(40));
    
    let iterations = 1_000_000;
    
    // Exponential growth benchmark
    let start = Instant::now();
    for i in 0..iterations {
        let _ = exponential_growth(100.0, 0.1, (i % 100) as f64 / 100.0);
    }
    let duration = start.elapsed();
    println!("Exponential Growth: {:.0} ops/sec", 
             iterations as f64 / duration.as_secs_f64());
    
    // Michaelis-Menten benchmark
    let start = Instant::now();
    for i in 0..iterations {
        let _ = michaelis_menten_velocity((i % 100) as f64, 10.0, 5.0);
    }
    let duration = start.elapsed();
    println!("Michaelis-Menten: {:.0} ops/sec", 
             iterations as f64 / duration.as_secs_f64());
    
    // L-system benchmark
    let start = Instant::now();
    let mut rules = HashMap::new();
    rules.insert('A', "AB".to_string());
    rules.insert('B', "A".to_string());
    for i in 0..(iterations / 10000) {
        let _ = l_system_expand("A", &rules, (i % 8) as usize);
    }
    let duration = start.elapsed();
    println!("L-system Expansion: {:.0} ops/sec", 
             (iterations / 10000) as f64 / duration.as_secs_f64());
    
    // Hodgkin-Huxley benchmark
    let start = Instant::now();
    let mut v = -65.0;
    let mut n = 0.318;
    let mut m = 0.053;
    let mut h = 0.596;
    for _ in 0..(iterations / 1000) {
        v = hodgkin_huxley_voltage(v, n, m, h, 0.0, 0.01);
        n = hodgkin_huxley_gate_n(v, n, 0.01);
        m = hodgkin_huxley_gate_m(v, m, 0.01);
        h = hodgkin_huxley_gate_h(v, h, 0.01);
    }
    let duration = start.elapsed();
    println!("Hodgkin-Huxley Step: {:.0} steps/sec", 
             (iterations / 1000) as f64 / duration.as_secs_f64());
    
    println!();
    println!("🧬 BIOLOGY DOMAIN COMPREHENSIVE DEMO COMPLETE! 🧬");
    println!("All major biological systems successfully simulated:");
    println!("✓ Growth Models & Population Dynamics");
    println!("✓ L-Systems & Plant Morphogenesis");
    println!("✓ Neural Dynamics & Action Potentials");
    println!("✓ Cellular Automata & Pattern Formation");
    println!("✓ Biochemical Kinetics & Enzyme Systems");
    println!("✓ Evolutionary Genetics & Hardy-Weinberg");
    println!("✓ Ecological Diversity & Biomass Pyramids");
    println!("✓ Reaction-Diffusion & Turing Patterns");
    println!("✓ Metabolic Scaling & Allometric Laws");
    println!("✓ Protein Folding Thermodynamics");
    println!("{}", "=".repeat(80));
}