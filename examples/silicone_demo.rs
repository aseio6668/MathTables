use mathtables::prelude::*;
use std::time::Instant;

fn main() {
    println!("{}", "=".repeat(80));
    println!("                    SILICONE DOMAIN COMPREHENSIVE DEMO");
    println!("{}", "=".repeat(80));
    println!();
    
    // Initialize silicone domain
    let silicone = SiliconeDomain::new();
    
    // 1. SILOXANE CHAIN PROPERTIES
    println!("🔗 SILOXANE CHAIN PROPERTIES");
    println!("{}", "-".repeat(40));
    
    // Chain length distribution
    let average_length = 1000.0;
    let polydispersity = 2.5;
    let std_dev = siloxane_chain_length_distribution(average_length, polydispersity).unwrap();
    
    println!("Chain Length Distribution:");
    println!("  Average length: {:.0} repeat units", average_length);
    println!("  Polydispersity index: {:.1}", polydispersity);
    println!("  Standard deviation: {:.1} repeat units", std_dev);
    
    // Molecular weight and glass transition
    let molecular_weight = 25000.0; // g/mol
    let crosslink_density = 0.001; // mol/cm³
    let tg = siloxane_glass_transition_temperature(molecular_weight, crosslink_density).unwrap();
    
    println!("\nGlass Transition Properties:");
    println!("  Molecular weight: {:.0} g/mol", molecular_weight);
    println!("  Crosslink density: {:.3} mol/cm³", crosslink_density);
    println!("  Glass transition temperature: {:.1} K ({:.1} °C)", tg, tg - 273.15);
    
    println!();
    
    // 2. RHEOLOGICAL PROPERTIES
    println!("🌡️ RHEOLOGICAL PROPERTIES");
    println!("{}", "-".repeat(40));
    
    let temperatures = [298.15, 323.15, 373.15, 423.15]; // K
    let molecular_weights = [5000.0, 25000.0, 100000.0]; // g/mol
    let shear_rates = [0.1, 1.0, 10.0, 100.0]; // s⁻¹
    
    println!("Viscosity vs Temperature (MW = 25,000 g/mol, γ̇ = 1.0 s⁻¹):");
    println!("  Temp (°C)  Viscosity (Pa·s)");
    for &temp in &temperatures {
        let viscosity = silicone.calculate_viscosity(temp, 25000.0, 1.0).unwrap();
        println!("  {:8.0}   {:12.3e}", temp - 273.15, viscosity);
    }
    
    println!("\nViscosity vs Molecular Weight (25°C, γ̇ = 1.0 s⁻¹):");
    println!("  MW (g/mol)  Viscosity (Pa·s)");
    for &mw in &molecular_weights {
        let viscosity = silicone.calculate_viscosity(298.15, mw, 1.0).unwrap();
        println!("  {:9.0}   {:12.3e}", mw, viscosity);
    }
    
    println!("\nShear Thinning Behavior (25°C, MW = 25,000 g/mol):");
    println!("  Shear Rate (s⁻¹)  Viscosity (Pa·s)");
    for &shear_rate in &shear_rates {
        let viscosity = silicone.calculate_viscosity(298.15, 25000.0, shear_rate).unwrap();
        println!("  {:14.1}   {:12.3e}", shear_rate, viscosity);
    }
    
    println!();
    
    // 3. THERMAL PROPERTIES
    println!("🔥 THERMAL PROPERTIES");
    println!("{}", "-".repeat(40));
    
    // Thermal expansion
    let original_length = 100.0; // mm
    let expansion_coeff = 2.4e-4; // /K
    let temp_changes = [-50.0, -25.0, 0.0, 25.0, 50.0, 100.0]; // K
    
    println!("Thermal Expansion (α = 2.4×10⁻⁴ /K):");
    println!("  ΔT (K)  Length (mm)  Expansion (%)");
    for &delta_t in &temp_changes {
        let new_length = siloxane_thermal_expansion(original_length, delta_t, expansion_coeff);
        let expansion_percent = ((new_length - original_length) / original_length) * 100.0;
        println!("  {:6.0}   {:9.3}     {:8.3}", delta_t, new_length, expansion_percent);
    }
    
    // Thermal conductivity
    let filler_fractions = [0.0, 0.1, 0.2, 0.3, 0.4];
    
    println!("\nThermal Conductivity vs Filler Content (25°C):");
    println!("  Filler (%)  Conductivity (W/m·K)");
    for &filler in &filler_fractions {
        let k = siloxane_thermal_conductivity(298.15, 0.005, filler).unwrap();
        println!("  {:9.1}   {:15.3}", filler * 100.0, k);
    }
    
    println!();
    
    // 4. MECHANICAL PROPERTIES
    println!("🔧 MECHANICAL PROPERTIES");
    println!("{}", "-".repeat(40));
    
    // Elastic modulus vs crosslinking
    let crosslink_densities = [0.001, 0.005, 0.01, 0.02, 0.05];
    
    println!("Elastic Modulus vs Crosslink Density (25°C, no filler):");
    println!("  Crosslink (mol/cm³)  Modulus (MPa)");
    for &crosslink in &crosslink_densities {
        let modulus = silicone_elastic_modulus(298.15, crosslink, 0.0).unwrap();
        println!("  {:17.3}   {:11.2}", crosslink, modulus / 1e6);
    }
    
    // Modulus vs filler content
    println!("\nElastic Modulus vs Filler Content (25°C, ρx = 0.005 mol/cm³):");
    println!("  Filler (%)  Modulus (MPa)");
    for &filler in &filler_fractions {
        let modulus = silicone_elastic_modulus(298.15, 0.005, filler).unwrap();
        println!("  {:9.1}   {:11.2}", filler * 100.0, modulus / 1e6);
    }
    
    // Chain entanglement effects
    let molecular_weights_entangle = [5000.0, 15000.0, 25000.0, 50000.0, 100000.0];
    
    println!("\nChain Entanglement Modulus (25°C):");
    println!("  MW (g/mol)  Entanglement Modulus (kPa)");
    for &mw in &molecular_weights_entangle {
        let ent_modulus = siloxane_chain_entanglement_modulus(mw, 298.15).unwrap();
        println!("  {:9.0}   {:18.1}", mw, ent_modulus / 1000.0);
    }
    
    println!();
    
    // 5. SURFACE PROPERTIES
    println!("💧 SURFACE PROPERTIES");
    println!("{}", "-".repeat(40));
    
    // Surface energy
    let chain_flexibilities = [0.2, 0.5, 0.8, 1.0];
    
    println!("Surface Energy vs Chain Flexibility (25°C):");
    println!("  Flexibility  Surface Energy (mJ/m²)");
    for &flexibility in &chain_flexibilities {
        let surface_energy = silicone_surface_energy(298.15, flexibility).unwrap();
        println!("  {:11.1}   {:18.1}", flexibility, surface_energy);
    }
    
    // Contact angles with water
    let surface_energies = [15.0, 20.0, 25.0, 30.0]; // mJ/m²
    let water_surface_tension = 72.8; // mJ/m²
    let interfacial_tensions = [45.0, 40.0, 35.0, 30.0]; // mJ/m²
    
    println!("\nContact Angle with Water:");
    println!("  Surface Energy (mJ/m²)  Contact Angle (°)  Wettability");
    for (i, &se) in surface_energies.iter().enumerate() {
        let contact_angle = siloxane_contact_angle(se, water_surface_tension, interfacial_tensions[i]).unwrap();
        let wettability = if contact_angle < 90.0 { "Hydrophilic" } else { "Hydrophobic" };
        println!("  {:20.1}   {:14.1}  {}", se, contact_angle, wettability);
    }
    
    println!();
    
    // 6. ELECTRICAL PROPERTIES
    println!("⚡ ELECTRICAL PROPERTIES");
    println!("{}", "-".repeat(40));
    
    // Dielectric constant vs frequency
    let frequencies = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]; // Hz
    
    println!("Dielectric Constant vs Frequency (25°C, dry):");
    println!("  Frequency (Hz)  Dielectric Constant");
    for &freq in &frequencies {
        let dielectric = siloxane_dielectric_constant(freq, 298.15, 0.0).unwrap();
        println!("  {:12.0e}   {:17.2}", freq, dielectric);
    }
    
    // Effect of moisture
    let moisture_contents = [0.0, 0.01, 0.05, 0.1]; // fraction
    
    println!("\nDielectric Constant vs Moisture Content (1 kHz, 25°C):");
    println!("  Moisture (%)  Dielectric Constant");
    for &moisture in &moisture_contents {
        let dielectric = siloxane_dielectric_constant(1e3, 298.15, moisture).unwrap();
        println!("  {:11.1}   {:17.2}", moisture * 100.0, dielectric);
    }
    
    println!();
    
    // 7. OPTICAL PROPERTIES
    println!("🌈 OPTICAL PROPERTIES");
    println!("{}", "-".repeat(40));
    
    // Refractive index vs wavelength
    let wavelengths = [400e-9, 500e-9, 589e-9, 650e-9, 750e-9]; // m
    let wavelength_names = ["Blue", "Green", "Yellow", "Red", "NIR"];
    
    println!("Refractive Index vs Wavelength (25°C):");
    println!("  Color    Wavelength (nm)  Refractive Index");
    for (i, &wl) in wavelengths.iter().enumerate() {
        let n = siloxane_refractive_index(wl, 298.15, 0.005).unwrap();
        println!("  {:7}   {:14.0}   {:14.4}", wavelength_names[i], wl * 1e9, n);
    }
    
    println!();
    
    // 8. GAS PERMEABILITY
    println!("💨 GAS PERMEABILITY");
    println!("{}", "-".repeat(40));
    
    // Different gases
    let gases = [
        ("H₂", 2.89),  // Hydrogen
        ("He", 2.60),  // Helium
        ("CO₂", 3.30), // Carbon dioxide
        ("O₂", 3.46),  // Oxygen
        ("N₂", 3.64),  // Nitrogen
        ("CH₄", 3.80), // Methane
    ];
    
    println!("Gas Permeability (25°C, ρx = 0.005 mol/cm³):");
    println!("  Gas  Molecule Size (Å)  Permeability (cm³·cm/cm²·s·cmHg)");
    for (name, size) in &gases {
        let permeability = siloxane_gas_permeability(*size, 298.15, 0.005).unwrap();
        println!("  {:4}   {:15.2}   {:30.2e}", name, size, permeability);
    }
    
    println!();
    
    // 9. SWELLING AND SOLVENT RESISTANCE
    println!("🧪 SWELLING AND SOLVENT RESISTANCE");
    println!("{}", "-".repeat(40));
    
    // Swelling in different solvents
    let solvents = [
        ("Hexane", 14.9),
        ("Toluene", 18.2),
        ("Ethanol", 26.5),
        ("Water", 47.8),
    ];
    let pdms_parameter = 15.5; // (J/cm³)^0.5
    
    println!("Swelling Ratio in Different Solvents (ρx = 0.001 mol/cm³):");
    println!("  Solvent   Solubility Parameter  Swelling Ratio");
    for (name, param) in &solvents {
        let swelling = siloxane_swelling_ratio(*param, pdms_parameter, 0.001).unwrap();
        println!("  {:8}   {:18.1}   {:12.1}", name, param, swelling);
    }
    
    println!();
    
    // 10. CURE KINETICS
    println!("⏱️ CURE KINETICS");
    println!("{}", "-".repeat(40));
    
    // Cure kinetics at different temperatures
    let cure_temperatures = [298.15, 323.15, 373.15, 423.15]; // K
    let cure_time = 3600.0; // seconds (1 hour)
    let catalyst_conc = 0.01;
    let activation_energy = 50000.0; // J/mol
    
    println!("Degree of Cure vs Temperature (1 hour, 1% catalyst):");
    println!("  Temperature (°C)  Degree of Cure (%)");
    for &temp in &cure_temperatures {
        let cure_degree = siloxane_cure_kinetics(temp, catalyst_conc, cure_time, activation_energy).unwrap();
        println!("  {:14.0}   {:16.1}", temp - 273.15, cure_degree * 100.0);
    }
    
    // Cure kinetics vs time
    let cure_times = [600.0, 1800.0, 3600.0, 7200.0, 14400.0]; // seconds
    
    println!("\nDegree of Cure vs Time (100°C, 1% catalyst):");
    println!("  Time (min)  Degree of Cure (%)");
    for &time in &cure_times {
        let cure_degree = siloxane_cure_kinetics(373.15, catalyst_conc, time, activation_energy).unwrap();
        println!("  {:9.0}   {:16.1}", time / 60.0, cure_degree * 100.0);
    }
    
    println!();
    
    // 11. BIOCOMPATIBILITY
    println!("🧬 BIOCOMPATIBILITY ASSESSMENT");
    println!("{}", "-".repeat(40));
    
    // Different silicone grades
    let silicone_grades = [
        ("Industrial", 0.1, 5.0, 0.6),     // High leachables, rough surface
        ("Commercial", 0.01, 1.0, 0.75),   // Medium quality
        ("Medical", 0.001, 0.1, 0.85),     // High purity, smooth
        ("Implant", 0.0001, 0.01, 0.9),    // Ultra-pure, very smooth
    ];
    
    println!("Biocompatibility Scores for Different Grades:");
    println!("  Grade       Leachables (%)  Roughness (μm)  Hydrophobicity  Score (/100)");
    for (grade, leach, rough, hydro) in &silicone_grades {
        let score = siloxane_biocompatibility_score(*leach, *rough, *hydro).unwrap();
        println!("  {:10}   {:12.4}   {:13.2}   {:12.1}   {:9.1}", 
                 grade, leach * 100.0, rough, hydro, score);
    }
    
    println!();
    
    // 12. DEGRADATION AND AGING
    println!("⚠️ DEGRADATION AND AGING");
    println!("{}", "-".repeat(40));
    
    // Environmental conditions
    let environmental_conditions = [
        ("Room temp, dark", 298.15, 0.0, 0.0),
        ("Elevated temp", 373.15, 0.0, 0.0),
        ("UV exposure", 298.15, 5.0, 0.21),
        ("Outdoor aging", 323.15, 2.0, 0.21),
        ("Accelerated aging", 423.15, 10.0, 0.21),
    ];
    
    println!("Degradation Rates under Different Conditions:");
    println!("  Condition          Temperature (°C)  UV (W/m²)  Degradation Rate (/year)");
    for (condition, temp, uv, o2) in &environmental_conditions {
        let rate = siloxane_degradation_rate(*temp, *uv, *o2).unwrap();
        let rate_per_year = rate * 365.25 * 24.0 * 3600.0;
        println!("  {:17}   {:14.0}   {:8.1}   {:18.2e}", 
                 condition, temp - 273.15, uv, rate_per_year);
    }
    
    println!();
    
    // 13. PERFORMANCE BENCHMARKS
    println!("⚡ PERFORMANCE BENCHMARKS");
    println!("{}", "-".repeat(40));
    
    let iterations = 100_000;
    
    // Viscosity calculation benchmark
    let start = Instant::now();
    for i in 0..iterations {
        let temp = 298.15 + (i % 100) as f64;
        let _ = silicone.calculate_viscosity(temp, 25000.0, 1.0).unwrap();
    }
    let duration = start.elapsed();
    println!("Viscosity Calculation: {:.0} ops/sec", 
             iterations as f64 / duration.as_secs_f64());
    
    // Elastic modulus benchmark
    let start = Instant::now();
    for i in 0..iterations {
        let crosslink = 0.001 + (i % 100) as f64 * 1e-6;
        let _ = silicone_elastic_modulus(298.15, crosslink, 0.1).unwrap();
    }
    let duration = start.elapsed();
    println!("Elastic Modulus: {:.0} ops/sec", 
             iterations as f64 / duration.as_secs_f64());
    
    // Gas permeability benchmark
    let start = Instant::now();
    for i in 0..iterations {
        let size = 3.0 + (i % 100) as f64 * 0.01;
        let _ = siloxane_gas_permeability(size, 298.15, 0.005).unwrap();
    }
    let duration = start.elapsed();
    println!("Gas Permeability: {:.0} ops/sec", 
             iterations as f64 / duration.as_secs_f64());
    
    // Thermal conductivity benchmark
    let start = Instant::now();
    for i in 0..iterations {
        let filler = (i % 50) as f64 * 0.01;
        let _ = siloxane_thermal_conductivity(298.15, 0.005, filler).unwrap();
    }
    let duration = start.elapsed();
    println!("Thermal Conductivity: {:.0} ops/sec", 
             iterations as f64 / duration.as_secs_f64());
    
    println!();
    
    // 14. MEDICAL GRADE SILICONE ANALYSIS
    println!("🏥 MEDICAL GRADE SILICONE ANALYSIS");
    println!("{}", "-".repeat(40));
    
    // Typical medical grade parameters
    let med_temp = 310.15; // Body temperature
    let med_mw = 50000.0;
    let med_crosslink = 0.003;
    let med_filler = 0.05; // Some reinforcement
    
    println!("Medical Grade Silicone Properties (37°C):");
    
    // Calculate all relevant properties
    let med_viscosity = silicone.calculate_viscosity(med_temp, med_mw, 0.0).unwrap();
    let med_modulus = silicone_elastic_modulus(med_temp, med_crosslink, med_filler).unwrap();
    let med_tg = siloxane_glass_transition_temperature(med_mw, med_crosslink).unwrap();
    let med_surface_energy = silicone_surface_energy(med_temp, 0.8).unwrap();
    let med_biocompat = siloxane_biocompatibility_score(0.0005, 0.1, 0.85).unwrap();
    let med_contact_angle = siloxane_contact_angle(20.0, 72.8, 40.0).unwrap();
    let med_o2_perm = siloxane_gas_permeability(3.46, med_temp, med_crosslink).unwrap();
    let med_thermal_k = siloxane_thermal_conductivity(med_temp, med_crosslink, med_filler).unwrap();
    
    println!("  Viscosity (zero shear): {:.2e} Pa·s", med_viscosity);
    println!("  Elastic modulus: {:.1} MPa", med_modulus / 1e6);
    println!("  Glass transition: {:.1} K ({:.1}°C)", med_tg, med_tg - 273.15);
    println!("  Surface energy: {:.1} mJ/m²", med_surface_energy);
    println!("  Contact angle (water): {:.1}°", med_contact_angle);
    println!("  Biocompatibility score: {:.1}/100", med_biocompat);
    println!("  Oxygen permeability: {:.2e} cm³·cm/cm²·s·cmHg", med_o2_perm);
    println!("  Thermal conductivity: {:.3} W/m·K", med_thermal_k);
    
    // Performance assessment
    println!("\nMedical Performance Assessment:");
    println!("  Flexibility: {}", if med_modulus < 2e6 { "✓ Excellent" } else { "⚠ Stiff" });
    println!("  Biocompatibility: {}", if med_biocompat > 80.0 { "✓ Excellent" } else { "⚠ Fair" });
    println!("  Hydrophobicity: {}", if med_contact_angle > 100.0 { "✓ Good" } else { "⚠ Moderate" });
    println!("  Temperature resistance: {}", if med_tg < 250.0 { "✓ Suitable" } else { "⚠ Brittle" });
    
    println!();
    println!("🧪 SILICONE DOMAIN COMPREHENSIVE DEMO COMPLETE! 🧪");
    println!("All major silicone properties successfully modeled:");
    println!("✓ Siloxane Chain Architecture & Molecular Weight Effects");
    println!("✓ Rheological Properties & Shear Thinning Behavior");
    println!("✓ Thermal Properties & Expansion Coefficients");
    println!("✓ Mechanical Properties & Crosslinking Effects");
    println!("✓ Surface Properties & Hydrophobic Behavior");
    println!("✓ Electrical Properties & Dielectric Performance");
    println!("✓ Optical Properties & Wavelength Dependence");
    println!("✓ Gas Permeability & Membrane Applications");
    println!("✓ Solvent Swelling & Chemical Resistance");
    println!("✓ Cure Kinetics & Processing Parameters");
    println!("✓ Biocompatibility Assessment & Medical Applications");
    println!("✓ Degradation & Environmental Aging");
    println!("{}", "=".repeat(80));
}