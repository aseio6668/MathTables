use mathtables::core::{TPEMaterial, BlockCopolymer};
use mathtables::domains::TPEDomain;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Thermoplastic Elastomer (TPE) Domain Demo");
    println!("===========================================\n");
    
    // ===== Material Database Showcase =====
    println!("📚 TPE Material Database:");
    println!("-------------------------");
    
    let materials = vec![
        ("TPU Standard", TPEMaterial::tpu_standard()),
        ("TPO Automotive", TPEMaterial::tpo_automotive()),
        ("TPS Consumer", TPEMaterial::tps_consumer()),
    ];
    
    for (name, material) in &materials {
        println!("  {} Properties:", name);
        println!("    • Hardness (Shore A): {:.0}", material.hardness_shore_a);
        println!("    • Tensile Strength: {:.1} MPa", material.tensile_strength / 1e6);
        println!("    • Elongation at Break: {:.0}%", material.elongation_at_break);
        println!("    • Elastic Modulus: {:.1} MPa", material.elastic_modulus / 1e6);
        println!("    • Density: {:.0} kg/m³", material.density);
        println!("    • Glass Transition: {:.1}°C", material.glass_transition_temp - 273.15);
        println!("    • Melting Point: {:.1}°C", material.melting_temp - 273.15);
        println!("    • Thermal Expansion: {:.0} µm/m/K", material.thermal_expansion_coeff * 1e6);
        println!("    • Thermal Conductivity: {:.2} W/(m·K)", material.thermal_conductivity);
        println!("    • Melt Flow Index: {:.1} g/10min", material.melt_flow_index);
        println!();
    }
    
    // ===== Thermal Response Analysis =====
    println!("🌡️ Thermal Response Analysis:");
    println!("------------------------------");
    
    let cable_length = 50.0; // 50 meter cable
    let tpu = TPEMaterial::tpu_standard();
    
    println!("  Cable Thermal Expansion Analysis:");
    let temperature_scenarios = vec![
        ("Arctic Winter", -40.0),
        ("Room Temperature", 20.0),
        ("Desert Summer", 60.0),
        ("Industrial Heat", 100.0),
    ];
    
    let reference_temp = 20.0; // Room temperature baseline
    for (scenario, temp) in &temperature_scenarios {
        let temp_change = temp - reference_temp;
        let expansion = TPEDomain::thermal_expansion(
            cable_length, 
            tpu.thermal_expansion_coeff, 
            temp_change
        );
        
        println!("    • {}: {:.0}°C → Δ{:.1} cm", 
                 scenario, temp, expansion * 100.0);
    }
    
    // Volumetric expansion for a connector housing
    let housing_volume = 0.00001; // 10 cm³
    let temp_swing = 80.0; // -40°C to +40°C
    let volume_expansion = TPEDomain::volumetric_thermal_expansion(
        housing_volume, tpu.thermal_expansion_coeff, temp_swing
    );
    println!("    • Connector Housing: {:.2} mm³ volume change", volume_expansion * 1e9);
    println!();
    
    // Glass transition behavior
    println!("  Glass Transition Temperature Analysis:");
    let cooling_rates = vec![0.1, 1.0, 10.0, 100.0]; // K/min
    let reference_rate = 1.0; // K/min
    
    for &rate in &cooling_rates {
        let shifted_tg = TPEDomain::glass_transition_shift(
            tpu.glass_transition_temp, rate, reference_rate
        )?;
        println!("    • Cooling at {:.1} K/min: Tg = {:.1}°C", 
                 rate, shifted_tg - 273.15);
    }
    println!();
    
    // ===== Heat Transfer Analysis =====
    println!("🔥 Heat Transfer Analysis:");
    println!("-------------------------");
    
    let surface_area = 0.01; // 100 cm²
    let thickness = 0.002; // 2 mm wall thickness
    
    // Conduction through TPE wall
    let temperature_gradient = (80.0 - 20.0) / thickness; // K/m
    let conduction_rate = TPEDomain::heat_conduction_rate(
        tpu.thermal_conductivity, surface_area, temperature_gradient
    );
    println!("  Heat conduction through TPE wall:");
    println!("    • Wall thickness: {:.1} mm", thickness * 1000.0);
    println!("    • Temperature difference: 60°C");
    println!("    • Heat flow rate: {:.2} W", conduction_rate.abs());
    
    // Convective cooling
    let convection_coefficient = 25.0; // W/(m²·K) - natural convection
    let temp_difference = 30.0; // 30K difference
    let convection_rate = TPEDomain::convective_heat_transfer(
        convection_coefficient, surface_area, temp_difference
    );
    println!("    • Convective cooling: {:.2} W", convection_rate);
    
    // Radiative heat transfer (high-temperature application)
    let emissivity = 0.85; // TPE emissivity
    let hot_temp = 373.15; // 100°C
    let ambient_temp = 293.15; // 20°C
    let radiation_rate = TPEDomain::radiative_heat_transfer(
        emissivity, surface_area, hot_temp, ambient_temp
    );
    println!("    • Radiative cooling: {:.2} W", radiation_rate);
    println!();
    
    // ===== Microstructure Simulation =====
    println!("🧬 Microstructure Simulation:");
    println!("-----------------------------");
    
    // Block copolymer analysis
    let molecular_weights = vec![20000.0, 50000.0, 100000.0, 200000.0];
    let chi_parameter = 0.08; // Moderate phase separation
    let composition = 0.25; // 25% hard phase
    
    println!("  Block Copolymer Domain Spacing:");
    for &mw in &molecular_weights {
        let domain_spacing = TPEDomain::block_copolymer_domain_spacing(
            mw, chi_parameter, composition
        )?;
        println!("    • MW {:.0} kDa: Domain spacing = {:.1} nm", 
                 mw / 1000.0, domain_spacing);
    }
    
    // Effective modulus calculation
    let copolymer = BlockCopolymer {
        hard_block_fraction: 0.25,
        soft_block_fraction: 0.75,
        domain_spacing: 15.0, // nm
        hard_block_modulus: 3e9, // 3 GPa (glassy phase)
        soft_block_modulus: 1e6, // 1 MPa (rubbery phase)
        interface_thickness: 2.0, // nm
    };
    
    let effective_modulus = TPEDomain::block_copolymer_modulus(&copolymer);
    println!("    • Effective modulus: {:.1} MPa", effective_modulus / 1e6);
    println!("    • Hard phase contribution: {:.1}%", 
             copolymer.hard_block_fraction * copolymer.hard_block_modulus / effective_modulus * 100.0);
    println!();
    
    // Polymer chain statistics
    println!("  Polymer Chain Statistics:");
    let segment_length = 1.5e-9; // 1.5 nm typical
    let chain_lengths = vec![100, 500, 1000, 5000];
    
    for &n in &chain_lengths {
        let end_to_end = TPEDomain::chain_end_to_end_distance(segment_length, n);
        let radius_of_gyration = TPEDomain::radius_of_gyration(segment_length, n);
        println!("    • {} segments: R_e2e = {:.1} nm, R_g = {:.1} nm", 
                 n, end_to_end * 1e9, radius_of_gyration * 1e9);
    }
    
    // Entropic elasticity
    let temperature = 298.15; // 25°C
    let extension = 80e-9; // 80 nm
    let contour_length = 150e-9; // 150 nm
    let elastic_force = TPEDomain::entropic_elastic_force(
        temperature, segment_length, extension, contour_length
    )?;
    println!("    • Entropic elastic force: {:.2} pN", elastic_force * 1e12);
    
    // Crystallinity analysis
    let amorphous_density = 1150.0; // kg/m³
    let crystalline_density = 1300.0; // kg/m³
    let measured_densities = vec![1150.0, 1190.0, 1225.0, 1260.0, 1300.0];
    
    println!("    • Crystallinity from density:");
    for &density in &measured_densities {
        let crystallinity = TPEDomain::crystallinity_from_density(
            density, amorphous_density, crystalline_density
        )?;
        println!("      - {:.0} kg/m³ → {:.0}% crystalline", density, crystallinity * 100.0);
    }
    println!();
    
    // ===== Injection Molding Analysis =====
    println!("🏭 Injection Molding & Processing:");
    println!("----------------------------------");
    
    // Melt flow analysis
    let melt_viscosity = 800.0; // Pa·s at processing temperature
    let channel_length = 0.15; // 15 cm flow path
    let pressure_drop = 5e6; // 5 MPa injection pressure
    
    // Circular sprue
    let sprue_radius = 0.004; // 4 mm radius
    let sprue_flow = TPEDomain::melt_flow_rate_circular(
        sprue_radius, pressure_drop, melt_viscosity, channel_length
    )?;
    println!("  Melt Flow Analysis:");
    println!("    • Circular sprue (Ø{:.0} mm): {:.2} cm³/s", 
             sprue_radius * 2000.0, sprue_flow * 1e6);
    
    // Rectangular runner
    let runner_width = 0.006; // 6 mm
    let runner_height = 0.003; // 3 mm
    let runner_flow = TPEDomain::melt_flow_rate_rectangular(
        runner_width, runner_height, pressure_drop, melt_viscosity, channel_length
    )?;
    println!("    • Rectangular runner ({}×{} mm): {:.2} cm³/s", 
             runner_width * 1000.0, runner_height * 1000.0, runner_flow * 1e6);
    
    // Temperature-dependent viscosity
    let processing_temps = vec![180.0, 200.0, 220.0, 240.0]; // °C
    let reference_temp = 200.0 + 273.15; // K
    let activation_energy = 45000.0; // J/mol
    
    println!("    • Temperature-dependent viscosity:");
    for &temp_c in &processing_temps {
        let temp_k = temp_c + 273.15;
        let viscosity = TPEDomain::temperature_viscosity(
            melt_viscosity, activation_energy, temp_k, reference_temp
        )?;
        println!("      - {:.0}°C: {:.0} Pa·s", temp_c, viscosity);
    }
    
    // Cooling analysis
    let mold_temp = 60.0; // °C
    let melt_temp = 200.0; // °C
    let time_constant = 45.0; // seconds
    
    println!("    • Cooling curve analysis:");
    let cooling_times = vec![10.0, 20.0, 30.0, 60.0, 120.0]; // seconds
    for &time in &cooling_times {
        let temp = TPEDomain::cooling_curve(melt_temp, mold_temp, time, time_constant);
        println!("      - t = {:.0}s: T = {:.1}°C", time, temp);
    }
    
    // Time to reach ejection temperature
    let ejection_temp = 80.0; // °C
    let cooling_time = TPEDomain::cooling_time_to_temperature(
        melt_temp, mold_temp, ejection_temp, time_constant
    )?;
    println!("    • Time to ejection temperature ({:.0}°C): {:.1} seconds", 
             ejection_temp, cooling_time);
    println!();
    
    // ===== Material Blending & Optimization =====
    println!("⚗️ Material Blending & Optimization:");
    println!("-----------------------------------");
    
    let tpu = TPEMaterial::tpu_standard();
    let tpo = TPEMaterial::tpo_automotive();
    
    // Create blends with different ratios
    let blend_ratios = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    println!("  TPU-TPO Blend Properties:");
    for &ratio in &blend_ratios {
        let blend = TPEDomain::linear_material_blend(&tpu, &tpo, ratio)?;
        println!("    • {:.0}% TPU / {:.0}% TPO:", ratio * 100.0, (1.0 - ratio) * 100.0);
        println!("      - Hardness: {:.1} Shore A", blend.hardness_shore_a);
        println!("      - Tensile: {:.1} MPa", blend.tensile_strength / 1e6);
        println!("      - Elongation: {:.0}%", blend.elongation_at_break);
        println!("      - Modulus: {:.1} MPa", blend.elastic_modulus / 1e6);
    }
    
    // Optimization for target properties
    let target_modulus = 200e6; // 200 MPa target (between TPU and TPO)
    let optimal_ratio = TPEDomain::optimize_composition_for_modulus(
        target_modulus, tpu.elastic_modulus, tpo.elastic_modulus
    )?;
    println!("    • Optimal blend for {:.0} MPa modulus: {:.1}% TPU / {:.1}% TPO", 
             target_modulus / 1e6, optimal_ratio * 100.0, (1.0 - optimal_ratio) * 100.0);
    
    // Performance scoring
    let target_hardness = 80.0; // Shore A
    let target_strength = 25e6; // Pa
    let target_elongation = 400.0; // %
    let weights = (0.3, 0.5, 0.2); // Prioritize strength
    
    println!("    • Material performance scores (target: {}A, {} MPa, {}%):", 
             target_hardness, target_strength / 1e6, target_elongation);
    
    for (name, material) in &materials {
        let score = TPEDomain::material_performance_score(
            material, target_hardness, target_strength, target_elongation, weights
        );
        println!("      - {}: {:.3}", name, score);
    }
    println!();
    
    // ===== Practical Applications =====
    println!("🔧 Practical Applications:");
    println!("--------------------------");
    
    // Cable jacket flexibility
    println!("  Cable Jacket Design:");
    let cable_diameters = vec![0.005, 0.008, 0.012, 0.020]; // Various cable sizes
    let min_bend_radius = 0.05; // 50 mm minimum bend
    let allowable_stress = 5e6; // 5 MPa
    
    for &diameter in &cable_diameters {
        let flexibility = TPEDomain::cable_flexibility_factor(
            min_bend_radius, diameter, tpu.elastic_modulus, allowable_stress
        )?;
        let status = if flexibility > 1.0 { "✅ Flexible" } else { "⚠️ Stiff" };
        println!("    • Ø{:.1} mm cable: Flexibility factor = {:.2} {}", 
                 diameter * 1000.0, flexibility, status);
    }
    
    // Seal compression analysis
    println!("    • O-Ring seal compression:");
    let initial_thickness = 0.003; // 3 mm O-ring
    let compression_levels = vec![0.1, 0.15, 0.2, 0.25]; // 10-25% compression
    let contact_area = 0.0005; // 5 cm²
    
    for &compression in &compression_levels {
        let compressed_thickness = initial_thickness * (1.0 - compression);
        let force = TPEDomain::seal_compression_force(
            initial_thickness, compressed_thickness, contact_area, tpu.elastic_modulus
        )?;
        println!("      - {:.0}% compression: {:.1} N force", compression * 100.0, force);
    }
    
    // Grip surface analysis
    println!("    • Grip surface friction:");
    let surface_conditions = vec![
        ("Smooth", 5.0),
        ("Textured", 25.0),
        ("Aggressive", 100.0),
    ];
    let contact_pressure = 2e5; // 200 kPa grip pressure
    
    for (condition, roughness) in &surface_conditions {
        let friction_coeff = TPEDomain::grip_friction_coefficient(
            *roughness, contact_pressure, tpu.hardness_shore_a
        );
        println!("      - {} surface ({:.0} µm): µ = {:.2}", condition, roughness, friction_coeff);
    }
    println!();
    
    // ===== Advanced Materials Engineering =====
    println!("🎯 Advanced Materials Engineering:");
    println!("---------------------------------");
    
    // Temperature-dependent property prediction
    let base_modulus = tpu.elastic_modulus;
    let reference_temp = 23.0 + 273.15; // 23°C
    let temp_coefficient = -0.002; // -0.2%/K
    
    println!("  Temperature-dependent modulus prediction:");
    let service_temps = vec![-40.0, 0.0, 23.0, 60.0, 100.0];
    for &temp_c in &service_temps {
        let temp_k = temp_c + 273.15;
        let adjusted_modulus = TPEDomain::temperature_property_interpolation(
            base_modulus, temp_k, reference_temp, temp_coefficient
        );
        println!("    • {:.0}°C: {:.1} MPa ({:+.1}%)", 
                 temp_c, adjusted_modulus / 1e6, 
                 (adjusted_modulus / base_modulus - 1.0) * 100.0);
    }
    
    // Phase transition energy calculations
    println!("    • Phase transition energy requirements:");
    let part_mass = 0.05; // 50g part
    let melting_enthalpy = 80000.0; // J/kg typical for TPE
    let crystallization_enthalpy = 60000.0; // J/kg
    
    let melt_energy = TPEDomain::phase_transition_energy(part_mass, melting_enthalpy);
    let crystallization_energy = TPEDomain::phase_transition_energy(part_mass, crystallization_enthalpy);
    
    println!("      - Melting energy: {:.1} kJ", melt_energy / 1000.0);
    println!("      - Crystallization energy: {:.1} kJ", crystallization_energy / 1000.0);
    println!();
    
    // ===== Performance Benchmarking =====
    println!("⚡ Performance Benchmarking:");
    println!("===========================");
    
    let start_time = std::time::Instant::now();
    let iterations = 100000;
    
    // Thermal expansion benchmark
    for _ in 0..iterations {
        let _ = TPEDomain::thermal_expansion(1.0, 150e-6, 50.0);
    }
    let thermal_time = start_time.elapsed();
    
    // Block copolymer benchmark
    let block_start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = TPEDomain::block_copolymer_domain_spacing(50000.0, 0.1, 0.3);
    }
    let block_time = block_start.elapsed();
    
    // Melt flow benchmark
    let flow_start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = TPEDomain::melt_flow_rate_circular(0.005, 5e6, 1000.0, 0.1);
    }
    let flow_time = flow_start.elapsed();
    
    println!("  Computational Performance ({} iterations):", iterations);
    println!("    • Thermal expansion: {:.0} calculations/second", 
             iterations as f64 / thermal_time.as_secs_f64());
    println!("    • Block copolymer analysis: {:.0} calculations/second", 
             iterations as f64 / block_time.as_secs_f64());
    println!("    • Melt flow analysis: {:.0} calculations/second", 
             iterations as f64 / flow_time.as_secs_f64());
    println!();
    
    // ===== Summary =====
    println!("✨ TPE Domain Demo Complete!");
    println!("============================");
    println!("Features Demonstrated:");
    println!("  ✅ Comprehensive TPE material database (TPU, TPO, TPS)");
    println!("  ✅ Thermal response modeling (expansion, heat transfer, phase transitions)");
    println!("  ✅ Microstructure simulation (block copolymers, chain statistics)");
    println!("  ✅ Injection molding mathematics (flow rates, cooling curves)");
    println!("  ✅ Material blending and property optimization");
    println!("  ✅ Practical applications (cables, seals, grips)");
    println!("  ✅ Temperature-dependent property prediction");
    println!("  ✅ High-performance calculations for real-time analysis");
    println!();
    println!("🎯 Applications Ready:");
    println!("  • Cable and wire jacketing design");
    println!("  • Automotive sealing systems");
    println!("  • Consumer product grips and handles");
    println!("  • Medical device components");
    println!("  • Industrial gaskets and bushings");
    println!("  • 3D printing and additive manufacturing");
    println!("  • Materials research and development");
    
    Ok(())
}