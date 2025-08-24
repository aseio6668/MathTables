use mathtables::core::{TPEMaterial, BlockCopolymer, TransitionType, ThermalTransition};
use mathtables::domains::TPEDomain;
use std::f64::consts::PI;

#[test]
fn test_tpe_material_presets() {
    let tpu = TPEMaterial::tpu_standard();
    assert_eq!(tpu.name, "TPU Standard Grade");
    assert_eq!(tpu.hardness_shore_a, 85.0);
    assert!((tpu.tensile_strength - 35e6).abs() < 1e3);
    assert_eq!(tpu.elongation_at_break, 450.0);
    
    let tpo = TPEMaterial::tpo_automotive();
    assert_eq!(tpo.name, "TPO Automotive Grade");
    assert_eq!(tpo.hardness_shore_a, 75.0);
    assert!((tpo.density - 900.0).abs() < 1.0);
    
    let tps = TPEMaterial::tps_consumer();
    assert_eq!(tps.name, "TPS Consumer Grade");
    assert_eq!(tps.hardness_shore_a, 70.0);
    assert!(tps.elongation_at_break > 400.0);
}

#[test]
fn test_thermal_expansion() {
    let original_length = 1.0; // 1 meter
    let expansion_coeff = 150e-6; // 150 µm/m/K (typical TPE)
    let temp_change = 50.0; // 50K temperature rise
    
    let expansion = TPEDomain::thermal_expansion(original_length, expansion_coeff, temp_change);
    let expected_expansion = 150e-6 * 1.0 * 50.0;
    
    assert!((expansion - expected_expansion).abs() < 1e-10);
    assert_eq!(expansion, 7.5e-3); // 7.5 mm expansion
}

#[test]
fn test_volumetric_thermal_expansion() {
    let original_volume = 0.001; // 1 liter
    let expansion_coeff = 180e-6; // TPU coefficient
    let temp_change = 30.0; // 30K rise
    
    let volume_expansion = TPEDomain::volumetric_thermal_expansion(
        original_volume, expansion_coeff, temp_change
    );
    
    let expected = 3.0 * expansion_coeff * original_volume * temp_change;
    assert!((volume_expansion - expected).abs() < 1e-12);
}

#[test]
fn test_heat_transfer_functions() {
    // Heat conduction
    let thermal_conductivity = 0.25; // W/(m·K) - TPU
    let area = 0.01; // 100 cm²
    let temperature_gradient = 1000.0; // 1000 K/m
    
    let conduction_rate = TPEDomain::heat_conduction_rate(
        thermal_conductivity, area, temperature_gradient
    );
    assert!((conduction_rate + 2.5).abs() < 1e-10); // Negative due to direction
    
    // Convective heat transfer
    let h_coefficient = 25.0; // W/(m²·K)
    let temp_diff = 50.0; // 50K difference
    
    let convection_rate = TPEDomain::convective_heat_transfer(h_coefficient, area, temp_diff);
    assert!((convection_rate - 12.5).abs() < 1e-10);
    
    // Radiative heat transfer
    let emissivity = 0.9; // High emissivity
    let t1 = 373.15; // 100°C
    let t2 = 293.15; // 20°C
    
    let radiation_rate = TPEDomain::radiative_heat_transfer(emissivity, area, t1, t2);
    assert!(radiation_rate > 0.0);
    assert!(radiation_rate < 100.0); // Sanity check
}

#[test]
fn test_glass_transition_shift() {
    let base_tg = 213.15; // -60°C in Kelvin
    let cooling_rate = 10.0; // K/min
    let reference_rate = 1.0; // K/min
    
    let shifted_tg = TPEDomain::glass_transition_shift(base_tg, cooling_rate, reference_rate).unwrap();
    
    // Higher cooling rate should increase Tg
    assert!(shifted_tg > base_tg);
    
    // Test error conditions
    assert!(TPEDomain::glass_transition_shift(base_tg, -1.0, reference_rate).is_err());
    assert!(TPEDomain::glass_transition_shift(base_tg, cooling_rate, 0.0).is_err());
}

#[test]
fn test_phase_transition_energy() {
    let mass = 0.5; // 500g
    let specific_enthalpy = 50000.0; // 50 kJ/kg (typical for TPE melting)
    
    let total_energy = TPEDomain::phase_transition_energy(mass, specific_enthalpy);
    assert_eq!(total_energy, 25000.0); // 25 kJ
}

#[test]
fn test_block_copolymer_domain_spacing() {
    let molecular_weight = 50000.0; // 50 kDa
    let chi_parameter = 0.1; // Moderate incompatibility
    let composition = 0.3; // 30% hard phase
    
    let domain_spacing = TPEDomain::block_copolymer_domain_spacing(
        molecular_weight, chi_parameter, composition
    ).unwrap();
    
    assert!(domain_spacing > 0.0);
    assert!(domain_spacing < 1000.0); // Reasonable nanoscale size
    
    // Test error conditions
    assert!(TPEDomain::block_copolymer_domain_spacing(molecular_weight, chi_parameter, 0.0).is_err());
    assert!(TPEDomain::block_copolymer_domain_spacing(molecular_weight, chi_parameter, 1.0).is_err());
    assert!(TPEDomain::block_copolymer_domain_spacing(molecular_weight, -0.1, composition).is_err());
}

#[test]
fn test_block_copolymer_modulus() {
    let copolymer = BlockCopolymer {
        hard_block_fraction: 0.3,
        soft_block_fraction: 0.7,
        domain_spacing: 20.0, // nm
        hard_block_modulus: 2e9, // 2 GPa
        soft_block_modulus: 10e6, // 10 MPa
        interface_thickness: 2.0, // nm
    };
    
    let effective_modulus = TPEDomain::block_copolymer_modulus(&copolymer);
    let expected = 0.3 * 2e9 + 0.7 * 10e6;
    
    assert!((effective_modulus - expected).abs() < 1e3);
    assert!(effective_modulus > copolymer.soft_block_modulus);
    assert!(effective_modulus < copolymer.hard_block_modulus);
}

#[test]
fn test_polymer_chain_statistics() {
    let segment_length = 1.5e-9; // 1.5 nm (typical C-C bond)
    let n_segments = 1000;
    
    // End-to-end distance
    let end_to_end = TPEDomain::chain_end_to_end_distance(segment_length, n_segments);
    let expected_e2e = segment_length * (n_segments as f64).sqrt();
    assert!((end_to_end - expected_e2e).abs() < 1e-12);
    
    // Radius of gyration
    let rg = TPEDomain::radius_of_gyration(segment_length, n_segments);
    let expected_rg = segment_length * (n_segments as f64 / 6.0).sqrt();
    assert!((rg - expected_rg).abs() < 1e-12);
    
    // Relationship check: Rg < R_e2e
    assert!(rg < end_to_end);
}

#[test]
fn test_entropic_elastic_force() {
    let temperature = 300.0; // K (room temperature)
    let segment_length = 1e-9; // 1 nm
    let extension = 50e-9; // 50 nm
    let contour_length = 100e-9; // 100 nm
    
    let force = TPEDomain::entropic_elastic_force(
        temperature, segment_length, extension, contour_length
    ).unwrap();
    
    assert!(force > 0.0); // Extension should create restoring force
    assert!(force < 1e-9); // Should be in picoNewton range
    
    // Test error condition: extension >= contour length
    assert!(TPEDomain::entropic_elastic_force(
        temperature, segment_length, contour_length, contour_length
    ).is_err());
}

#[test]
fn test_crystallinity_from_density() {
    let amorphous_density = 900.0; // kg/m³
    let crystalline_density = 1000.0; // kg/m³
    let measured_density = 950.0; // kg/m³ (50% crystalline)
    
    let crystallinity = TPEDomain::crystallinity_from_density(
        measured_density, amorphous_density, crystalline_density
    ).unwrap();
    
    assert!((crystallinity - 0.5).abs() < 1e-10);
    
    // Test edge cases
    let fully_amorphous = TPEDomain::crystallinity_from_density(
        amorphous_density, amorphous_density, crystalline_density
    ).unwrap();
    assert!((fully_amorphous - 0.0).abs() < 1e-10);
    
    let fully_crystalline = TPEDomain::crystallinity_from_density(
        crystalline_density, amorphous_density, crystalline_density
    ).unwrap();
    assert!((fully_crystalline - 1.0).abs() < 1e-10);
    
    // Test error condition
    assert!(TPEDomain::crystallinity_from_density(
        950.0, 1000.0, 900.0 // Invalid: crystalline < amorphous
    ).is_err());
}

#[test]
fn test_melt_flow_rates() {
    // Circular channel (Poiseuille flow)
    let radius = 0.005; // 5 mm radius
    let pressure_drop = 1e6; // 1 MPa
    let viscosity = 1000.0; // 1000 Pa·s (typical TPE melt)
    let length = 0.1; // 10 cm
    
    let circular_flow = TPEDomain::melt_flow_rate_circular(
        radius, pressure_drop, viscosity, length
    ).unwrap();
    
    let expected_circular = (PI * radius.powi(4) * pressure_drop) / (8.0 * viscosity * length);
    assert!((circular_flow - expected_circular).abs() < 1e-12);
    
    // Rectangular channel
    let width = 0.02; // 20 mm
    let height = 0.002; // 2 mm
    
    let rectangular_flow = TPEDomain::melt_flow_rate_rectangular(
        width, height, pressure_drop, viscosity, length
    ).unwrap();
    
    let expected_rectangular = (width * height.powi(3) * pressure_drop) / (12.0 * viscosity * length);
    assert!((rectangular_flow - expected_rectangular).abs() < 1e-12);
    
    // Test error conditions
    assert!(TPEDomain::melt_flow_rate_circular(0.0, pressure_drop, viscosity, length).is_err());
    assert!(TPEDomain::melt_flow_rate_circular(radius, pressure_drop, 0.0, length).is_err());
}

#[test]
fn test_temperature_viscosity() {
    let reference_viscosity = 1000.0; // Pa·s
    let activation_energy = 50000.0; // J/mol
    let temperature = 473.15; // 200°C
    let reference_temperature = 453.15; // 180°C
    
    let viscosity = TPEDomain::temperature_viscosity(
        reference_viscosity, activation_energy, temperature, reference_temperature
    ).unwrap();
    
    // Higher temperature should reduce viscosity
    assert!(viscosity < reference_viscosity);
    
    // Test same temperature
    let same_temp_viscosity = TPEDomain::temperature_viscosity(
        reference_viscosity, activation_energy, reference_temperature, reference_temperature
    ).unwrap();
    assert!((same_temp_viscosity - reference_viscosity).abs() < 1e-6);
    
    // Test error condition
    assert!(TPEDomain::temperature_viscosity(
        reference_viscosity, activation_energy, -10.0, reference_temperature
    ).is_err());
}

#[test]
fn test_cooling_curves() {
    let initial_temp = 200.0; // °C
    let ambient_temp = 25.0; // °C
    let time_constant = 300.0; // 5 minutes
    
    // Test at t = 0
    let temp_at_zero = TPEDomain::cooling_curve(initial_temp, ambient_temp, 0.0, time_constant);
    assert!((temp_at_zero - initial_temp).abs() < 1e-10);
    
    // Test at large time (should approach ambient)
    let temp_at_large_time = TPEDomain::cooling_curve(initial_temp, ambient_temp, 3000.0, time_constant);
    assert!(temp_at_large_time < initial_temp);
    assert!((temp_at_large_time - ambient_temp).abs() < 1.0);
    
    // Test cooling time calculation
    let target_temp = 100.0; // °C
    let cooling_time = TPEDomain::cooling_time_to_temperature(
        initial_temp, ambient_temp, target_temp, time_constant
    ).unwrap();
    
    assert!(cooling_time > 0.0);
    
    // Verify the calculation by using it in cooling curve
    let verify_temp = TPEDomain::cooling_curve(initial_temp, ambient_temp, cooling_time, time_constant);
    assert!((verify_temp - target_temp).abs() < 0.1);
}

#[test]
fn test_solidification_front_velocity() {
    let thermal_diffusivity = 1e-7; // m²/s
    let latent_heat = 100000.0; // J/kg
    let specific_heat = 2000.0; // J/(kg·K)
    let temperature_gradient = 1000.0; // K/m
    
    let velocity = TPEDomain::solidification_front_velocity(
        thermal_diffusivity, latent_heat, specific_heat, temperature_gradient
    ).unwrap();
    
    assert!(velocity > 0.0);
    assert!(velocity < 1e-3); // Should be mm/s scale
    
    // Test error conditions
    assert!(TPEDomain::solidification_front_velocity(
        0.0, latent_heat, specific_heat, temperature_gradient
    ).is_err());
    assert!(TPEDomain::solidification_front_velocity(
        thermal_diffusivity, 0.0, specific_heat, temperature_gradient
    ).is_err());
}

#[test]
fn test_material_blending() {
    let tpu = TPEMaterial::tpu_standard();
    let tpo = TPEMaterial::tpo_automotive();
    let fraction_tpu = 0.7;
    
    let blend = TPEDomain::linear_material_blend(&tpu, &tpo, fraction_tpu).unwrap();
    
    // Check that blended properties are weighted averages
    let expected_hardness = tpu.hardness_shore_a * fraction_tpu + tpo.hardness_shore_a * (1.0 - fraction_tpu);
    assert!((blend.hardness_shore_a - expected_hardness).abs() < 1e-10);
    
    let expected_density = tpu.density * fraction_tpu + tpo.density * (1.0 - fraction_tpu);
    assert!((blend.density - expected_density).abs() < 1e-10);
    
    // Test error condition
    assert!(TPEDomain::linear_material_blend(&tpu, &tpo, 1.5).is_err());
}

#[test]
fn test_temperature_property_interpolation() {
    let base_property = 25e6; // 25 MPa elastic modulus
    let temperature = 323.15; // 50°C
    let reference_temperature = 298.15; // 25°C
    let temperature_coefficient = -0.002; // -0.2%/K (typical polymer behavior)
    
    let adjusted_property = TPEDomain::temperature_property_interpolation(
        base_property, temperature, reference_temperature, temperature_coefficient
    );
    
    let expected = base_property * (1.0 + temperature_coefficient * (temperature - reference_temperature));
    assert!((adjusted_property - expected).abs() < 1e-6);
    
    // Higher temperature should reduce modulus for typical polymers
    assert!(adjusted_property < base_property);
}

#[test]
fn test_composition_optimization() {
    let target_modulus = 15e6; // 15 MPa target
    let material_a_modulus = 25e6; // 25 MPa
    let material_b_modulus = 8e6; // 8 MPa
    
    let optimal_fraction = TPEDomain::optimize_composition_for_modulus(
        target_modulus, material_a_modulus, material_b_modulus
    ).unwrap();
    
    let expected_fraction = (target_modulus - material_b_modulus) / 
                           (material_a_modulus - material_b_modulus);
    assert!((optimal_fraction - expected_fraction).abs() < 1e-10);
    
    // Verify the result
    let achieved_modulus = material_a_modulus * optimal_fraction + 
                          material_b_modulus * (1.0 - optimal_fraction);
    assert!((achieved_modulus - target_modulus).abs() < 1e-6);
    
    // Test unachievable target
    assert!(TPEDomain::optimize_composition_for_modulus(
        30e6, material_a_modulus, material_b_modulus
    ).is_err());
}

#[test]
fn test_material_performance_score() {
    let material = TPEMaterial::tpu_standard();
    let target_hardness = 85.0; // Exact match
    let target_strength = 35e6; // Exact match
    let target_elongation = 450.0; // Exact match
    let weights = (0.3, 0.4, 0.3); // Equal-ish weighting
    
    let perfect_score = TPEDomain::material_performance_score(
        &material, target_hardness, target_strength, target_elongation, weights
    );
    
    // Perfect match should give score close to 1.0
    assert!((perfect_score - 1.0).abs() < 1e-10);
    
    // Test with mismatched targets
    let imperfect_score = TPEDomain::material_performance_score(
        &material, 50.0, 20e6, 300.0, weights
    );
    
    assert!(imperfect_score < perfect_score);
    assert!(imperfect_score >= 0.0);
}

#[test]
fn test_cable_flexibility_analysis() {
    let bend_radius = 0.05; // 50 mm bend radius
    let cable_diameter = 0.01; // 10 mm diameter
    let elastic_modulus = 25e6; // 25 MPa
    let target_stress = 5e6; // 5 MPa allowable stress
    
    let flexibility_factor = TPEDomain::cable_flexibility_factor(
        bend_radius, cable_diameter, elastic_modulus, target_stress
    ).unwrap();
    
    assert!(flexibility_factor > 0.0);
    
    // If flexibility factor > 1, the cable can handle the bend
    // If < 1, the bend is too severe
    
    // Test error conditions
    assert!(TPEDomain::cable_flexibility_factor(0.0, cable_diameter, elastic_modulus, target_stress).is_err());
}

#[test]
fn test_seal_compression_analysis() {
    let initial_thickness = 0.005; // 5 mm
    let compressed_thickness = 0.004; // 4 mm (20% compression)
    let contact_area = 0.001; // 10 cm²
    let elastic_modulus = 10e6; // 10 MPa
    
    let compression_force = TPEDomain::seal_compression_force(
        initial_thickness, compressed_thickness, contact_area, elastic_modulus
    ).unwrap();
    
    let expected_strain = (initial_thickness - compressed_thickness) / initial_thickness;
    let expected_stress = elastic_modulus * expected_strain;
    let expected_force = expected_stress * contact_area;
    
    assert!((compression_force - expected_force).abs() < 1e-6);
    assert!(compression_force > 0.0);
    
    // Test error condition
    assert!(TPEDomain::seal_compression_force(
        initial_thickness, initial_thickness + 0.001, contact_area, elastic_modulus
    ).is_err());
}

#[test]
fn test_grip_friction_coefficient() {
    let surface_roughness = 25.0; // µm
    let contact_pressure = 1e5; // 100 kPa
    let hardness_shore_a = 70.0; // Medium hardness
    
    let friction_coeff = TPEDomain::grip_friction_coefficient(
        surface_roughness, contact_pressure, hardness_shore_a
    );
    
    assert!(friction_coeff > 0.0);
    assert!(friction_coeff < 2.0); // Reasonable upper bound
    
    // Softer material should have higher friction
    let soft_friction = TPEDomain::grip_friction_coefficient(
        surface_roughness, contact_pressure, 40.0
    );
    assert!(soft_friction > friction_coeff);
    
    // Rougher surface should have higher friction
    let rough_friction = TPEDomain::grip_friction_coefficient(
        100.0, contact_pressure, hardness_shore_a
    );
    assert!(rough_friction > friction_coeff);
}

#[test]
fn test_integration_cable_jacket_design() {
    // Real-world cable jacket design scenario
    let tpu = TPEMaterial::tpu_standard();
    let cable_diameter = 0.008; // 8 mm cable
    let min_bend_radius = 0.05; // 50 mm minimum bend
    let max_stress = 5e6; // 5 MPa allowable stress
    
    // Check flexibility
    let flexibility = TPEDomain::cable_flexibility_factor(
        min_bend_radius, cable_diameter, tpu.elastic_modulus, max_stress
    ).unwrap();
    
    // Check thermal expansion over temperature range
    let cable_length = 10.0; // 10 meter cable
    let temp_range = 80.0; // -40°C to +40°C
    let thermal_expansion = TPEDomain::thermal_expansion(
        cable_length, tpu.thermal_expansion_coeff, temp_range
    );
    
    // Design should be feasible
    assert!(flexibility > 0.5); // Reasonable flexibility
    assert!(thermal_expansion < 0.15); // < 15 cm expansion over 10m (realistic for TPE)
    
    println!("Cable Design Results:");
    println!("  Material: {}", tpu.name);
    println!("  Flexibility Factor: {:.2}", flexibility);
    println!("  Thermal Expansion: {:.1} mm over {}m", thermal_expansion * 1000.0, cable_length);
    println!("  Status: {}", if flexibility > 1.0 { "✅ Flexible enough" } else { "⚠️ Check design" });
}