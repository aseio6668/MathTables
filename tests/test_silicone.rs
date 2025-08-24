use mathtables::prelude::*;

#[test]
fn test_silicone_domain_initialization() {
    let domain = SiliconeDomain::new();
    
    let viscosity = domain.calculate_viscosity(298.15, 10000.0, 1.0).unwrap();
    assert!(viscosity > 0.0);
    
    // Test with different parameters
    let high_temp_viscosity = domain.calculate_viscosity(373.15, 10000.0, 1.0).unwrap();
    assert!(high_temp_viscosity < viscosity); // Viscosity decreases with temperature
}

#[test]
fn test_silicone_viscosity_temperature_dependence() {
    let domain = SiliconeDomain::new();
    
    let viscosity_25c = domain.calculate_viscosity(298.15, 50000.0, 0.0).unwrap();
    let viscosity_100c = domain.calculate_viscosity(373.15, 50000.0, 0.0).unwrap();
    
    assert!(viscosity_100c < viscosity_25c);
    assert!(viscosity_25c > 0.0);
    assert!(viscosity_100c > 0.0);
}

#[test]
fn test_silicone_viscosity_molecular_weight_dependence() {
    let domain = SiliconeDomain::new();
    
    let low_mw_viscosity = domain.calculate_viscosity(298.15, 1000.0, 0.0).unwrap();
    let high_mw_viscosity = domain.calculate_viscosity(298.15, 100000.0, 0.0).unwrap();
    
    assert!(high_mw_viscosity > low_mw_viscosity);
}

#[test]
fn test_silicone_viscosity_shear_thinning() {
    let domain = SiliconeDomain::new();
    
    let low_shear_viscosity = domain.calculate_viscosity(298.15, 50000.0, 0.1).unwrap();
    let high_shear_viscosity = domain.calculate_viscosity(298.15, 50000.0, 100.0).unwrap();
    
    assert!(high_shear_viscosity < low_shear_viscosity); // Shear thinning behavior
}

#[test]
fn test_siloxane_chain_length_distribution() {
    let std_dev = siloxane_chain_length_distribution(1000.0, 2.0).unwrap();
    assert!(std_dev > 0.0);
    
    // Higher polydispersity should give higher standard deviation
    let high_pdi_std_dev = siloxane_chain_length_distribution(1000.0, 3.0).unwrap();
    assert!(high_pdi_std_dev > std_dev);
    
    // Test error conditions
    assert!(siloxane_chain_length_distribution(-100.0, 2.0).is_err());
    assert!(siloxane_chain_length_distribution(1000.0, 0.5).is_err());
}

#[test]
fn test_siloxane_thermal_expansion() {
    let original_length = 100.0;
    let temperature_change = 50.0;
    let expansion_coeff = 2.4e-4; // Typical for PDMS
    
    let expanded_length = siloxane_thermal_expansion(original_length, temperature_change, expansion_coeff);
    let expected = original_length * (1.0 + expansion_coeff * temperature_change);
    
    assert!((expanded_length - expected).abs() < 1e-10);
    assert!(expanded_length > original_length);
    
    // Cooling should shrink
    let cooled_length = siloxane_thermal_expansion(original_length, -50.0, expansion_coeff);
    assert!(cooled_length < original_length);
}

#[test]
fn test_siloxane_thermal_conductivity() {
    let k_base = siloxane_thermal_conductivity(298.15, 0.0, 0.0).unwrap();
    let k_crosslinked = siloxane_thermal_conductivity(298.15, 0.1, 0.0).unwrap();
    let k_filled = siloxane_thermal_conductivity(298.15, 0.0, 0.3).unwrap();
    
    assert!(k_base > 0.0);
    assert!(k_crosslinked > k_base); // Crosslinking increases conductivity
    assert!(k_filled > k_base); // Filler increases conductivity significantly
    
    // Test error conditions
    assert!(siloxane_thermal_conductivity(-100.0, 0.1, 0.2).is_err());
    assert!(siloxane_thermal_conductivity(298.15, -0.1, 0.2).is_err());
    assert!(siloxane_thermal_conductivity(298.15, 0.1, 1.5).is_err());
}

#[test]
fn test_siloxane_gas_permeability() {
    // Test with different gas molecule sizes
    let perm_h2 = siloxane_gas_permeability(2.89, 298.15, 0.1).unwrap(); // Hydrogen
    let perm_o2 = siloxane_gas_permeability(3.46, 298.15, 0.1).unwrap(); // Oxygen
    let perm_n2 = siloxane_gas_permeability(3.64, 298.15, 0.1).unwrap(); // Nitrogen
    
    assert!(perm_h2 > perm_o2); // Smaller molecules permeate faster
    assert!(perm_o2 > perm_n2);
    assert!(perm_n2 > 0.0);
    
    // Higher temperature should increase permeability
    let perm_high_temp = siloxane_gas_permeability(3.46, 373.15, 0.1).unwrap();
    assert!(perm_high_temp > perm_o2);
    
    // Higher crosslinking should decrease permeability
    let perm_high_crosslink = siloxane_gas_permeability(3.46, 298.15, 0.5).unwrap();
    assert!(perm_high_crosslink < perm_o2);
}

#[test]
fn test_silicone_elastic_modulus() {
    let modulus_base = silicone_elastic_modulus(298.15, 0.001, 0.0).unwrap();
    let modulus_crosslinked = silicone_elastic_modulus(298.15, 0.01, 0.0).unwrap();
    let modulus_filled = silicone_elastic_modulus(298.15, 0.001, 0.2).unwrap();
    
    assert!(modulus_base > 0.0);
    assert!(modulus_crosslinked > modulus_base); // Higher crosslinking increases modulus
    assert!(modulus_filled > modulus_base); // Filler increases modulus
    
    // Temperature effect
    let modulus_high_temp = silicone_elastic_modulus(373.15, 0.001, 0.0).unwrap();
    assert!(modulus_high_temp > modulus_base); // Modulus increases with temperature in rubber
}

#[test]
fn test_siloxane_glass_transition_temperature() {
    let tg_low_mw = siloxane_glass_transition_temperature(1000.0, 0.0).unwrap();
    let tg_high_mw = siloxane_glass_transition_temperature(100000.0, 0.0).unwrap();
    let tg_crosslinked = siloxane_glass_transition_temperature(10000.0, 0.01).unwrap();
    
    assert!(tg_high_mw < tg_low_mw); // Higher MW gives lower Tg (more flexible)
    assert!(tg_crosslinked > siloxane_glass_transition_temperature(10000.0, 0.0).unwrap()); // Crosslinking raises Tg
}

#[test]
fn test_silicone_surface_energy() {
    let surface_energy = silicone_surface_energy(298.15, 0.8).unwrap();
    let surface_energy_high_temp = silicone_surface_energy(373.15, 0.8).unwrap();
    let surface_energy_low_flexibility = silicone_surface_energy(298.15, 0.2).unwrap();
    
    assert!(surface_energy > 0.0);
    assert!(surface_energy_high_temp < surface_energy); // Surface energy decreases with temperature
    assert!(surface_energy_low_flexibility > surface_energy); // Lower flexibility = higher surface energy
}

#[test]
fn test_siloxane_dielectric_constant() {
    let dielectric_low_freq = siloxane_dielectric_constant(1000.0, 298.15, 0.0).unwrap();
    let dielectric_high_freq = siloxane_dielectric_constant(1e9, 298.15, 0.0).unwrap();
    let dielectric_wet = siloxane_dielectric_constant(1000.0, 298.15, 0.1).unwrap();
    
    assert!(dielectric_low_freq > 2.0);
    assert!(dielectric_high_freq < dielectric_low_freq); // Frequency dependence
    assert!(dielectric_wet > dielectric_low_freq); // Moisture increases dielectric constant
}

#[test]
fn test_siloxane_biocompatibility_score() {
    let score_good = siloxane_biocompatibility_score(0.001, 0.1, 0.7).unwrap();
    let score_poor = siloxane_biocompatibility_score(0.1, 10.0, 0.3).unwrap();
    
    assert!(score_good >= 0.0 && score_good <= 100.0);
    assert!(score_poor >= 0.0 && score_poor <= 100.0);
    assert!(score_good > score_poor); // Better conditions = higher score
    
    // Test maximum score doesn't exceed 100
    let score_perfect = siloxane_biocompatibility_score(0.0, 0.0, 1.0).unwrap();
    assert!(score_perfect <= 100.0);
}

#[test]
fn test_siloxane_cure_kinetics() {
    // Test at different temperatures with very short time and high activation energy
    let cure_low_temp = siloxane_cure_kinetics(298.15, 0.00001, 0.1, 80000.0).unwrap();
    let cure_high_temp = siloxane_cure_kinetics(373.15, 0.00001, 0.1, 80000.0).unwrap();
    
    assert!(cure_low_temp >= 0.0 && cure_low_temp <= 1.0);
    assert!(cure_high_temp >= 0.0 && cure_high_temp <= 1.0);
    assert!(cure_high_temp > cure_low_temp); // Higher temperature = faster cure
    
    // Test with different catalyst concentrations
    let cure_low_cat = siloxane_cure_kinetics(298.15, 0.00001, 0.1, 80000.0).unwrap();
    let cure_high_cat = siloxane_cure_kinetics(298.15, 0.0001, 0.1, 80000.0).unwrap();
    
    assert!(cure_high_cat > cure_low_cat); // More catalyst = faster cure
    
    // Test time progression
    let cure_short = siloxane_cure_kinetics(298.15, 0.00001, 0.01, 80000.0).unwrap();
    let cure_long = siloxane_cure_kinetics(298.15, 0.00001, 1.0, 80000.0).unwrap();
    
    assert!(cure_long > cure_short); // Longer time = more cure
}

#[test]
fn test_siloxane_degradation_rate() {
    let base_rate = siloxane_degradation_rate(298.15, 0.0, 0.0).unwrap();
    let thermal_rate = siloxane_degradation_rate(373.15, 0.0, 0.0).unwrap();
    let uv_rate = siloxane_degradation_rate(298.15, 10.0, 0.0).unwrap();
    let oxidative_rate = siloxane_degradation_rate(298.15, 0.0, 0.21).unwrap(); // Air oxygen
    
    assert!(base_rate > 0.0);
    assert!(thermal_rate > base_rate); // Higher temperature = faster degradation
    assert!(uv_rate > base_rate); // UV exposure = faster degradation
    assert!(oxidative_rate > base_rate); // Oxygen = faster degradation
}

#[test]
fn test_siloxane_contact_angle() {
    // Water on silicone (hydrophobic)
    let angle_water = siloxane_contact_angle(20.0, 72.8, 40.0).unwrap();
    assert!(angle_water >= 90.0); // Should be hydrophobic
    assert!(angle_water <= 180.0);
    
    // Test edge cases
    let angle_perfect_wetting = siloxane_contact_angle(100.0, 72.8, 27.8).unwrap();
    assert!(angle_perfect_wetting >= 0.0);
    
    let angle_complete_non_wetting = siloxane_contact_angle(10.0, 72.8, 82.8).unwrap();
    assert!(angle_complete_non_wetting <= 180.0);
}

#[test]
fn test_siloxane_chain_entanglement_modulus() {
    // Below entanglement molecular weight
    let modulus_low_mw = siloxane_chain_entanglement_modulus(5000.0, 298.15).unwrap();
    assert_eq!(modulus_low_mw, 0.0); // No entanglements
    
    // Above entanglement molecular weight
    let modulus_high_mw = siloxane_chain_entanglement_modulus(50000.0, 298.15).unwrap();
    assert!(modulus_high_mw > 0.0);
    
    // Temperature dependence
    let modulus_high_temp = siloxane_chain_entanglement_modulus(50000.0, 373.15).unwrap();
    assert!(modulus_high_temp > modulus_high_mw); // Modulus increases with temperature
}

#[test]
fn test_siloxane_swelling_ratio() {
    // Good solvent (similar solubility parameter)
    let swelling_good = siloxane_swelling_ratio(15.5, 15.5, 0.001).unwrap();
    assert!(swelling_good > 1.0);
    
    // Poor solvent (different solubility parameter)
    let swelling_poor = siloxane_swelling_ratio(20.0, 15.5, 0.001).unwrap();
    assert!(swelling_poor >= 1.0);
    assert!(swelling_poor < swelling_good); // Good solvent swells more
    
    // Higher crosslinking reduces swelling
    let swelling_crosslinked = siloxane_swelling_ratio(15.5, 15.5, 0.01).unwrap();
    assert!(swelling_crosslinked < swelling_good);
}

#[test]
fn test_siloxane_refractive_index() {
    // Visible light wavelength
    let n_visible = siloxane_refractive_index(589e-9, 298.15, 0.0).unwrap();
    assert!(n_visible > 1.0);
    assert!(n_visible < 2.0); // Reasonable range for polymers
    
    // Different wavelengths
    let n_blue = siloxane_refractive_index(400e-9, 298.15, 0.0).unwrap();
    let n_red = siloxane_refractive_index(700e-9, 298.15, 0.0).unwrap();
    assert!(n_blue > n_red); // Normal dispersion
    
    // Temperature dependence
    let n_high_temp = siloxane_refractive_index(589e-9, 373.15, 0.0).unwrap();
    assert!(n_high_temp < n_visible); // dn/dT is negative for polymers
    
    // Crosslinking effect
    let n_crosslinked = siloxane_refractive_index(589e-9, 298.15, 0.1).unwrap();
    assert!(n_crosslinked > n_visible); // Crosslinking increases density and n
}

#[test]
fn test_silicone_integration_properties() {
    let domain = SiliconeDomain::new();
    
    // Create a typical medical grade silicone scenario
    let temperature = 310.15; // Body temperature
    let molecular_weight = 25000.0;
    let crosslink_density = 0.005;
    let filler_fraction = 0.1;
    
    // Calculate various properties
    let viscosity = domain.calculate_viscosity(temperature, molecular_weight, 0.0).unwrap();
    let modulus = silicone_elastic_modulus(temperature, crosslink_density, filler_fraction).unwrap();
    let tg = siloxane_glass_transition_temperature(molecular_weight, crosslink_density).unwrap();
    let biocompat = siloxane_biocompatibility_score(0.001, 0.5, 0.8).unwrap();
    
    // Medical grade silicone should have reasonable properties
    assert!(viscosity > 0.0);
    assert!(modulus > 100000.0); // At least 100 kPa for structural integrity
    assert!(tg < temperature); // Should be rubbery at body temperature
    assert!(biocompat > 50.0); // Should have decent biocompatibility
    
    println!("Medical Grade Silicone Properties:");
    println!("  Viscosity: {:.2e} Pa·s", viscosity);
    println!("  Elastic Modulus: {:.0} Pa", modulus);
    println!("  Glass Transition: {:.1} K", tg);
    println!("  Biocompatibility Score: {:.1}/100", biocompat);
}

#[test]
fn test_silicone_error_handling() {
    let domain = SiliconeDomain::new();
    
    // Test invalid inputs
    assert!(domain.calculate_viscosity(-100.0, 10000.0, 1.0).is_err());
    assert!(domain.calculate_viscosity(298.15, -10000.0, 1.0).is_err());
    
    assert!(siloxane_thermal_conductivity(-100.0, 0.1, 0.2).is_err());
    assert!(siloxane_thermal_conductivity(298.15, -0.1, 0.2).is_err());
    assert!(siloxane_thermal_conductivity(298.15, 0.1, 1.5).is_err());
    
    assert!(siloxane_gas_permeability(-1.0, 298.15, 0.1).is_err());
    assert!(siloxane_gas_permeability(3.6, -100.0, 0.1).is_err());
    
    assert!(silicone_elastic_modulus(-100.0, 0.001, 0.1).is_err());
    assert!(silicone_elastic_modulus(298.15, -0.001, 0.1).is_err());
    assert!(silicone_elastic_modulus(298.15, 0.001, -0.1).is_err());
}