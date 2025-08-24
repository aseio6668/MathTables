use crate::core::types::*;
use std::f64::consts::{E, PI};

#[derive(Clone, Debug)]
pub struct SiliconeDomain;

impl SiliconeDomain {
    pub fn new() -> Self {
        Self
    }

    pub fn calculate_viscosity(&self, temperature: f64, molecular_weight: f64, 
                              shear_rate: f64) -> MathResult<f64> {
        if temperature <= 0.0 {
            return Err(MathError::InvalidArgument("Temperature must be positive".to_string()));
        }
        if molecular_weight <= 0.0 {
            return Err(MathError::InvalidArgument("Molecular weight must be positive".to_string()));
        }
        
        // Power law model for silicone viscosity
        let reference_temp = 298.15; // K
        let temp_factor = E.powf(2000.0 * (1.0/temperature - 1.0/reference_temp)); // Fixed sign
        let mw_factor = (molecular_weight / 10000.0).powf(3.4); // Normalized
        let shear_factor = if shear_rate > 0.0 {
            shear_rate.powf(-0.8)
        } else {
            1.0
        };
        
        Ok(0.001 * mw_factor * temp_factor * shear_factor)
    }

    pub fn crosslink_density(&self, silicone: &SiloxaneChain) -> f64 {
        silicone.crosslink_density
    }
}

pub fn siloxane_chain_length_distribution(average_length: f64, polydispersity: f64) -> MathResult<f64> {
    if average_length <= 0.0 {
        return Err(MathError::InvalidArgument("Average chain length must be positive".to_string()));
    }
    if polydispersity < 1.0 {
        return Err(MathError::InvalidArgument("Polydispersity must be >= 1.0".to_string()));
    }
    
    // Most probable distribution for chain length
    let variance = (polydispersity - 1.0) * average_length.powi(2);
    Ok(variance.sqrt())
}

pub fn siloxane_thermal_expansion(original_length: f64, temperature_change: f64, 
                                expansion_coefficient: f64) -> f64 {
    original_length * (1.0 + expansion_coefficient * temperature_change)
}

pub fn siloxane_thermal_conductivity(temperature: f64, crosslink_density: f64, 
                                   filler_fraction: f64) -> MathResult<f64> {
    if temperature <= 0.0 {
        return Err(MathError::InvalidArgument("Temperature must be positive".to_string()));
    }
    if crosslink_density < 0.0 || filler_fraction < 0.0 || filler_fraction > 1.0 {
        return Err(MathError::InvalidArgument("Invalid material parameters".to_string()));
    }
    
    // Base thermal conductivity of silicone (W/m·K)
    let base_conductivity = 0.15;
    
    // Temperature dependence
    let temp_factor = 1.0 + 0.0001 * (temperature - 298.15);
    
    // Crosslink density effect
    let crosslink_factor = 1.0 + 0.1 * crosslink_density;
    
    // Filler effect (thermal enhancement)
    let filler_factor = 1.0 + 10.0 * filler_fraction;
    
    Ok(base_conductivity * temp_factor * crosslink_factor * filler_factor)
}

pub fn siloxane_gas_permeability(gas_molecule_size: f64, temperature: f64, 
                                crosslink_density: f64) -> MathResult<f64> {
    if gas_molecule_size <= 0.0 {
        return Err(MathError::InvalidArgument("Gas molecule size must be positive".to_string()));
    }
    if temperature <= 0.0 {
        return Err(MathError::InvalidArgument("Temperature must be positive".to_string()));
    }
    
    // Arrhenius temperature dependence
    let activation_energy = 15000.0; // J/mol
    let gas_constant = 8.314; // J/(mol·K)
    let temp_factor = E.powf(-activation_energy / (gas_constant * temperature));
    
    // Free volume model for gas permeability
    let base_permeability = 1e-10; // cm³(STP)·cm/(cm²·s·cmHg)
    let size_factor = E.powf(-gas_molecule_size / 0.5); // size selectivity
    let crosslink_factor = E.powf(-2.0 * crosslink_density); // crosslinking reduces permeability
    
    Ok(base_permeability * temp_factor * size_factor * crosslink_factor)
}

pub fn silicone_elastic_modulus(temperature: f64, crosslink_density: f64, 
                               filler_fraction: f64) -> MathResult<f64> {
    if temperature <= 0.0 {
        return Err(MathError::InvalidArgument("Temperature must be positive".to_string()));
    }
    if crosslink_density < 0.0 {
        return Err(MathError::InvalidArgument("Crosslink density must be non-negative".to_string()));
    }
    if filler_fraction < 0.0 || filler_fraction > 1.0 {
        return Err(MathError::InvalidArgument("Filler fraction must be between 0 and 1".to_string()));
    }
    
    // Rubber elasticity theory
    let gas_constant = 8.314; // J/(mol·K)
    let density = 970.0; // kg/m³ for PDMS
    let molar_mass = 0.07416; // kg/mol for SiO repeat unit
    
    // Network modulus from crosslink density (mol/m³)
    let crosslink_mol_density = crosslink_density * density / molar_mass;
    let network_modulus = crosslink_mol_density * gas_constant * temperature;
    
    // Base modulus for entanglements and physical interactions
    let base_modulus = 50000.0; // Pa, typical for uncrosslinked PDMS
    
    // Filler reinforcement (Einstein-Guth-Gold equation)
    let filler_factor = 1.0 + 2.5 * filler_fraction + 14.1 * filler_fraction.powi(2);
    
    Ok((base_modulus + network_modulus) * filler_factor)
}

pub fn siloxane_glass_transition_temperature(molecular_weight: f64, 
                                           crosslink_density: f64) -> MathResult<f64> {
    if molecular_weight <= 0.0 {
        return Err(MathError::InvalidArgument("Molecular weight must be positive".to_string()));
    }
    
    // Fox equation for polymer blends/networks
    let base_tg = 148.15; // K, for infinite molecular weight PDMS
    let mw_correction = 17.0 / molecular_weight.powf(0.5);
    let crosslink_correction = 20.0 * crosslink_density;
    
    Ok(base_tg + mw_correction + crosslink_correction)
}

pub fn silicone_surface_energy(temperature: f64, chain_flexibility: f64) -> MathResult<f64> {
    if temperature <= 0.0 {
        return Err(MathError::InvalidArgument("Temperature must be positive".to_string()));
    }
    if chain_flexibility < 0.0 || chain_flexibility > 1.0 {
        return Err(MathError::InvalidArgument("Chain flexibility must be between 0 and 1".to_string()));
    }
    
    // Temperature-dependent surface energy
    let surface_energy_298 = 20.4; // mJ/m² at 298K
    let temp_coefficient = -0.07; // mJ/(m²·K)
    
    let temp_contribution = surface_energy_298 + temp_coefficient * (temperature - 298.15);
    let flexibility_factor = 1.0 - 0.3 * chain_flexibility; // more flexible = lower surface energy
    
    Ok(temp_contribution * flexibility_factor)
}

pub fn siloxane_dielectric_constant(frequency: f64, temperature: f64, 
                                  moisture_content: f64) -> MathResult<f64> {
    if frequency <= 0.0 {
        return Err(MathError::InvalidArgument("Frequency must be positive".to_string()));
    }
    if temperature <= 0.0 {
        return Err(MathError::InvalidArgument("Temperature must be positive".to_string()));
    }
    if moisture_content < 0.0 || moisture_content > 1.0 {
        return Err(MathError::InvalidArgument("Moisture content must be between 0 and 1".to_string()));
    }
    
    // Base dielectric constant for dry silicone
    let base_dielectric = 2.7;
    
    // Frequency dependence (Debye relaxation)
    let relaxation_freq = 1e6; // Hz
    let freq_factor = 1.0 / (1.0 + (frequency / relaxation_freq).powi(2)).sqrt();
    
    // Temperature dependence
    let temp_factor = 1.0 - 0.0005 * (temperature - 298.15);
    
    // Moisture effect
    let moisture_factor = 1.0 + 10.0 * moisture_content; // water has high dielectric constant
    
    Ok(base_dielectric * freq_factor * temp_factor * moisture_factor)
}

pub fn siloxane_biocompatibility_score(leachable_content: f64, surface_roughness: f64, 
                                     hydrophobicity: f64) -> MathResult<f64> {
    if leachable_content < 0.0 || surface_roughness < 0.0 {
        return Err(MathError::InvalidArgument("Content and roughness must be non-negative".to_string()));
    }
    if hydrophobicity < 0.0 || hydrophobicity > 1.0 {
        return Err(MathError::InvalidArgument("Hydrophobicity must be between 0 and 1".to_string()));
    }
    
    // Biocompatibility scoring (higher is better)
    let leachable_penalty = E.powf(-10.0 * leachable_content);
    let roughness_penalty = E.powf(-0.1 * surface_roughness);
    let hydrophobic_bonus = 0.5 + 0.5 * hydrophobicity; // moderate hydrophobicity is good
    
    let score = 100.0 * leachable_penalty * roughness_penalty * hydrophobic_bonus;
    Ok(score.min(100.0))
}

pub fn siloxane_cure_kinetics(temperature: f64, catalyst_concentration: f64, 
                            time: f64, activation_energy: f64) -> MathResult<f64> {
    if temperature <= 0.0 {
        return Err(MathError::InvalidArgument("Temperature must be positive".to_string()));
    }
    if catalyst_concentration < 0.0 {
        return Err(MathError::InvalidArgument("Catalyst concentration must be non-negative".to_string()));
    }
    if time < 0.0 {
        return Err(MathError::InvalidArgument("Time must be non-negative".to_string()));
    }
    
    // Arrhenius kinetics for curing reaction
    let gas_constant = 8.314; // J/(mol·K)
    let pre_exponential = 1e10; // s⁻¹
    
    let rate_constant = pre_exponential * E.powf(-activation_energy / (gas_constant * temperature));
    let effective_rate = rate_constant * (1.0 + catalyst_concentration);
    
    // First-order kinetics for degree of cure
    let degree_of_cure = 1.0 - E.powf(-effective_rate * time);
    
    Ok(degree_of_cure.min(1.0))
}

pub fn siloxane_degradation_rate(temperature: f64, uv_intensity: f64, 
                                oxygen_partial_pressure: f64) -> MathResult<f64> {
    if temperature <= 0.0 {
        return Err(MathError::InvalidArgument("Temperature must be positive".to_string()));
    }
    if uv_intensity < 0.0 || oxygen_partial_pressure < 0.0 {
        return Err(MathError::InvalidArgument("Intensity and pressure must be non-negative".to_string()));
    }
    
    // Multi-factor degradation model
    let base_rate = 1e-8; // s⁻¹ at reference conditions
    let activation_energy = 80000.0; // J/mol
    let gas_constant = 8.314; // J/(mol·K)
    
    // Thermal degradation (Arrhenius)
    let thermal_factor = E.powf(-activation_energy / (gas_constant * temperature));
    
    // UV degradation (linear with intensity)
    let uv_factor = 1.0 + 0.1 * uv_intensity;
    
    // Oxidative degradation (square root dependence on O2)
    let oxidative_factor = 1.0 + 0.05 * oxygen_partial_pressure.sqrt();
    
    Ok(base_rate * thermal_factor * uv_factor * oxidative_factor)
}

pub fn siloxane_contact_angle(surface_energy: f64, liquid_surface_tension: f64, 
                            interfacial_tension: f64) -> MathResult<f64> {
    if surface_energy <= 0.0 || liquid_surface_tension <= 0.0 {
        return Err(MathError::InvalidArgument("Surface energies must be positive".to_string()));
    }
    
    // Young's equation: γLV cos θ = γSV - γSL
    let cos_theta = (surface_energy - interfacial_tension) / liquid_surface_tension;
    
    // Clamp to valid range for cosine
    let cos_theta_clamped = cos_theta.max(-1.0).min(1.0);
    let contact_angle = cos_theta_clamped.acos();
    
    // Convert from radians to degrees
    Ok(contact_angle * 180.0 / PI)
}

pub fn siloxane_chain_entanglement_modulus(molecular_weight: f64, 
                                         temperature: f64) -> MathResult<f64> {
    if molecular_weight <= 0.0 {
        return Err(MathError::InvalidArgument("Molecular weight must be positive".to_string()));
    }
    if temperature <= 0.0 {
        return Err(MathError::InvalidArgument("Temperature must be positive".to_string()));
    }
    
    // Entanglement molecular weight for PDMS
    let me = 12000.0; // g/mol
    
    if molecular_weight < me {
        // Below entanglement threshold
        Ok(0.0)
    } else {
        // Plateau modulus from entanglement theory
        let density = 0.97; // g/cm³
        let gas_constant = 8.314; // J/(mol·K)
        
        let entanglement_density = density / me;
        let plateau_modulus = entanglement_density * gas_constant * temperature;
        
        Ok(plateau_modulus * 1000.0) // Convert to Pa
    }
}

pub fn siloxane_swelling_ratio(solvent_parameter: f64, polymer_parameter: f64, 
                              crosslink_density: f64) -> MathResult<f64> {
    if crosslink_density <= 0.0 {
        return Err(MathError::InvalidArgument("Crosslink density must be positive".to_string()));
    }
    
    // Flory-Rehner theory for swelling
    let chi = ((solvent_parameter - polymer_parameter) / 2.0).powi(2);
    
    // For PDMS: solvent parameter ~ 15.5 (J/cm³)^0.5
    let polymer_vol_fraction = if chi < 0.5 {
        // Good solvent - more swelling
        0.1 * (crosslink_density / 0.001).powf(0.6)
    } else {
        // Poor solvent - less swelling
        0.3 * (crosslink_density / 0.001).powf(0.3)
    };
    
    // Ensure volume fraction is between 0 and 1
    let polymer_vol_fraction_clamped = polymer_vol_fraction.min(0.99).max(0.01);
    let swelling_ratio = 1.0 / polymer_vol_fraction_clamped;
    
    Ok(swelling_ratio)
}

pub fn siloxane_refractive_index(wavelength: f64, temperature: f64, 
                               crosslink_density: f64) -> MathResult<f64> {
    if wavelength <= 0.0 {
        return Err(MathError::InvalidArgument("Wavelength must be positive".to_string()));
    }
    if temperature <= 0.0 {
        return Err(MathError::InvalidArgument("Temperature must be positive".to_string()));
    }
    
    // Sellmeier equation for wavelength dependence
    let base_n = 1.4118; // at 589 nm, 25°C
    let wavelength_nm = wavelength * 1e9; // convert to nm
    
    let wavelength_correction = 0.001 * (589.0 - wavelength_nm) / wavelength_nm;
    
    // Temperature dependence
    let temp_correction = -4.5e-4 * (temperature - 298.15);
    
    // Crosslink density effect
    let crosslink_correction = 0.01 * crosslink_density;
    
    let refractive_index = base_n + wavelength_correction + temp_correction + crosslink_correction;
    
    Ok(refractive_index.max(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_silicone_domain_initialization() {
        let domain = SiliconeDomain::new();
        
        let viscosity = domain.calculate_viscosity(298.15, 10000.0, 1.0).unwrap();
        assert!(viscosity > 0.0);
    }
    
    #[test]
    fn test_chain_length_distribution() {
        let std_dev = siloxane_chain_length_distribution(1000.0, 2.0).unwrap();
        assert!(std_dev > 0.0);
    }
    
    #[test]
    fn test_thermal_expansion() {
        let expanded = siloxane_thermal_expansion(100.0, 50.0, 2.4e-4);
        assert!((expanded - 101.2).abs() < 0.01);
    }
    
    #[test]
    fn test_thermal_conductivity() {
        let k = siloxane_thermal_conductivity(298.15, 0.1, 0.2).unwrap();
        assert!(k > 0.15); // Should be higher than base due to crosslinking and filler
    }
    
    #[test]
    fn test_gas_permeability() {
        let perm = siloxane_gas_permeability(3.6, 298.15, 0.1).unwrap();
        assert!(perm > 0.0);
        assert!(perm < 1e-10); // Should be less than base due to molecule size and crosslinking
    }
    
    #[test]
    fn test_elastic_modulus() {
        let modulus = silicone_elastic_modulus(298.15, 0.001, 0.1).unwrap();
        assert!(modulus > 0.0);
    }
    
    #[test]
    fn test_glass_transition() {
        let tg = siloxane_glass_transition_temperature(10000.0, 0.01).unwrap();
        assert!(tg > 148.0); // Should be above base Tg
    }
    
    #[test]
    fn test_biocompatibility() {
        let score = siloxane_biocompatibility_score(0.01, 1.0, 0.8).unwrap();
        assert!(score >= 0.0 && score <= 100.0);
    }
    
    #[test]
    fn test_cure_kinetics() {
        let cure = siloxane_cure_kinetics(373.15, 0.01, 3600.0, 50000.0).unwrap();
        assert!(cure >= 0.0 && cure <= 1.0);
    }
    
    #[test]
    fn test_contact_angle() {
        let angle = siloxane_contact_angle(20.0, 72.8, 40.0).unwrap();
        assert!(angle >= 0.0 && angle <= 180.0);
    }
}