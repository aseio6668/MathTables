use crate::core::{
    MathDomain, MathResult, MathError, Point2D, Point3D, Vector3D,
    TPEMaterial, BlockCopolymer, ThermalTransition, TransitionType, ProcessingConditions
};
use std::any::Any;
use std::f64::consts::PI;

pub struct TPEDomain;

impl TPEMaterial {
    /// Create TPU (Thermoplastic Polyurethane) material
    pub fn tpu_standard() -> Self {
        Self {
            name: "TPU Standard Grade".to_string(),
            hardness_shore_a: 85.0,
            tensile_strength: 35e6, // 35 MPa
            elongation_at_break: 450.0, // 450%
            elastic_modulus: 25e6, // 25 MPa
            density: 1200.0, // kg/m³
            glass_transition_temp: 213.15, // -60°C
            melting_temp: 453.15, // 180°C
            thermal_expansion_coeff: 180e-6, // 1/K
            thermal_conductivity: 0.25, // W/(m·K)
            specific_heat: 1800.0, // J/(kg·K)
            melt_flow_index: 15.0, // g/10min
        }
    }
    
    /// Create TPO (Thermoplastic Olefin) material
    pub fn tpo_automotive() -> Self {
        Self {
            name: "TPO Automotive Grade".to_string(),
            hardness_shore_a: 75.0,
            tensile_strength: 18e6, // 18 MPa
            elongation_at_break: 300.0, // 300%
            elastic_modulus: 450e6, // 450 MPa
            density: 900.0, // kg/m³
            glass_transition_temp: 233.15, // -40°C
            melting_temp: 438.15, // 165°C
            thermal_expansion_coeff: 150e-6, // 1/K
            thermal_conductivity: 0.2, // W/(m·K)
            specific_heat: 2000.0, // J/(kg·K)
            melt_flow_index: 8.0, // g/10min
        }
    }
    
    /// Create TPS (Thermoplastic Styrenic) material  
    pub fn tps_consumer() -> Self {
        Self {
            name: "TPS Consumer Grade".to_string(),
            hardness_shore_a: 70.0,
            tensile_strength: 20e6, // 20 MPa
            elongation_at_break: 500.0, // 500%
            elastic_modulus: 8e6, // 8 MPa
            density: 950.0, // kg/m³
            glass_transition_temp: 253.15, // -20°C
            melting_temp: 443.15, // 170°C
            thermal_expansion_coeff: 120e-6, // 1/K
            thermal_conductivity: 0.15, // W/(m·K)
            specific_heat: 1600.0, // J/(kg·K)
            melt_flow_index: 12.0, // g/10min
        }
    }
}

impl TPEDomain {
    pub fn new() -> Self {
        Self
    }
    
    // ===== Thermal Response Functions =====
    
    /// Calculate thermal expansion: ΔL = α·L·ΔT
    pub fn thermal_expansion(
        original_length: f64,
        expansion_coefficient: f64,
        temperature_change: f64
    ) -> f64 {
        expansion_coefficient * original_length * temperature_change
    }
    
    /// Calculate volumetric thermal expansion: ΔV = 3α·V·ΔT (approximate)
    pub fn volumetric_thermal_expansion(
        original_volume: f64,
        expansion_coefficient: f64,
        temperature_change: f64
    ) -> f64 {
        3.0 * expansion_coefficient * original_volume * temperature_change
    }
    
    /// Heat conduction equation: q = -k·A·(dT/dx)
    pub fn heat_conduction_rate(
        thermal_conductivity: f64,
        area: f64,
        temperature_gradient: f64
    ) -> f64 {
        -thermal_conductivity * area * temperature_gradient
    }
    
    /// Convective heat transfer: q = h·A·ΔT
    pub fn convective_heat_transfer(
        heat_transfer_coefficient: f64,
        area: f64,
        temperature_difference: f64
    ) -> f64 {
        heat_transfer_coefficient * area * temperature_difference
    }
    
    /// Radiative heat transfer: q = ε·σ·A·(T₁⁴ - T₂⁴)
    pub fn radiative_heat_transfer(
        emissivity: f64,
        area: f64,
        temperature_1: f64,
        temperature_2: f64
    ) -> f64 {
        let stefan_boltzmann = 5.670374419e-8; // W/(m²·K⁴)
        emissivity * stefan_boltzmann * area * (temperature_1.powi(4) - temperature_2.powi(4))
    }
    
    /// Glass transition temperature shift with cooling rate (WLF equation approximation)
    pub fn glass_transition_shift(
        base_tg: f64,
        cooling_rate: f64,
        reference_rate: f64
    ) -> MathResult<f64> {
        if cooling_rate <= 0.0 || reference_rate <= 0.0 {
            return Err(MathError::InvalidArgument("Cooling rates must be positive".to_string()));
        }
        
        let rate_ratio = cooling_rate / reference_rate;
        let shift = 5.0 * rate_ratio.ln(); // Empirical approximation
        Ok(base_tg + shift)
    }
    
    /// Phase transition enthalpy calculation
    pub fn phase_transition_energy(
        mass: f64,
        specific_enthalpy: f64
    ) -> f64 {
        mass * specific_enthalpy
    }
    
    // ===== Microstructure Simulation =====
    
    /// Calculate domain spacing in block copolymers using scaling theory
    pub fn block_copolymer_domain_spacing(
        molecular_weight: f64,
        chi_parameter: f64, // Flory-Huggins interaction parameter
        composition: f64    // Volume fraction (0-1)
    ) -> MathResult<f64> {
        if composition <= 0.0 || composition >= 1.0 {
            return Err(MathError::InvalidArgument("Composition must be between 0 and 1".to_string()));
        }
        
        if chi_parameter <= 0.0 {
            return Err(MathError::InvalidArgument("Chi parameter must be positive".to_string()));
        }
        
        // Scaling theory: d ~ N^(2/3) * chi^(1/6)
        let degree_of_polymerization = molecular_weight / 100.0; // Approximate monomer MW
        let domain_spacing = degree_of_polymerization.powf(2.0/3.0) * chi_parameter.powf(1.0/6.0);
        Ok(domain_spacing) // nm
    }
    
    /// Effective modulus of block copolymer using rule of mixtures
    pub fn block_copolymer_modulus(copolymer: &BlockCopolymer) -> f64 {
        let hard_contribution = copolymer.hard_block_fraction * copolymer.hard_block_modulus;
        let soft_contribution = copolymer.soft_block_fraction * copolymer.soft_block_modulus;
        hard_contribution + soft_contribution
    }
    
    /// Polymer chain end-to-end distance (freely-jointed chain model)
    pub fn chain_end_to_end_distance(
        segment_length: f64,
        number_of_segments: usize
    ) -> f64 {
        segment_length * (number_of_segments as f64).sqrt()
    }
    
    /// Polymer chain radius of gyration
    pub fn radius_of_gyration(
        segment_length: f64,
        number_of_segments: usize
    ) -> f64 {
        segment_length * (number_of_segments as f64 / 6.0).sqrt()
    }
    
    /// Entropic elasticity force: F = kT/b * (r/L) / (1 - r/L)  (Langevin chain)
    pub fn entropic_elastic_force(
        temperature: f64,
        segment_length: f64,
        extension: f64,
        contour_length: f64
    ) -> MathResult<f64> {
        if extension >= contour_length {
            return Err(MathError::InvalidArgument("Extension cannot exceed contour length".to_string()));
        }
        
        let kb = 1.380649e-23; // Boltzmann constant
        let relative_extension = extension / contour_length;
        
        if relative_extension >= 1.0 {
            return Err(MathError::InvalidArgument("Relative extension must be < 1".to_string()));
        }
        
        let force = (kb * temperature / segment_length) * 
                   relative_extension / (1.0 - relative_extension);
        Ok(force)
    }
    
    /// Crystallinity degree estimation from density
    pub fn crystallinity_from_density(
        measured_density: f64,
        amorphous_density: f64,
        crystalline_density: f64
    ) -> MathResult<f64> {
        if crystalline_density <= amorphous_density {
            return Err(MathError::InvalidArgument("Crystalline density must be > amorphous density".to_string()));
        }
        
        let crystallinity = (measured_density - amorphous_density) / 
                           (crystalline_density - amorphous_density);
        Ok(crystallinity.max(0.0).min(1.0))
    }
    
    // ===== Injection Molding & Fabrication Math =====
    
    /// Poiseuille flow for melt flow in channels: Q = (π·r⁴·ΔP)/(8·η·L)
    pub fn melt_flow_rate_circular(
        radius: f64,
        pressure_drop: f64,
        viscosity: f64,
        length: f64
    ) -> MathResult<f64> {
        if radius <= 0.0 || viscosity <= 0.0 || length <= 0.0 {
            return Err(MathError::InvalidArgument("Physical parameters must be positive".to_string()));
        }
        
        let flow_rate = (PI * radius.powi(4) * pressure_drop) / (8.0 * viscosity * length);
        Ok(flow_rate)
    }
    
    /// Rectangular channel flow: Q = (w·h³·ΔP)/(12·η·L)
    pub fn melt_flow_rate_rectangular(
        width: f64,
        height: f64,
        pressure_drop: f64,
        viscosity: f64,
        length: f64
    ) -> MathResult<f64> {
        if width <= 0.0 || height <= 0.0 || viscosity <= 0.0 || length <= 0.0 {
            return Err(MathError::InvalidArgument("Physical parameters must be positive".to_string()));
        }
        
        let flow_rate = (width * height.powi(3) * pressure_drop) / (12.0 * viscosity * length);
        Ok(flow_rate)
    }
    
    /// Temperature-dependent viscosity (Arrhenius model): η = η₀·exp(E/RT)
    pub fn temperature_viscosity(
        reference_viscosity: f64,
        activation_energy: f64,
        temperature: f64,
        reference_temperature: f64
    ) -> MathResult<f64> {
        if temperature <= 0.0 || reference_temperature <= 0.0 {
            return Err(MathError::InvalidArgument("Temperatures must be positive (Kelvin)".to_string()));
        }
        
        let gas_constant = 8.314; // J/(mol·K)
        let exponent = activation_energy / gas_constant * (1.0/temperature - 1.0/reference_temperature);
        Ok(reference_viscosity * exponent.exp())
    }
    
    /// Cooling curve: T(t) = T_ambient + (T_initial - T_ambient)·exp(-t/τ)
    pub fn cooling_curve(
        initial_temperature: f64,
        ambient_temperature: f64,
        time: f64,
        time_constant: f64
    ) -> f64 {
        ambient_temperature + (initial_temperature - ambient_temperature) * (-time / time_constant).exp()
    }
    
    /// Cooling time to specific temperature
    pub fn cooling_time_to_temperature(
        initial_temperature: f64,
        ambient_temperature: f64,
        target_temperature: f64,
        time_constant: f64
    ) -> MathResult<f64> {
        if target_temperature <= ambient_temperature || initial_temperature <= ambient_temperature {
            return Err(MathError::InvalidArgument("Invalid temperature relationship".to_string()));
        }
        
        let ratio = (target_temperature - ambient_temperature) / (initial_temperature - ambient_temperature);
        if ratio <= 0.0 {
            return Err(MathError::InvalidArgument("Target temperature invalid".to_string()));
        }
        
        Ok(-time_constant * ratio.ln())
    }
    
    /// Solidification front velocity (Stefan problem approximation)
    pub fn solidification_front_velocity(
        thermal_diffusivity: f64,
        latent_heat: f64,
        specific_heat: f64,
        temperature_gradient: f64
    ) -> MathResult<f64> {
        if thermal_diffusivity <= 0.0 || latent_heat <= 0.0 || specific_heat <= 0.0 {
            return Err(MathError::InvalidArgument("Material properties must be positive".to_string()));
        }
        
        let stefan_number = specific_heat * temperature_gradient / latent_heat;
        let velocity = thermal_diffusivity * stefan_number;
        Ok(velocity)
    }
    
    // ===== Material Property Interpolation =====
    
    /// Linear interpolation between two materials
    pub fn linear_material_blend(
        material_a: &TPEMaterial,
        material_b: &TPEMaterial,
        fraction_a: f64
    ) -> MathResult<TPEMaterial> {
        if fraction_a < 0.0 || fraction_a > 1.0 {
            return Err(MathError::InvalidArgument("Fraction must be between 0 and 1".to_string()));
        }
        
        let fraction_b = 1.0 - fraction_a;
        
        Ok(TPEMaterial {
            name: format!("{}_{}_Blend", material_a.name, material_b.name),
            hardness_shore_a: material_a.hardness_shore_a * fraction_a + material_b.hardness_shore_a * fraction_b,
            tensile_strength: material_a.tensile_strength * fraction_a + material_b.tensile_strength * fraction_b,
            elongation_at_break: material_a.elongation_at_break * fraction_a + material_b.elongation_at_break * fraction_b,
            elastic_modulus: material_a.elastic_modulus * fraction_a + material_b.elastic_modulus * fraction_b,
            density: material_a.density * fraction_a + material_b.density * fraction_b,
            glass_transition_temp: material_a.glass_transition_temp * fraction_a + material_b.glass_transition_temp * fraction_b,
            melting_temp: material_a.melting_temp * fraction_a + material_b.melting_temp * fraction_b,
            thermal_expansion_coeff: material_a.thermal_expansion_coeff * fraction_a + material_b.thermal_expansion_coeff * fraction_b,
            thermal_conductivity: material_a.thermal_conductivity * fraction_a + material_b.thermal_conductivity * fraction_b,
            specific_heat: material_a.specific_heat * fraction_a + material_b.specific_heat * fraction_b,
            melt_flow_index: material_a.melt_flow_index * fraction_a + material_b.melt_flow_index * fraction_b,
        })
    }
    
    /// Property interpolation with temperature dependence
    pub fn temperature_property_interpolation(
        base_property: f64,
        temperature: f64,
        reference_temperature: f64,
        temperature_coefficient: f64
    ) -> f64 {
        base_property * (1.0 + temperature_coefficient * (temperature - reference_temperature))
    }
    
    /// Optimize material composition for target properties (simplified)
    pub fn optimize_composition_for_modulus(
        target_modulus: f64,
        material_a_modulus: f64,
        material_b_modulus: f64
    ) -> MathResult<f64> {
        if material_a_modulus == material_b_modulus {
            return Ok(0.5); // Any ratio works
        }
        
        let fraction_a = (target_modulus - material_b_modulus) / 
                        (material_a_modulus - material_b_modulus);
        
        if fraction_a < 0.0 || fraction_a > 1.0 {
            return Err(MathError::InvalidArgument("Target modulus not achievable with these materials".to_string()));
        }
        
        Ok(fraction_a)
    }
    
    /// Multi-objective optimization score (weighted sum)
    pub fn material_performance_score(
        material: &TPEMaterial,
        target_hardness: f64,
        target_strength: f64,
        target_elongation: f64,
        weights: (f64, f64, f64) // (hardness, strength, elongation) weights
    ) -> f64 {
        let hardness_score = 1.0 - (material.hardness_shore_a - target_hardness).abs() / target_hardness;
        let strength_score = 1.0 - (material.tensile_strength - target_strength).abs() / target_strength;
        let elongation_score = 1.0 - (material.elongation_at_break - target_elongation).abs() / target_elongation;
        
        weights.0 * hardness_score.max(0.0) + 
        weights.1 * strength_score.max(0.0) + 
        weights.2 * elongation_score.max(0.0)
    }
    
    // ===== Specialized TPE Applications =====
    
    /// Cable jacket flexibility analysis
    pub fn cable_flexibility_factor(
        bend_radius: f64,
        cable_diameter: f64,
        elastic_modulus: f64,
        target_stress: f64
    ) -> MathResult<f64> {
        if bend_radius <= 0.0 || cable_diameter <= 0.0 || elastic_modulus <= 0.0 {
            return Err(MathError::InvalidArgument("Physical parameters must be positive".to_string()));
        }
        
        // Bending stress: σ = E·d/(2·R)
        let bending_stress = elastic_modulus * cable_diameter / (2.0 * bend_radius);
        let flexibility_factor = target_stress / bending_stress;
        Ok(flexibility_factor)
    }
    
    /// Seal compression analysis
    pub fn seal_compression_force(
        initial_thickness: f64,
        compressed_thickness: f64,
        contact_area: f64,
        elastic_modulus: f64
    ) -> MathResult<f64> {
        if compressed_thickness >= initial_thickness {
            return Err(MathError::InvalidArgument("Compressed thickness must be less than initial".to_string()));
        }
        
        let strain = (initial_thickness - compressed_thickness) / initial_thickness;
        let stress = elastic_modulus * strain;
        let force = stress * contact_area;
        Ok(force)
    }
    
    /// Grip surface friction coefficient estimation
    pub fn grip_friction_coefficient(
        surface_roughness: f64,
        contact_pressure: f64,
        hardness_shore_a: f64
    ) -> f64 {
        // Empirical relationship for TPE grips
        let base_friction = 0.3; // Baseline coefficient
        let roughness_factor = (surface_roughness / 50.0).sqrt(); // Roughness in µm
        let pressure_factor = (contact_pressure / 1e5).powf(0.1); // Pressure in Pa
        let hardness_factor = (100.0 - hardness_shore_a) / 100.0; // Softer = higher friction
        
        base_friction * roughness_factor * pressure_factor * (1.0 + hardness_factor)
    }
}

impl MathDomain for TPEDomain {
    fn name(&self) -> &str { "TPE (Thermoplastic Elastomers)" }
    fn description(&self) -> &str { "Comprehensive thermoplastic elastomer analysis including thermal response, microstructure simulation, fabrication mathematics, and material property optimization" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "thermal_expansion" => {
                if args.len() != 3 {
                    return Err(MathError::InvalidArgument("thermal_expansion requires 3 arguments".to_string()));
                }
                let length = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let coeff = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                let temp_change = args[2].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be f64".to_string()))?;
                Ok(Box::new(Self::thermal_expansion(*length, *coeff, *temp_change)))
            },
            "block_copolymer_domain_spacing" => {
                if args.len() != 3 {
                    return Err(MathError::InvalidArgument("block_copolymer_domain_spacing requires 3 arguments".to_string()));
                }
                let mw = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let chi = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                let comp = args[2].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be f64".to_string()))?;
                Ok(Box::new(Self::block_copolymer_domain_spacing(*mw, *chi, *comp)?))
            },
            "melt_flow_rate_circular" => {
                if args.len() != 4 {
                    return Err(MathError::InvalidArgument("melt_flow_rate_circular requires 4 arguments".to_string()));
                }
                let radius = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let pressure = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                let viscosity = args[2].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be f64".to_string()))?;
                let length = args[3].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Fourth argument must be f64".to_string()))?;
                Ok(Box::new(Self::melt_flow_rate_circular(*radius, *pressure, *viscosity, *length)?))
            },
            "cable_flexibility_factor" => {
                if args.len() != 4 {
                    return Err(MathError::InvalidArgument("cable_flexibility_factor requires 4 arguments".to_string()));
                }
                let bend_radius = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let diameter = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                let modulus = args[2].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be f64".to_string()))?;
                let target_stress = args[3].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Fourth argument must be f64".to_string()))?;
                Ok(Box::new(Self::cable_flexibility_factor(*bend_radius, *diameter, *modulus, *target_stress)?))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "thermal_expansion".to_string(),
            "volumetric_thermal_expansion".to_string(),
            "heat_conduction_rate".to_string(),
            "convective_heat_transfer".to_string(),
            "radiative_heat_transfer".to_string(),
            "glass_transition_shift".to_string(),
            "block_copolymer_domain_spacing".to_string(),
            "block_copolymer_modulus".to_string(),
            "chain_end_to_end_distance".to_string(),
            "entropic_elastic_force".to_string(),
            "crystallinity_from_density".to_string(),
            "melt_flow_rate_circular".to_string(),
            "melt_flow_rate_rectangular".to_string(),
            "temperature_viscosity".to_string(),
            "cooling_curve".to_string(),
            "linear_material_blend".to_string(),
            "cable_flexibility_factor".to_string(),
            "seal_compression_force".to_string(),
            "grip_friction_coefficient".to_string(),
        ]
    }
}