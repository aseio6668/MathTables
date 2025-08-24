use crate::core::{
    MathDomain, MathResult, MathError,
    CosmologyParams, GravitationalWave
};
use std::f64::consts::PI;

pub struct CosmologyDomain;

impl CosmologyDomain {
    pub fn new() -> Self {
        Self
    }
    
    /// Standard cosmological parameters (Planck 2018)
    pub fn planck_2018_params() -> CosmologyParams {
        CosmologyParams {
            hubble_constant: 67.4,           // H0 in km/s/Mpc
            omega_matter: 0.315,             // Ωm
            omega_lambda: 0.685,             // ΩΛ
            omega_radiation: 9.2e-5,         // Ωr
            age_universe: 13.8e9,            // t0 in years
            cmb_temperature: 2.7255,         // T0 in Kelvin
        }
    }
}

impl CosmologyDomain {
    /// Hubble parameter H(z) at redshift z
    /// H(z) = H0 * sqrt(Ωm*(1+z)³ + ΩΛ + Ωr*(1+z)⁴)
    pub fn hubble_parameter(params: &CosmologyParams, redshift: f64) -> MathResult<f64> {
        if redshift < -1.0 {
            return Err(MathError::InvalidArgument("Redshift must be > -1".to_string()));
        }
        
        let z_plus_1 = 1.0 + redshift;
        let matter_term = params.omega_matter * z_plus_1.powi(3);
        let lambda_term = params.omega_lambda;
        let radiation_term = params.omega_radiation * z_plus_1.powi(4);
        
        let h_z = params.hubble_constant * (matter_term + lambda_term + radiation_term).sqrt();
        Ok(h_z)
    }
    
    /// Luminosity distance DL(z) in Mpc
    /// DL(z) = (1+z) * comoving_distance(z)
    pub fn luminosity_distance(params: &CosmologyParams, redshift: f64) -> MathResult<f64> {
        if redshift < 0.0 {
            return Err(MathError::InvalidArgument("Redshift must be non-negative".to_string()));
        }
        
        let dc = Self::comoving_distance(params, redshift)?;
        let dl = (1.0 + redshift) * dc;
        Ok(dl)
    }
    
    /// Angular diameter distance DA(z) in Mpc
    /// DA(z) = comoving_distance(z) / (1+z)
    pub fn angular_diameter_distance(params: &CosmologyParams, redshift: f64) -> MathResult<f64> {
        if redshift < 0.0 {
            return Err(MathError::InvalidArgument("Redshift must be non-negative".to_string()));
        }
        
        let dc = Self::comoving_distance(params, redshift)?;
        let da = dc / (1.0 + redshift);
        Ok(da)
    }
    
    /// Comoving distance DC(z) in Mpc (numerical integration)
    pub fn comoving_distance(params: &CosmologyParams, redshift: f64) -> MathResult<f64> {
        if redshift < 0.0 {
            return Err(MathError::InvalidArgument("Redshift must be non-negative".to_string()));
        }
        
        const C_OVER_H0: f64 = 2997.92458; // c/H0 in Mpc for H0 in km/s/Mpc
        
        // Numerical integration using Simpson's rule
        let n_steps = 1000;
        let dz = redshift / n_steps as f64;
        let mut integral = 0.0;
        
        for i in 0..=n_steps {
            let z = i as f64 * dz;
            let h_z = Self::hubble_parameter(params, z)?;
            let integrand = params.hubble_constant / h_z;
            
            let weight = if i == 0 || i == n_steps {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            
            integral += weight * integrand;
        }
        
        integral *= dz / 3.0;
        let dc = C_OVER_H0 * integral;
        Ok(dc)
    }
    
    /// Age of universe at redshift z in years
    /// t(z) = ∫[z,∞] dz' / ((1+z') * H(z'))
    pub fn age_at_redshift(params: &CosmologyParams, redshift: f64) -> MathResult<f64> {
        if redshift < 0.0 {
            return Err(MathError::InvalidArgument("Redshift must be non-negative".to_string()));
        }
        
        // Hubble time in years
        const H0_TO_YEARS: f64 = 9.777752e9; // 1/(H0 in s^-1) to years conversion
        
        // Numerical integration from z to large redshift
        let z_max = 1000.0; // Approximate infinity
        let n_steps = 1000;
        let dz = (z_max - redshift) / n_steps as f64;
        let mut integral = 0.0;
        
        for i in 0..=n_steps {
            let z = redshift + i as f64 * dz;
            let h_z = Self::hubble_parameter(params, z)?;
            let integrand = 1.0 / ((1.0 + z) * h_z);
            
            let weight = if i == 0 || i == n_steps {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            
            integral += weight * integrand;
        }
        
        integral *= dz / 3.0;
        let age = H0_TO_YEARS * integral;
        Ok(age)
    }
    
    /// Critical density at redshift z in g/cm³
    /// ρc(z) = 3 * H(z)² / (8π * G)
    pub fn critical_density(params: &CosmologyParams, redshift: f64) -> MathResult<f64> {
        let h_z = Self::hubble_parameter(params, redshift)?;
        
        // Convert H(z) from km/s/Mpc to 1/s
        let h_z_si = h_z * 3.24078e-20; // km/s/Mpc to 1/s
        
        // Critical density: ρc = 3H²/(8πG)
        const G: f64 = 6.67430e-8; // cm³/g/s²
        let rho_c = 3.0 * h_z_si.powi(2) / (8.0 * PI * G);
        Ok(rho_c)
    }
    
    /// Dark matter density parameter Ωm(z)
    pub fn omega_matter_z(params: &CosmologyParams, redshift: f64) -> MathResult<f64> {
        let h_z = Self::hubble_parameter(params, redshift)?;
        let h0_ratio = params.hubble_constant / h_z;
        let omega_m_z = params.omega_matter * (1.0 + redshift).powi(3) * h0_ratio.powi(2);
        Ok(omega_m_z)
    }
    
    /// Dark energy density parameter ΩΛ(z) 
    pub fn omega_lambda_z(params: &CosmologyParams, redshift: f64) -> MathResult<f64> {
        let h_z = Self::hubble_parameter(params, redshift)?;
        let h0_ratio = params.hubble_constant / h_z;
        let omega_lambda_z = params.omega_lambda * h0_ratio.powi(2);
        Ok(omega_lambda_z)
    }
    
    /// Jeans wavelength for dark matter perturbations
    /// λJ = π * cs / G / ρ^(1/2), where cs is sound speed
    pub fn jeans_wavelength(
        density: f64, 
        sound_speed: f64, 
        temperature: f64
    ) -> MathResult<f64> {
        if density <= 0.0 || sound_speed <= 0.0 || temperature <= 0.0 {
            return Err(MathError::InvalidArgument("All parameters must be positive".to_string()));
        }
        
        const G: f64 = 6.67430e-11; // m³/kg/s²
        let jeans_length = PI * sound_speed / (G * density).sqrt();
        Ok(jeans_length)
    }
    
    /// Gravitational wave strain amplitude from binary merger
    /// h = (4 * G * M_chirp^(5/3) * (π*f)^(2/3)) / (c^4 * d)
    pub fn gw_strain_amplitude(
        chirp_mass_solar: f64,
        frequency_hz: f64,
        distance_mpc: f64
    ) -> MathResult<f64> {
        if chirp_mass_solar <= 0.0 || frequency_hz <= 0.0 || distance_mpc <= 0.0 {
            return Err(MathError::InvalidArgument("All parameters must be positive".to_string()));
        }
        
        const G: f64 = 6.67430e-11; // m³/kg/s²
        const C: f64 = 2.998e8;     // m/s
        const M_SUN: f64 = 1.989e30; // kg
        const MPC_TO_M: f64 = 3.086e22; // m
        
        let chirp_mass = chirp_mass_solar * M_SUN;
        let distance = distance_mpc * MPC_TO_M;
        
        let numerator = 4.0 * G * chirp_mass.powf(5.0/3.0) * (PI * frequency_hz).powf(2.0/3.0);
        let denominator = C.powi(4) * distance;
        
        let strain = numerator / denominator;
        Ok(strain)
    }
    
    /// Gravitational wave frequency evolution (chirp)
    /// df/dt = (96/5) * π^(8/3) * (G*M_chirp/c³)^(5/3) * f^(11/3)
    pub fn gw_frequency_evolution(
        chirp_mass_solar: f64,
        frequency_hz: f64
    ) -> MathResult<f64> {
        if chirp_mass_solar <= 0.0 || frequency_hz <= 0.0 {
            return Err(MathError::InvalidArgument("Parameters must be positive".to_string()));
        }
        
        const G: f64 = 6.67430e-11; // m³/kg/s²
        const C: f64 = 2.998e8;     // m/s
        const M_SUN: f64 = 1.989e30; // kg
        
        let chirp_mass = chirp_mass_solar * M_SUN;
        
        let coefficient = (96.0 / 5.0) * PI.powf(8.0/3.0);
        let mass_term = (G * chirp_mass / C.powi(3)).powf(5.0/3.0);
        let frequency_term = frequency_hz.powf(11.0/3.0);
        
        let dfdt = coefficient * mass_term * frequency_term;
        Ok(dfdt)
    }
    
    /// Time to merger for gravitational wave source
    /// t_merge = (5/256) * c⁵ / G⁵ / M_chirp⁵ * (π*f)^(-8/3)
    pub fn gw_time_to_merger(
        chirp_mass_solar: f64,
        frequency_hz: f64
    ) -> MathResult<f64> {
        if chirp_mass_solar <= 0.0 || frequency_hz <= 0.0 {
            return Err(MathError::InvalidArgument("Parameters must be positive".to_string()));
        }
        
        const G: f64 = 6.67430e-11; // m³/kg/s²
        const C: f64 = 2.998e8;     // m/s
        const M_SUN: f64 = 1.989e30; // kg
        
        let chirp_mass = chirp_mass_solar * M_SUN;
        
        let coefficient = 5.0 / 256.0;
        let c_term = C.powi(5);
        let g_term = G.powi(5);
        let mass_term = chirp_mass.powi(5);
        let freq_term = (PI * frequency_hz).powf(-8.0/3.0);
        
        let t_merger = coefficient * c_term * freq_term / (g_term * mass_term);
        Ok(t_merger)
    }
    
    /// Cosmic microwave background temperature at redshift z
    /// T(z) = T0 * (1 + z)
    pub fn cmb_temperature_z(params: &CosmologyParams, redshift: f64) -> MathResult<f64> {
        if redshift < 0.0 {
            return Err(MathError::InvalidArgument("Redshift must be non-negative".to_string()));
        }
        
        let temp_z = params.cmb_temperature * (1.0 + redshift);
        Ok(temp_z)
    }
    
    /// Scale factor a(t) from redshift: a = 1/(1+z)
    pub fn scale_factor(redshift: f64) -> MathResult<f64> {
        if redshift < -1.0 {
            return Err(MathError::InvalidArgument("Redshift must be > -1".to_string()));
        }
        
        let scale_factor = 1.0 / (1.0 + redshift);
        Ok(scale_factor)
    }
    
    /// Comoving volume element dV/dz in Mpc³/sr
    /// dV/dz = c * D_M²(z) / H(z), where D_M is comoving distance
    pub fn comoving_volume_element(params: &CosmologyParams, redshift: f64) -> MathResult<f64> {
        let dm = Self::comoving_distance(params, redshift)?;
        let h_z = Self::hubble_parameter(params, redshift)?;
        
        const C_KM_S: f64 = 299792.458; // c in km/s
        
        let dv_dz = C_KM_S * dm.powi(2) / h_z;
        Ok(dv_dz)
    }
}

impl MathDomain for CosmologyDomain {
    fn name(&self) -> &str {
        "Cosmology"
    }
    
    fn description(&self) -> &str {
        "Mathematical models for cosmological phenomena including distances, ages, dark matter, and gravitational waves"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "hubble_parameter".to_string(),
            "luminosity_distance".to_string(),
            "angular_diameter_distance".to_string(),
            "comoving_distance".to_string(),
            "age_at_redshift".to_string(),
            "critical_density".to_string(),
            "omega_matter_z".to_string(),
            "omega_lambda_z".to_string(),
            "jeans_wavelength".to_string(),
            "gw_strain_amplitude".to_string(),
            "gw_frequency_evolution".to_string(),
            "gw_time_to_merger".to_string(),
            "cmb_temperature_z".to_string(),
            "scale_factor".to_string(),
            "comoving_volume_element".to_string(),
        ]
    }
    
    fn compute(&self, _operation: &str, _args: &[&dyn std::any::Any]) -> MathResult<Box<dyn std::any::Any>> {
        Err(MathError::NotImplemented("Generic compute not implemented for Cosmology domain".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hubble_parameter() {
        let params = CosmologyDomain::planck_2018_params();
        
        // At z=0, H(z) = H0
        let h0 = CosmologyDomain::hubble_parameter(&params, 0.0).unwrap();
        assert_relative_eq!(h0, params.hubble_constant, epsilon = 1e-10);
        
        // At high redshift, H(z) should be much larger
        let h_high_z = CosmologyDomain::hubble_parameter(&params, 10.0).unwrap();
        assert!(h_high_z > params.hubble_constant);
    }

    #[test]
    fn test_distances() {
        let params = CosmologyDomain::planck_2018_params();
        
        let z = 1.0;
        let dl = CosmologyDomain::luminosity_distance(&params, z).unwrap();
        let da = CosmologyDomain::angular_diameter_distance(&params, z).unwrap();
        let dc = CosmologyDomain::comoving_distance(&params, z).unwrap();
        
        // Relationships: DL = (1+z)*DC, DA = DC/(1+z)
        assert_relative_eq!(dl, (1.0 + z) * dc, epsilon = 1e-6);
        assert_relative_eq!(da, dc / (1.0 + z), epsilon = 1e-6);
        
        // All distances should be positive
        assert!(dl > 0.0);
        assert!(da > 0.0);
        assert!(dc > 0.0);
    }

    #[test]
    fn test_age_calculation() {
        let params = CosmologyDomain::planck_2018_params();
        
        // Age at z=0 should be close to current age
        let age_now = CosmologyDomain::age_at_redshift(&params, 0.0).unwrap();
        assert!(age_now > 10e9 && age_now < 20e9); // Between 10-20 Gyr
        
        // Age at high redshift should be much smaller
        let age_high_z = CosmologyDomain::age_at_redshift(&params, 5.0).unwrap();
        assert!(age_high_z < age_now);
    }

    #[test]
    fn test_gravitational_waves() {
        let chirp_mass = 30.0; // Solar masses
        let frequency = 100.0; // Hz
        let distance = 100.0; // Mpc
        
        let strain = CosmologyDomain::gw_strain_amplitude(chirp_mass, frequency, distance).unwrap();
        assert!(strain > 0.0 && strain < 1e-18);
        
        let dfdt = CosmologyDomain::gw_frequency_evolution(chirp_mass, frequency).unwrap();
        assert!(dfdt > 0.0);
        
        let t_merger = CosmologyDomain::gw_time_to_merger(chirp_mass, frequency).unwrap();
        assert!(t_merger > 0.0);
    }

    #[test]
    fn test_scale_factor() {
        let a0 = CosmologyDomain::scale_factor(0.0).unwrap();
        assert_relative_eq!(a0, 1.0, epsilon = 1e-10);
        
        let a1 = CosmologyDomain::scale_factor(1.0).unwrap();
        assert_relative_eq!(a1, 0.5, epsilon = 1e-10);
        
        let a_high = CosmologyDomain::scale_factor(9.0).unwrap();
        assert_relative_eq!(a_high, 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_cmb_temperature() {
        let params = CosmologyDomain::planck_2018_params();
        
        let t0 = CosmologyDomain::cmb_temperature_z(&params, 0.0).unwrap();
        assert_relative_eq!(t0, params.cmb_temperature, epsilon = 1e-10);
        
        let t_recomb = CosmologyDomain::cmb_temperature_z(&params, 1100.0).unwrap();
        assert_relative_eq!(t_recomb, params.cmb_temperature * 1101.0, epsilon = 1e-6);
    }

    #[test]
    fn test_critical_density() {
        let params = CosmologyDomain::planck_2018_params();
        
        let rho_c = CosmologyDomain::critical_density(&params, 0.0).unwrap();
        assert!(rho_c > 1e-29 && rho_c < 1e-28); // g/cm³
        
        // Critical density should increase with redshift
        let rho_c_high = CosmologyDomain::critical_density(&params, 2.0).unwrap();
        assert!(rho_c_high > rho_c);
    }

    #[test]
    fn test_omega_evolution() {
        let params = CosmologyDomain::planck_2018_params();
        
        let om_0 = CosmologyDomain::omega_matter_z(&params, 0.0).unwrap();
        let ol_0 = CosmologyDomain::omega_lambda_z(&params, 0.0).unwrap();
        
        assert_relative_eq!(om_0, params.omega_matter, epsilon = 1e-6);
        assert_relative_eq!(ol_0, params.omega_lambda, epsilon = 1e-6);
        
        // At high redshift, matter should dominate
        let om_high = CosmologyDomain::omega_matter_z(&params, 10.0).unwrap();
        let ol_high = CosmologyDomain::omega_lambda_z(&params, 10.0).unwrap();
        
        assert!(om_high > ol_high);
    }
}