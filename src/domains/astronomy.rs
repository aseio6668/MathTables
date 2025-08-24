use crate::core::{
    MathDomain, MathResult, MathError, Point3D, Vector3D,
    StellarProperties, OrbitalElements
};
use std::f64::consts::PI;

pub struct AstronomyDomain;

impl AstronomyDomain {
    pub fn new() -> Self {
        Self
    }
}

impl AstronomyDomain {
    /// Kepler's third law: P² = (4π²/GM) * a³
    /// Returns orbital period in seconds for given semi-major axis and central mass
    pub fn orbital_period(
        semi_major_axis_m: f64,
        central_mass_kg: f64
    ) -> MathResult<f64> {
        if semi_major_axis_m <= 0.0 || central_mass_kg <= 0.0 {
            return Err(MathError::InvalidArgument("Parameters must be positive".to_string()));
        }
        
        const G: f64 = 6.67430e-11; // m³/kg/s²
        
        let period_squared = (4.0 * PI.powi(2) / (G * central_mass_kg)) * semi_major_axis_m.powi(3);
        let period = period_squared.sqrt();
        Ok(period)
    }
    
    /// Orbital velocity at distance r: v = sqrt(GM/r)
    pub fn orbital_velocity(
        distance_m: f64,
        central_mass_kg: f64
    ) -> MathResult<f64> {
        if distance_m <= 0.0 || central_mass_kg <= 0.0 {
            return Err(MathError::InvalidArgument("Parameters must be positive".to_string()));
        }
        
        const G: f64 = 6.67430e-11; // m³/kg/s²
        
        let velocity = (G * central_mass_kg / distance_m).sqrt();
        Ok(velocity)
    }
    
    /// Escape velocity: v_esc = sqrt(2GM/r)
    pub fn escape_velocity(
        distance_m: f64,
        central_mass_kg: f64
    ) -> MathResult<f64> {
        if distance_m <= 0.0 || central_mass_kg <= 0.0 {
            return Err(MathError::InvalidArgument("Parameters must be positive".to_string()));
        }
        
        const G: f64 = 6.67430e-11; // m³/kg/s²
        
        let v_escape = (2.0 * G * central_mass_kg / distance_m).sqrt();
        Ok(v_escape)
    }
    
    /// Roche limit for fluid body: d = 2.44 * R_primary * (ρ_primary / ρ_satellite)^(1/3)
    pub fn roche_limit(
        primary_radius_m: f64,
        primary_density_kg_m3: f64,
        satellite_density_kg_m3: f64
    ) -> MathResult<f64> {
        if primary_radius_m <= 0.0 || primary_density_kg_m3 <= 0.0 || satellite_density_kg_m3 <= 0.0 {
            return Err(MathError::InvalidArgument("All parameters must be positive".to_string()));
        }
        
        let roche_distance = 2.44 * primary_radius_m * 
                            (primary_density_kg_m3 / satellite_density_kg_m3).powf(1.0/3.0);
        Ok(roche_distance)
    }
    
    /// Hill sphere radius: r_Hill = a * (m_satellite / (3 * m_star))^(1/3)
    pub fn hill_sphere_radius(
        orbital_distance_m: f64,
        satellite_mass_kg: f64,
        star_mass_kg: f64
    ) -> MathResult<f64> {
        if orbital_distance_m <= 0.0 || satellite_mass_kg <= 0.0 || star_mass_kg <= 0.0 {
            return Err(MathError::InvalidArgument("All parameters must be positive".to_string()));
        }
        
        let r_hill = orbital_distance_m * (satellite_mass_kg / (3.0 * star_mass_kg)).powf(1.0/3.0);
        Ok(r_hill)
    }
    
    /// Stellar luminosity from Stefan-Boltzmann law: L = 4πR²σT⁴
    pub fn stellar_luminosity(
        radius_m: f64,
        temperature_k: f64
    ) -> MathResult<f64> {
        if radius_m <= 0.0 || temperature_k <= 0.0 {
            return Err(MathError::InvalidArgument("Parameters must be positive".to_string()));
        }
        
        const STEFAN_BOLTZMANN: f64 = 5.670374419e-8; // W/m²/K⁴
        
        let luminosity = 4.0 * PI * radius_m.powi(2) * STEFAN_BOLTZMANN * temperature_k.powi(4);
        Ok(luminosity)
    }
    
    /// Mass-luminosity relation: L/L_sun = (M/M_sun)^α, where α ≈ 3.5 for main sequence
    pub fn mass_luminosity_relation(
        mass_solar: f64,
        alpha: f64
    ) -> MathResult<f64> {
        if mass_solar <= 0.0 {
            return Err(MathError::InvalidArgument("Mass must be positive".to_string()));
        }
        
        let luminosity_solar = mass_solar.powf(alpha);
        Ok(luminosity_solar)
    }
    
    /// Main sequence lifetime: τ = 10^10 * (M/M_sun)^(-2.5) years
    pub fn main_sequence_lifetime(mass_solar: f64) -> MathResult<f64> {
        if mass_solar <= 0.0 {
            return Err(MathError::InvalidArgument("Mass must be positive".to_string()));
        }
        
        let lifetime_years = 1e10 * mass_solar.powf(-2.5);
        Ok(lifetime_years)
    }
    
    /// Schwarzschild radius: Rs = 2GM/c²
    pub fn schwarzschild_radius(mass_kg: f64) -> MathResult<f64> {
        if mass_kg <= 0.0 {
            return Err(MathError::InvalidArgument("Mass must be positive".to_string()));
        }
        
        const G: f64 = 6.67430e-11; // m³/kg/s²
        const C: f64 = 2.998e8;     // m/s
        
        let rs = 2.0 * G * mass_kg / (C * C);
        Ok(rs)
    }
    
    /// Chandrasekhar limit for white dwarf mass: M_Ch ≈ 1.4 M_sun
    pub fn chandrasekhar_mass(mean_molecular_weight: f64) -> MathResult<f64> {
        if mean_molecular_weight <= 0.0 {
            return Err(MathError::InvalidArgument("Molecular weight must be positive".to_string()));
        }
        
        const M_SUN: f64 = 1.989e30; // kg
        
        // Chandrasekhar mass formula (approximate)
        let m_ch_solar = 1.4 * (2.0 / mean_molecular_weight).powi(2);
        let m_ch_kg = m_ch_solar * M_SUN;
        Ok(m_ch_kg)
    }
    
    /// Tidal force on satellite: F_tidal = 2 * G * M * m * r / d³
    pub fn tidal_force(
        primary_mass_kg: f64,
        satellite_mass_kg: f64,
        satellite_radius_m: f64,
        orbital_distance_m: f64
    ) -> MathResult<f64> {
        if primary_mass_kg <= 0.0 || satellite_mass_kg <= 0.0 || 
           satellite_radius_m <= 0.0 || orbital_distance_m <= 0.0 {
            return Err(MathError::InvalidArgument("All parameters must be positive".to_string()));
        }
        
        const G: f64 = 6.67430e-11; // m³/kg/s²
        
        let tidal_force = 2.0 * G * primary_mass_kg * satellite_mass_kg * 
                         satellite_radius_m / orbital_distance_m.powi(3);
        Ok(tidal_force)
    }
    
    /// Synodic period: 1/P_syn = |1/P₁ - 1/P₂|
    pub fn synodic_period(period1_s: f64, period2_s: f64) -> MathResult<f64> {
        if period1_s <= 0.0 || period2_s <= 0.0 {
            return Err(MathError::InvalidArgument("Periods must be positive".to_string()));
        }
        
        let p_syn_inv = (1.0 / period1_s - 1.0 / period2_s).abs();
        if p_syn_inv == 0.0 {
            return Err(MathError::InvalidArgument("Periods cannot be equal".to_string()));
        }
        
        let synodic_period = 1.0 / p_syn_inv;
        Ok(synodic_period)
    }
    
    /// Solve Kepler's equation: M = E - e*sin(E) using Newton-Raphson
    pub fn solve_keplers_equation(
        mean_anomaly: f64,
        eccentricity: f64,
        tolerance: f64
    ) -> MathResult<f64> {
        if eccentricity < 0.0 || eccentricity >= 1.0 {
            return Err(MathError::InvalidArgument("Eccentricity must be in [0,1)".to_string()));
        }
        if tolerance <= 0.0 {
            return Err(MathError::InvalidArgument("Tolerance must be positive".to_string()));
        }
        
        let mut eccentric_anomaly = mean_anomaly; // Initial guess
        let max_iterations = 100;
        
        for _ in 0..max_iterations {
            let f = eccentric_anomaly - eccentricity * eccentric_anomaly.sin() - mean_anomaly;
            let df = 1.0 - eccentricity * eccentric_anomaly.cos();
            
            if df.abs() < 1e-15 {
                return Err(MathError::ComputationError("Derivative too small in Newton-Raphson".to_string()));
            }
            
            let delta = f / df;
            eccentric_anomaly -= delta;
            
            if delta.abs() < tolerance {
                return Ok(eccentric_anomaly);
            }
        }
        
        Err(MathError::ComputationError("Failed to converge in Kepler equation".to_string()))
    }
    
    /// True anomaly from eccentric anomaly: ν = 2*atan(sqrt((1+e)/(1-e))*tan(E/2))
    pub fn true_anomaly_from_eccentric(
        eccentric_anomaly: f64,
        eccentricity: f64
    ) -> MathResult<f64> {
        if eccentricity < 0.0 || eccentricity >= 1.0 {
            return Err(MathError::InvalidArgument("Eccentricity must be in [0,1)".to_string()));
        }
        
        let sqrt_factor = ((1.0 + eccentricity) / (1.0 - eccentricity)).sqrt();
        let tan_half_e = (eccentric_anomaly / 2.0).tan();
        let true_anomaly = 2.0 * (sqrt_factor * tan_half_e).atan();
        
        Ok(true_anomaly)
    }
    
    /// Position in orbital plane from orbital elements
    pub fn orbital_position_2d(
        elements: &OrbitalElements,
        time: f64
    ) -> MathResult<(f64, f64)> {
        // Calculate mean anomaly at given time
        let n = 2.0 * PI / (365.25 * 24.0 * 3600.0); // Approximate mean motion (rad/s)
        let mean_anomaly = elements.mean_anomaly + n * (time - elements.epoch);
        
        // Solve Kepler's equation
        let eccentric_anomaly = Self::solve_keplers_equation(mean_anomaly, elements.eccentricity, 1e-12)?;
        
        // Calculate true anomaly
        let true_anomaly = Self::true_anomaly_from_eccentric(eccentric_anomaly, elements.eccentricity)?;
        
        // Calculate distance
        let r = elements.semi_major_axis * (1.0 - elements.eccentricity * eccentric_anomaly.cos());
        
        // Position in orbital plane
        let x = r * (true_anomaly + elements.argument_periapsis).cos();
        let y = r * (true_anomaly + elements.argument_periapsis).sin();
        
        Ok((x, y))
    }
    
    /// Habitable zone boundaries (inner and outer edges)
    /// Based on stellar luminosity and assuming Earth-like atmosphere
    pub fn habitable_zone_boundaries(stellar_luminosity_solar: f64) -> MathResult<(f64, f64)> {
        if stellar_luminosity_solar <= 0.0 {
            return Err(MathError::InvalidArgument("Luminosity must be positive".to_string()));
        }
        
        const AU: f64 = 1.496e11; // m
        
        // Conservative habitable zone (Kasting et al. 1993)
        let inner_edge_au = 0.95 * stellar_luminosity_solar.sqrt();
        let outer_edge_au = 1.37 * stellar_luminosity_solar.sqrt();
        
        let inner_edge_m = inner_edge_au * AU;
        let outer_edge_m = outer_edge_au * AU;
        
        Ok((inner_edge_m, outer_edge_m))
    }
    
    /// Transit depth for exoplanet: depth = (R_planet / R_star)²
    pub fn transit_depth(
        planet_radius_m: f64,
        star_radius_m: f64
    ) -> MathResult<f64> {
        if planet_radius_m <= 0.0 || star_radius_m <= 0.0 {
            return Err(MathError::InvalidArgument("Radii must be positive".to_string()));
        }
        
        let depth = (planet_radius_m / star_radius_m).powi(2);
        Ok(depth)
    }
    
    /// Transit duration for circular orbit
    pub fn transit_duration(
        orbital_period_s: f64,
        star_radius_m: f64,
        orbital_radius_m: f64,
        impact_parameter: f64
    ) -> MathResult<f64> {
        if orbital_period_s <= 0.0 || star_radius_m <= 0.0 || orbital_radius_m <= 0.0 {
            return Err(MathError::InvalidArgument("Physical parameters must be positive".to_string()));
        }
        if impact_parameter < 0.0 || impact_parameter > 1.0 {
            return Err(MathError::InvalidArgument("Impact parameter must be in [0,1]".to_string()));
        }
        
        let transit_chord = 2.0 * star_radius_m * (1.0 - impact_parameter.powi(2)).sqrt();
        let orbital_velocity = 2.0 * PI * orbital_radius_m / orbital_period_s;
        let duration = transit_chord / orbital_velocity;
        
        Ok(duration)
    }
}

impl MathDomain for AstronomyDomain {
    fn name(&self) -> &str {
        "Astronomy"
    }
    
    fn description(&self) -> &str {
        "Mathematical models for stellar dynamics, orbital mechanics, and astrophysical phenomena"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "orbital_period".to_string(),
            "orbital_velocity".to_string(),
            "escape_velocity".to_string(),
            "roche_limit".to_string(),
            "hill_sphere_radius".to_string(),
            "stellar_luminosity".to_string(),
            "mass_luminosity_relation".to_string(),
            "main_sequence_lifetime".to_string(),
            "schwarzschild_radius".to_string(),
            "chandrasekhar_mass".to_string(),
            "tidal_force".to_string(),
            "synodic_period".to_string(),
            "solve_keplers_equation".to_string(),
            "true_anomaly_from_eccentric".to_string(),
            "habitable_zone_boundaries".to_string(),
            "transit_depth".to_string(),
            "transit_duration".to_string(),
        ]
    }
    
    fn compute(&self, _operation: &str, _args: &[&dyn std::any::Any]) -> MathResult<Box<dyn std::any::Any>> {
        Err(MathError::NotImplemented("Generic compute not implemented for Astronomy domain".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_orbital_mechanics() {
        const M_SUN: f64 = 1.989e30; // kg
        const AU: f64 = 1.496e11; // m
        
        // Earth's orbital period should be approximately 1 year
        let period = AstronomyDomain::orbital_period(AU, M_SUN).unwrap();
        let year_seconds = 365.25 * 24.0 * 3600.0;
        assert_relative_eq!(period, year_seconds, epsilon = 1e-2);
        
        // Earth's orbital velocity should be ~30 km/s
        let velocity = AstronomyDomain::orbital_velocity(AU, M_SUN).unwrap();
        assert!(velocity > 25e3 && velocity < 35e3);
    }

    #[test]
    fn test_escape_velocity() {
        const M_EARTH: f64 = 5.972e24; // kg
        const R_EARTH: f64 = 6.371e6; // m
        
        let v_esc = AstronomyDomain::escape_velocity(R_EARTH, M_EARTH).unwrap();
        assert!(v_esc > 10e3 && v_esc < 12e3); // ~11.2 km/s
    }

    #[test]
    fn test_stellar_properties() {
        const R_SUN: f64 = 6.96e8; // m
        const T_SUN: f64 = 5778.0; // K
        const L_SUN: f64 = 3.828e26; // W
        
        let luminosity = AstronomyDomain::stellar_luminosity(R_SUN, T_SUN).unwrap();
        assert_relative_eq!(luminosity, L_SUN, epsilon = 1e-2);
        
        // Main sequence lifetime for Sun should be ~10 Gyr
        let lifetime = AstronomyDomain::main_sequence_lifetime(1.0).unwrap();
        assert!(lifetime > 8e9 && lifetime < 12e9);
        
        // Mass-luminosity relation for Sun
        let l_ratio = AstronomyDomain::mass_luminosity_relation(1.0, 3.5).unwrap();
        assert_relative_eq!(l_ratio, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_schwarzschild_radius() {
        const M_SUN: f64 = 1.989e30; // kg
        
        let rs_sun = AstronomyDomain::schwarzschild_radius(M_SUN).unwrap();
        assert!(rs_sun > 2e3 && rs_sun < 3e3); // ~2.95 km
    }

    #[test]
    fn test_keplers_equation() {
        let mean_anomaly = PI / 4.0; // 45 degrees
        let eccentricity = 0.1;
        
        let ecc_anomaly = AstronomyDomain::solve_keplers_equation(mean_anomaly, eccentricity, 1e-12).unwrap();
        
        // Verify the solution
        let check = ecc_anomaly - eccentricity * ecc_anomaly.sin();
        assert_relative_eq!(check, mean_anomaly, epsilon = 1e-10);
        
        // True anomaly calculation
        let true_anomaly = AstronomyDomain::true_anomaly_from_eccentric(ecc_anomaly, eccentricity).unwrap();
        assert!(true_anomaly > mean_anomaly); // Should be slightly larger for e > 0
    }

    #[test]
    fn test_habitable_zone() {
        // Sun's habitable zone
        let (inner, outer) = AstronomyDomain::habitable_zone_boundaries(1.0).unwrap();
        
        const AU: f64 = 1.496e11; // m
        let inner_au = inner / AU;
        let outer_au = outer / AU;
        
        assert!(inner_au > 0.8 && inner_au < 1.0);
        assert!(outer_au > 1.2 && outer_au < 1.5);
    }

    #[test]
    fn test_transit_calculations() {
        const R_JUPITER: f64 = 6.9911e7; // m
        const R_SUN: f64 = 6.96e8; // m
        
        let depth = AstronomyDomain::transit_depth(R_JUPITER, R_SUN).unwrap();
        assert!(depth > 0.001 && depth < 0.01); // ~1% depth
        
        let duration = AstronomyDomain::transit_duration(
            365.25 * 24.0 * 3600.0, // 1 year period
            R_SUN,
            1.496e11, // 1 AU
            0.0 // Central transit
        ).unwrap();
        assert!(duration > 10.0 * 3600.0 && duration < 20.0 * 3600.0); // 10-20 hours
    }

    #[test]
    fn test_synodic_period() {
        // Mars-Earth synodic period
        let earth_period = 365.25 * 24.0 * 3600.0; // s
        let mars_period = 687.0 * 24.0 * 3600.0; // s
        
        let synodic = AstronomyDomain::synodic_period(earth_period, mars_period).unwrap();
        let synodic_days = synodic / (24.0 * 3600.0);
        
        assert!(synodic_days > 700.0 && synodic_days < 800.0); // ~780 days
    }

    #[test]
    fn test_roche_limit() {
        const R_EARTH: f64 = 6.371e6; // m
        const RHO_EARTH: f64 = 5515.0; // kg/m³
        const RHO_WATER: f64 = 1000.0; // kg/m³
        
        let roche = AstronomyDomain::roche_limit(R_EARTH, RHO_EARTH, RHO_WATER).unwrap();
        assert!(roche > 15e6 && roche < 25e6); // ~2.44 Earth radii
    }

    #[test]
    fn test_tidal_force() {
        const M_EARTH: f64 = 5.972e24; // kg
        const M_MOON: f64 = 7.342e22; // kg
        const R_MOON: f64 = 1.737e6; // m
        const EARTH_MOON_DIST: f64 = 3.844e8; // m
        
        let tidal = AstronomyDomain::tidal_force(M_EARTH, M_MOON, R_MOON, EARTH_MOON_DIST).unwrap();
        assert!(tidal > 0.0);
    }
}