use std::f64::consts;

pub const PI: f64 = consts::PI;
pub const E: f64 = consts::E;
pub const TAU: f64 = consts::TAU;
pub const SQRT_2: f64 = consts::SQRT_2;
pub const SQRT_3: f64 = 1.7320508075688772;
pub const PHI: f64 = 1.618033988749895; // Golden ratio
pub const LN_2: f64 = consts::LN_2;
pub const LN_10: f64 = consts::LN_10;

pub const PLANCK_CONSTANT: f64 = 6.62607015e-34;
pub const SPEED_OF_LIGHT: f64 = 299792458.0;
pub const GRAVITATIONAL_CONSTANT: f64 = 6.67430e-11;
pub const AVOGADRO_NUMBER: f64 = 6.02214076e23;

pub const DEGREES_TO_RADIANS: f64 = PI / 180.0;
pub const RADIANS_TO_DEGREES: f64 = 180.0 / PI;

pub struct MathematicalConstants;

impl MathematicalConstants {
    pub fn catalan() -> f64 {
        0.915965594177219015054603514932384110774
    }
    
    pub fn euler_mascheroni() -> f64 {
        0.5772156649015328606065120900824024310422
    }
    
    pub fn apery() -> f64 {
        1.2020569031595942853997381615114499907649
    }
}