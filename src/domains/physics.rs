use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;

#[derive(Debug, Clone)]
pub struct QuantumState {
    pub amplitudes: Vec<(f64, f64)>, // (real, imaginary) components
    pub basis_labels: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AngularMomentum {
    pub l: f64, // orbital quantum number
    pub m: f64, // magnetic quantum number
    pub j: f64, // total angular momentum
    pub spin: f64,
}

#[derive(Debug, Clone)]
pub struct Angle {
    pub radians: f64,
    pub degrees: f64,
    pub angle_type: AngleType,
}

#[derive(Debug, Clone)]
pub enum AngleType {
    Acute,
    Right,
    Obtuse,
    Straight,
    Reflex,
    Full,
}

pub struct PhysicsDomain;

impl PhysicsDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn create_quantum_state(amplitudes: Vec<(f64, f64)>, labels: Vec<String>) -> MathResult<QuantumState> {
        if amplitudes.len() != labels.len() {
            return Err(MathError::InvalidArgument("Amplitudes and labels must have same length".to_string()));
        }
        
        let norm_squared: f64 = amplitudes.iter()
            .map(|(r, i)| r * r + i * i)
            .sum();
            
        if (norm_squared - 1.0).abs() > 1e-10 {
            return Err(MathError::InvalidArgument("Quantum state must be normalized".to_string()));
        }
        
        Ok(QuantumState {
            amplitudes,
            basis_labels: labels,
        })
    }
    
    pub fn quantum_probability(state: &QuantumState, basis_index: usize) -> MathResult<f64> {
        if basis_index >= state.amplitudes.len() {
            return Err(MathError::InvalidArgument("Basis index out of bounds".to_string()));
        }
        
        let (r, i) = state.amplitudes[basis_index];
        Ok(r * r + i * i)
    }
    
    pub fn create_angle(degrees: f64) -> Angle {
        let radians = degrees.to_radians();
        let normalized_degrees = degrees % 360.0;
        
        let angle_type = match normalized_degrees.abs() {
            d if d == 0.0 => AngleType::Straight,
            d if d > 0.0 && d < 90.0 => AngleType::Acute,
            d if d == 90.0 => AngleType::Right,
            d if d > 90.0 && d < 180.0 => AngleType::Obtuse,
            d if d == 180.0 => AngleType::Straight,
            d if d > 180.0 && d < 360.0 => AngleType::Reflex,
            _ => AngleType::Full,
        };
        
        Angle {
            radians,
            degrees: normalized_degrees,
            angle_type,
        }
    }
    
    pub fn create_angular_momentum(l: f64, m: f64, spin: f64) -> MathResult<AngularMomentum> {
        if m.abs() > l {
            return Err(MathError::InvalidArgument("Magnetic quantum number |m| must be ≤ l".to_string()));
        }
        
        let j = (l * (l + 1.0)).sqrt();
        
        Ok(AngularMomentum { l, m, j, spin })
    }
    
    pub fn angular_momentum_z_component(am: &AngularMomentum) -> f64 {
        am.m
    }
    
    pub fn angular_momentum_magnitude(am: &AngularMomentum) -> f64 {
        (am.l * (am.l + 1.0)).sqrt()
    }
    
    pub fn commutator_uncertainty(am: &AngularMomentum) -> f64 {
        am.m.abs()
    }
}

impl MathDomain for PhysicsDomain {
    fn name(&self) -> &str { "Physics" }
    fn description(&self) -> &str { "Mathematical physics including quantum mechanics, angles, and angular momentum" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "create_quantum_state" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("create_quantum_state requires 2 arguments".to_string()));
                }
                let amplitudes = args[0].downcast_ref::<Vec<(f64, f64)>>().ok_or_else(|| MathError::InvalidArgument("First argument must be Vec<(f64, f64)>".to_string()))?;
                let labels = args[1].downcast_ref::<Vec<String>>().ok_or_else(|| MathError::InvalidArgument("Second argument must be Vec<String>".to_string()))?;
                Ok(Box::new(Self::create_quantum_state(amplitudes.clone(), labels.clone())?))
            },
            "quantum_probability" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("quantum_probability requires 2 arguments".to_string()));
                }
                let state = args[0].downcast_ref::<QuantumState>().ok_or_else(|| MathError::InvalidArgument("First argument must be QuantumState".to_string()))?;
                let index = args[1].downcast_ref::<usize>().ok_or_else(|| MathError::InvalidArgument("Second argument must be usize".to_string()))?;
                Ok(Box::new(Self::quantum_probability(state, *index)?))
            },
            "create_angle" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("create_angle requires 1 argument".to_string()));
                }
                let degrees = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Argument must be f64".to_string()))?;
                Ok(Box::new(Self::create_angle(*degrees)))
            },
            "create_angular_momentum" => {
                if args.len() != 3 {
                    return Err(MathError::InvalidArgument("create_angular_momentum requires 3 arguments".to_string()));
                }
                let l = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let m = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                let spin = args[2].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be f64".to_string()))?;
                Ok(Box::new(Self::create_angular_momentum(*l, *m, *spin)?))
            },
            "angular_momentum_magnitude" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("angular_momentum_magnitude requires 1 argument".to_string()));
                }
                let am = args[0].downcast_ref::<AngularMomentum>().ok_or_else(|| MathError::InvalidArgument("Argument must be AngularMomentum".to_string()))?;
                Ok(Box::new(Self::angular_momentum_magnitude(am)))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "create_quantum_state".to_string(),
            "quantum_probability".to_string(),
            "create_angle".to_string(),
            "create_angular_momentum".to_string(),
            "angular_momentum_magnitude".to_string(),
            "angular_momentum_z_component".to_string(),
            "commutator_uncertainty".to_string(),
        ]
    }
}