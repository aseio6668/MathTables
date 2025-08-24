use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;
use std::ops::{Add, Sub, Neg};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct Multivector {
    pub coefficients: HashMap<BasisElement, f64>,
    pub dimension: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BasisElement {
    pub indices: Vec<usize>, // Sorted indices of basis vectors
}

#[derive(Debug, Clone)]
pub struct GeometricSpace {
    pub dimension: usize,
    pub signature: Vec<i8>, // Metric signature: +1, -1, or 0
    pub basis_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Rotor {
    pub multivector: Multivector,
}

#[derive(Debug, Clone)]
pub struct Motor {
    pub multivector: Multivector,
}

#[derive(Debug, Clone)]
pub struct Versor {
    pub multivector: Multivector,
}

pub struct GeometricAlgebraDomain;

impl BasisElement {
    pub fn scalar() -> Self {
        BasisElement { indices: vec![] }
    }
    
    pub fn vector(index: usize) -> Self {
        BasisElement { indices: vec![index] }
    }
    
    pub fn bivector(i: usize, j: usize) -> Self {
        let mut indices = vec![i, j];
        indices.sort();
        BasisElement { indices }
    }
    
    pub fn pseudoscalar(dimension: usize) -> Self {
        BasisElement { indices: (0..dimension).collect() }
    }
    
    pub fn grade(&self) -> usize {
        self.indices.len()
    }
    
    pub fn geometric_product(&self, other: &BasisElement) -> (BasisElement, f64, Vec<i8>) {
        let mut result_indices = self.indices.clone();
        let mut sign = 1.0;
        let mut metric_factors = Vec::new();
        
        for &other_idx in &other.indices {
            let mut position = result_indices.len();
            let mut swaps = 0;
            
            // Find insertion position and count swaps
            for (i, &idx) in result_indices.iter().enumerate() {
                if other_idx == idx {
                    // Same basis vector - contributes metric signature
                    result_indices.remove(i);
                    metric_factors.push(other_idx as i8);
                    position = usize::MAX; // Mark for removal
                    break;
                } else if other_idx < idx {
                    position = i;
                    swaps = result_indices.len() - i;
                    break;
                }
            }
            
            if position != usize::MAX {
                result_indices.insert(position, other_idx);
                // Apply sign from anticommutation
                if swaps % 2 == 1 {
                    sign *= -1.0;
                }
            }
        }
        
        (BasisElement { indices: result_indices }, sign, metric_factors)
    }
    
    pub fn reverse(&self) -> f64 {
        let k = self.grade();
        if (k * (k - 1) / 2) % 2 == 0 { 1.0 } else { -1.0 }
    }
    
    pub fn conjugate(&self) -> f64 {
        let k = self.grade();
        if k % 2 == 0 { 1.0 } else { -1.0 }
    }
    
    pub fn involute(&self) -> f64 {
        let k = self.grade();
        if (k * (k + 1) / 2) % 2 == 0 { 1.0 } else { -1.0 }
    }
}

impl Multivector {
    pub fn new(dimension: usize) -> Self {
        Multivector {
            coefficients: HashMap::new(),
            dimension,
        }
    }
    
    pub fn scalar(value: f64, dimension: usize) -> Self {
        let mut mv = Multivector::new(dimension);
        mv.coefficients.insert(BasisElement::scalar(), value);
        mv
    }
    
    pub fn vector(components: &[f64]) -> Self {
        let dimension = components.len();
        let mut mv = Multivector::new(dimension);
        
        for (i, &component) in components.iter().enumerate() {
            if component != 0.0 {
                mv.coefficients.insert(BasisElement::vector(i), component);
            }
        }
        mv
    }
    
    pub fn bivector(i: usize, j: usize, value: f64, dimension: usize) -> Self {
        let mut mv = Multivector::new(dimension);
        if value != 0.0 {
            mv.coefficients.insert(BasisElement::bivector(i, j), value);
        }
        mv
    }
    
    pub fn get_coefficient(&self, basis: &BasisElement) -> f64 {
        self.coefficients.get(basis).copied().unwrap_or(0.0)
    }
    
    pub fn set_coefficient(&mut self, basis: BasisElement, value: f64) {
        if value == 0.0 {
            self.coefficients.remove(&basis);
        } else {
            self.coefficients.insert(basis, value);
        }
    }
    
    pub fn grade_part(&self, k: usize) -> Multivector {
        let mut result = Multivector::new(self.dimension);
        
        for (basis, &coeff) in &self.coefficients {
            if basis.grade() == k {
                result.coefficients.insert(basis.clone(), coeff);
            }
        }
        
        result
    }
    
    pub fn scalar_part(&self) -> f64 {
        self.get_coefficient(&BasisElement::scalar())
    }
    
    pub fn vector_part(&self) -> Vec<f64> {
        let mut components = vec![0.0; self.dimension];
        
        for (basis, &coeff) in &self.coefficients {
            if basis.grade() == 1 && !basis.indices.is_empty() {
                components[basis.indices[0]] = coeff;
            }
        }
        
        components
    }
    
    pub fn magnitude_squared(&self, signature: &[i8]) -> f64 {
        let conjugate = self.conjugate();
        let product = self.geometric_product(&conjugate, signature);
        product.scalar_part()
    }
    
    pub fn magnitude(&self, signature: &[i8]) -> f64 {
        self.magnitude_squared(signature).sqrt()
    }
    
    pub fn normalize(&self, signature: &[i8]) -> MathResult<Multivector> {
        let mag = self.magnitude(signature);
        if mag == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        
        let mut result = self.clone();
        for (_, coeff) in result.coefficients.iter_mut() {
            *coeff /= mag;
        }
        
        Ok(result)
    }
    
    pub fn reverse(&self) -> Multivector {
        let mut result = Multivector::new(self.dimension);
        
        for (basis, &coeff) in &self.coefficients {
            let sign = basis.reverse();
            result.coefficients.insert(basis.clone(), coeff * sign);
        }
        
        result
    }
    
    pub fn conjugate(&self) -> Multivector {
        let mut result = Multivector::new(self.dimension);
        
        for (basis, &coeff) in &self.coefficients {
            let sign = basis.conjugate();
            result.coefficients.insert(basis.clone(), coeff * sign);
        }
        
        result
    }
    
    pub fn involute(&self) -> Multivector {
        let mut result = Multivector::new(self.dimension);
        
        for (basis, &coeff) in &self.coefficients {
            let sign = basis.involute();
            result.coefficients.insert(basis.clone(), coeff * sign);
        }
        
        result
    }
    
    pub fn geometric_product(&self, other: &Multivector, signature: &[i8]) -> Multivector {
        let mut result = Multivector::new(self.dimension.max(other.dimension));
        
        for (basis_a, &coeff_a) in &self.coefficients {
            for (basis_b, &coeff_b) in &other.coefficients {
                let (product_basis, sign, metric_factors) = basis_a.geometric_product(basis_b);
                
                // Apply metric signature
                let mut metric_sign = 1.0;
                for &factor_idx in &metric_factors {
                    if (factor_idx as usize) < signature.len() {
                        match signature[factor_idx as usize] {
                            1 => {}, // Positive - no change
                            -1 => metric_sign *= -1.0,
                            0 => metric_sign = 0.0, // Degenerate
                            _ => {},
                        }
                    }
                }
                
                let final_coeff = coeff_a * coeff_b * sign * metric_sign;
                
                if final_coeff != 0.0 {
                    let current = result.get_coefficient(&product_basis);
                    result.set_coefficient(product_basis, current + final_coeff);
                }
            }
        }
        
        result
    }
    
    pub fn wedge_product(&self, other: &Multivector) -> Multivector {
        let mut result = Multivector::new(self.dimension.max(other.dimension));
        
        for (basis_a, &coeff_a) in &self.coefficients {
            for (basis_b, &coeff_b) in &other.coefficients {
                // Check if indices overlap (wedge is zero if they do)
                let mut overlap = false;
                for &idx_a in &basis_a.indices {
                    if basis_b.indices.contains(&idx_a) {
                        overlap = true;
                        break;
                    }
                }
                
                if !overlap {
                    let (product_basis, sign, _) = basis_a.geometric_product(basis_b);
                    let final_coeff = coeff_a * coeff_b * sign;
                    
                    if final_coeff != 0.0 {
                        let current = result.get_coefficient(&product_basis);
                        result.set_coefficient(product_basis, current + final_coeff);
                    }
                }
            }
        }
        
        result
    }
    
    pub fn dot_product(&self, other: &Multivector, signature: &[i8]) -> Multivector {
        let geometric = self.geometric_product(other, signature);
        let wedge = self.wedge_product(other);
        &geometric - &wedge
    }
    
    pub fn commutator_product(&self, other: &Multivector, signature: &[i8]) -> Multivector {
        let ab = self.geometric_product(other, signature);
        let ba = other.geometric_product(self, signature);
        let half = Multivector::scalar(0.5, self.dimension);
        (&ab - &ba).geometric_product(&half, signature)
    }
    
    pub fn exp(&self, signature: &[i8]) -> MathResult<Multivector> {
        // Series expansion: exp(A) = 1 + A + A^2/2! + A^3/3! + ...
        let mut result = Multivector::scalar(1.0, self.dimension);
        let mut term = Multivector::scalar(1.0, self.dimension);
        let max_terms = 20;
        
        for n in 1..=max_terms {
            term = term.geometric_product(self, signature);
            let factorial = (1..=n).fold(1.0, |acc, x| acc * x as f64);
            let scaled_term = term.scalar_multiply(1.0 / factorial);
            result = &result + &scaled_term;
            
            // Check convergence
            if scaled_term.magnitude(signature) < 1e-12 {
                break;
            }
        }
        
        Ok(result)
    }
    
    pub fn ln(&self, signature: &[i8]) -> MathResult<Multivector> {
        // For rotors: ln(R) = θ/2 * B where R = cos(θ/2) + sin(θ/2)*B
        let scalar = self.scalar_part();
        let vector_part = self.grade_part(2); // Bivector part for rotors
        
        if scalar <= 0.0 {
            return Err(MathError::DomainError("Logarithm undefined for this multivector".to_string()));
        }
        
        let bivector_mag = vector_part.magnitude(signature);
        
        if bivector_mag == 0.0 {
            // Pure scalar
            Ok(Multivector::scalar(scalar.ln(), self.dimension))
        } else {
            let angle = bivector_mag.atan2(scalar);
            let normalized_bivector = vector_part.scalar_multiply(angle / bivector_mag);
            Ok(normalized_bivector)
        }
    }
    
    pub fn scalar_multiply(&self, scalar: f64) -> Multivector {
        let mut result = Multivector::new(self.dimension);
        
        for (basis, &coeff) in &self.coefficients {
            result.coefficients.insert(basis.clone(), coeff * scalar);
        }
        
        result
    }
}

impl Add for &Multivector {
    type Output = Multivector;
    
    fn add(self, other: &Multivector) -> Multivector {
        let mut result = Multivector::new(self.dimension.max(other.dimension));
        
        // Add coefficients from self
        for (basis, &coeff) in &self.coefficients {
            result.coefficients.insert(basis.clone(), coeff);
        }
        
        // Add coefficients from other
        for (basis, &coeff) in &other.coefficients {
            let current = result.get_coefficient(basis);
            result.set_coefficient(basis.clone(), current + coeff);
        }
        
        result
    }
}

impl Sub for &Multivector {
    type Output = Multivector;
    
    fn sub(self, other: &Multivector) -> Multivector {
        let mut result = Multivector::new(self.dimension.max(other.dimension));
        
        // Add coefficients from self
        for (basis, &coeff) in &self.coefficients {
            result.coefficients.insert(basis.clone(), coeff);
        }
        
        // Subtract coefficients from other
        for (basis, &coeff) in &other.coefficients {
            let current = result.get_coefficient(basis);
            result.set_coefficient(basis.clone(), current - coeff);
        }
        
        result
    }
}

impl Neg for &Multivector {
    type Output = Multivector;
    
    fn neg(self) -> Multivector {
        let mut result = Multivector::new(self.dimension);
        
        for (basis, &coeff) in &self.coefficients {
            result.coefficients.insert(basis.clone(), -coeff);
        }
        
        result
    }
}

impl GeometricSpace {
    pub fn euclidean(dimension: usize) -> Self {
        GeometricSpace {
            dimension,
            signature: vec![1; dimension],
            basis_names: (0..dimension).map(|i| format!("e{}", i)).collect(),
        }
    }
    
    pub fn minkowski(space_dims: usize) -> Self {
        let mut signature = vec![-1]; // Time dimension
        signature.extend(vec![1; space_dims]); // Space dimensions
        
        let mut basis_names = vec!["e0".to_string()];
        basis_names.extend((1..=space_dims).map(|i| format!("e{}", i)));
        
        GeometricSpace {
            dimension: space_dims + 1,
            signature,
            basis_names,
        }
    }
    
    pub fn conformal(space_dims: usize) -> Self {
        // Conformal space adds two null dimensions
        let total_dims = space_dims + 2;
        let mut signature = vec![1; space_dims];
        signature.extend(vec![1, -1]); // e+ and e-
        
        let mut basis_names: Vec<String> = (0..space_dims).map(|i| format!("e{}", i)).collect();
        basis_names.extend(vec!["e+".to_string(), "e-".to_string()]);
        
        GeometricSpace {
            dimension: total_dims,
            signature,
            basis_names,
        }
    }
}

impl GeometricAlgebraDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn create_rotor(angle: f64, bivector: &Multivector, signature: &[i8]) -> MathResult<Rotor> {
        let half_angle = angle / 2.0;
        let normalized_bivector = bivector.normalize(signature)?;
        let bivector_scaled = normalized_bivector.scalar_multiply(half_angle);
        let rotor_mv = bivector_scaled.exp(signature)?;
        
        Ok(Rotor { multivector: rotor_mv })
    }
    
    pub fn rotate_vector(rotor: &Rotor, vector: &Multivector, signature: &[i8]) -> Multivector {
        let rotor_reverse = rotor.multivector.reverse();
        let temp = rotor.multivector.geometric_product(vector, signature);
        temp.geometric_product(&rotor_reverse, signature)
    }
    
    pub fn reflect_vector(vector: &Multivector, mirror: &Multivector, signature: &[i8]) -> MathResult<Multivector> {
        let mirror_normalized = mirror.normalize(signature)?;
        let temp = mirror_normalized.geometric_product(vector, signature);
        Ok(temp.geometric_product(&mirror_normalized, signature))
    }
    
    pub fn meet_join_duality(
        a: &Multivector,
        b: &Multivector,
        pseudoscalar: &Multivector,
        signature: &[i8],
    ) -> (Multivector, Multivector) {
        let meet = a.wedge_product(b);
        
        // Join via duality: A ∨ B = (A* ∧ B*)*
        let a_dual = Self::dual(a, pseudoscalar, signature);
        let b_dual = Self::dual(b, pseudoscalar, signature);
        let join_dual = a_dual.wedge_product(&b_dual);
        let join = Self::dual(&join_dual, pseudoscalar, signature);
        
        (meet, join)
    }
    
    pub fn dual(mv: &Multivector, pseudoscalar: &Multivector, signature: &[i8]) -> Multivector {
        mv.geometric_product(pseudoscalar, signature)
    }
    
    pub fn geometric_mean(multivectors: &[Multivector], signature: &[i8]) -> MathResult<Multivector> {
        if multivectors.is_empty() {
            return Err(MathError::InvalidArgument("Cannot compute mean of empty set".to_string()));
        }
        
        // Simple arithmetic mean for now (geometric mean is more complex)
        let mut sum = Multivector::new(multivectors[0].dimension);
        
        for mv in multivectors {
            sum = &sum + mv;
        }
        
        let n = multivectors.len() as f64;
        Ok(sum.scalar_multiply(1.0 / n))
    }
    
    pub fn blade_factorization(blade: &Multivector, signature: &[i8]) -> MathResult<Vec<Multivector>> {
        // Simplified factorization for simple blades
        let grade = blade.coefficients.keys().map(|b| b.grade()).max().unwrap_or(0);
        
        if grade <= 1 {
            return Ok(vec![blade.clone()]);
        }
        
        // For higher grades, this is a complex problem
        // Simplified implementation returns the blade itself
        Ok(vec![blade.clone()])
    }
    
    pub fn motor_from_line_rotation(
        angle: f64,
        line_point: &Multivector,
        line_direction: &Multivector,
        signature: &[i8],
    ) -> MathResult<Motor> {
        // Create motor for rotation about a line
        let rotor_part = Self::create_rotor(angle, line_direction, signature)?;
        
        // Translation part (simplified)
        let translation = line_point.geometric_product(line_direction, signature);
        let translator_part = translation.scalar_multiply(angle / 2.0);
        
        let motor_mv = &rotor_part.multivector + &translator_part;
        Ok(Motor { multivector: motor_mv })
    }
    
    pub fn conformal_point_from_euclidean(euclidean_point: &[f64]) -> MathResult<Multivector> {
        let n = euclidean_point.len();
        let conformal_dim = n + 2;
        let mut conformal = Multivector::new(conformal_dim);
        
        // Set scalar part to 1
        conformal.set_coefficient(BasisElement::scalar(), 1.0);
        
        // Set euclidean coordinates
        for (i, &coord) in euclidean_point.iter().enumerate() {
            conformal.set_coefficient(BasisElement::vector(i), coord);
        }
        
        // Set null coordinates
        let norm_squared: f64 = euclidean_point.iter().map(|&x| x * x).sum();
        conformal.set_coefficient(BasisElement::vector(n), norm_squared / 2.0); // e+
        conformal.set_coefficient(BasisElement::vector(n + 1), norm_squared / 2.0); // e-
        
        Ok(conformal)
    }
    
    pub fn conformal_sphere(center: &[f64], radius: f64) -> MathResult<Multivector> {
        let point = Self::conformal_point_from_euclidean(center)?;
        let n = center.len();
        let conformal_dim = n + 2;
        let mut sphere = point;
        
        // Adjust for radius
        let radius_term = radius * radius / 2.0;
        let current_eplus = sphere.get_coefficient(&BasisElement::vector(n));
        sphere.set_coefficient(BasisElement::vector(n), current_eplus - radius_term);
        
        Ok(sphere)
    }
    
    pub fn sandwich_product(
        operator: &Multivector,
        operand: &Multivector,
        signature: &[i8],
    ) -> Multivector {
        let temp = operator.geometric_product(operand, signature);
        let operator_reverse = operator.reverse();
        temp.geometric_product(&operator_reverse, signature)
    }
}

impl MathDomain for GeometricAlgebraDomain {
    fn name(&self) -> &str { "Geometric Algebra" }
    fn description(&self) -> &str { "Geometric algebra with multivectors, rotors, and conformal geometry" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, _args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            _ => Err(MathError::InvalidOperation(format!("Operation {} not implemented in compute interface", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "multivector_addition".to_string(),
            "geometric_product".to_string(),
            "wedge_product".to_string(),
            "dot_product".to_string(),
            "create_rotor".to_string(),
            "rotate_vector".to_string(),
            "reflect_vector".to_string(),
            "dual_operation".to_string(),
            "meet_join".to_string(),
            "conformal_point".to_string(),
            "conformal_sphere".to_string(),
            "motor_transformation".to_string(),
            "sandwich_product".to_string(),
            "blade_factorization".to_string(),
        ]
    }
}