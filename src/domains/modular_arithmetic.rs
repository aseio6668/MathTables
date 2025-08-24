use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModularInteger {
    pub value: i64,
    pub modulus: i64,
}

#[derive(Debug, Clone)]
pub struct ChineseRemainderTheorem {
    pub remainders: Vec<i64>,
    pub moduli: Vec<i64>,
    pub solution: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct QuadraticResidue {
    pub number: i64,
    pub modulus: i64,
    pub is_residue: bool,
    pub square_roots: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct PrimitiveRoot {
    pub modulus: i64,
    pub roots: Vec<i64>,
    pub order: usize,
}

#[derive(Debug, Clone)]
pub struct DiscreteLogarithm {
    pub base: i64,
    pub result: i64,
    pub modulus: i64,
    pub logarithm: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct ModularPolynomial {
    pub coefficients: Vec<i64>, // coefficients[i] is coefficient of x^i
    pub modulus: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EllipticCurve {
    pub a: i64, // coefficient of x
    pub b: i64, // constant term (y² = x³ + ax + b)
    pub modulus: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EllipticCurvePoint {
    pub x: Option<i64>, // None for point at infinity
    pub y: Option<i64>,
    pub curve: EllipticCurve,
}

#[derive(Debug, Clone)]
pub struct FiniteField {
    pub characteristic: i64,
    pub degree: usize,
    pub primitive_polynomial: Vec<i64>,
}

pub struct ModularArithmeticDomain;

impl ModularInteger {
    pub fn new(value: i64, modulus: i64) -> MathResult<Self> {
        if modulus <= 0 {
            return Err(MathError::InvalidArgument("Modulus must be positive".to_string()));
        }
        
        let normalized_value = value.rem_euclid(modulus);
        Ok(ModularInteger {
            value: normalized_value,
            modulus,
        })
    }
    
    pub fn add(&self, other: &ModularInteger) -> MathResult<ModularInteger> {
        if self.modulus != other.modulus {
            return Err(MathError::InvalidArgument("Moduli must be equal".to_string()));
        }
        
        let result = (self.value + other.value) % self.modulus;
        Ok(ModularInteger {
            value: result,
            modulus: self.modulus,
        })
    }
    
    pub fn subtract(&self, other: &ModularInteger) -> MathResult<ModularInteger> {
        if self.modulus != other.modulus {
            return Err(MathError::InvalidArgument("Moduli must be equal".to_string()));
        }
        
        let result = (self.value - other.value).rem_euclid(self.modulus);
        Ok(ModularInteger {
            value: result,
            modulus: self.modulus,
        })
    }
    
    pub fn multiply(&self, other: &ModularInteger) -> MathResult<ModularInteger> {
        if self.modulus != other.modulus {
            return Err(MathError::InvalidArgument("Moduli must be equal".to_string()));
        }
        
        let result = (self.value * other.value) % self.modulus;
        Ok(ModularInteger {
            value: result,
            modulus: self.modulus,
        })
    }
    
    pub fn power(&self, exponent: i64) -> MathResult<ModularInteger> {
        if exponent < 0 {
            let inverse = self.modular_inverse()?;
            return inverse.power(-exponent);
        }
        
        let mut result = 1;
        let mut base = self.value;
        let mut exp = exponent;
        
        while exp > 0 {
            if exp % 2 == 1 {
                result = (result * base) % self.modulus;
            }
            base = (base * base) % self.modulus;
            exp /= 2;
        }
        
        Ok(ModularInteger {
            value: result,
            modulus: self.modulus,
        })
    }
    
    pub fn modular_inverse(&self) -> MathResult<ModularInteger> {
        let (gcd, x, _) = Self::extended_gcd(self.value, self.modulus);
        
        if gcd != 1 {
            return Err(MathError::InvalidArgument("Modular inverse does not exist".to_string()));
        }
        
        let inverse = x.rem_euclid(self.modulus);
        Ok(ModularInteger {
            value: inverse,
            modulus: self.modulus,
        })
    }
    
    fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
        if b == 0 {
            (a, 1, 0)
        } else {
            let (gcd, x1, y1) = Self::extended_gcd(b, a % b);
            (gcd, y1, x1 - (a / b) * y1)
        }
    }
    
    pub fn order(&self) -> MathResult<i64> {
        if self.value == 0 {
            return Ok(0);
        }
        
        let mut current = self.clone();
        let identity = ModularInteger::new(1, self.modulus)?;
        
        for order in 1..=self.modulus {
            if current == identity {
                return Ok(order);
            }
            current = current.multiply(self)?;
        }
        
        Err(MathError::ComputationError("Order computation failed".to_string()))
    }
}

impl ModularArithmeticDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn chinese_remainder_theorem(remainders: &[i64], moduli: &[i64]) -> MathResult<ChineseRemainderTheorem> {
        if remainders.len() != moduli.len() {
            return Err(MathError::InvalidArgument("Remainders and moduli must have same length".to_string()));
        }
        
        if moduli.is_empty() {
            return Err(MathError::InvalidArgument("Empty input".to_string()));
        }
        
        // Check that moduli are pairwise coprime
        for i in 0..moduli.len() {
            for j in (i+1)..moduli.len() {
                let (gcd, _, _) = ModularInteger::extended_gcd(moduli[i], moduli[j]);
                if gcd != 1 {
                    return Err(MathError::InvalidArgument("Moduli must be pairwise coprime".to_string()));
                }
            }
        }
        
        let product: i64 = moduli.iter().product();
        let mut solution = 0;
        
        for i in 0..remainders.len() {
            let ni = product / moduli[i];
            let (_, mi, _) = ModularInteger::extended_gcd(ni, moduli[i]);
            solution += remainders[i] * ni * mi;
        }
        
        solution = solution.rem_euclid(product);
        
        Ok(ChineseRemainderTheorem {
            remainders: remainders.to_vec(),
            moduli: moduli.to_vec(),
            solution: Some(solution),
        })
    }
    
    pub fn euler_totient(n: i64) -> MathResult<i64> {
        if n <= 0 {
            return Err(MathError::InvalidArgument("n must be positive".to_string()));
        }
        
        if n == 1 {
            return Ok(1);
        }
        
        let mut result = n;
        let mut temp_n = n;
        
        // Find all prime factors
        for p in 2..=((n as f64).sqrt() as i64 + 1) {
            if temp_n % p == 0 {
                while temp_n % p == 0 {
                    temp_n /= p;
                }
                result -= result / p;
            }
        }
        
        if temp_n > 1 {
            result -= result / temp_n;
        }
        
        Ok(result)
    }
    
    pub fn carmichael_function(n: i64) -> MathResult<i64> {
        if n <= 0 {
            return Err(MathError::InvalidArgument("n must be positive".to_string()));
        }
        
        if n == 1 {
            return Ok(1);
        }
        
        // Simplified implementation for demonstration
        // Full implementation would factor n and compute lambda(p^k) for each prime power
        let totient = Self::euler_totient(n)?;
        
        // For most cases, Carmichael function divides Euler's totient
        // This is a placeholder - proper implementation needs prime factorization
        Ok(totient)
    }
    
    pub fn quadratic_residue(a: i64, p: i64) -> MathResult<QuadraticResidue> {
        if p <= 1 {
            return Err(MathError::InvalidArgument("p must be greater than 1".to_string()));
        }
        
        let a_mod = a.rem_euclid(p);
        let mut square_roots = Vec::new();
        let mut is_residue = false;
        
        // Check if a is a quadratic residue by testing all possible values
        for x in 0..p {
            if (x * x) % p == a_mod {
                square_roots.push(x);
                is_residue = true;
            }
        }
        
        Ok(QuadraticResidue {
            number: a,
            modulus: p,
            is_residue,
            square_roots,
        })
    }
    
    pub fn legendre_symbol(a: i64, p: i64) -> MathResult<i8> {
        if p <= 2 || p % 2 == 0 {
            return Err(MathError::InvalidArgument("p must be an odd prime".to_string()));
        }
        
        let a_mod = a.rem_euclid(p);
        
        if a_mod == 0 {
            return Ok(0);
        }
        
        // Use Euler's criterion: (a/p) ≡ a^((p-1)/2) (mod p)
        let modular_a = ModularInteger::new(a_mod, p)?;
        let result = modular_a.power((p - 1) / 2)?;
        
        if result.value == 1 {
            Ok(1)
        } else if result.value == p - 1 {
            Ok(-1)
        } else {
            Ok(0)
        }
    }
    
    pub fn jacobi_symbol(a: i64, n: i64) -> MathResult<i8> {
        if n <= 0 || n % 2 == 0 {
            return Err(MathError::InvalidArgument("n must be a positive odd integer".to_string()));
        }
        
        let mut a = a.rem_euclid(n);
        let mut n = n;
        let mut result = 1i8;
        
        while a != 0 {
            while a % 2 == 0 {
                a /= 2;
                if n % 8 == 3 || n % 8 == 5 {
                    result = -result;
                }
            }
            
            std::mem::swap(&mut a, &mut n);
            
            if a % 4 == 3 && n % 4 == 3 {
                result = -result;
            }
            
            a %= n;
        }
        
        if n == 1 {
            Ok(result)
        } else {
            Ok(0)
        }
    }
    
    pub fn primitive_roots(modulus: i64) -> MathResult<PrimitiveRoot> {
        if modulus <= 1 {
            return Err(MathError::InvalidArgument("Modulus must be greater than 1".to_string()));
        }
        
        let totient = Self::euler_totient(modulus)?;
        let mut roots = Vec::new();
        
        for g in 1..modulus {
            let gcd = Self::gcd(g, modulus);
            if gcd == 1 {
                let modular_g = ModularInteger::new(g, modulus)?;
                if let Ok(order) = modular_g.order() {
                    if order == totient {
                        roots.push(g);
                    }
                }
            }
        }
        
        Ok(PrimitiveRoot {
            modulus,
            roots,
            order: totient as usize,
        })
    }
    
    pub fn discrete_logarithm(base: i64, result: i64, modulus: i64) -> MathResult<DiscreteLogarithm> {
        if modulus <= 1 {
            return Err(MathError::InvalidArgument("Modulus must be greater than 1".to_string()));
        }
        
        let base_mod = base.rem_euclid(modulus);
        let result_mod = result.rem_euclid(modulus);
        
        // Baby-step giant-step algorithm (simplified)
        let m = ((modulus as f64).sqrt() as i64) + 1;
        let mut baby_steps = HashMap::new();
        
        let mut gamma = 1;
        for j in 0..m {
            baby_steps.insert(gamma, j);
            gamma = (gamma * base_mod) % modulus;
        }
        
        let base_m_inv = ModularInteger::new(base_mod, modulus)?.power(m)?.modular_inverse()?;
        let mut y = result_mod;
        
        for i in 0..m {
            if let Some(&j) = baby_steps.get(&y) {
                let logarithm = i * m + j;
                return Ok(DiscreteLogarithm {
                    base,
                    result,
                    modulus,
                    logarithm: Some(logarithm),
                });
            }
            y = (y * base_m_inv.value) % modulus;
        }
        
        Ok(DiscreteLogarithm {
            base,
            result,
            modulus,
            logarithm: None,
        })
    }
    
    pub fn quadratic_residues(modulus: i64) -> MathResult<Vec<i64>> {
        if modulus <= 1 {
            return Err(MathError::InvalidArgument("Modulus must be greater than 1".to_string()));
        }
        
        let mut residues = Vec::new();
        let mut seen = std::collections::HashSet::new();
        
        for x in 0..modulus {
            let square = (x * x) % modulus;
            if !seen.contains(&square) {
                residues.push(square);
                seen.insert(square);
            }
        }
        
        residues.sort();
        Ok(residues)
    }
    
    pub fn solve_congruence(a: i64, b: i64, m: i64) -> MathResult<Vec<i64>> {
        // Solve ax ≡ b (mod m)
        let gcd = Self::gcd(a, m);
        
        if b % gcd != 0 {
            return Ok(Vec::new()); // No solutions
        }
        
        let a_reduced = a / gcd;
        let b_reduced = b / gcd;
        let m_reduced = m / gcd;
        
        let a_mod = ModularInteger::new(a_reduced, m_reduced)?;
        let a_inv = a_mod.modular_inverse()?;
        let solution = (a_inv.value * b_reduced) % m_reduced;
        
        let mut solutions = Vec::new();
        for i in 0..gcd {
            solutions.push(solution + i * m_reduced);
        }
        
        Ok(solutions)
    }
    
    fn gcd(a: i64, b: i64) -> i64 {
        let mut a = a.abs();
        let mut b = b.abs();
        
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        
        a
    }
    
    pub fn multiplicative_order(a: i64, m: i64) -> MathResult<i64> {
        if m <= 0 {
            return Err(MathError::InvalidArgument("Modulus must be positive".to_string()));
        }
        
        if Self::gcd(a, m) != 1 {
            return Err(MathError::InvalidArgument("a and m must be coprime".to_string()));
        }
        
        let modular_a = ModularInteger::new(a, m)?;
        modular_a.order()
    }
    
    pub fn wilson_theorem_check(p: i64) -> MathResult<bool> {
        if p <= 1 {
            return Err(MathError::InvalidArgument("p must be greater than 1".to_string()));
        }
        
        if p == 2 {
            return Ok(true);
        }
        
        // Wilson's theorem: (p-1)! ≡ -1 (mod p) iff p is prime
        let mut factorial = 1;
        for i in 1..(p) {
            factorial = (factorial * i) % p;
        }
        
        Ok(factorial == p - 1)
    }
    
    pub fn fermat_little_theorem_check(a: i64, p: i64) -> MathResult<bool> {
        if p <= 1 {
            return Err(MathError::InvalidArgument("p must be greater than 1".to_string()));
        }
        
        if a % p == 0 {
            return Ok(true); // a ≡ 0 (mod p)
        }
        
        let modular_a = ModularInteger::new(a, p)?;
        let result = modular_a.power(p - 1)?;
        
        Ok(result.value == 1)
    }
}

impl ModularPolynomial {
    pub fn new(coefficients: Vec<i64>, modulus: i64) -> MathResult<Self> {
        if modulus <= 0 {
            return Err(MathError::InvalidArgument("Modulus must be positive".to_string()));
        }
        
        let normalized_coeffs: Vec<i64> = coefficients.iter()
            .map(|&c| c.rem_euclid(modulus))
            .collect();
        
        Ok(ModularPolynomial {
            coefficients: normalized_coeffs,
            modulus,
        })
    }
    
    pub fn evaluate(&self, x: i64) -> i64 {
        let mut result = 0;
        let mut x_power = 1;
        
        for &coeff in &self.coefficients {
            result = (result + coeff * x_power) % self.modulus;
            x_power = (x_power * x) % self.modulus;
        }
        
        result
    }
    
    pub fn add(&self, other: &ModularPolynomial) -> MathResult<ModularPolynomial> {
        if self.modulus != other.modulus {
            return Err(MathError::InvalidArgument("Moduli must be equal".to_string()));
        }
        
        let max_len = self.coefficients.len().max(other.coefficients.len());
        let mut result_coeffs = vec![0; max_len];
        
        for i in 0..max_len {
            let self_coeff = self.coefficients.get(i).copied().unwrap_or(0);
            let other_coeff = other.coefficients.get(i).copied().unwrap_or(0);
            result_coeffs[i] = (self_coeff + other_coeff) % self.modulus;
        }
        
        Ok(ModularPolynomial {
            coefficients: result_coeffs,
            modulus: self.modulus,
        })
    }
    
    pub fn multiply(&self, other: &ModularPolynomial) -> MathResult<ModularPolynomial> {
        if self.modulus != other.modulus {
            return Err(MathError::InvalidArgument("Moduli must be equal".to_string()));
        }
        
        if self.coefficients.is_empty() || other.coefficients.is_empty() {
            return Ok(ModularPolynomial {
                coefficients: vec![0],
                modulus: self.modulus,
            });
        }
        
        let result_len = self.coefficients.len() + other.coefficients.len() - 1;
        let mut result_coeffs = vec![0; result_len];
        
        for i in 0..self.coefficients.len() {
            for j in 0..other.coefficients.len() {
                result_coeffs[i + j] = (result_coeffs[i + j] + 
                                      self.coefficients[i] * other.coefficients[j]) % self.modulus;
            }
        }
        
        Ok(ModularPolynomial {
            coefficients: result_coeffs,
            modulus: self.modulus,
        })
    }
}

impl MathDomain for ModularArithmeticDomain {
    fn name(&self) -> &str { "Enhanced Modular Arithmetic" }
    fn description(&self) -> &str { "Advanced modular arithmetic, number theory, quadratic residues, and primitive roots" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "modular_add" => {
                if args.len() != 4 {
                    return Err(MathError::InvalidArgument("modular_add requires 4 arguments".to_string()));
                }
                let a = args[0].downcast_ref::<i64>().ok_or_else(|| MathError::InvalidArgument("First argument must be i64".to_string()))?;
                let b = args[1].downcast_ref::<i64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be i64".to_string()))?;
                let m = args[2].downcast_ref::<i64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be i64".to_string()))?;
                
                let mod_a = ModularInteger::new(*a, *m)?;
                let mod_b = ModularInteger::new(*b, *m)?;
                let result = mod_a.add(&mod_b)?;
                Ok(Box::new(result.value))
            },
            "euler_totient" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("euler_totient requires 1 argument".to_string()));
                }
                let n = args[0].downcast_ref::<i64>().ok_or_else(|| MathError::InvalidArgument("Argument must be i64".to_string()))?;
                let result = Self::euler_totient(*n)?;
                Ok(Box::new(result))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "modular_add".to_string(),
            "modular_multiply".to_string(),
            "modular_power".to_string(),
            "modular_inverse".to_string(),
            "chinese_remainder_theorem".to_string(),
            "euler_totient".to_string(),
            "carmichael_function".to_string(),
            "quadratic_residue".to_string(),
            "legendre_symbol".to_string(),
            "jacobi_symbol".to_string(),
            "primitive_roots".to_string(),
            "discrete_logarithm".to_string(),
            "multiplicative_order".to_string(),
            "solve_congruence".to_string(),
            "wilson_theorem".to_string(),
            "fermat_little_theorem".to_string(),
            "quadratic_residues_list".to_string(),
            "polynomial_arithmetic".to_string(),
        ]
    }
}