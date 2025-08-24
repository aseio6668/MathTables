use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;

#[derive(Debug, Clone, PartialEq)]
pub struct PAdicNumber {
    pub digits: Vec<i64>,  // p-adic digits (least significant first)
    pub valuation: i64,    // p-adic valuation (power of p)
    pub prime: i64,        // The prime p
    pub precision: usize,  // Number of significant p-adic digits
}

#[derive(Debug, Clone)]
pub struct PAdicInteger {
    pub padic: PAdicNumber,
}

#[derive(Debug, Clone)]
pub struct PAdicRational {
    pub numerator: PAdicInteger,
    pub denominator: PAdicInteger,
}

pub struct PAdicNumbersDomain;

impl PAdicNumber {
    pub fn new(prime: i64, precision: usize) -> MathResult<Self> {
        if !Self::is_prime(prime) {
            return Err(MathError::InvalidArgument("Base must be prime".to_string()));
        }
        
        Ok(PAdicNumber {
            digits: vec![0; precision],
            valuation: 0,
            prime,
            precision,
        })
    }
    
    pub fn from_integer(value: i64, prime: i64, precision: usize) -> MathResult<Self> {
        if !Self::is_prime(prime) {
            return Err(MathError::InvalidArgument("Base must be prime".to_string()));
        }
        
        let mut padic = PAdicNumber::new(prime, precision)?;
        
        if value == 0 {
            return Ok(padic);
        }
        
        let mut n = value.abs();
        let mut valuation = 0;
        
        // Calculate p-adic valuation
        while n % prime == 0 {
            n /= prime;
            valuation += 1;
        }
        
        padic.valuation = if value < 0 { -valuation } else { valuation };
        
        // Extract p-adic digits
        for i in 0..precision {
            if n == 0 {
                break;
            }
            padic.digits[i] = n % prime;
            n /= prime;
        }
        
        // Handle negative numbers using p-adic complement
        if value < 0 {
            padic = padic.p_adic_negate()?;
        }
        
        Ok(padic)
    }
    
    pub fn from_rational(numerator: i64, denominator: i64, prime: i64, precision: usize) -> MathResult<Self> {
        if denominator == 0 {
            return Err(MathError::DivisionByZero);
        }
        
        if !Self::is_prime(prime) {
            return Err(MathError::InvalidArgument("Base must be prime".to_string()));
        }
        
        // Check if denominator is coprime to p
        if Self::gcd(denominator.abs(), prime) != 1 {
            return Err(MathError::InvalidArgument("Denominator must be coprime to p for p-adic representation".to_string()));
        }
        
        let num_padic = Self::from_integer(numerator, prime, precision)?;
        let den_padic = Self::from_integer(denominator, prime, precision)?;
        
        num_padic.p_adic_divide(&den_padic)
    }
    
    fn is_prime(n: i64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }
        
        let sqrt_n = (n as f64).sqrt() as i64;
        for i in (3..=sqrt_n).step_by(2) {
            if n % i == 0 {
                return false;
            }
        }
        true
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
    
    pub fn p_adic_valuation(&self) -> i64 {
        self.valuation
    }
    
    pub fn p_adic_norm(&self) -> f64 {
        if self.is_zero() {
            0.0
        } else {
            (self.prime as f64).powf(-self.valuation as f64)
        }
    }
    
    pub fn is_zero(&self) -> bool {
        self.digits.iter().all(|&d| d == 0)
    }
    
    pub fn p_adic_distance(&self, other: &PAdicNumber) -> MathResult<f64> {
        if self.prime != other.prime {
            return Err(MathError::InvalidArgument("p-adic numbers must have same prime".to_string()));
        }
        
        let difference = self.p_adic_subtract(other)?;
        Ok(difference.p_adic_norm())
    }
    
    pub fn p_adic_add(&self, other: &PAdicNumber) -> MathResult<PAdicNumber> {
        if self.prime != other.prime {
            return Err(MathError::InvalidArgument("p-adic numbers must have same prime".to_string()));
        }
        
        let mut result = PAdicNumber::new(self.prime, self.precision.max(other.precision))?;
        let min_val = self.valuation.min(other.valuation);
        result.valuation = min_val;
        
        let mut carry = 0;
        let max_len = self.digits.len().max(other.digits.len());
        
        for i in 0..max_len {
            let self_digit = if i < self.digits.len() { self.digits[i] } else { 0 };
            let other_digit = if i < other.digits.len() { other.digits[i] } else { 0 };
            
            let sum = self_digit + other_digit + carry;
            if i < result.digits.len() {
                result.digits[i] = sum % self.prime;
            }
            carry = sum / self.prime;
        }
        
        Ok(result)
    }
    
    pub fn p_adic_subtract(&self, other: &PAdicNumber) -> MathResult<PAdicNumber> {
        let neg_other = other.p_adic_negate()?;
        self.p_adic_add(&neg_other)
    }
    
    pub fn p_adic_multiply(&self, other: &PAdicNumber) -> MathResult<PAdicNumber> {
        if self.prime != other.prime {
            return Err(MathError::InvalidArgument("p-adic numbers must have same prime".to_string()));
        }
        
        let mut result = PAdicNumber::new(self.prime, self.precision.max(other.precision))?;
        result.valuation = self.valuation + other.valuation;
        
        // Multiply digit by digit with carries
        for i in 0..self.digits.len() {
            for j in 0..other.digits.len() {
                if i + j < result.digits.len() {
                    let product = self.digits[i] * other.digits[j];
                    let mut pos = i + j;
                    let mut carry = product;
                    
                    while carry > 0 && pos < result.digits.len() {
                        let sum = result.digits[pos] + carry;
                        result.digits[pos] = sum % self.prime;
                        carry = sum / self.prime;
                        pos += 1;
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    pub fn p_adic_divide(&self, other: &PAdicNumber) -> MathResult<PAdicNumber> {
        if other.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        
        if self.prime != other.prime {
            return Err(MathError::InvalidArgument("p-adic numbers must have same prime".to_string()));
        }
        
        // Division in p-adics: a/b = a * b^(-1)
        let inverse = other.p_adic_inverse()?;
        self.p_adic_multiply(&inverse)
    }
    
    pub fn p_adic_inverse(&self) -> MathResult<PAdicNumber> {
        if self.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        
        // Use Hensel lifting to compute inverse
        let mut result = PAdicNumber::new(self.prime, self.precision)?;
        result.valuation = -self.valuation;
        
        // Find initial approximation (inverse of first non-zero digit mod p)
        let first_digit = self.digits.iter().find(|&&d| d != 0).copied().unwrap_or(1);
        let mut inv_digit = 1;
        
        // Find modular inverse of first_digit mod prime
        for i in 1..self.prime {
            if (first_digit * i) % self.prime == 1 {
                inv_digit = i;
                break;
            }
        }
        
        result.digits[0] = inv_digit;
        
        // Hensel lifting for higher precision
        for precision in 1..self.precision {
            let product = self.p_adic_multiply(&result)?;
            let one = PAdicNumber::from_integer(1, self.prime, self.precision)?;
            let diff = one.p_adic_subtract(&product)?;
            
            if precision < diff.digits.len() && diff.digits[precision] != 0 {
                let correction = (diff.digits[precision] * inv_digit) % self.prime;
                if precision < result.digits.len() {
                    result.digits[precision] = correction;
                }
            }
        }
        
        Ok(result)
    }
    
    pub fn p_adic_negate(&self) -> MathResult<PAdicNumber> {
        if self.is_zero() {
            return Ok(self.clone());
        }
        
        let mut result = PAdicNumber::new(self.prime, self.precision)?;
        result.valuation = self.valuation;
        
        // Two's complement in base p
        let mut borrow = 0;
        for i in 0..self.digits.len() {
            let digit = self.prime - self.digits[i] - borrow;
            result.digits[i] = digit % self.prime;
            borrow = if digit >= self.prime { 0 } else { 1 };
        }
        
        Ok(result)
    }
    
    pub fn p_adic_power(&self, exponent: i64) -> MathResult<PAdicNumber> {
        if exponent == 0 {
            return PAdicNumber::from_integer(1, self.prime, self.precision);
        }
        
        if exponent < 0 {
            let inverse = self.p_adic_inverse()?;
            return inverse.p_adic_power(-exponent);
        }
        
        let mut result = PAdicNumber::from_integer(1, self.prime, self.precision)?;
        let mut base = self.clone();
        let mut exp = exponent;
        
        while exp > 0 {
            if exp % 2 == 1 {
                result = result.p_adic_multiply(&base)?;
            }
            base = base.p_adic_multiply(&base)?;
            exp /= 2;
        }
        
        Ok(result)
    }
    
    pub fn p_adic_sqrt(&self) -> MathResult<PAdicNumber> {
        if self.is_zero() {
            return Ok(self.clone());
        }
        
        // Check if square root exists (Hensel's lemma conditions)
        if self.valuation % 2 != 0 {
            return Err(MathError::DomainError("Square root does not exist (odd valuation)".to_string()));
        }
        
        // Use Hensel lifting for square root
        let mut result = PAdicNumber::new(self.prime, self.precision)?;
        result.valuation = self.valuation / 2;
        
        // Find initial square root mod p
        let first_digit = self.digits[0];
        let mut sqrt_digit = 0;
        
        for i in 0..self.prime {
            if (i * i) % self.prime == first_digit % self.prime {
                sqrt_digit = i;
                break;
            }
        }
        
        if sqrt_digit == 0 && first_digit != 0 {
            return Err(MathError::DomainError("Square root does not exist".to_string()));
        }
        
        result.digits[0] = sqrt_digit;
        
        // Hensel lifting for higher precision
        for precision in 1..self.precision {
            let square = result.p_adic_multiply(&result)?;
            let diff = self.p_adic_subtract(&square)?;
            
            if precision < diff.digits.len() && diff.digits[precision] != 0 {
                let two_inv = if self.prime == 2 { 1 } else { (self.prime + 1) / 2 };
                let correction = (diff.digits[precision] * two_inv) % self.prime;
                if precision < result.digits.len() {
                    result.digits[precision] = correction;
                }
            }
        }
        
        Ok(result)
    }
    
    pub fn p_adic_exp(&self) -> MathResult<PAdicNumber> {
        // p-adic exponential using power series
        // exp(x) = 1 + x + x²/2! + x³/3! + ...
        
        if self.p_adic_norm() >= 1.0 {
            return Err(MathError::DomainError("p-adic exponential requires |x|_p < 1".to_string()));
        }
        
        let mut result = PAdicNumber::from_integer(1, self.prime, self.precision)?;
        let mut term = self.clone();
        let mut factorial = 1;
        
        for n in 1..self.precision {
            factorial *= n as i64;
            let factorial_padic = PAdicNumber::from_integer(factorial, self.prime, self.precision)?;
            let term_divided = term.p_adic_divide(&factorial_padic)?;
            result = result.p_adic_add(&term_divided)?;
            
            if n < self.precision - 1 {
                term = term.p_adic_multiply(self)?;
            }
        }
        
        Ok(result)
    }
    
    pub fn p_adic_log(&self) -> MathResult<PAdicNumber> {
        // p-adic logarithm using power series
        // log(1+x) = x - x²/2 + x³/3 - x⁴/4 + ...
        
        let one = PAdicNumber::from_integer(1, self.prime, self.precision)?;
        let x = self.p_adic_subtract(&one)?;
        
        if x.p_adic_norm() >= 1.0 {
            return Err(MathError::DomainError("p-adic logarithm requires |x-1|_p < 1".to_string()));
        }
        
        let mut result = PAdicNumber::new(self.prime, self.precision)?;
        let mut term = x.clone();
        
        for n in 1..self.precision {
            let n_padic = PAdicNumber::from_integer(n as i64, self.prime, self.precision)?;
            let term_divided = term.p_adic_divide(&n_padic)?;
            
            if n % 2 == 1 {
                result = result.p_adic_add(&term_divided)?;
            } else {
                result = result.p_adic_subtract(&term_divided)?;
            }
            
            if n < self.precision - 1 {
                term = term.p_adic_multiply(&x)?;
            }
        }
        
        Ok(result)
    }
}

impl PAdicNumbersDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn create_padic_integer(value: i64, prime: i64, precision: usize) -> MathResult<PAdicNumber> {
        PAdicNumber::from_integer(value, prime, precision)
    }
    
    pub fn create_padic_rational(num: i64, den: i64, prime: i64, precision: usize) -> MathResult<PAdicNumber> {
        PAdicNumber::from_rational(num, den, prime, precision)
    }
    
    pub fn padic_arithmetic(
        a: &PAdicNumber,
        b: &PAdicNumber,
        operation: &str,
    ) -> MathResult<PAdicNumber> {
        match operation {
            "add" => a.p_adic_add(b),
            "subtract" => a.p_adic_subtract(b),
            "multiply" => a.p_adic_multiply(b),
            "divide" => a.p_adic_divide(b),
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation))),
        }
    }
    
    pub fn padic_convergence_test(sequence: &[PAdicNumber], tolerance: f64) -> MathResult<bool> {
        if sequence.len() < 2 {
            return Ok(true);
        }
        
        let prime = sequence[0].prime;
        
        for i in 1..sequence.len() {
            if sequence[i].prime != prime {
                return Err(MathError::InvalidArgument("All p-adic numbers must have same prime".to_string()));
            }
            
            let distance = sequence[i].p_adic_distance(&sequence[i - 1])?;
            if distance > tolerance {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    pub fn padic_series_sum(terms: &[PAdicNumber]) -> MathResult<PAdicNumber> {
        if terms.is_empty() {
            return Err(MathError::InvalidArgument("Cannot sum empty series".to_string()));
        }
        
        let prime = terms[0].prime;
        let precision = terms[0].precision;
        let mut sum = PAdicNumber::new(prime, precision)?;
        
        for term in terms {
            if term.prime != prime {
                return Err(MathError::InvalidArgument("All terms must have same prime".to_string()));
            }
            sum = sum.p_adic_add(term)?;
        }
        
        Ok(sum)
    }
    
    pub fn hensel_lifting(
        polynomial_coeffs: &[i64],
        root_mod_p: i64,
        prime: i64,
        precision: usize,
    ) -> MathResult<PAdicNumber> {
        if polynomial_coeffs.is_empty() {
            return Err(MathError::InvalidArgument("Polynomial cannot be empty".to_string()));
        }
        
        let mut root = PAdicNumber::from_integer(root_mod_p, prime, precision)?;
        
        // Hensel lifting: x_{n+1} = x_n - f(x_n)/f'(x_n) mod p^{n+1}
        for _ in 1..precision {
            let f_value = Self::evaluate_polynomial(polynomial_coeffs, &root)?;
            let df_value = Self::evaluate_polynomial_derivative(polynomial_coeffs, &root)?;
            
            if df_value.is_zero() {
                return Err(MathError::ComputationError("Derivative is zero - cannot continue lifting".to_string()));
            }
            
            let correction = f_value.p_adic_divide(&df_value)?;
            root = root.p_adic_subtract(&correction)?;
        }
        
        Ok(root)
    }
    
    fn evaluate_polynomial(coeffs: &[i64], x: &PAdicNumber) -> MathResult<PAdicNumber> {
        let mut result = PAdicNumber::new(x.prime, x.precision)?;
        let mut power = PAdicNumber::from_integer(1, x.prime, x.precision)?;
        
        for &coeff in coeffs {
            let coeff_padic = PAdicNumber::from_integer(coeff, x.prime, x.precision)?;
            let term = coeff_padic.p_adic_multiply(&power)?;
            result = result.p_adic_add(&term)?;
            power = power.p_adic_multiply(x)?;
        }
        
        Ok(result)
    }
    
    fn evaluate_polynomial_derivative(coeffs: &[i64], x: &PAdicNumber) -> MathResult<PAdicNumber> {
        if coeffs.len() <= 1 {
            return PAdicNumber::new(x.prime, x.precision);
        }
        
        let mut result = PAdicNumber::new(x.prime, x.precision)?;
        let mut power = PAdicNumber::from_integer(1, x.prime, x.precision)?;
        
        for (i, &coeff) in coeffs.iter().enumerate().skip(1) {
            let coeff_padic = PAdicNumber::from_integer(coeff * i as i64, x.prime, x.precision)?;
            let term = coeff_padic.p_adic_multiply(&power)?;
            result = result.p_adic_add(&term)?;
            if i < coeffs.len() - 1 {
                power = power.p_adic_multiply(x)?;
            }
        }
        
        Ok(result)
    }
    
    pub fn padic_completion_of_rationals(
        rational_sequence: &[(i64, i64)],
        prime: i64,
        precision: usize,
    ) -> MathResult<Vec<PAdicNumber>> {
        let mut result = Vec::new();
        
        for &(num, den) in rational_sequence {
            let padic = PAdicNumber::from_rational(num, den, prime, precision)?;
            result.push(padic);
        }
        
        Ok(result)
    }
}

impl MathDomain for PAdicNumbersDomain {
    fn name(&self) -> &str { "p-adic Numbers" }
    fn description(&self) -> &str { "p-adic number arithmetic, analysis, and Hensel lifting" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "create_padic" => {
                if args.len() != 3 {
                    return Err(MathError::InvalidArgument("create_padic requires 3 arguments".to_string()));
                }
                let value = args[0].downcast_ref::<i64>().ok_or_else(|| MathError::InvalidArgument("First argument must be i64".to_string()))?;
                let prime = args[1].downcast_ref::<i64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be i64".to_string()))?;
                let precision = args[2].downcast_ref::<usize>().ok_or_else(|| MathError::InvalidArgument("Third argument must be usize".to_string()))?;
                Ok(Box::new(Self::create_padic_integer(*value, *prime, *precision)?))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "create_padic_integer".to_string(),
            "create_padic_rational".to_string(),
            "padic_add".to_string(),
            "padic_multiply".to_string(),
            "padic_divide".to_string(),
            "padic_power".to_string(),
            "padic_sqrt".to_string(),
            "padic_exp".to_string(),
            "padic_log".to_string(),
            "padic_norm".to_string(),
            "padic_distance".to_string(),
            "hensel_lifting".to_string(),
            "padic_series_sum".to_string(),
            "padic_convergence_test".to_string(),
        ]
    }
}