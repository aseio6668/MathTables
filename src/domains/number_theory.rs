use crate::core::{MathDomain, MathResult, MathError};
use num_bigint::BigInt;
use std::any::Any;

pub struct NumberTheoryDomain;

impl NumberTheoryDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn gcd(a: i64, b: i64) -> i64 {
        if b == 0 { a.abs() } else { Self::gcd(b, a % b) }
    }
    
    pub fn lcm(a: i64, b: i64) -> i64 {
        if a == 0 && b == 0 { 0 } else { (a * b).abs() / Self::gcd(a, b) }
    }
    
    pub fn is_prime(n: u64) -> bool {
        if n < 2 { return false; }
        if n == 2 { return true; }
        if n % 2 == 0 { return false; }
        
        let limit = (n as f64).sqrt() as u64 + 1;
        for i in (3..=limit).step_by(2) {
            if n % i == 0 { return false; }
        }
        true
    }
    
    pub fn prime_factors(n: u64) -> Vec<u64> {
        let mut factors = Vec::new();
        let mut num = n;
        let mut divisor = 2;
        
        while divisor * divisor <= num {
            while num % divisor == 0 {
                factors.push(divisor);
                num /= divisor;
            }
            divisor += 1;
        }
        
        if num > 1 {
            factors.push(num);
        }
        
        factors
    }
    
    pub fn fibonacci(n: u64) -> BigInt {
        if n <= 1 { return BigInt::from(n); }
        
        let mut a = BigInt::from(0);
        let mut b = BigInt::from(1);
        
        for _ in 2..=n {
            let temp = &a + &b;
            a = std::mem::replace(&mut b, temp);
        }
        
        b
    }
    
    pub fn euler_totient(n: u64) -> u64 {
        if n == 0 { return 0; }
        
        let mut result = n;
        let factors = Self::prime_factors(n);
        let unique_factors: std::collections::HashSet<_> = factors.into_iter().collect();
        
        for p in unique_factors {
            result = result / p * (p - 1);
        }
        
        result
    }
}

impl MathDomain for NumberTheoryDomain {
    fn name(&self) -> &str { "Number Theory" }
    fn description(&self) -> &str { "Mathematical domain focused on properties of integers and prime numbers" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "gcd" => {
                if args.len() != 2 { 
                    return Err(MathError::InvalidArgument("gcd requires 2 arguments".to_string())); 
                }
                let a = args[0].downcast_ref::<i64>().ok_or_else(|| MathError::InvalidArgument("First argument must be i64".to_string()))?;
                let b = args[1].downcast_ref::<i64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be i64".to_string()))?;
                Ok(Box::new(Self::gcd(*a, *b)))
            },
            "lcm" => {
                if args.len() != 2 { 
                    return Err(MathError::InvalidArgument("lcm requires 2 arguments".to_string())); 
                }
                let a = args[0].downcast_ref::<i64>().ok_or_else(|| MathError::InvalidArgument("First argument must be i64".to_string()))?;
                let b = args[1].downcast_ref::<i64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be i64".to_string()))?;
                Ok(Box::new(Self::lcm(*a, *b)))
            },
            "is_prime" => {
                if args.len() != 1 { 
                    return Err(MathError::InvalidArgument("is_prime requires 1 argument".to_string())); 
                }
                let n = args[0].downcast_ref::<u64>().ok_or_else(|| MathError::InvalidArgument("Argument must be u64".to_string()))?;
                Ok(Box::new(Self::is_prime(*n)))
            },
            "prime_factors" => {
                if args.len() != 1 { 
                    return Err(MathError::InvalidArgument("prime_factors requires 1 argument".to_string())); 
                }
                let n = args[0].downcast_ref::<u64>().ok_or_else(|| MathError::InvalidArgument("Argument must be u64".to_string()))?;
                Ok(Box::new(Self::prime_factors(*n)))
            },
            "fibonacci" => {
                if args.len() != 1 { 
                    return Err(MathError::InvalidArgument("fibonacci requires 1 argument".to_string())); 
                }
                let n = args[0].downcast_ref::<u64>().ok_or_else(|| MathError::InvalidArgument("Argument must be u64".to_string()))?;
                Ok(Box::new(Self::fibonacci(*n)))
            },
            "euler_totient" => {
                if args.len() != 1 { 
                    return Err(MathError::InvalidArgument("euler_totient requires 1 argument".to_string())); 
                }
                let n = args[0].downcast_ref::<u64>().ok_or_else(|| MathError::InvalidArgument("Argument must be u64".to_string()))?;
                Ok(Box::new(Self::euler_totient(*n)))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "gcd".to_string(),
            "lcm".to_string(),
            "is_prime".to_string(),
            "prime_factors".to_string(),
            "fibonacci".to_string(),
            "euler_totient".to_string(),
        ]
    }
}