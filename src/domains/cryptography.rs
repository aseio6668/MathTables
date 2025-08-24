use crate::core::{MathDomain, MathResult, MathError};
use num_bigint::BigUint;
use std::any::Any;

#[derive(Debug, Clone)]
pub struct RSAKeyPair {
    pub public_key: RSAPublicKey,
    pub private_key: RSAPrivateKey,
}

#[derive(Debug, Clone)]
pub struct RSAPublicKey {
    pub n: BigUint,
    pub e: BigUint,
}

#[derive(Debug, Clone)]
pub struct RSAPrivateKey {
    pub n: BigUint,
    pub d: BigUint,
}

#[derive(Debug, Clone)]
pub struct CipherResult {
    pub ciphertext: Vec<u8>,
    pub algorithm: String,
}

#[derive(Debug, Clone)]
pub struct HashResult {
    pub digest: Vec<u8>,
    pub algorithm: String,
}

pub struct CryptographyDomain;

impl CryptographyDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn generate_rsa_keypair(bit_length: usize) -> MathResult<RSAKeyPair> {
        if bit_length < 512 {
            return Err(MathError::InvalidArgument("RSA key length must be at least 512 bits".to_string()));
        }
        
        let p = Self::generate_prime(bit_length / 2)?;
        let q = Self::generate_prime(bit_length / 2)?;
        
        let n = &p * &q;
        let phi_n = (&p - 1u32) * (&q - 1u32);
        
        let e = BigUint::from(65537u32); // Common choice for e
        
        if Self::gcd(&e, &phi_n) != BigUint::from(1u32) {
            return Err(MathError::ComputationError("e and phi(n) are not coprime".to_string()));
        }
        
        let d = Self::mod_inverse(&e, &phi_n)?;
        
        Ok(RSAKeyPair {
            public_key: RSAPublicKey {
                n: n.clone(),
                e,
            },
            private_key: RSAPrivateKey {
                n,
                d,
            },
        })
    }
    
    pub fn rsa_encrypt(message: &[u8], public_key: &RSAPublicKey) -> MathResult<Vec<u8>> {
        let message_int = BigUint::from_bytes_be(message);
        
        if message_int >= public_key.n {
            return Err(MathError::InvalidArgument("Message too large for RSA modulus".to_string()));
        }
        
        let ciphertext_int = Self::mod_pow(&message_int, &public_key.e, &public_key.n);
        Ok(ciphertext_int.to_bytes_be())
    }
    
    pub fn rsa_decrypt(ciphertext: &[u8], private_key: &RSAPrivateKey) -> MathResult<Vec<u8>> {
        let ciphertext_int = BigUint::from_bytes_be(ciphertext);
        
        if ciphertext_int >= private_key.n {
            return Err(MathError::InvalidArgument("Ciphertext too large for RSA modulus".to_string()));
        }
        
        let message_int = Self::mod_pow(&ciphertext_int, &private_key.d, &private_key.n);
        Ok(message_int.to_bytes_be())
    }
    
    pub fn miller_rabin_primality_test(n: &BigUint, iterations: usize) -> bool {
        if *n == BigUint::from(2u32) || *n == BigUint::from(3u32) {
            return true;
        }
        
        if *n < BigUint::from(2u32) || n % 2u32 == BigUint::from(0u32) {
            return false;
        }
        
        let mut d = n - 1u32;
        let mut r = 0;
        
        while &d % 2u32 == BigUint::from(0u32) {
            d /= 2u32;
            r += 1;
        }
        
        for _ in 0..iterations {
            let a = Self::random_bigint_range(&BigUint::from(2u32), &(n - 2u32));
            let mut x = Self::mod_pow(&a, &d, n);
            
            if x == BigUint::from(1u32) || x == n - 1u32 {
                continue;
            }
            
            let mut is_composite = true;
            for _ in 0..(r - 1) {
                x = Self::mod_pow(&x, &BigUint::from(2u32), n);
                if x == n - 1u32 {
                    is_composite = false;
                    break;
                }
            }
            
            if is_composite {
                return false;
            }
        }
        
        true
    }
    
    pub fn generate_prime(bit_length: usize) -> MathResult<BigUint> {
        if bit_length < 2 {
            return Err(MathError::InvalidArgument("Prime bit length must be at least 2".to_string()));
        }
        
        let max_attempts = 1000;
        
        for _ in 0..max_attempts {
            let mut candidate = Self::random_bigint(bit_length);
            
            candidate |= BigUint::from(1u32); // Make odd
            candidate |= BigUint::from(1u32) << (bit_length - 1); // Set MSB
            
            if Self::miller_rabin_primality_test(&candidate, 10) {
                return Ok(candidate);
            }
        }
        
        Err(MathError::ComputationError("Failed to generate prime after maximum attempts".to_string()))
    }
    
    fn random_bigint(bit_length: usize) -> BigUint {
        let byte_length = (bit_length + 7) / 8;
        let mut bytes = vec![0u8; byte_length];
        
        for byte in &mut bytes {
            *byte = rand::random::<u8>();
        }
        
        if bit_length % 8 != 0 {
            let mask = (1u8 << (bit_length % 8)) - 1;
            bytes[0] &= mask;
        }
        
        BigUint::from_bytes_be(&bytes)
    }
    
    fn random_bigint_range(min: &BigUint, max: &BigUint) -> BigUint {
        if min >= max {
            return min.clone();
        }
        
        let range = max - min;
        let bit_length = range.bits();
        
        loop {
            let candidate = Self::random_bigint(bit_length as usize);
            if candidate < range {
                return min + candidate;
            }
        }
    }
    
    pub fn gcd(a: &BigUint, b: &BigUint) -> BigUint {
        let mut a = a.clone();
        let mut b = b.clone();
        
        while b != BigUint::from(0u32) {
            let temp = b.clone();
            b = &a % &b;
            a = temp;
        }
        
        a
    }
    
    pub fn extended_gcd(a: &BigUint, b: &BigUint) -> (BigUint, BigUint, BigUint) {
        if *b == BigUint::from(0u32) {
            return (a.clone(), BigUint::from(1u32), BigUint::from(0u32));
        }
        
        let (gcd, x1, y1) = Self::extended_gcd(b, &(a % b));
        let x = y1.clone();
        let y = if x1 >= (a / b) * &y1 {
            &x1 - (a / b) * &y1
        } else {
            BigUint::from(0u32) // Handle underflow case
        };
        
        (gcd, x, y)
    }
    
    pub fn mod_inverse(a: &BigUint, m: &BigUint) -> MathResult<BigUint> {
        let (gcd, x, _) = Self::extended_gcd(a, m);
        
        if gcd != BigUint::from(1u32) {
            return Err(MathError::ComputationError("Modular inverse does not exist".to_string()));
        }
        
        Ok(x % m)
    }
    
    pub fn mod_pow(base: &BigUint, exp: &BigUint, modulus: &BigUint) -> BigUint {
        if *modulus == BigUint::from(1u32) {
            return BigUint::from(0u32);
        }
        
        let mut result = BigUint::from(1u32);
        let mut base = base % modulus;
        let mut exp = exp.clone();
        
        while exp > BigUint::from(0u32) {
            if &exp % 2u32 == BigUint::from(1u32) {
                result = (&result * &base) % modulus;
            }
            exp >>= 1;
            base = (&base * &base) % modulus;
        }
        
        result
    }
    
    pub fn caesar_cipher(text: &str, shift: i32) -> String {
        text.chars()
            .map(|c| {
                if c.is_ascii_alphabetic() {
                    let base = if c.is_ascii_lowercase() { b'a' } else { b'A' };
                    let shifted = ((c as u8 - base) as i32 + shift).rem_euclid(26) as u8;
                    (base + shifted) as char
                } else {
                    c
                }
            })
            .collect()
    }
    
    pub fn vigenere_cipher(text: &str, key: &str, encrypt: bool) -> MathResult<String> {
        if key.is_empty() {
            return Err(MathError::InvalidArgument("Vigenère key cannot be empty".to_string()));
        }
        
        let key_bytes: Vec<u8> = key.to_uppercase()
            .chars()
            .filter(|c| c.is_ascii_alphabetic())
            .map(|c| (c as u8) - b'A')
            .collect();
        
        if key_bytes.is_empty() {
            return Err(MathError::InvalidArgument("Vigenère key must contain alphabetic characters".to_string()));
        }
        
        let mut result = String::new();
        let mut key_index = 0;
        
        for c in text.chars() {
            if c.is_ascii_alphabetic() {
                let is_lowercase = c.is_ascii_lowercase();
                let base = if is_lowercase { b'a' } else { b'A' };
                let char_val = (c.to_ascii_uppercase() as u8) - b'A';
                let key_val = key_bytes[key_index % key_bytes.len()];
                
                let shifted_val = if encrypt {
                    (char_val + key_val) % 26
                } else {
                    (char_val + 26 - key_val) % 26
                };
                
                let shifted_char = (base + shifted_val) as char;
                result.push(if is_lowercase {
                    shifted_char.to_ascii_lowercase()
                } else {
                    shifted_char
                });
                
                key_index += 1;
            } else {
                result.push(c);
            }
        }
        
        Ok(result)
    }
    
    pub fn simple_hash(data: &[u8]) -> HashResult {
        let mut hash: u64 = 5381;
        
        for &byte in data {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        
        HashResult {
            digest: hash.to_be_bytes().to_vec(),
            algorithm: "SimpleHash".to_string(),
        }
    }
    
    pub fn xor_cipher(data: &[u8], key: &[u8]) -> MathResult<Vec<u8>> {
        if key.is_empty() {
            return Err(MathError::InvalidArgument("XOR key cannot be empty".to_string()));
        }
        
        let result: Vec<u8> = data.iter()
            .enumerate()
            .map(|(i, &byte)| byte ^ key[i % key.len()])
            .collect();
        
        Ok(result)
    }
    
    pub fn one_time_pad(message: &[u8], key: &[u8]) -> MathResult<Vec<u8>> {
        if message.len() != key.len() {
            return Err(MathError::InvalidArgument("One-time pad requires key length equal to message length".to_string()));
        }
        
        let result: Vec<u8> = message.iter()
            .zip(key.iter())
            .map(|(&m, &k)| m ^ k)
            .collect();
        
        Ok(result)
    }
}

impl MathDomain for CryptographyDomain {
    fn name(&self) -> &str { "Cryptography" }
    fn description(&self) -> &str { "Cryptographic algorithms including RSA, classical ciphers, and number theory for cryptography" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "caesar_cipher" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("caesar_cipher requires 2 arguments".to_string()));
                }
                let text = args[0].downcast_ref::<String>().ok_or_else(|| MathError::InvalidArgument("First argument must be String".to_string()))?;
                let shift = args[1].downcast_ref::<i32>().ok_or_else(|| MathError::InvalidArgument("Second argument must be i32".to_string()))?;
                Ok(Box::new(Self::caesar_cipher(text, *shift)))
            },
            "simple_hash" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("simple_hash requires 1 argument".to_string()));
                }
                let data = args[0].downcast_ref::<Vec<u8>>().ok_or_else(|| MathError::InvalidArgument("Argument must be Vec<u8>".to_string()))?;
                Ok(Box::new(Self::simple_hash(data)))
            },
            "xor_cipher" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("xor_cipher requires 2 arguments".to_string()));
                }
                let data = args[0].downcast_ref::<Vec<u8>>().ok_or_else(|| MathError::InvalidArgument("First argument must be Vec<u8>".to_string()))?;
                let key = args[1].downcast_ref::<Vec<u8>>().ok_or_else(|| MathError::InvalidArgument("Second argument must be Vec<u8>".to_string()))?;
                Ok(Box::new(Self::xor_cipher(data, key)?))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "generate_rsa_keypair".to_string(),
            "rsa_encrypt".to_string(),
            "rsa_decrypt".to_string(),
            "miller_rabin_primality_test".to_string(),
            "generate_prime".to_string(),
            "gcd".to_string(),
            "extended_gcd".to_string(),
            "mod_inverse".to_string(),
            "mod_pow".to_string(),
            "caesar_cipher".to_string(),
            "vigenere_cipher".to_string(),
            "simple_hash".to_string(),
            "xor_cipher".to_string(),
            "one_time_pad".to_string(),
        ]
    }
}