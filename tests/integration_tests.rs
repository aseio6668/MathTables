use mathtables::prelude::*;

#[test]
fn test_framework_initialization() {
    let math = MathTables::new();
    assert!(math.is_initialized());
    assert!(math.list_domains().len() >= 5);
}

#[test]
fn test_number_theory_domain() {
    let math = MathTables::new();
    let domain = math.get_domain("number_theory").unwrap();
    
    // Test GCD
    let a = 48i64;
    let b = 18i64;
    let result = domain.compute("gcd", &[&a, &b]).unwrap();
    let gcd_value = result.downcast_ref::<i64>().unwrap();
    assert_eq!(*gcd_value, 6);
    
    // Test prime checking
    let prime = 17u64;
    let result = domain.compute("is_prime", &[&prime]).unwrap();
    let is_prime = result.downcast_ref::<bool>().unwrap();
    assert!(*is_prime);
    
    let composite = 15u64;
    let result = domain.compute("is_prime", &[&composite]).unwrap();
    let is_prime = result.downcast_ref::<bool>().unwrap();
    assert!(!*is_prime);
}

#[test]
fn test_geometry_domain() {
    let math = MathTables::new();
    let domain = math.get_domain("geometry").unwrap();
    
    // Test circle area
    let radius = 5.0f64;
    let result = domain.compute("circle_area", &[&radius]).unwrap();
    let area = result.downcast_ref::<f64>().unwrap();
    assert!((area - std::f64::consts::PI * 25.0).abs() < 1e-10);
    
    // Test distance calculation
    let p1 = Point2D { x: 0.0, y: 0.0 };
    let p2 = Point2D { x: 3.0, y: 4.0 };
    let result = domain.compute("distance_2d", &[&p1, &p2]).unwrap();
    let distance = result.downcast_ref::<f64>().unwrap();
    assert!((distance - 5.0).abs() < 1e-10);
}

#[test]
fn test_algebra_domain() {
    let math = MathTables::new();
    let domain = math.get_domain("algebra").unwrap();
    
    // Test matrix determinant
    let matrix = Matrix {
        data: vec![vec![2.0, 1.0], vec![3.0, 4.0]],
        rows: 2,
        cols: 2,
    };
    let result = domain.compute("matrix_determinant", &[&matrix]).unwrap();
    let det = result.downcast_ref::<f64>().unwrap();
    assert!((det - 5.0).abs() < 1e-10);
    
    // Test polynomial evaluation
    let poly = Polynomial {
        coefficients: vec![1.0, -2.0, 1.0], // x² - 2x + 1 = (x-1)²
    };
    let x = 3.0f64;
    let result = domain.compute("polynomial_evaluate", &[&poly, &x]).unwrap();
    let value = result.downcast_ref::<f64>().unwrap();
    assert!((value - 4.0).abs() < 1e-10); // (3-1)² = 4
}

#[test]
fn test_calculus_domain() {
    let math = MathTables::new();
    let domain = math.get_domain("calculus").unwrap();
    
    // Test Taylor series for sin(π/6) = 0.5
    let x = std::f64::consts::PI / 6.0;
    let terms = 10usize;
    let result = domain.compute("taylor_series_sin", &[&x, &terms]).unwrap();
    let sin_approx = result.downcast_ref::<f64>().unwrap();
    assert!((sin_approx - 0.5).abs() < 1e-6);
    
    // Test Taylor series for cos(π/3) = 0.5
    let x = std::f64::consts::PI / 3.0;
    let result = domain.compute("taylor_series_cos", &[&x, &terms]).unwrap();
    let cos_approx = result.downcast_ref::<f64>().unwrap();
    assert!((cos_approx - 0.5).abs() < 1e-4);
}

#[test]
fn test_discrete_domain() {
    let math = MathTables::new();
    let domain = math.get_domain("discrete").unwrap();
    
    // Test factorial
    let n = 5u64;
    let result = domain.compute("factorial", &[&n]).unwrap();
    let factorial = result.downcast_ref::<u64>().unwrap();
    assert_eq!(*factorial, 120);
    
    // Test combination C(5,2) = 10
    let n = 5u64;
    let k = 2u64;
    let result = domain.compute("combination", &[&n, &k]).unwrap();
    let combination = result.downcast_ref::<u64>().unwrap();
    assert_eq!(*combination, 10);
    
    // Test Catalan number C_3 = 5
    let n = 3u64;
    let result = domain.compute("catalan_number", &[&n]).unwrap();
    let catalan = result.downcast_ref::<u64>().unwrap();
    assert_eq!(*catalan, 5);
}

#[test]
fn test_plugin_system() {
    let mut math = MathTables::new();
    
    // Create a simple test plugin
    let mut test_plugin = mathtables::plugins::Plugin::new(
        "test".to_string(),
        "1.0.0".to_string(),
        "Test plugin".to_string(),
    );
    
    // Add a simple addition function
    let add_function: mathtables::plugins::PluginFunction = Box::new(|args| {
        if let (Some(a), Some(b)) = (
            args.get(0).and_then(|arg| arg.downcast_ref::<f64>()),
            args.get(1).and_then(|arg| arg.downcast_ref::<f64>()),
        ) {
            Box::new(a + b)
        } else {
            Box::new(0.0f64)
        }
    });
    
    test_plugin.add_function("add".to_string(), add_function);
    math.plugin_registry().register_plugin(test_plugin);
    
    // Test the plugin
    assert!(math.plugin_registry().list_plugins().contains(&"test"));
    
    let a = 3.0f64;
    let b = 4.0f64;
    if let Some(result) = math.plugin_registry().call_plugin_function("test", "add", &[&a, &b]) {
        let sum = result.downcast_ref::<f64>().unwrap();
        assert!((sum - 7.0).abs() < 1e-10);
    } else {
        panic!("Plugin function call failed");
    }
}

#[test]
fn test_error_handling() {
    let math = MathTables::new();
    let domain = math.get_domain("number_theory").unwrap();
    
    // Test invalid operation
    let result = domain.compute("nonexistent_operation", &[]);
    assert!(result.is_err());
    
    // Test invalid arguments
    let wrong_type = "not a number";
    let result = domain.compute("gcd", &[&wrong_type]);
    assert!(result.is_err());
}

#[test]
fn test_mathematical_constants() {
    use mathtables::utils::*;
    
    assert!((PI - std::f64::consts::PI).abs() < 1e-10);
    assert!((E - std::f64::consts::E).abs() < 1e-10);
    assert!((SQRT_2 - std::f64::consts::SQRT_2).abs() < 1e-10);
}

#[test]
fn test_conversions() {
    use mathtables::utils::*;
    
    // Test degree/radian conversion
    let degrees = 180.0;
    let radians = degrees_to_radians(degrees);
    assert!((radians - std::f64::consts::PI).abs() < 1e-10);
    
    let back_to_degrees = radians_to_degrees(radians);
    assert!((back_to_degrees - degrees).abs() < 1e-10);
    
    // Test temperature conversions
    let celsius = 0.0;
    let fahrenheit = celsius_to_fahrenheit(celsius);
    assert!((fahrenheit - 32.0).abs() < 1e-10);
    
    let kelvin = celsius_to_kelvin(celsius);
    assert!((kelvin - 273.15).abs() < 1e-10);
}

#[test]
fn test_validations() {
    use mathtables::utils::*;
    
    assert!(is_finite_number(5.0));
    assert!(!is_finite_number(f64::INFINITY));
    assert!(!is_finite_number(f64::NAN));
    
    assert!(is_positive(5.0));
    assert!(!is_positive(-5.0));
    assert!(!is_positive(0.0));
    
    assert!(is_valid_triangle(3.0, 4.0, 5.0));
    assert!(!is_valid_triangle(1.0, 2.0, 5.0));
    
    assert!(is_right_triangle(3.0, 4.0, 5.0));
    assert!(!is_right_triangle(2.0, 3.0, 4.0));
}

#[test]
fn test_cross_domain_integration() {
    use mathtables::domains::number_theory::NumberTheoryDomain;
    use mathtables::domains::geometry::GeometryDomain;
    
    // Test using results from one domain in another
    let n = 13u64;
    let is_prime = NumberTheoryDomain::is_prime(n);
    assert!(is_prime);
    
    // Use the prime number as a radius for circle area
    let radius = n as f64;
    let area_result = GeometryDomain::circle_area(radius);
    assert!(area_result.is_ok());
    
    if let Ok(area) = area_result {
        let expected_area = std::f64::consts::PI * (n * n) as f64;
        assert!((area - expected_area).abs() < 1e-10);
    }
}

#[test]
fn test_large_number_operations() {
    use mathtables::domains::number_theory::NumberTheoryDomain;
    
    // Test large Fibonacci numbers
    let large_fib = NumberTheoryDomain::fibonacci(50);
    
    // F(50) = 12586269025
    let expected = "12586269025".parse::<num_bigint::BigInt>().unwrap();
    assert_eq!(large_fib, expected);
    
    // Test large prime factorization
    let large_composite = 2_305_843_009_213_693_951u64; // 2^61 - 1 (Mersenne number, not prime)
    let factors = NumberTheoryDomain::prime_factors(large_composite);
    
    // Verify that the product of factors equals the original number
    let product: u64 = factors.iter().product();
    assert_eq!(product, large_composite);
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn benchmark_prime_checking() {
        use mathtables::domains::number_theory::NumberTheoryDomain;
        
        let primes_to_test: Vec<u64> = vec![
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
            1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061,
            10007, 10009, 10037, 10039, 10061, 10067, 10069, 10079, 10091, 10093
        ];
        
        let start = Instant::now();
        for &prime in &primes_to_test {
            assert!(NumberTheoryDomain::is_prime(prime));
        }
        let duration = start.elapsed();
        
        println!("Prime checking for {} numbers took: {:?}", primes_to_test.len(), duration);
        
        // Should be reasonably fast (under 1ms for these small primes)
        assert!(duration.as_millis() < 100);
    }
    
    #[test]
    fn benchmark_matrix_operations() {
        use mathtables::domains::algebra::AlgebraDomain;
        
        // Test with progressively larger matrices
        let sizes = vec![2, 4, 8, 16];
        
        for size in sizes {
            let matrix = Matrix {
                data: (0..size).map(|i| {
                    (0..size).map(|j| (i * size + j + 1) as f64).collect()
                }).collect(),
                rows: size,
                cols: size,
            };
            
            let start = Instant::now();
            let det_result = AlgebraDomain::matrix_determinant(&matrix);
            let duration = start.elapsed();
            
            assert!(det_result.is_ok());
            println!("{}x{} matrix determinant took: {:?}", size, size, duration);
        }
    }
}