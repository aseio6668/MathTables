use mathtables::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MathTables Framework Demo ===\n");
    
    let mut math_framework = MathTables::new();
    
    println!("Available domains: {:?}\n", math_framework.list_domains());
    
    // Number Theory Examples
    println!("=== Number Theory ===");
    if let Some(number_theory) = math_framework.get_domain("number_theory") {
        let a = 48i64;
        let b = 18i64;
        
        let gcd_result = number_theory.compute("gcd", &[&a, &b])?;
        println!("GCD of {} and {}: {:?}", a, b, gcd_result);
        
        let n = 17u64;
        let is_prime_result = number_theory.compute("is_prime", &[&n])?;
        println!("{} is prime: {:?}", n, is_prime_result);
        
        let fib_n = 10u64;
        let fibonacci_result = number_theory.compute("fibonacci", &[&fib_n])?;
        println!("Fibonacci({}): {:?}", fib_n, fibonacci_result);
    }
    
    // Geometry Examples
    println!("\n=== Geometry ===");
    if let Some(geometry) = math_framework.get_domain("geometry") {
        let p1 = Point2D { x: 0.0, y: 0.0 };
        let p2 = Point2D { x: 3.0, y: 4.0 };
        
        let distance_result = geometry.compute("distance_2d", &[&p1, &p2])?;
        println!("Distance between {:?} and {:?}: {:?}", p1, p2, distance_result);
        
        let radius = 5.0f64;
        let area_result = geometry.compute("circle_area", &[&radius])?;
        println!("Circle area (radius {}): {:?}", radius, area_result);
        
        let a = 3.0f64;
        let b = 4.0f64;
        let c = 5.0f64;
        let triangle_area_result = geometry.compute("triangle_area_heron", &[&a, &b, &c])?;
        println!("Triangle area (sides {}, {}, {}): {:?}", a, b, c, triangle_area_result);
    }
    
    // Algebra Examples
    println!("\n=== Algebra ===");
    if let Some(algebra) = math_framework.get_domain("algebra") {
        let matrix = Matrix {
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            rows: 2,
            cols: 2,
        };
        
        let det_result = algebra.compute("matrix_determinant", &[&matrix])?;
        println!("Matrix determinant: {:?}", det_result);
        
        let poly = Polynomial {
            coefficients: vec![1.0, -3.0, 2.0], // 2x^2 - 3x + 1
        };
        let x = 2.0f64;
        
        let eval_result = algebra.compute("polynomial_evaluate", &[&poly, &x])?;
        println!("Polynomial evaluation at x={}: {:?}", x, eval_result);
        
        let qa = 1.0f64;
        let qb = -5.0f64;
        let qc = 6.0f64;
        
        let roots_result = algebra.compute("quadratic_roots", &[&qa, &qb, &qc])?;
        println!("Quadratic roots ({}x² + {}x + {}): {:?}", qa, qb, qc, roots_result);
    }
    
    // Calculus Examples
    println!("\n=== Calculus ===");
    if let Some(calculus) = math_framework.get_domain("calculus") {
        let x = std::f64::consts::PI / 4.0;
        let terms = 10usize;
        
        let sin_result = calculus.compute("taylor_series_sin", &[&x, &terms])?;
        println!("Taylor series sin({:.4}) ≈ {:?}", x, sin_result);
        
        let cos_result = calculus.compute("taylor_series_cos", &[&x, &terms])?;
        println!("Taylor series cos({:.4}) ≈ {:?}", x, cos_result);
        
        let exp_x = 1.0f64;
        let exp_result = calculus.compute("taylor_series_exp", &[&exp_x, &terms])?;
        println!("Taylor series exp({}) ≈ {:?}", exp_x, exp_result);
    }
    
    // Discrete Mathematics Examples
    println!("\n=== Discrete Mathematics ===");
    if let Some(discrete) = math_framework.get_domain("discrete") {
        let n = 5u64;
        let factorial_result = discrete.compute("factorial", &[&n])?;
        println!("{}! = {:?}", n, factorial_result);
        
        let cn = 10u64;
        let ck = 3u64;
        let combination_result = discrete.compute("combination", &[&cn, &ck])?;
        println!("C({}, {}) = {:?}", cn, ck, combination_result);
        
        let catalan_n = 4u64;
        let catalan_result = discrete.compute("catalan_number", &[&catalan_n])?;
        println!("Catalan number C_{} = {:?}", catalan_n, catalan_result);
        
        let fib_seq_n = 8usize;
        let fib_seq_result = discrete.compute("fibonacci_sequence", &[&fib_seq_n])?;
        println!("First {} Fibonacci numbers: {:?}", fib_seq_n, fib_seq_result);
    }
    
    println!("\n=== Framework Information ===");
    println!("Framework initialized: {}", math_framework.is_initialized());
    println!("Available plugins: {:?}", math_framework.plugin_registry().list_plugins());
    
    Ok(())
}