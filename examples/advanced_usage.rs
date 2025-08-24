use mathtables::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MathTables Advanced Usage Demo ===\n");
    
    let math_framework = MathTables::new();
    
    // Advanced Number Theory Operations
    println!("=== Advanced Number Theory ===");
    demonstrate_number_theory_applications();
    
    // Advanced Geometry Operations
    println!("\n=== Advanced Geometry ===");
    demonstrate_geometry_applications();
    
    // Advanced Algebra Operations
    println!("\n=== Advanced Algebra ===");
    demonstrate_algebra_applications();
    
    // Advanced Calculus Operations
    println!("\n=== Advanced Calculus ===");
    demonstrate_calculus_applications();
    
    // Advanced Discrete Mathematics Operations
    println!("\n=== Advanced Discrete Mathematics ===");
    demonstrate_discrete_applications();
    
    // Integration Examples
    println!("\n=== Cross-Domain Integration ===");
    demonstrate_cross_domain_integration(&math_framework);
    
    Ok(())
}

fn demonstrate_number_theory_applications() {
    use mathtables::domains::number_theory::NumberTheoryDomain;
    
    // Prime number analysis
    println!("Prime Analysis:");
    let numbers = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
    for &n in &numbers {
        println!("  {} -> Prime: {}", n, NumberTheoryDomain::is_prime(n));
    }
    
    // Prime factorization
    let composite_numbers = vec![60, 100, 144, 210, 315];
    println!("\nPrime Factorizations:");
    for &n in &composite_numbers {
        let factors = NumberTheoryDomain::prime_factors(n);
        println!("  {} = {:?}", n, factors);
    }
    
    // Euler's totient function
    println!("\nEuler's Totient Function:");
    let totient_examples = vec![1, 2, 6, 12, 18, 30];
    for &n in &totient_examples {
        println!("  φ({}) = {}", n, NumberTheoryDomain::euler_totient(n));
    }
    
    // Fibonacci sequence analysis
    println!("\nLarge Fibonacci Numbers:");
    for i in 40..=50 {
        let fib = NumberTheoryDomain::fibonacci(i);
        println!("  F({}) = {}", i, fib);
    }
}

fn demonstrate_geometry_applications() {
    use mathtables::domains::geometry::GeometryDomain;
    use mathtables::core::types::{Point2D, Vector3D};
    
    // Complex geometric calculations
    let triangle_vertices = vec![
        Point2D { x: 0.0, y: 0.0 },
        Point2D { x: 8.0, y: 0.0 },
        Point2D { x: 4.0, y: 6.0 },
    ];
    
    println!("Triangle Analysis:");
    println!("  Vertices: {:?}", triangle_vertices);
    
    // Calculate all side lengths
    let side_a = GeometryDomain::distance_2d(&triangle_vertices[0], &triangle_vertices[1]);
    let side_b = GeometryDomain::distance_2d(&triangle_vertices[1], &triangle_vertices[2]);
    let side_c = GeometryDomain::distance_2d(&triangle_vertices[2], &triangle_vertices[0]);
    
    println!("  Side lengths: {:.2}, {:.2}, {:.2}", side_a, side_b, side_c);
    
    // Calculate area using Heron's formula
    if let Ok(area) = GeometryDomain::triangle_area_heron(side_a, side_b, side_c) {
        println!("  Area (Heron's formula): {:.2}", area);
    }
    
    // 3D vector operations
    println!("\n3D Vector Operations:");
    let v1 = Vector3D { x: 1.0, y: 2.0, z: 3.0 };
    let v2 = Vector3D { x: 4.0, y: 5.0, z: 6.0 };
    
    println!("  Vector 1: {:?}", v1);
    println!("  Vector 2: {:?}", v2);
    
    let dot_product = GeometryDomain::dot_product_3d(&v1, &v2);
    let cross_product = GeometryDomain::cross_product_3d(&v1, &v2);
    let magnitude_1 = GeometryDomain::vector_magnitude_3d(&v1);
    let magnitude_2 = GeometryDomain::vector_magnitude_3d(&v2);
    
    println!("  Dot product: {:.2}", dot_product);
    println!("  Cross product: {:?}", cross_product);
    println!("  Magnitudes: {:.2}, {:.2}", magnitude_1, magnitude_2);
    
    // Volume calculations
    println!("\nVolume Calculations:");
    let sphere_radius = 5.0;
    if let Ok(sphere_volume) = GeometryDomain::sphere_volume(sphere_radius) {
        println!("  Sphere (r={}): {:.2}", sphere_radius, sphere_volume);
    }
}

fn demonstrate_algebra_applications() {
    use mathtables::domains::algebra::AlgebraDomain;
    use mathtables::core::types::{Matrix, Polynomial};
    
    // Matrix operations
    println!("Matrix Operations:");
    let matrix_a = Matrix {
        data: vec![vec![2.0, 1.0], vec![1.0, 3.0]],
        rows: 2,
        cols: 2,
    };
    
    println!("  Matrix A: {:?}", matrix_a.data);
    
    if let Ok(det) = AlgebraDomain::matrix_determinant(&matrix_a) {
        println!("  Determinant: {:.2}", det);
    }
    
    // System of linear equations
    let constants = vec![5.0, 7.0];
    if let Ok(solution) = AlgebraDomain::solve_linear_system(&matrix_a, &constants) {
        println!("  Solution to Ax = b: {:?}", solution);
    }
    
    // Polynomial operations
    println!("\nPolynomial Operations:");
    let poly1 = Polynomial {
        coefficients: vec![1.0, -3.0, 2.0], // 2x² - 3x + 1
    };
    let poly2 = Polynomial {
        coefficients: vec![-1.0, 1.0], // x - 1
    };
    
    println!("  P1(x) = 2x² - 3x + 1");
    println!("  P2(x) = x - 1");
    
    let poly_sum = AlgebraDomain::polynomial_add(&poly1, &poly2);
    let poly_product = AlgebraDomain::polynomial_multiply(&poly1, &poly2);
    
    println!("  P1 + P2: {:?}", poly_sum.coefficients);
    println!("  P1 × P2: {:?}", poly_product.coefficients);
    
    // Evaluate polynomials at various points
    println!("  P1 evaluations:");
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        let value = AlgebraDomain::polynomial_evaluate(&poly1, x);
        println!("    P1({}) = {:.2}", x, value);
    }
    
    // Quadratic equation solver
    println!("\nQuadratic Equations:");
    let quadratics = vec![
        (1.0, -5.0, 6.0),   // x² - 5x + 6 = 0
        (2.0, -4.0, -6.0),  // 2x² - 4x - 6 = 0
        (1.0, 0.0, -4.0),   // x² - 4 = 0
        (1.0, -2.0, 1.0),   // x² - 2x + 1 = 0
    ];
    
    for (a, b, c) in quadratics {
        println!("  {}x² + {}x + {} = 0", a, b, c);
        if let Ok((root1, root2)) = AlgebraDomain::quadratic_roots(a, b, c) {
            match (root1, root2) {
                (Some(r1), Some(r2)) => println!("    Roots: {:.2}, {:.2}", r1, r2),
                (Some(r1), None) => println!("    Root: {:.2} (double)", r1),
                (None, None) => println!("    Complex roots"),
                _ => println!("    No real roots"),
            }
        }
    }
}

fn demonstrate_calculus_applications() {
    use mathtables::domains::calculus::CalculusDomain;
    use std::f64::consts::PI;
    
    // Taylor series approximations
    println!("Taylor Series Approximations:");
    let test_values = vec![0.0, PI/6.0, PI/4.0, PI/3.0, PI/2.0];
    
    for &x in &test_values {
        let terms_10 = 10;
        let terms_20 = 20;
        
        let sin_10 = CalculusDomain::taylor_series_sin(x, terms_10);
        let sin_20 = CalculusDomain::taylor_series_sin(x, terms_20);
        let sin_actual = x.sin();
        
        let cos_10 = CalculusDomain::taylor_series_cos(x, terms_10);
        let cos_20 = CalculusDomain::taylor_series_cos(x, terms_20);
        let cos_actual = x.cos();
        
        println!("  x = {:.4}:", x);
        println!("    sin(x): actual = {:.6}, taylor(10) = {:.6}, taylor(20) = {:.6}", 
                 sin_actual, sin_10, sin_20);
        println!("    cos(x): actual = {:.6}, taylor(10) = {:.6}, taylor(20) = {:.6}", 
                 cos_actual, cos_10, cos_20);
    }
    
    // Exponential approximations
    println!("\nExponential Approximations:");
    let exp_values = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0];
    
    for &x in &exp_values {
        let exp_taylor = CalculusDomain::taylor_series_exp(x, 20);
        let exp_actual = x.exp();
        let error = (exp_taylor - exp_actual).abs();
        
        println!("  exp({:.1}): actual = {:.6}, taylor = {:.6}, error = {:.2e}", 
                 x, exp_actual, exp_taylor, error);
    }
}

fn demonstrate_discrete_applications() {
    use mathtables::domains::discrete::DiscreteDomain;
    use std::collections::HashMap;
    
    // Combinatorics
    println!("Combinatorial Analysis:");
    
    // Pascal's triangle
    println!("  Pascal's Triangle (first 8 rows):");
    for n in 0..8 {
        print!("    ");
        for k in 0..=n {
            if let Ok(coeff) = DiscreteDomain::combination(n, k) {
                print!("{:4}", coeff);
            }
        }
        println!();
    }
    
    // Catalan numbers
    println!("\n  Catalan Numbers:");
    for n in 0..10 {
        let catalan = DiscreteDomain::catalan_number(n);
        println!("    C_{} = {}", n, catalan);
    }
    
    // Fibonacci analysis
    println!("\n  Fibonacci Sequence Analysis:");
    let fib_seq = DiscreteDomain::fibonacci_sequence(15);
    println!("    Sequence: {:?}", fib_seq);
    
    // Calculate ratios between consecutive terms
    println!("    Golden ratio approximations:");
    for i in 1..fib_seq.len() {
        if fib_seq[i-1] > 0 {
            let ratio = fib_seq[i] as f64 / fib_seq[i-1] as f64;
            println!("      F_{}/F_{} = {:.6}", i+1, i, ratio);
        }
    }
    
    // Graph theory examples
    println!("\nGraph Theory:");
    
    // Create a simple graph
    let mut graph: HashMap<usize, Vec<usize>> = HashMap::new();
    graph.insert(0, vec![1, 2]);
    graph.insert(1, vec![0, 2, 3]);
    graph.insert(2, vec![0, 1, 3]);
    graph.insert(3, vec![1, 2, 4]);
    graph.insert(4, vec![3]);
    
    println!("  Graph: {:?}", graph);
    
    // Find shortest path
    if let Some(path) = DiscreteDomain::graph_shortest_path(&graph, 0, 4) {
        println!("  Shortest path from 0 to 4: {:?}", path);
    }
    
    // Check if bipartite
    let is_bipartite = DiscreteDomain::is_bipartite(&graph);
    println!("  Is bipartite: {}", is_bipartite);
    
    // Create a directed acyclic graph for topological sorting
    let mut dag: HashMap<usize, Vec<usize>> = HashMap::new();
    dag.insert(0, vec![1, 2]);
    dag.insert(1, vec![3]);
    dag.insert(2, vec![3]);
    dag.insert(3, vec![4]);
    dag.insert(4, vec![]);
    
    if let Some(topo_sort) = DiscreteDomain::topological_sort(&dag) {
        println!("  Topological sort of DAG: {:?}", topo_sort);
    }
}

fn demonstrate_cross_domain_integration(math_framework: &MathTables) {
    println!("Cross-Domain Mathematical Applications:");
    
    // Example: Analyzing a pendulum (Physics + Calculus + Trigonometry)
    println!("\n1. Simple Pendulum Analysis:");
    
    let length = 1.0f64; // 1 meter
    let gravity = 9.81f64; // m/s²
    
    // Period calculation using small angle approximation
    let period = 2.0 * std::f64::consts::PI * (length / gravity).sqrt();
    println!("   Pendulum length: {} m", length);
    println!("   Period (small angles): {:.3} seconds", period);
    
    // Frequency and angular frequency
    let frequency = 1.0 / period;
    let angular_frequency = 2.0 * std::f64::consts::PI * frequency;
    println!("   Frequency: {:.3} Hz", frequency);
    println!("   Angular frequency: {:.3} rad/s", angular_frequency);
    
    // Example: Geometric series convergence (Algebra + Calculus + Number Theory)
    println!("\n2. Geometric Series Analysis:");
    
    let ratios: Vec<f64> = vec![0.5, 0.8, 0.9, 0.99];
    
    for &r in &ratios {
        if r < 1.0 {
            let sum = 1.0 / (1.0 - r);
            let partial_sums: Vec<f64> = (1..=10)
                .map(|n| (1.0 - r.powi(n as i32)) / (1.0 - r))
                .collect();
            
            println!("   Ratio r = {:.2}:", r);
            println!("     Theoretical sum: {:.3}", sum);
            println!("     10th partial sum: {:.3}", partial_sums[9]);
            println!("     Error: {:.2e}", (sum - partial_sums[9]).abs());
        }
    }
    
    // Example: Circle packing problem (Geometry + Optimization + Number Theory)
    println!("\n3. Circle Packing in Unit Square:");
    
    // Calculate maximum radius for n circles in unit square
    let circle_counts = vec![1, 4, 9, 16, 25];
    
    for &n in &circle_counts {
        // Simple grid arrangement
        let sqrt_n = (n as f64).sqrt();
        if sqrt_n.fract() == 0.0 {
            let radius = 1.0 / (2.0 * sqrt_n);
            let total_area = n as f64 * std::f64::consts::PI * radius * radius;
            let packing_density = total_area / 1.0; // Unit square area is 1
            
            println!("   {} circles: radius = {:.4}, packing density = {:.3}", 
                     n, radius, packing_density);
        }
    }
    
    // Example: Monte Carlo integration (Statistics + Calculus + Discrete)
    println!("\n4. Monte Carlo π Estimation:");
    
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let sample_sizes = vec![1000, 10000, 100000];
    
    for &samples in &sample_sizes {
        let mut inside_circle = 0;
        
        for _ in 0..samples {
            let x: f64 = rng.gen();
            let y: f64 = rng.gen();
            if x*x + y*y <= 1.0 {
                inside_circle += 1;
            }
        }
        
        let pi_estimate = 4.0 * inside_circle as f64 / samples as f64;
        let error = (pi_estimate - std::f64::consts::PI).abs();
        
        println!("   {} samples: π ≈ {:.6}, error = {:.4}", 
                 samples, pi_estimate, error);
    }
}