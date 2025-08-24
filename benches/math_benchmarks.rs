use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mathtables::prelude::*;
use mathtables::domains::*;

fn benchmark_number_theory(c: &mut Criterion) {
    let mut group = c.benchmark_group("number_theory");
    
    group.bench_function("gcd", |b| {
        b.iter(|| {
            number_theory::NumberTheoryDomain::gcd(black_box(1071), black_box(462))
        })
    });
    
    group.bench_function("is_prime_small", |b| {
        b.iter(|| {
            number_theory::NumberTheoryDomain::is_prime(black_box(1009))
        })
    });
    
    group.bench_function("is_prime_large", |b| {
        b.iter(|| {
            number_theory::NumberTheoryDomain::is_prime(black_box(1000003))
        })
    });
    
    group.bench_function("fibonacci_30", |b| {
        b.iter(|| {
            number_theory::NumberTheoryDomain::fibonacci(black_box(30))
        })
    });
    
    group.bench_function("prime_factors", |b| {
        b.iter(|| {
            number_theory::NumberTheoryDomain::prime_factors(black_box(123456789))
        })
    });
    
    group.finish();
}

fn benchmark_geometry(c: &mut Criterion) {
    let mut group = c.benchmark_group("geometry");
    
    let p1 = Point2D { x: 1.0, y: 2.0 };
    let p2 = Point2D { x: 4.0, y: 6.0 };
    let v1 = Vector3D { x: 1.0, y: 2.0, z: 3.0 };
    let v2 = Vector3D { x: 4.0, y: 5.0, z: 6.0 };
    
    group.bench_function("distance_2d", |b| {
        b.iter(|| {
            geometry::GeometryDomain::distance_2d(black_box(&p1), black_box(&p2))
        })
    });
    
    group.bench_function("dot_product_3d", |b| {
        b.iter(|| {
            geometry::GeometryDomain::dot_product_3d(black_box(&v1), black_box(&v2))
        })
    });
    
    group.bench_function("cross_product_3d", |b| {
        b.iter(|| {
            geometry::GeometryDomain::cross_product_3d(black_box(&v1), black_box(&v2))
        })
    });
    
    group.bench_function("circle_area", |b| {
        b.iter(|| {
            geometry::GeometryDomain::circle_area(black_box(5.0))
        })
    });
    
    group.bench_function("triangle_area_heron", |b| {
        b.iter(|| {
            geometry::GeometryDomain::triangle_area_heron(black_box(3.0), black_box(4.0), black_box(5.0))
        })
    });
    
    group.finish();
}

fn benchmark_algebra(c: &mut Criterion) {
    let mut group = c.benchmark_group("algebra");
    
    let matrix_2x2 = Matrix {
        data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        rows: 2,
        cols: 2,
    };
    
    let matrix_4x4 = Matrix {
        data: vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0, 16.0],
        ],
        rows: 4,
        cols: 4,
    };
    
    let poly = Polynomial {
        coefficients: vec![1.0, -3.0, 2.0, 1.0, -1.0], // 4th degree polynomial
    };
    
    group.bench_function("matrix_determinant_2x2", |b| {
        b.iter(|| {
            algebra::AlgebraDomain::matrix_determinant(black_box(&matrix_2x2))
        })
    });
    
    group.bench_function("matrix_determinant_4x4", |b| {
        b.iter(|| {
            algebra::AlgebraDomain::matrix_determinant(black_box(&matrix_4x4))
        })
    });
    
    group.bench_function("polynomial_evaluate", |b| {
        b.iter(|| {
            algebra::AlgebraDomain::polynomial_evaluate(black_box(&poly), black_box(2.5))
        })
    });
    
    group.bench_function("polynomial_multiply", |b| {
        b.iter(|| {
            algebra::AlgebraDomain::polynomial_multiply(black_box(&poly), black_box(&poly))
        })
    });
    
    group.bench_function("quadratic_roots", |b| {
        b.iter(|| {
            algebra::AlgebraDomain::quadratic_roots(black_box(1.0), black_box(-5.0), black_box(6.0))
        })
    });
    
    group.finish();
}

fn benchmark_calculus(c: &mut Criterion) {
    let mut group = c.benchmark_group("calculus");
    
    group.bench_function("taylor_series_sin", |b| {
        b.iter(|| {
            calculus::CalculusDomain::taylor_series_sin(black_box(1.0), black_box(10))
        })
    });
    
    group.bench_function("taylor_series_cos", |b| {
        b.iter(|| {
            calculus::CalculusDomain::taylor_series_cos(black_box(1.0), black_box(10))
        })
    });
    
    group.bench_function("taylor_series_exp", |b| {
        b.iter(|| {
            calculus::CalculusDomain::taylor_series_exp(black_box(2.0), black_box(15))
        })
    });
    
    // Test numerical integration performance
    let test_function = |x: f64| x * x + 2.0 * x + 1.0;
    
    group.bench_function("numerical_integral_trapezoidal", |b| {
        b.iter(|| {
            calculus::CalculusDomain::numerical_integral_trapezoidal(
                black_box(&test_function),
                black_box(0.0),
                black_box(10.0),
                black_box(1000)
            )
        })
    });
    
    group.finish();
}

fn benchmark_discrete(c: &mut Criterion) {
    let mut group = c.benchmark_group("discrete");
    
    group.bench_function("factorial", |b| {
        b.iter(|| {
            discrete::DiscreteDomain::factorial(black_box(15))
        })
    });
    
    group.bench_function("combination", |b| {
        b.iter(|| {
            discrete::DiscreteDomain::combination(black_box(20), black_box(10))
        })
    });
    
    group.bench_function("catalan_number", |b| {
        b.iter(|| {
            discrete::DiscreteDomain::catalan_number(black_box(10))
        })
    });
    
    group.bench_function("fibonacci_sequence", |b| {
        b.iter(|| {
            discrete::DiscreteDomain::fibonacci_sequence(black_box(20))
        })
    });
    
    // Graph algorithms benchmark
    use std::collections::HashMap;
    let mut graph: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..100 {
        let neighbors = if i < 99 { vec![i + 1] } else { vec![] };
        graph.insert(i, neighbors);
    }
    
    group.bench_function("graph_shortest_path", |b| {
        b.iter(|| {
            discrete::DiscreteDomain::graph_shortest_path(black_box(&graph), black_box(0), black_box(99))
        })
    });
    
    group.finish();
}

fn benchmark_framework_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("framework");
    
    group.bench_function("framework_initialization", |b| {
        b.iter(|| {
            MathTables::new()
        })
    });
    
    let math = MathTables::new();
    
    group.bench_function("domain_access", |b| {
        b.iter(|| {
            black_box(math.get_domain("number_theory"))
        })
    });
    
    let domain = math.get_domain("number_theory").unwrap();
    let a = 48i64;
    let b = 18i64;
    
    group.bench_function("domain_compute_gcd", |b| {
        b.iter(|| {
            domain.compute("gcd", &[black_box(&a), black_box(&b)])
        })
    });
    
    group.finish();
}

fn benchmark_plugin_system(c: &mut Criterion) {
    let mut group = c.benchmark_group("plugins");
    
    let mut math = MathTables::new();
    
    // Create a simple plugin
    let mut test_plugin = mathtables::plugins::Plugin::new(
        "benchmark_test".to_string(),
        "1.0.0".to_string(),
        "Benchmark test plugin".to_string(),
    );
    
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
    
    let a = 3.0f64;
    let b = 4.0f64;
    
    group.bench_function("plugin_function_call", |b| {
        b.iter(|| {
            math.plugin_registry().call_plugin_function(
                "benchmark_test",
                "add",
                &[black_box(&a), black_box(&b)]
            )
        })
    });
    
    group.finish();
}

fn benchmark_utilities(c: &mut Criterion) {
    let mut group = c.benchmark_group("utilities");
    
    group.bench_function("degrees_to_radians", |b| {
        b.iter(|| {
            mathtables::utils::degrees_to_radians(black_box(180.0))
        })
    });
    
    group.bench_function("cartesian_to_polar", |b| {
        let point = Point2D { x: 3.0, y: 4.0 };
        b.iter(|| {
            mathtables::utils::cartesian_to_polar(black_box(&point))
        })
    });
    
    let matrix = Matrix {
        data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        rows: 2,
        cols: 2,
    };
    
    group.bench_function("matrix_validation", |b| {
        b.iter(|| {
            mathtables::utils::is_valid_matrix(black_box(&matrix))
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_number_theory,
    benchmark_geometry,
    benchmark_algebra,
    benchmark_calculus,
    benchmark_discrete,
    benchmark_framework_operations,
    benchmark_plugin_system,
    benchmark_utilities
);

criterion_main!(benches);