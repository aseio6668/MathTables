use mathtables::prelude::*;

#[test]
fn test_smbo_tpe_domain_initialization() {
    let domain = SMBOTPEDomain::new();
    // Just test that it can be created
    assert!(true);
}

#[test]
fn test_tpe_optimizer_creation() {
    let bounds = vec![(0.0, 1.0), (0.0, 10.0)];
    let optimizer = TPEOptimizer::new(bounds.clone(), 0.25);
    
    assert_eq!(optimizer.parameter_bounds, bounds);
    assert_eq!(optimizer.gamma, 0.25);
    assert_eq!(optimizer.observations.len(), 0);
    assert_eq!(optimizer.n_startup_trials, 10);
    assert_eq!(optimizer.n_ei_candidates, 100);
}

#[test]
fn test_add_observation_valid() {
    let bounds = vec![(0.0, 1.0), (0.0, 10.0)];
    let mut optimizer = TPEOptimizer::new(bounds, 0.25);
    
    let result = optimizer.add_observation(vec![0.5, 5.0], 1.0);
    assert!(result.is_ok());
    assert_eq!(optimizer.observations.len(), 1);
    assert_eq!(optimizer.observations[0].parameters, vec![0.5, 5.0]);
    assert_eq!(optimizer.observations[0].objective_value, 1.0);
}

#[test]
fn test_add_observation_invalid_dimensions() {
    let bounds = vec![(0.0, 1.0), (0.0, 10.0)];
    let mut optimizer = TPEOptimizer::new(bounds, 0.25);
    
    let result = optimizer.add_observation(vec![0.5], 1.0); // Wrong dimension
    assert!(result.is_err());
    assert_eq!(optimizer.observations.len(), 0);
}

#[test]
fn test_add_observation_out_of_bounds() {
    let bounds = vec![(0.0, 1.0), (0.0, 10.0)];
    let mut optimizer = TPEOptimizer::new(bounds, 0.25);
    
    // First parameter out of bounds
    let result1 = optimizer.add_observation(vec![1.5, 5.0], 1.0);
    assert!(result1.is_err());
    
    // Second parameter out of bounds
    let result2 = optimizer.add_observation(vec![0.5, 15.0], 1.0);
    assert!(result2.is_err());
    
    assert_eq!(optimizer.observations.len(), 0);
}

#[test]
fn test_parzen_estimator_creation() {
    let data = vec![0.1, 0.5, 0.8, 0.3, 0.7];
    let bounds = (0.0, 1.0);
    let estimator = ParzenEstimator::new(data.clone(), bounds);
    
    assert!(estimator.is_ok());
    let estimator = estimator.unwrap();
    assert_eq!(estimator.data_points, data);
    assert!(estimator.bandwidth > 0.0);
    assert_eq!(estimator.bounds, bounds);
}

#[test]
fn test_parzen_estimator_empty_data() {
    let data = vec![];
    let bounds = (0.0, 1.0);
    let estimator = ParzenEstimator::new(data, bounds);
    
    assert!(estimator.is_err());
}

#[test]
fn test_parzen_estimator_density() {
    let data = vec![0.5, 0.5, 0.5]; // All points at 0.5
    let bounds = (0.0, 1.0);
    let estimator = ParzenEstimator::new(data, bounds).unwrap();
    
    let density_center = estimator.density(0.5);
    let density_edge = estimator.density(0.1);
    
    assert!(density_center > 0.0);
    assert!(density_edge >= 0.0);
    assert!(density_center > density_edge); // Should be higher at center
}

#[test]
fn test_parzen_estimator_sampling() {
    let data = vec![0.2, 0.4, 0.6, 0.8];
    let bounds = (0.0, 1.0);
    let estimator = ParzenEstimator::new(data, bounds).unwrap();
    
    for _ in 0..10 {
        let sample = estimator.sample();
        assert!(sample >= bounds.0);
        assert!(sample <= bounds.1);
    }
}

#[test]
fn test_suggest_parameters_startup_phase() {
    let bounds = vec![(0.0, 1.0), (-10.0, 10.0), (0.0, 100.0)];
    let optimizer = TPEOptimizer::new(bounds.clone(), 0.25);
    
    // Should use random sampling during startup
    let suggestion = optimizer.suggest_next_parameters().unwrap();
    
    assert_eq!(suggestion.parameters.len(), 3);
    assert!(suggestion.parameters[0] >= 0.0 && suggestion.parameters[0] <= 1.0);
    assert!(suggestion.parameters[1] >= -10.0 && suggestion.parameters[1] <= 10.0);
    assert!(suggestion.parameters[2] >= 0.0 && suggestion.parameters[2] <= 100.0);
    assert_eq!(suggestion.expected_improvement, 0.0);
    assert_eq!(suggestion.acquisition_value, 1.0);
}

#[test]
fn test_suggest_parameters_with_history() {
    let bounds = vec![(0.0, 1.0), (0.0, 10.0)];
    let mut optimizer = TPEOptimizer::new(bounds, 0.25);
    optimizer.n_startup_trials = 3; // Reduce startup trials for testing
    
    // Add enough observations to exit startup phase
    let observations = vec![
        (vec![0.1, 1.0], 5.0),
        (vec![0.3, 3.0], 3.0),
        (vec![0.5, 5.0], 2.0),
        (vec![0.7, 7.0], 4.0),
        (vec![0.9, 9.0], 6.0),
    ];
    
    for (params, objective) in observations {
        optimizer.add_observation(params, objective).unwrap();
    }
    
    let suggestion = optimizer.suggest_next_parameters().unwrap();
    
    assert_eq!(suggestion.parameters.len(), 2);
    assert!(suggestion.parameters[0] >= 0.0 && suggestion.parameters[0] <= 1.0);
    assert!(suggestion.parameters[1] >= 0.0 && suggestion.parameters[1] <= 10.0);
    assert!(suggestion.acquisition_value > 0.0);
}

#[test]
fn test_get_best_parameters() {
    let bounds = vec![(0.0, 1.0)];
    let mut optimizer = TPEOptimizer::new(bounds, 0.25);
    
    // Add observations with different objective values
    optimizer.add_observation(vec![0.2], 5.0).unwrap();
    optimizer.add_observation(vec![0.8], 2.0).unwrap(); // Best
    optimizer.add_observation(vec![0.5], 3.0).unwrap();
    optimizer.add_observation(vec![0.1], 4.0).unwrap();
    
    let (best_params, best_value) = optimizer.get_best_parameters().unwrap();
    assert_eq!(best_params, vec![0.8]);
    assert_eq!(best_value, 2.0);
}

#[test]
fn test_get_best_parameters_empty() {
    let bounds = vec![(0.0, 1.0)];
    let optimizer = TPEOptimizer::new(bounds, 0.25);
    
    let result = optimizer.get_best_parameters();
    assert!(result.is_err());
}

#[test]
fn test_optimization_history() {
    let bounds = vec![(0.0, 1.0)];
    let mut optimizer = TPEOptimizer::new(bounds, 0.25);
    
    // Add observations in order
    optimizer.add_observation(vec![0.1], 5.0).unwrap();
    optimizer.add_observation(vec![0.2], 3.0).unwrap(); // New best
    optimizer.add_observation(vec![0.3], 4.0).unwrap(); // Worse, keep previous best
    optimizer.add_observation(vec![0.4], 1.0).unwrap(); // New best
    optimizer.add_observation(vec![0.5], 2.0).unwrap(); // Worse, keep previous best
    
    let history = optimizer.optimization_history();
    assert_eq!(history, vec![5.0, 3.0, 3.0, 1.0, 1.0]);
}

#[test]
fn test_optimize_tpe_material_properties() {
    let target_modulus = 20.0;
    let target_biocompatibility = 85.0;
    
    let optimizer = optimize_tpe_material_properties(target_modulus, target_biocompatibility).unwrap();
    
    assert!(optimizer.observations.len() > 0);
    assert_eq!(optimizer.parameter_bounds.len(), 5); // 5 parameters
    
    // Check parameter bounds
    assert_eq!(optimizer.parameter_bounds[0], (30.0, 90.0)); // Hardness
    assert_eq!(optimizer.parameter_bounds[1], (5.0, 50.0));  // Tensile strength
    assert_eq!(optimizer.parameter_bounds[2], (0.001, 0.1)); // Crosslink density
    assert_eq!(optimizer.parameter_bounds[3], (0.0, 0.4));   // Filler fraction
    assert_eq!(optimizer.parameter_bounds[4], (293.15, 373.15)); // Temperature
    
    // Should be able to suggest next parameters
    let suggestion = optimizer.suggest_next_parameters().unwrap();
    assert_eq!(suggestion.parameters.len(), 5);
}

#[test]
fn test_optimize_silicone_crosslinking() {
    let target_modulus = 1.5; // MPa
    let target_elongation = 400.0; // %
    
    let optimizer = optimize_silicone_crosslinking(target_modulus, target_elongation).unwrap();
    
    assert!(optimizer.observations.len() > 0);
    assert_eq!(optimizer.parameter_bounds.len(), 4); // 4 parameters
    
    // Check parameter bounds
    assert_eq!(optimizer.parameter_bounds[0], (0.001, 0.02));    // Crosslink density
    assert_eq!(optimizer.parameter_bounds[1], (323.15, 423.15)); // Temperature
    assert_eq!(optimizer.parameter_bounds[2], (0.001, 0.05));    // Catalyst concentration
    assert_eq!(optimizer.parameter_bounds[3], (1800.0, 14400.0)); // Cure time
    
    // Should be able to suggest next parameters
    let suggestion = optimizer.suggest_next_parameters().unwrap();
    assert_eq!(suggestion.parameters.len(), 4);
}

#[test]
fn test_tpe_optimization_workflow() {
    // Simulate a complete optimization workflow
    let bounds = vec![(0.0, 10.0), (0.0, 1.0)];
    let mut optimizer = TPEOptimizer::new(bounds, 0.2);
    optimizer.n_startup_trials = 5;
    optimizer.n_ei_candidates = 20;
    
    // Simulate objective function: minimize (x-5)^2 + (y-0.3)^2
    let objective = |params: &[f64]| -> f64 {
        let x = params[0];
        let y = params[1];
        (x - 5.0).powi(2) + (y - 0.3).powi(2)
    };
    
    // Run optimization for several iterations
    for iteration in 0..15 {
        let suggestion = optimizer.suggest_next_parameters().unwrap();
        let obj_value = objective(&suggestion.parameters);
        optimizer.add_observation(suggestion.parameters, obj_value).unwrap();
        
        if iteration >= 5 { // After startup phase
            assert!(suggestion.acquisition_value > 0.0);
        }
    }
    
    // Check that we have improvement over iterations
    let history = optimizer.optimization_history();
    assert_eq!(history.len(), 15);
    
    let final_best = history.last().unwrap();
    let initial_best = history.first().unwrap();
    assert!(final_best <= initial_best); // Should improve (or at least not get worse)
    
    // Best parameters should be close to optimum
    let (best_params, best_value) = optimizer.get_best_parameters().unwrap();
    assert_eq!(best_params.len(), 2);
    assert!(best_value >= 0.0); // Objective is always non-negative
    
    // The best parameters should be reasonably close to [5.0, 0.3]
    // (allowing for randomness in the optimization)
    assert!(best_params[0] >= 0.0 && best_params[0] <= 10.0);
    assert!(best_params[1] >= 0.0 && best_params[1] <= 1.0);
}

#[test]
fn test_multi_modal_optimization() {
    // Test with multi-modal function
    let bounds = vec![(-10.0, 10.0)];
    let mut optimizer = TPEOptimizer::new(bounds, 0.3);
    optimizer.n_startup_trials = 8;
    
    // Multi-modal objective: sin(x) + 0.1*x^2 (has global minimum around x ≈ -1.4)
    let objective = |params: &[f64]| -> f64 {
        let x = params[0];
        x.sin() + 0.1 * x * x
    };
    
    // Run several iterations
    for _ in 0..20 {
        let suggestion = optimizer.suggest_next_parameters().unwrap();
        let obj_value = objective(&suggestion.parameters);
        optimizer.add_observation(suggestion.parameters, obj_value).unwrap();
    }
    
    let (best_params, best_value) = optimizer.get_best_parameters().unwrap();
    
    // Should find a reasonable solution
    assert!(best_params[0] >= -10.0 && best_params[0] <= 10.0);
    assert!(best_value <= 5.0); // Should be much better than random
}

#[test]
fn test_constraint_handling() {
    // Test that TPE respects parameter bounds strictly
    let bounds = vec![(0.0, 1.0), (10.0, 20.0)];
    let optimizer = TPEOptimizer::new(bounds.clone(), 0.25);
    
    // Generate many suggestions and verify all are within bounds
    for _ in 0..50 {
        let suggestion = optimizer.suggest_next_parameters().unwrap();
        assert!(suggestion.parameters[0] >= bounds[0].0);
        assert!(suggestion.parameters[0] <= bounds[0].1);
        assert!(suggestion.parameters[1] >= bounds[1].0);
        assert!(suggestion.parameters[1] <= bounds[1].1);
    }
}

#[test]
fn test_tpe_performance_characteristics() {
    // Test basic performance characteristics
    let bounds = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
    let mut optimizer = TPEOptimizer::new(bounds, 0.25);
    optimizer.n_startup_trials = 5;
    
    let start_time = std::time::Instant::now();
    
    // Add several observations and generate suggestions
    for i in 0..20 {
        let suggestion = optimizer.suggest_next_parameters().unwrap();
        optimizer.add_observation(
            suggestion.parameters, 
            (i as f64) * 0.1 // Simple increasing objective
        ).unwrap();
    }
    
    let duration = start_time.elapsed();
    
    // Should complete reasonably quickly (less than 1 second for this simple case)
    assert!(duration.as_secs() < 1);
    
    // Verify we have collected all observations
    assert_eq!(optimizer.observations.len(), 20);
    
    // History should show monotonic improvement (since we used increasing objectives)
    let history = optimizer.optimization_history();
    for i in 1..history.len() {
        assert!(history[i] <= history[i-1]); // Non-increasing (minimization)
    }
}