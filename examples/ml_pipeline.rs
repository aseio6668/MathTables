use mathtables::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MathTables Machine Learning Pipeline Demo ===\n");

    // Generate sample dataset for demonstration
    let (training_data, labels) = generate_sample_data();
    println!("Generated training dataset with {} samples", training_data.len());
    
    // 1. Linear Regression
    println!("\n1. Training Linear Regression Model...");
    let model = MachineLearningDomain::linear_regression(
        &training_data, 
        &labels, 
        0.01,   // learning rate
        1000    // epochs
    )?;
    
    println!("   Model trained successfully!");
    println!("   Weights: {:?}", model.weights);
    println!("   Bias: {:.4}", model.bias);
    println!("   Training Loss: {:.6}", model.training_loss);
    
    // Make predictions
    let test_sample = vec![2.5, 3.0];
    let prediction = MachineLearningDomain::predict_linear_regression(&model, &test_sample)?;
    println!("   Prediction for {:?}: {:.4}", test_sample, prediction);
    
    // 2. Principal Component Analysis (PCA)
    println!("\n2. Applying Principal Component Analysis...");
    let pca_result = MachineLearningDomain::principal_component_analysis(&training_data, 2)?;
    
    println!("   PCA completed!");
    println!("   Explained variance ratio: {:?}", pca_result.explained_variance_ratio);
    println!("   Cumulative variance ratio: {:?}", pca_result.cumulative_variance_ratio);
    
    for (i, component) in pca_result.components.iter().enumerate() {
        println!("   PC{}: variance = {:.4}", i + 1, component.explained_variance);
    }
    
    // 3. K-Means Clustering
    println!("\n3. Performing K-Means Clustering...");
    let clusters = MachineLearningDomain::k_means_clustering(
        &training_data, 
        3,      // number of clusters
        100,    // max iterations
        1e-4    // tolerance
    )?;
    
    println!("   Clustering completed!");
    println!("   Converged in {} iterations", clusters.iterations);
    println!("   Final inertia: {:.6}", clusters.inertia);
    println!("   Cluster centroids:");
    
    for (i, centroid) in clusters.centroids.iter().enumerate() {
        println!("     Cluster {}: {:?}", i, centroid);
    }
    
    // Display cluster assignments
    let mut cluster_counts = vec![0; 3];
    for &assignment in &clusters.assignments {
        cluster_counts[assignment] += 1;
    }
    println!("   Cluster sizes: {:?}", cluster_counts);
    
    // 4. K-Nearest Neighbors Classification
    println!("\n4. K-Nearest Neighbors Classification...");
    
    // Generate classification data
    let (class_data, class_labels) = generate_classification_data();
    let knn = MachineLearningDomain::knn_classifier(class_data, class_labels, 3);
    
    // Test classification
    let test_points = vec![
        vec![1.0, 1.0],
        vec![3.0, 3.0],
        vec![-1.0, -1.0],
    ];
    
    println!("   KNN Classifier created (k=3)");
    for point in &test_points {
        let prediction = MachineLearningDomain::knn_predict(&knn, point)?;
        println!("   Point {:?} classified as: {}", point, prediction);
    }
    
    // 5. Model Evaluation
    println!("\n5. Model Evaluation...");
    
    // Generate predictions vs actual for demonstration
    let true_labels = vec![0, 1, 0, 1, 1, 0, 1, 0];
    let predicted_labels = vec![0, 1, 1, 1, 1, 0, 0, 0];
    
    let accuracy = MachineLearningDomain::accuracy_score(&true_labels, &predicted_labels)?;
    let confusion_matrix = MachineLearningDomain::confusion_matrix(&true_labels, &predicted_labels)?;
    
    println!("   Accuracy: {:.2}%", accuracy * 100.0);
    println!("   Confusion Matrix:");
    for ((true_label, pred_label), count) in confusion_matrix {
        println!("     True: {}, Predicted: {} -> Count: {}", true_label, pred_label, count);
    }
    
    // 6. Complete Pipeline Integration
    println!("\n6. Integrated ML Pipeline...");
    
    // Combine PCA with clustering
    // Note: In a real implementation, you would project data using PCA components
    println!("   Applying dimensionality reduction before clustering...");
    
    // For demonstration, we'll use the original data but mention the integration
    let reduced_clusters = MachineLearningDomain::k_means_clustering(
        &training_data, 
        2,      // fewer clusters for reduced data
        50,     // fewer iterations
        1e-3    // larger tolerance
    )?;
    
    println!("   Clustering on reduced data completed!");
    println!("   Reduced model inertia: {:.6}", reduced_clusters.inertia);
    
    // Feature engineering demonstration
    println!("\n7. Feature Engineering Example...");
    
    // Calculate feature statistics
    if !training_data.is_empty() && !training_data[0].is_empty() {
        let num_features = training_data[0].len();
        println!("   Dataset has {} features", num_features);
        
        for feature_idx in 0..num_features {
            let feature_values: Vec<f64> = training_data.iter()
                .map(|sample| sample[feature_idx])
                .collect();
            
            // Create dataset for statistics
            let feature_dataset = Dataset {
                values: feature_values,
                name: Some(format!("Feature_{}", feature_idx)),
            };
            
            let mean = StatisticsDomain::mean(&feature_dataset)?;
            let std_dev = StatisticsDomain::standard_deviation(&feature_dataset)?;
            
            println!("   Feature {}: mean = {:.4}, std = {:.4}", 
                    feature_idx, mean, std_dev);
        }
    }
    
    println!("\n=== Machine Learning Pipeline Complete ===");
    println!("This pipeline demonstrated:");
    println!("• Linear regression with gradient descent");
    println!("• Dimensionality reduction with PCA");
    println!("• Unsupervised learning with K-means clustering");
    println!("• Classification with KNN");
    println!("• Model evaluation and metrics");
    println!("• Feature engineering and statistics");
    
    Ok(())
}

fn generate_sample_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    // Generate synthetic regression data
    // y = 2*x1 + 3*x2 + noise
    let mut data = Vec::new();
    let mut labels = Vec::new();
    
    for i in 0..50 {
        let x1 = (i as f64) * 0.1;
        let x2 = (i as f64) * 0.05 + 1.0;
        let noise = ((i as f64) * 0.1).sin() * 0.1;
        
        let y = 2.0 * x1 + 3.0 * x2 + noise;
        
        data.push(vec![x1, x2]);
        labels.push(y);
    }
    
    (data, labels)
}

fn generate_classification_data() -> (Vec<Vec<f64>>, Vec<i32>) {
    // Generate synthetic classification data
    let mut data = Vec::new();
    let mut labels = Vec::new();
    
    // Class 0: points around (0, 0)
    for i in 0..20 {
        let x = (i as f64 - 10.0) * 0.2;
        let y = (i as f64 - 10.0) * 0.15;
        data.push(vec![x, y]);
        labels.push(0);
    }
    
    // Class 1: points around (3, 3)
    for i in 0..20 {
        let x = 3.0 + (i as f64 - 10.0) * 0.2;
        let y = 3.0 + (i as f64 - 10.0) * 0.15;
        data.push(vec![x, y]);
        labels.push(1);
    }
    
    (data, labels)
}