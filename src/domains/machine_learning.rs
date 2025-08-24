use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct LinearRegressionModel {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub training_loss: f64,
}

#[derive(Debug, Clone)]
pub struct KMeansResult {
    pub centroids: Vec<Vec<f64>>,
    pub assignments: Vec<usize>,
    pub inertia: f64,
    pub iterations: usize,
}

#[derive(Debug, Clone)]
pub struct KNNClassifier {
    pub training_data: Vec<Vec<f64>>,
    pub training_labels: Vec<i32>,
    pub k: usize,
}

#[derive(Debug, Clone)]
pub struct PrincipalComponent {
    pub component: Vec<f64>,
    pub explained_variance: f64,
}

#[derive(Debug, Clone)]
pub struct PCAResult {
    pub components: Vec<PrincipalComponent>,
    pub explained_variance_ratio: Vec<f64>,
    pub cumulative_variance_ratio: Vec<f64>,
}

pub struct MachineLearningDomain;

impl MachineLearningDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn linear_regression(
        x_data: &[Vec<f64>],
        y_data: &[f64],
        learning_rate: f64,
        epochs: usize,
    ) -> MathResult<LinearRegressionModel> {
        if x_data.is_empty() || x_data.len() != y_data.len() {
            return Err(MathError::InvalidArgument("Data dimensions mismatch".to_string()));
        }
        
        let n_features = x_data[0].len();
        let n_samples = x_data.len();
        
        let mut weights = vec![0.0; n_features];
        let mut bias = 0.0;
        
        for _epoch in 0..epochs {
            let mut dw = vec![0.0; n_features];
            let mut db = 0.0;
            let mut _total_loss = 0.0;
            
            for i in 0..n_samples {
                let mut prediction = bias;
                for j in 0..n_features {
                    prediction += weights[j] * x_data[i][j];
                }
                
                let error = prediction - y_data[i];
                _total_loss += error * error;
                
                db += error;
                for j in 0..n_features {
                    dw[j] += error * x_data[i][j];
                }
            }
            
            let n_samples_f = n_samples as f64;
            for j in 0..n_features {
                weights[j] -= learning_rate * (dw[j] / n_samples_f);
            }
            bias -= learning_rate * (db / n_samples_f);
        }
        
        let mut final_loss = 0.0;
        for i in 0..n_samples {
            let mut prediction = bias;
            for j in 0..n_features {
                prediction += weights[j] * x_data[i][j];
            }
            let error = prediction - y_data[i];
            final_loss += error * error;
        }
        final_loss /= n_samples as f64;
        
        Ok(LinearRegressionModel {
            weights,
            bias,
            training_loss: final_loss,
        })
    }
    
    pub fn predict_linear_regression(model: &LinearRegressionModel, features: &[f64]) -> MathResult<f64> {
        if features.len() != model.weights.len() {
            return Err(MathError::InvalidArgument("Feature dimension mismatch".to_string()));
        }
        
        let mut prediction = model.bias;
        for (i, &feature) in features.iter().enumerate() {
            prediction += model.weights[i] * feature;
        }
        
        Ok(prediction)
    }
    
    pub fn k_means_clustering(
        data: &[Vec<f64>],
        k: usize,
        max_iterations: usize,
        tolerance: f64,
    ) -> MathResult<KMeansResult> {
        if data.is_empty() || k == 0 || k > data.len() {
            return Err(MathError::InvalidArgument("Invalid clustering parameters".to_string()));
        }
        
        let n_features = data[0].len();
        let mut centroids = Vec::new();
        
        for i in 0..k {
            centroids.push(data[i % data.len()].clone());
        }
        
        let mut assignments = vec![0; data.len()];
        let mut iterations = 0;
        
        for iter in 0..max_iterations {
            let mut changed = false;
            
            for (i, point) in data.iter().enumerate() {
                let mut min_distance = f64::INFINITY;
                let mut best_cluster = 0;
                
                for (c, centroid) in centroids.iter().enumerate() {
                    let distance = Self::euclidean_distance(point, centroid)?;
                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = c;
                    }
                }
                
                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }
            
            let old_centroids = centroids.clone();
            
            for c in 0..k {
                let cluster_points: Vec<_> = data.iter()
                    .enumerate()
                    .filter(|(i, _)| assignments[*i] == c)
                    .map(|(_, point)| point)
                    .collect();
                
                if !cluster_points.is_empty() {
                    for j in 0..n_features {
                        centroids[c][j] = cluster_points.iter()
                            .map(|point| point[j])
                            .sum::<f64>() / cluster_points.len() as f64;
                    }
                }
            }
            
            let centroid_movement: f64 = centroids.iter()
                .zip(old_centroids.iter())
                .map(|(new, old)| Self::euclidean_distance(new, old).unwrap_or(0.0))
                .sum();
            
            iterations = iter + 1;
            
            if !changed || centroid_movement < tolerance {
                break;
            }
        }
        
        let mut inertia = 0.0;
        for (i, point) in data.iter().enumerate() {
            let distance = Self::euclidean_distance(point, &centroids[assignments[i]])?;
            inertia += distance * distance;
        }
        
        Ok(KMeansResult {
            centroids,
            assignments,
            inertia,
            iterations,
        })
    }
    
    pub fn knn_classifier(training_data: Vec<Vec<f64>>, training_labels: Vec<i32>, k: usize) -> KNNClassifier {
        KNNClassifier {
            training_data,
            training_labels,
            k,
        }
    }
    
    pub fn knn_predict(classifier: &KNNClassifier, query_point: &[f64]) -> MathResult<i32> {
        if classifier.training_data.is_empty() {
            return Err(MathError::InvalidArgument("Empty training data".to_string()));
        }
        
        let mut distances: Vec<(f64, i32)> = Vec::new();
        
        for (i, training_point) in classifier.training_data.iter().enumerate() {
            let distance = Self::euclidean_distance(query_point, training_point)?;
            distances.push((distance, classifier.training_labels[i]));
        }
        
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        let k_neighbors = std::cmp::min(classifier.k, distances.len());
        let mut label_counts = HashMap::new();
        
        for i in 0..k_neighbors {
            *label_counts.entry(distances[i].1).or_insert(0) += 1;
        }
        
        let predicted_label = label_counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(label, _)| label)
            .unwrap_or(0);
        
        Ok(predicted_label)
    }
    
    pub fn principal_component_analysis(
        data: &[Vec<f64>],
        n_components: usize,
    ) -> MathResult<PCAResult> {
        if data.is_empty() || n_components == 0 {
            return Err(MathError::InvalidArgument("Invalid PCA parameters".to_string()));
        }
        
        let n_samples = data.len();
        let n_features = data[0].len();
        
        if n_components > n_features {
            return Err(MathError::InvalidArgument("Number of components exceeds feature count".to_string()));
        }
        
        let means: Vec<f64> = (0..n_features)
            .map(|j| data.iter().map(|row| row[j]).sum::<f64>() / n_samples as f64)
            .collect();
        
        let centered_data: Vec<Vec<f64>> = data.iter()
            .map(|row| row.iter().enumerate().map(|(j, &val)| val - means[j]).collect())
            .collect();
        
        let mut covariance_matrix = vec![vec![0.0; n_features]; n_features];
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sum = 0.0;
                for sample in &centered_data {
                    sum += sample[i] * sample[j];
                }
                covariance_matrix[i][j] = sum / (n_samples - 1) as f64;
            }
        }
        
        let (eigenvalues, eigenvectors) = Self::power_iteration_eigenvalues(&covariance_matrix, n_components)?;
        
        let total_variance: f64 = eigenvalues.iter().sum();
        let explained_variance_ratio: Vec<f64> = eigenvalues.iter()
            .map(|&val| val / total_variance)
            .collect();
        
        let mut cumulative_variance_ratio = Vec::new();
        let mut cumulative_sum = 0.0;
        for &ratio in &explained_variance_ratio {
            cumulative_sum += ratio;
            cumulative_variance_ratio.push(cumulative_sum);
        }
        
        let components: Vec<PrincipalComponent> = eigenvalues.iter()
            .zip(eigenvectors.iter())
            .map(|(&eigenval, eigenvec)| PrincipalComponent {
                component: eigenvec.clone(),
                explained_variance: eigenval,
            })
            .collect();
        
        Ok(PCAResult {
            components,
            explained_variance_ratio,
            cumulative_variance_ratio,
        })
    }
    
    fn power_iteration_eigenvalues(matrix: &[Vec<f64>], n_components: usize) -> MathResult<(Vec<f64>, Vec<Vec<f64>>)> {
        let n = matrix.len();
        let mut eigenvalues = Vec::new();
        let mut eigenvectors = Vec::new();
        let mut remaining_matrix = matrix.iter().map(|row| row.clone()).collect::<Vec<_>>();
        
        for _ in 0..n_components {
            let mut v = vec![1.0; n];
            let max_iterations = 1000;
            let tolerance = 1e-10;
            
            for _iter in 0..max_iterations {
                let mut new_v = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        new_v[i] += remaining_matrix[i][j] * v[j];
                    }
                }
                
                let norm = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < tolerance {
                    break;
                }
                
                for i in 0..n {
                    new_v[i] /= norm;
                }
                
                let convergence = v.iter().zip(new_v.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, |acc, x| acc.max(x));
                
                v = new_v;
                
                if convergence < tolerance {
                    break;
                }
            }
            
            let mut eigenvalue = 0.0;
            for i in 0..n {
                for j in 0..n {
                    eigenvalue += v[i] * remaining_matrix[i][j] * v[j];
                }
            }
            
            eigenvalues.push(eigenvalue);
            eigenvectors.push(v.clone());
            
            for i in 0..n {
                for j in 0..n {
                    remaining_matrix[i][j] -= eigenvalue * v[i] * v[j];
                }
            }
        }
        
        Ok((eigenvalues, eigenvectors))
    }
    
    fn euclidean_distance(a: &[f64], b: &[f64]) -> MathResult<f64> {
        if a.len() != b.len() {
            return Err(MathError::InvalidArgument("Vector dimensions mismatch".to_string()));
        }
        
        let distance = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt();
        
        Ok(distance)
    }
    
    pub fn confusion_matrix(true_labels: &[i32], predicted_labels: &[i32]) -> MathResult<HashMap<(i32, i32), i32>> {
        if true_labels.len() != predicted_labels.len() {
            return Err(MathError::InvalidArgument("Label arrays must have same length".to_string()));
        }
        
        let mut matrix = HashMap::new();
        
        for (&true_label, &pred_label) in true_labels.iter().zip(predicted_labels.iter()) {
            *matrix.entry((true_label, pred_label)).or_insert(0) += 1;
        }
        
        Ok(matrix)
    }
    
    pub fn accuracy_score(true_labels: &[i32], predicted_labels: &[i32]) -> MathResult<f64> {
        if true_labels.is_empty() || true_labels.len() != predicted_labels.len() {
            return Err(MathError::InvalidArgument("Invalid label arrays".to_string()));
        }
        
        let correct = true_labels.iter()
            .zip(predicted_labels.iter())
            .filter(|(&true_label, &pred_label)| true_label == pred_label)
            .count();
        
        Ok(correct as f64 / true_labels.len() as f64)
    }
}

impl MathDomain for MachineLearningDomain {
    fn name(&self) -> &str { "Machine Learning" }
    fn description(&self) -> &str { "Basic machine learning algorithms including regression, clustering, and classification" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "accuracy_score" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("accuracy_score requires 2 arguments".to_string()));
                }
                let true_labels = args[0].downcast_ref::<Vec<i32>>().ok_or_else(|| MathError::InvalidArgument("First argument must be Vec<i32>".to_string()))?;
                let pred_labels = args[1].downcast_ref::<Vec<i32>>().ok_or_else(|| MathError::InvalidArgument("Second argument must be Vec<i32>".to_string()))?;
                Ok(Box::new(Self::accuracy_score(true_labels, pred_labels)?))
            },
            "confusion_matrix" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("confusion_matrix requires 2 arguments".to_string()));
                }
                let true_labels = args[0].downcast_ref::<Vec<i32>>().ok_or_else(|| MathError::InvalidArgument("First argument must be Vec<i32>".to_string()))?;
                let pred_labels = args[1].downcast_ref::<Vec<i32>>().ok_or_else(|| MathError::InvalidArgument("Second argument must be Vec<i32>".to_string()))?;
                Ok(Box::new(Self::confusion_matrix(true_labels, pred_labels)?))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "linear_regression".to_string(),
            "predict_linear_regression".to_string(),
            "k_means_clustering".to_string(),
            "knn_classifier".to_string(),
            "knn_predict".to_string(),
            "principal_component_analysis".to_string(),
            "confusion_matrix".to_string(),
            "accuracy_score".to_string(),
        ]
    }
}