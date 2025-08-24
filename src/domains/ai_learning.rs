use crate::core::{
    MathDomain, MathResult, MathError,
    Tensor, AttentionField, TransformerConfig, LearningSchedule, OptimizerConfig
};
use std::f64::consts::PI;

pub struct AILearningDomain;

impl AILearningDomain {
    pub fn new() -> Self {
        Self
    }
}

// Tensor operations
impl Tensor {
    /// Create new tensor with given shape
    pub fn new(shape: Vec<usize>) -> Self {
        let total_size = shape.iter().product();
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride = 1;
        for &dim in shape.iter().rev() {
            strides.push(stride);
            stride *= dim;
        }
        strides.reverse();
        
        Tensor {
            data: vec![0.0; total_size],
            shape,
            strides,
        }
    }
    
    /// Create tensor filled with random values from normal distribution
    pub fn randn(shape: Vec<usize>, mean: f64, std: f64) -> Self {
        let total_size = shape.iter().product();
        let mut tensor = Self::new(shape);
        
        // Box-Muller transform for normal distribution
        for i in (0..total_size).step_by(2) {
            let u1: f64 = (i as f64 + 1.0) / (total_size as f64 + 1.0);
            let u2: f64 = (i as f64 + 2.0) / (total_size as f64 + 1.0);
            
            let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            let z2 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).sin();
            
            tensor.data[i] = mean + std * z1;
            if i + 1 < total_size {
                tensor.data[i + 1] = mean + std * z2;
            }
        }
        
        tensor
    }
    
    /// Reshape tensor (must preserve total size)
    pub fn reshape(&self, new_shape: Vec<usize>) -> MathResult<Self> {
        let old_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();
        
        if old_size != new_size {
            return Err(MathError::InvalidArgument("Cannot reshape: size mismatch".to_string()));
        }
        
        let mut new_strides = Vec::with_capacity(new_shape.len());
        let mut stride = 1;
        for &dim in new_shape.iter().rev() {
            new_strides.push(stride);
            stride *= dim;
        }
        new_strides.reverse();
        
        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
        })
    }
    
    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> MathResult<Tensor> {
        if self.shape != other.shape {
            return Err(MathError::InvalidArgument("Shape mismatch for addition".to_string()));
        }
        
        let mut result = self.clone();
        for (i, &val) in other.data.iter().enumerate() {
            result.data[i] += val;
        }
        
        Ok(result)
    }
    
    /// Matrix multiplication (2D tensors only)
    pub fn matmul(&self, other: &Tensor) -> MathResult<Tensor> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(MathError::InvalidArgument("Matrix multiplication requires 2D tensors".to_string()));
        }
        
        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        
        if k1 != k2 {
            return Err(MathError::InvalidArgument("Inner dimensions must match".to_string()));
        }
        
        let mut result = Tensor::new(vec![m, n]);
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..k1 {
                    sum += self.data[i * k1 + k] * other.data[k * n + j];
                }
                result.data[i * n + j] = sum;
            }
        }
        
        Ok(result)
    }
}

impl AILearningDomain {
    /// Softmax activation function
    /// softmax(x_i) = exp(x_i) / Σ exp(x_j)
    pub fn softmax(logits: &[f64], temperature: f64) -> MathResult<Vec<f64>> {
        if logits.is_empty() {
            return Err(MathError::InvalidArgument("Logits cannot be empty".to_string()));
        }
        if temperature <= 0.0 {
            return Err(MathError::InvalidArgument("Temperature must be positive".to_string()));
        }
        
        // Numerically stable softmax
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let scaled_logits: Vec<f64> = logits.iter().map(|x| (x - max_logit) / temperature).collect();
        
        let exp_sum: f64 = scaled_logits.iter().map(|x| x.exp()).sum();
        let probabilities: Vec<f64> = scaled_logits.iter().map(|x| x.exp() / exp_sum).collect();
        
        Ok(probabilities)
    }
    
    /// ReLU activation function
    pub fn relu(x: f64) -> f64 {
        x.max(0.0)
    }
    
    /// GELU activation function (Gaussian Error Linear Unit)
    /// GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    pub fn gelu(x: f64) -> f64 {
        let sqrt_2_pi = (2.0 / PI).sqrt();
        let tanh_arg = sqrt_2_pi * (x + 0.044715 * x.powi(3));
        0.5 * x * (1.0 + tanh_arg.tanh())
    }
    
    /// Swish/SiLU activation function
    /// Swish(x) = x * sigmoid(x)
    pub fn swish(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }
    
    /// Cross-entropy loss between predicted probabilities and true labels
    /// CE = -Σ y_i * log(p_i)
    pub fn cross_entropy_loss(predictions: &[f64], targets: &[f64]) -> MathResult<f64> {
        if predictions.len() != targets.len() {
            return Err(MathError::InvalidArgument("Predictions and targets must have same length".to_string()));
        }
        
        let mut loss = 0.0;
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            if *pred <= 0.0 || *pred >= 1.0 {
                return Err(MathError::InvalidArgument("Predictions must be in (0,1)".to_string()));
            }
            loss -= target * pred.ln();
        }
        
        Ok(loss)
    }
    
    /// Mean squared error loss
    /// MSE = (1/n) * Σ (y_i - ŷ_i)²
    pub fn mse_loss(predictions: &[f64], targets: &[f64]) -> MathResult<f64> {
        if predictions.len() != targets.len() {
            return Err(MathError::InvalidArgument("Predictions and targets must have same length".to_string()));
        }
        
        let n = predictions.len() as f64;
        let mse = predictions.iter()
            .zip(targets.iter())
            .map(|(pred, target)| (pred - target).powi(2))
            .sum::<f64>() / n;
        
        Ok(mse)
    }
    
    /// Scaled dot-product attention
    /// Attention(Q,K,V) = softmax(QK^T / √d_k) * V
    pub fn scaled_dot_product_attention(
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        scale: f64
    ) -> MathResult<Tensor> {
        if queries.shape.len() != 2 || keys.shape.len() != 2 || values.shape.len() != 2 {
            return Err(MathError::InvalidArgument("Q, K, V must be 2D tensors".to_string()));
        }
        
        let d_k = keys.shape[1];
        if queries.shape[1] != d_k || values.shape[0] != keys.shape[0] {
            return Err(MathError::InvalidArgument("Incompatible attention dimensions".to_string()));
        }
        
        // Compute attention scores: Q @ K^T
        let keys_t = Self::transpose_2d(keys)?;
        let scores = queries.matmul(&keys_t)?;
        
        // Scale scores
        let mut scaled_scores = scores.clone();
        let scaling_factor = scale / (d_k as f64).sqrt();
        for score in &mut scaled_scores.data {
            *score *= scaling_factor;
        }
        
        // Apply softmax to each row
        let seq_len = scaled_scores.shape[1];
        for i in 0..scaled_scores.shape[0] {
            let row_start = i * seq_len;
            let row_end = row_start + seq_len;
            let row_scores = &scaled_scores.data[row_start..row_end];
            let attention_weights = Self::softmax(row_scores, 1.0)?;
            
            for (j, &weight) in attention_weights.iter().enumerate() {
                scaled_scores.data[row_start + j] = weight;
            }
        }
        
        // Apply attention weights to values
        let output = scaled_scores.matmul(values)?;
        Ok(output)
    }
    
    /// Transpose 2D tensor
    fn transpose_2d(tensor: &Tensor) -> MathResult<Tensor> {
        if tensor.shape.len() != 2 {
            return Err(MathError::InvalidArgument("Transpose requires 2D tensor".to_string()));
        }
        
        let (m, n) = (tensor.shape[0], tensor.shape[1]);
        let mut result = Tensor::new(vec![n, m]);
        
        for i in 0..m {
            for j in 0..n {
                result.data[j * m + i] = tensor.data[i * n + j];
            }
        }
        
        Ok(result)
    }
    
    /// Layer normalization
    /// LayerNorm(x) = γ * (x - μ) / σ + β
    pub fn layer_norm(
        input: &[f64],
        gamma: &[f64],
        beta: &[f64],
        epsilon: f64
    ) -> MathResult<Vec<f64>> {
        if input.len() != gamma.len() || input.len() != beta.len() {
            return Err(MathError::InvalidArgument("Input, gamma, and beta must have same length".to_string()));
        }
        
        let n = input.len() as f64;
        let mean = input.iter().sum::<f64>() / n;
        let variance = input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = (variance + epsilon).sqrt();
        
        let normalized: Vec<f64> = input.iter()
            .zip(gamma.iter())
            .zip(beta.iter())
            .map(|((x, g), b)| g * (x - mean) / std_dev + b)
            .collect();
        
        Ok(normalized)
    }
    
    /// Batch normalization (simplified version)
    pub fn batch_norm(
        batch: &[Vec<f64>],
        gamma: f64,
        beta: f64,
        epsilon: f64
    ) -> MathResult<Vec<Vec<f64>>> {
        if batch.is_empty() {
            return Err(MathError::InvalidArgument("Batch cannot be empty".to_string()));
        }
        
        let batch_size = batch.len();
        let feature_size = batch[0].len();
        
        // Check all samples have same feature size
        for sample in batch {
            if sample.len() != feature_size {
                return Err(MathError::InvalidArgument("All samples must have same feature size".to_string()));
            }
        }
        
        let mut normalized_batch = vec![vec![0.0; feature_size]; batch_size];
        
        // Compute mean and variance for each feature
        for feat_idx in 0..feature_size {
            let feature_values: Vec<f64> = batch.iter().map(|sample| sample[feat_idx]).collect();
            let mean = feature_values.iter().sum::<f64>() / batch_size as f64;
            let variance = feature_values.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / batch_size as f64;
            let std_dev = (variance + epsilon).sqrt();
            
            // Normalize each sample's feature
            for (sample_idx, sample) in batch.iter().enumerate() {
                normalized_batch[sample_idx][feat_idx] = 
                    gamma * (sample[feat_idx] - mean) / std_dev + beta;
            }
        }
        
        Ok(normalized_batch)
    }
    
    /// Dropout simulation (returns mask)
    pub fn dropout_mask(size: usize, dropout_rate: f64) -> MathResult<Vec<f64>> {
        if dropout_rate < 0.0 || dropout_rate >= 1.0 {
            return Err(MathError::InvalidArgument("Dropout rate must be in [0,1)".to_string()));
        }
        
        let keep_prob = 1.0 - dropout_rate;
        let mut mask = Vec::with_capacity(size);
        
        // Simple pseudo-random number generation
        for i in 0..size {
            let pseudo_rand = ((i * 1664525 + 1013904223) % (1 << 32)) as f64 / (1u64 << 32) as f64;
            mask.push(if pseudo_rand < keep_prob { 1.0 / keep_prob } else { 0.0 });
        }
        
        Ok(mask)
    }
    
    /// Adam optimizer step
    /// Updates: m_t = β₁*m_{t-1} + (1-β₁)*g_t
    ///         v_t = β₂*v_{t-1} + (1-β₂)*g_t²
    ///         θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
    pub fn adam_update(
        params: &mut [f64],
        gradients: &[f64],
        m_state: &mut [f64],
        v_state: &mut [f64],
        config: &OptimizerConfig,
        step: usize
    ) -> MathResult<()> {
        if params.len() != gradients.len() || params.len() != m_state.len() || params.len() != v_state.len() {
            return Err(MathError::InvalidArgument("All arrays must have same length".to_string()));
        }
        
        let bias_correction1 = 1.0 - config.beta1.powi(step as i32);
        let bias_correction2 = 1.0 - config.beta2.powi(step as i32);
        
        for i in 0..params.len() {
            // Update biased first moment estimate
            m_state[i] = config.beta1 * m_state[i] + (1.0 - config.beta1) * gradients[i];
            
            // Update biased second raw moment estimate
            v_state[i] = config.beta2 * v_state[i] + (1.0 - config.beta2) * gradients[i].powi(2);
            
            // Compute bias-corrected first moment estimate
            let m_hat = m_state[i] / bias_correction1;
            
            // Compute bias-corrected second raw moment estimate
            let v_hat = v_state[i] / bias_correction2;
            
            // Update parameters
            params[i] -= config.learning_rate * m_hat / (v_hat.sqrt() + config.epsilon);
            
            // Apply weight decay (L2 regularization)
            if config.weight_decay > 0.0 {
                params[i] *= 1.0 - config.learning_rate * config.weight_decay;
            }
        }
        
        Ok(())
    }
    
    /// Learning rate schedule evaluation
    pub fn evaluate_learning_schedule(
        schedule: &LearningSchedule,
        step: usize,
        total_steps: usize
    ) -> f64 {
        match schedule {
            LearningSchedule::Constant(rate) => *rate,
            LearningSchedule::Exponential { initial, decay } => {
                initial * decay.powf(step as f64)
            },
            LearningSchedule::Cosine { initial, min_rate } => {
                let progress = step as f64 / total_steps as f64;
                min_rate + 0.5 * (initial - min_rate) * (1.0 + (PI * progress).cos())
            },
            LearningSchedule::Polynomial { initial, power } => {
                initial * (1.0 - step as f64 / total_steps as f64).powf(*power)
            },
            LearningSchedule::WarmupCosine { warmup_steps, max_rate, min_rate } => {
                if step < *warmup_steps {
                    // Linear warmup
                    max_rate * step as f64 / *warmup_steps as f64
                } else {
                    // Cosine annealing
                    let progress = (step - warmup_steps) as f64 / (total_steps - warmup_steps) as f64;
                    min_rate + 0.5 * (max_rate - min_rate) * (1.0 + (PI * progress).cos())
                }
            }
        }
    }
    
    /// Gradient clipping by norm
    pub fn clip_gradients_by_norm(gradients: &mut [f64], max_norm: f64) -> MathResult<f64> {
        if max_norm <= 0.0 {
            return Err(MathError::InvalidArgument("Max norm must be positive".to_string()));
        }
        
        let grad_norm = gradients.iter().map(|g| g.powi(2)).sum::<f64>().sqrt();
        
        if grad_norm > max_norm {
            let scale = max_norm / grad_norm;
            for grad in gradients {
                *grad *= scale;
            }
        }
        
        Ok(grad_norm)
    }
    
    /// Cosine similarity between two vectors
    /// cos_sim(a,b) = (a·b) / (||a|| * ||b||)
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> MathResult<f64> {
        if a.len() != b.len() {
            return Err(MathError::InvalidArgument("Vectors must have same length".to_string()));
        }
        
        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let norm_b = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return Err(MathError::InvalidArgument("Cannot compute similarity with zero vector".to_string()));
        }
        
        Ok(dot_product / (norm_a * norm_b))
    }
    
    /// Principal Component Analysis (simplified eigenvalue approach)
    pub fn pca_transform_2d(
        data: &[Vec<f64>],
        n_components: usize
    ) -> MathResult<Vec<Vec<f64>>> {
        if data.is_empty() || n_components == 0 {
            return Err(MathError::InvalidArgument("Data cannot be empty and components > 0".to_string()));
        }
        
        let n_samples = data.len();
        let n_features = data[0].len();
        
        if n_components > n_features {
            return Err(MathError::InvalidArgument("Components cannot exceed features".to_string()));
        }
        
        // Center the data
        let mut means = vec![0.0; n_features];
        for sample in data {
            for (i, &val) in sample.iter().enumerate() {
                means[i] += val;
            }
        }
        for mean in &mut means {
            *mean /= n_samples as f64;
        }
        
        let mut centered_data = Vec::new();
        for sample in data {
            let centered: Vec<f64> = sample.iter()
                .zip(&means)
                .map(|(val, mean)| val - mean)
                .collect();
            centered_data.push(centered);
        }
        
        // For simplicity, return first n_components features (not true PCA)
        // Real PCA would require eigenvalue decomposition
        let transformed: Vec<Vec<f64>> = centered_data.iter()
            .map(|sample| sample[..n_components].to_vec())
            .collect();
        
        Ok(transformed)
    }
}

impl MathDomain for AILearningDomain {
    fn name(&self) -> &str {
        "AI Learning"
    }
    
    fn description(&self) -> &str {
        "Mathematical foundations for artificial intelligence and machine learning including optimization, neural networks, and attention mechanisms"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "softmax".to_string(),
            "relu".to_string(),
            "gelu".to_string(),
            "swish".to_string(),
            "cross_entropy_loss".to_string(),
            "mse_loss".to_string(),
            "layer_norm".to_string(),
            "cosine_similarity".to_string(),
        ]
    }
    
    fn compute(&self, _operation: &str, _args: &[&dyn std::any::Any]) -> MathResult<Box<dyn std::any::Any>> {
        Err(MathError::NotImplemented("Generic compute not implemented for AI Learning domain".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tensor_operations() {
        let tensor = Tensor::new(vec![2, 3]);
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.data.len(), 6);
        
        let reshaped = tensor.reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped.shape, vec![3, 2]);
    }

    #[test]
    fn test_activation_functions() {
        // ReLU
        assert_eq!(AILearningDomain::relu(1.0), 1.0);
        assert_eq!(AILearningDomain::relu(-1.0), 0.0);
        
        // GELU should be smooth
        let gelu_pos = AILearningDomain::gelu(1.0);
        let gelu_neg = AILearningDomain::gelu(-1.0);
        assert!(gelu_pos > 0.8);
        assert!(gelu_neg < -0.1);
        
        // Swish should be x for large positive x
        let swish_large = AILearningDomain::swish(10.0);
        assert_relative_eq!(swish_large, 10.0, epsilon = 1e-3);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = AILearningDomain::softmax(&logits, 1.0).unwrap();
        
        // Probabilities should sum to 1
        let sum: f64 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        
        // Largest logit should have largest probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_loss_functions() {
        let predictions = vec![0.7, 0.2, 0.1];
        let targets = vec![1.0, 0.0, 0.0];
        
        let ce_loss = AILearningDomain::cross_entropy_loss(&predictions, &targets).unwrap();
        assert!(ce_loss > 0.0);
        
        let mse_loss = AILearningDomain::mse_loss(&predictions, &targets).unwrap();
        assert!(mse_loss > 0.0);
    }

    #[test]
    fn test_layer_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];
        
        let normalized = AILearningDomain::layer_norm(&input, &gamma, &beta, 1e-5).unwrap();
        
        // Mean should be approximately 0
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = AILearningDomain::cosine_similarity(&a, &b).unwrap();
        assert_relative_eq!(sim, 1.0, epsilon = 1e-10);
        
        let c = vec![0.0, 1.0, 0.0];
        let sim2 = AILearningDomain::cosine_similarity(&a, &c).unwrap();
        assert_relative_eq!(sim2, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dropout_mask() {
        let mask = AILearningDomain::dropout_mask(100, 0.5).unwrap();
        assert_eq!(mask.len(), 100);
        
        let num_zeros = mask.iter().filter(|&&x| x == 0.0).count();
        let num_nonzeros = mask.len() - num_zeros;
        
        // Should have roughly 50% dropout
        assert!(num_zeros > 30 && num_zeros < 70);
        assert!(num_nonzeros > 30);
    }

    #[test]
    fn test_learning_schedules() {
        let constant = LearningSchedule::Constant(0.001);
        assert_eq!(AILearningDomain::evaluate_learning_schedule(&constant, 100, 1000), 0.001);
        
        let exponential = LearningSchedule::Exponential { initial: 0.01, decay: 0.9 };
        let lr = AILearningDomain::evaluate_learning_schedule(&exponential, 10, 1000);
        assert!(lr < 0.01);
        
        let cosine = LearningSchedule::Cosine { initial: 0.01, min_rate: 0.0001 };
        let lr_start = AILearningDomain::evaluate_learning_schedule(&cosine, 0, 1000);
        let lr_end = AILearningDomain::evaluate_learning_schedule(&cosine, 1000, 1000);
        assert_relative_eq!(lr_start, 0.01, epsilon = 1e-6);
        assert_relative_eq!(lr_end, 0.0001, epsilon = 1e-6);
    }

    #[test]
    fn test_gradient_clipping() {
        let mut gradients = vec![10.0, -5.0, 3.0];
        let norm = AILearningDomain::clip_gradients_by_norm(&mut gradients, 5.0).unwrap();
        
        assert!(norm > 10.0); // Original norm
        
        let new_norm = gradients.iter().map(|g| g.powi(2)).sum::<f64>().sqrt();
        assert_relative_eq!(new_norm, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tensor_matrix_multiplication() {
        let a = Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
            strides: vec![2, 1],
        };
        let b = Tensor {
            data: vec![2.0, 0.0, 1.0, 3.0],
            shape: vec![2, 2],
            strides: vec![2, 1],
        };
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        // [1,2] @ [2,0] = [4, 6]
        // [3,4]   [1,3]   [10,12]
        assert_eq!(result.data, vec![4.0, 6.0, 10.0, 12.0]);
    }
}