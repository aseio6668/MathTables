use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Dataset {
    pub values: Vec<f64>,
    pub name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Distribution {
    pub distribution_type: DistributionType,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub enum DistributionType {
    Normal,
    Uniform,
    Exponential,
    Poisson,
    Binomial,
}

#[derive(Debug, Clone)]
pub struct Functor<T, U> {
    pub source_category: String,
    pub target_category: String,
    pub object_mapping: HashMap<T, U>,
    pub morphism_mapping: HashMap<String, String>,
}

pub struct StatisticsDomain;

impl StatisticsDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn create_dataset(values: Vec<f64>, name: Option<String>) -> Dataset {
        Dataset { values, name }
    }
    
    pub fn mean(dataset: &Dataset) -> MathResult<f64> {
        if dataset.values.is_empty() {
            return Err(MathError::InvalidArgument("Cannot calculate mean of empty dataset".to_string()));
        }
        
        let sum: f64 = dataset.values.iter().sum();
        Ok(sum / dataset.values.len() as f64)
    }
    
    pub fn median(dataset: &Dataset) -> MathResult<f64> {
        if dataset.values.is_empty() {
            return Err(MathError::InvalidArgument("Cannot calculate median of empty dataset".to_string()));
        }
        
        let mut sorted = dataset.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted.len();
        if n % 2 == 0 {
            Ok((sorted[n/2 - 1] + sorted[n/2]) / 2.0)
        } else {
            Ok(sorted[n/2])
        }
    }
    
    pub fn mode(dataset: &Dataset) -> MathResult<Vec<f64>> {
        if dataset.values.is_empty() {
            return Err(MathError::InvalidArgument("Cannot calculate mode of empty dataset".to_string()));
        }
        
        let mut freq_map = HashMap::new();
        for &value in &dataset.values {
            *freq_map.entry(value.to_bits()).or_insert(0) += 1;
        }
        
        let max_freq = *freq_map.values().max().unwrap();
        let modes: Vec<f64> = freq_map.iter()
            .filter(|(_, &freq)| freq == max_freq)
            .map(|(&bits, _)| f64::from_bits(bits))
            .collect();
            
        Ok(modes)
    }
    
    pub fn variance(dataset: &Dataset) -> MathResult<f64> {
        if dataset.values.len() < 2 {
            return Err(MathError::InvalidArgument("Variance requires at least 2 data points".to_string()));
        }
        
        let mean_val = Self::mean(dataset)?;
        let sum_squares: f64 = dataset.values.iter()
            .map(|x| (x - mean_val).powi(2))
            .sum();
            
        Ok(sum_squares / (dataset.values.len() - 1) as f64)
    }
    
    pub fn standard_deviation(dataset: &Dataset) -> MathResult<f64> {
        Ok(Self::variance(dataset)?.sqrt())
    }
    
    pub fn correlation_coefficient(x: &Dataset, y: &Dataset) -> MathResult<f64> {
        if x.values.len() != y.values.len() {
            return Err(MathError::InvalidArgument("Datasets must have same length".to_string()));
        }
        
        if x.values.len() < 2 {
            return Err(MathError::InvalidArgument("Correlation requires at least 2 data points".to_string()));
        }
        
        let mean_x = Self::mean(x)?;
        let mean_y = Self::mean(y)?;
        
        let numerator: f64 = x.values.iter().zip(&y.values)
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();
            
        let sum_x_sq: f64 = x.values.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let sum_y_sq: f64 = y.values.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        
        let denominator = (sum_x_sq * sum_y_sq).sqrt();
        
        if denominator == 0.0 {
            return Err(MathError::ComputationError("Cannot compute correlation with zero variance".to_string()));
        }
        
        Ok(numerator / denominator)
    }
    
    pub fn z_score(value: f64, dataset: &Dataset) -> MathResult<f64> {
        let mean_val = Self::mean(dataset)?;
        let std_dev = Self::standard_deviation(dataset)?;
        
        if std_dev == 0.0 {
            return Err(MathError::ComputationError("Cannot compute z-score with zero standard deviation".to_string()));
        }
        
        Ok((value - mean_val) / std_dev)
    }
    
    pub fn create_normal_distribution(mean: f64, std_dev: f64) -> MathResult<Distribution> {
        if std_dev <= 0.0 {
            return Err(MathError::InvalidArgument("Standard deviation must be positive".to_string()));
        }
        
        let mut params = HashMap::new();
        params.insert("mean".to_string(), mean);
        params.insert("std_dev".to_string(), std_dev);
        
        Ok(Distribution {
            distribution_type: DistributionType::Normal,
            parameters: params,
        })
    }
    
    pub fn normal_pdf(x: f64, distribution: &Distribution) -> MathResult<f64> {
        match distribution.distribution_type {
            DistributionType::Normal => {
                let mean = distribution.parameters.get("mean").unwrap_or(&0.0);
                let std_dev = distribution.parameters.get("std_dev").unwrap_or(&1.0);
                
                let coefficient = 1.0 / (std_dev * (2.0 * std::f64::consts::PI).sqrt());
                let exponent = -0.5 * ((x - mean) / std_dev).powi(2);
                
                Ok(coefficient * exponent.exp())
            },
            _ => Err(MathError::InvalidArgument("PDF calculation only supported for normal distribution".to_string()))
        }
    }
    
    pub fn create_functor<T: Clone + Eq + std::hash::Hash, U: Clone>(
        source: &str,
        target: &str,
        object_map: HashMap<T, U>,
        morphism_map: HashMap<String, String>,
    ) -> Functor<T, U> {
        Functor {
            source_category: source.to_string(),
            target_category: target.to_string(),
            object_mapping: object_map,
            morphism_mapping: morphism_map,
        }
    }
}

impl MathDomain for StatisticsDomain {
    fn name(&self) -> &str { "Statistics and Functors" }
    fn description(&self) -> &str { "Statistical analysis, probability distributions, and category theory functors" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "create_dataset" => {
                if args.len() < 1 || args.len() > 2 {
                    return Err(MathError::InvalidArgument("create_dataset requires 1 or 2 arguments".to_string()));
                }
                let values = args[0].downcast_ref::<Vec<f64>>().ok_or_else(|| MathError::InvalidArgument("First argument must be Vec<f64>".to_string()))?;
                let name = if args.len() == 2 {
                    args[1].downcast_ref::<String>().map(|s| s.clone())
                } else {
                    None
                };
                Ok(Box::new(Self::create_dataset(values.clone(), name)))
            },
            "mean" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("mean requires 1 argument".to_string()));
                }
                let dataset = args[0].downcast_ref::<Dataset>().ok_or_else(|| MathError::InvalidArgument("Argument must be Dataset".to_string()))?;
                Ok(Box::new(Self::mean(dataset)?))
            },
            "median" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("median requires 1 argument".to_string()));
                }
                let dataset = args[0].downcast_ref::<Dataset>().ok_or_else(|| MathError::InvalidArgument("Argument must be Dataset".to_string()))?;
                Ok(Box::new(Self::median(dataset)?))
            },
            "variance" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("variance requires 1 argument".to_string()));
                }
                let dataset = args[0].downcast_ref::<Dataset>().ok_or_else(|| MathError::InvalidArgument("Argument must be Dataset".to_string()))?;
                Ok(Box::new(Self::variance(dataset)?))
            },
            "standard_deviation" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("standard_deviation requires 1 argument".to_string()));
                }
                let dataset = args[0].downcast_ref::<Dataset>().ok_or_else(|| MathError::InvalidArgument("Argument must be Dataset".to_string()))?;
                Ok(Box::new(Self::standard_deviation(dataset)?))
            },
            "create_normal_distribution" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("create_normal_distribution requires 2 arguments".to_string()));
                }
                let mean = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let std_dev = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                Ok(Box::new(Self::create_normal_distribution(*mean, *std_dev)?))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "create_dataset".to_string(),
            "mean".to_string(),
            "median".to_string(),
            "mode".to_string(),
            "variance".to_string(),
            "standard_deviation".to_string(),
            "correlation_coefficient".to_string(),
            "z_score".to_string(),
            "create_normal_distribution".to_string(),
            "normal_pdf".to_string(),
            "create_functor".to_string(),
        ]
    }
}