use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct FuzzySet {
    pub universe: Vec<f64>,
    pub membership_function: Vec<f64>, // membership values for each element in universe
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct FuzzyNumber {
    pub alpha_cuts: HashMap<u32, (f64, f64)>, // alpha level -> (left, right) bounds
    pub support: (f64, f64),
    pub core: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct LinguisticVariable {
    pub name: String,
    pub universe: (f64, f64),
    pub terms: HashMap<String, FuzzySet>,
}

#[derive(Debug, Clone)]
pub struct FuzzyRule {
    pub conditions: Vec<(String, String)>, // (variable_name, term_name)
    pub conclusion: (String, String),
    pub certainty_factor: f64,
}

#[derive(Debug, Clone)]
pub struct FuzzyInferenceSystem {
    pub variables: HashMap<String, LinguisticVariable>,
    pub rules: Vec<FuzzyRule>,
    pub defuzzification_method: DefuzzificationMethod,
}

#[derive(Debug, Clone)]
pub enum DefuzzificationMethod {
    Centroid,
    BisectorOfArea,
    MeanOfMaxima,
    SmallestOfMaxima,
    LargestOfMaxima,
}

#[derive(Debug, Clone)]
pub enum TNorm {
    Minimum,
    Product,
    LukasiewiczTNorm,
    DrasticProduct,
    NilpotentMinimum,
    HamacherProduct(f64), // parameter γ
}

#[derive(Debug, Clone)]
pub enum SNorm {
    Maximum,
    ProbabilisticSum,
    LukasiewiczSNorm,
    DrasticSum,
    NilpotentMaximum,
    HamacherSum(f64), // parameter γ
}

#[derive(Debug, Clone)]
pub struct FuzzyRelation {
    pub domain1: Vec<f64>,
    pub domain2: Vec<f64>,
    pub relation_matrix: Vec<Vec<f64>>,
}

pub struct FuzzyLogicDomain;

impl FuzzySet {
    pub fn new(name: String, universe: Vec<f64>) -> Self {
        let membership_function = vec![0.0; universe.len()];
        FuzzySet {
            universe,
            membership_function,
            name,
        }
    }
    
    pub fn triangular(name: String, universe: Vec<f64>, a: f64, b: f64, c: f64) -> MathResult<Self> {
        if a >= b || b >= c {
            return Err(MathError::InvalidArgument("Parameters must satisfy a < b < c".to_string()));
        }
        
        let membership_function: Vec<f64> = universe.iter().map(|&x| {
            if x <= a || x >= c {
                0.0
            } else if x <= b {
                (x - a) / (b - a)
            } else {
                (c - x) / (c - b)
            }
        }).collect();
        
        Ok(FuzzySet {
            universe,
            membership_function,
            name,
        })
    }
    
    pub fn trapezoidal(name: String, universe: Vec<f64>, a: f64, b: f64, c: f64, d: f64) -> MathResult<Self> {
        if a >= b || b >= c || c >= d {
            return Err(MathError::InvalidArgument("Parameters must satisfy a < b ≤ c < d".to_string()));
        }
        
        let membership_function: Vec<f64> = universe.iter().map(|&x| {
            if x <= a || x >= d {
                0.0
            } else if x <= b {
                (x - a) / (b - a)
            } else if x <= c {
                1.0
            } else {
                (d - x) / (d - c)
            }
        }).collect();
        
        Ok(FuzzySet {
            universe,
            membership_function,
            name,
        })
    }
    
    pub fn gaussian(name: String, universe: Vec<f64>, center: f64, sigma: f64) -> MathResult<Self> {
        if sigma <= 0.0 {
            return Err(MathError::InvalidArgument("Sigma must be positive".to_string()));
        }
        
        let membership_function: Vec<f64> = universe.iter().map(|&x| {
            (-(x - center).powi(2) / (2.0 * sigma.powi(2))).exp()
        }).collect();
        
        Ok(FuzzySet {
            universe,
            membership_function,
            name,
        })
    }
    
    pub fn membership(&self, x: f64) -> f64 {
        if let Some(index) = self.universe.iter().position(|&val| (val - x).abs() < 1e-10) {
            self.membership_function[index]
        } else {
            // Interpolate if not exact match
            for i in 0..self.universe.len() - 1 {
                if self.universe[i] <= x && x <= self.universe[i + 1] {
                    let ratio = (x - self.universe[i]) / (self.universe[i + 1] - self.universe[i]);
                    return self.membership_function[i] + 
                           ratio * (self.membership_function[i + 1] - self.membership_function[i]);
                }
            }
            0.0
        }
    }
    
    pub fn union(&self, other: &FuzzySet, s_norm: &SNorm) -> MathResult<FuzzySet> {
        if self.universe != other.universe {
            return Err(MathError::InvalidArgument("Fuzzy sets must have same universe".to_string()));
        }
        
        let membership_function: Vec<f64> = self.membership_function.iter()
            .zip(other.membership_function.iter())
            .map(|(&a, &b)| Self::apply_s_norm(a, b, s_norm))
            .collect();
        
        Ok(FuzzySet {
            universe: self.universe.clone(),
            membership_function,
            name: format!("{} ∪ {}", self.name, other.name),
        })
    }
    
    pub fn intersection(&self, other: &FuzzySet, t_norm: &TNorm) -> MathResult<FuzzySet> {
        if self.universe != other.universe {
            return Err(MathError::InvalidArgument("Fuzzy sets must have same universe".to_string()));
        }
        
        let membership_function: Vec<f64> = self.membership_function.iter()
            .zip(other.membership_function.iter())
            .map(|(&a, &b)| Self::apply_t_norm(a, b, t_norm))
            .collect();
        
        Ok(FuzzySet {
            universe: self.universe.clone(),
            membership_function,
            name: format!("{} ∩ {}", self.name, other.name),
        })
    }
    
    pub fn complement(&self) -> FuzzySet {
        let membership_function: Vec<f64> = self.membership_function.iter()
            .map(|&val| 1.0 - val)
            .collect();
        
        FuzzySet {
            universe: self.universe.clone(),
            membership_function,
            name: format!("¬{}", self.name),
        }
    }
    
    pub fn alpha_cut(&self, alpha: f64) -> Vec<f64> {
        if alpha < 0.0 || alpha > 1.0 {
            return Vec::new();
        }
        
        self.universe.iter()
            .zip(self.membership_function.iter())
            .filter_map(|(&x, &membership)| {
                if membership >= alpha {
                    Some(x)
                } else {
                    None
                }
            })
            .collect()
    }
    
    pub fn support(&self) -> Vec<f64> {
        self.alpha_cut(0.0)
    }
    
    pub fn core(&self) -> Vec<f64> {
        self.alpha_cut(1.0)
    }
    
    pub fn height(&self) -> f64 {
        self.membership_function.iter().fold(0.0, |max, &val| max.max(val))
    }
    
    pub fn cardinality(&self) -> f64 {
        self.membership_function.iter().sum()
    }
    
    fn apply_t_norm(a: f64, b: f64, t_norm: &TNorm) -> f64 {
        match t_norm {
            TNorm::Minimum => a.min(b),
            TNorm::Product => a * b,
            TNorm::LukasiewiczTNorm => (a + b - 1.0).max(0.0),
            TNorm::DrasticProduct => {
                if a == 1.0 { b }
                else if b == 1.0 { a }
                else { 0.0 }
            },
            TNorm::NilpotentMinimum => {
                if a + b > 1.0 { a.min(b) }
                else { 0.0 }
            },
            TNorm::HamacherProduct(gamma) => {
                if a == 0.0 && b == 0.0 { 0.0 }
                else { (a * b) / (gamma + (1.0 - gamma) * (a + b - a * b)) }
            },
        }
    }
    
    fn apply_s_norm(a: f64, b: f64, s_norm: &SNorm) -> f64 {
        match s_norm {
            SNorm::Maximum => a.max(b),
            SNorm::ProbabilisticSum => a + b - a * b,
            SNorm::LukasiewiczSNorm => (a + b).min(1.0),
            SNorm::DrasticSum => {
                if a == 0.0 { b }
                else if b == 0.0 { a }
                else { 1.0 }
            },
            SNorm::NilpotentMaximum => {
                if a + b < 1.0 { a.max(b) }
                else { 1.0 }
            },
            SNorm::HamacherSum(gamma) => {
                (a + b + (gamma - 2.0) * a * b) / (1.0 + (gamma - 1.0) * a * b)
            },
        }
    }
}

impl FuzzyLogicDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn create_linguistic_variable(name: String, universe: (f64, f64)) -> LinguisticVariable {
        LinguisticVariable {
            name,
            universe,
            terms: HashMap::new(),
        }
    }
    
    pub fn mamdani_inference(
        system: &FuzzyInferenceSystem,
        inputs: &HashMap<String, f64>
    ) -> MathResult<HashMap<String, f64>> {
        let mut output_sets: HashMap<String, Vec<FuzzySet>> = HashMap::new();
        
        // Apply each rule
        for rule in &system.rules {
            let mut activation_level: f64 = 1.0;
            
            // Calculate activation level for rule conditions
            for (var_name, term_name) in &rule.conditions {
                if let (Some(input_value), Some(variable)) = (inputs.get(var_name), system.variables.get(var_name)) {
                    if let Some(term) = variable.terms.get(term_name) {
                        let membership = term.membership(*input_value);
                        activation_level = activation_level.min(membership);
                    }
                }
            }
            
            activation_level *= rule.certainty_factor;
            
            // Apply implication to conclusion
            let (output_var, output_term) = &rule.conclusion;
            if let Some(variable) = system.variables.get(output_var) {
                if let Some(term) = variable.terms.get(output_term) {
                    let implied_set = Self::implication_min(term, activation_level);
                    output_sets.entry(output_var.clone()).or_insert_with(Vec::new).push(implied_set);
                }
            }
        }
        
        // Aggregate and defuzzify
        let mut results = HashMap::new();
        for (var_name, sets) in output_sets {
            let aggregated = Self::aggregate_sets(&sets)?;
            let defuzzified = Self::defuzzify(&aggregated, &system.defuzzification_method)?;
            results.insert(var_name, defuzzified);
        }
        
        Ok(results)
    }
    
    fn implication_min(antecedent: &FuzzySet, activation: f64) -> FuzzySet {
        let membership_function: Vec<f64> = antecedent.membership_function.iter()
            .map(|&val| val.min(activation))
            .collect();
        
        FuzzySet {
            universe: antecedent.universe.clone(),
            membership_function,
            name: format!("implied_{}", antecedent.name),
        }
    }
    
    fn aggregate_sets(sets: &[FuzzySet]) -> MathResult<FuzzySet> {
        if sets.is_empty() {
            return Err(MathError::InvalidArgument("Cannot aggregate empty set list".to_string()));
        }
        
        let mut result = sets[0].clone();
        
        for set in sets.iter().skip(1) {
            result = result.union(set, &SNorm::Maximum)?;
        }
        
        Ok(result)
    }
    
    fn defuzzify(set: &FuzzySet, method: &DefuzzificationMethod) -> MathResult<f64> {
        match method {
            DefuzzificationMethod::Centroid => {
                let numerator: f64 = set.universe.iter()
                    .zip(set.membership_function.iter())
                    .map(|(&x, &membership)| x * membership)
                    .sum();
                
                let denominator: f64 = set.membership_function.iter().sum();
                
                if denominator == 0.0 {
                    Ok(0.0)
                } else {
                    Ok(numerator / denominator)
                }
            },
            DefuzzificationMethod::MeanOfMaxima => {
                let max_membership = set.height();
                let max_elements: Vec<f64> = set.universe.iter()
                    .zip(set.membership_function.iter())
                    .filter_map(|(&x, &membership)| {
                        if (membership - max_membership).abs() < 1e-10 {
                            Some(x)
                        } else {
                            None
                        }
                    })
                    .collect();
                
                if max_elements.is_empty() {
                    Ok(0.0)
                } else {
                    Ok(max_elements.iter().sum::<f64>() / max_elements.len() as f64)
                }
            },
            DefuzzificationMethod::SmallestOfMaxima => {
                let max_membership = set.height();
                set.universe.iter()
                    .zip(set.membership_function.iter())
                    .find(|(_, &membership)| (membership - max_membership).abs() < 1e-10)
                    .map(|(&x, _)| x)
                    .ok_or_else(|| MathError::ComputationError("No maximum found".to_string()))
            },
            DefuzzificationMethod::LargestOfMaxima => {
                let max_membership = set.height();
                set.universe.iter()
                    .zip(set.membership_function.iter())
                    .rev()
                    .find(|(_, &membership)| (membership - max_membership).abs() < 1e-10)
                    .map(|(&x, _)| x)
                    .ok_or_else(|| MathError::ComputationError("No maximum found".to_string()))
            },
            DefuzzificationMethod::BisectorOfArea => {
                let total_area: f64 = set.membership_function.iter().sum();
                let target_area = total_area / 2.0;
                
                let mut cumulative_area = 0.0;
                for (i, &membership) in set.membership_function.iter().enumerate() {
                    cumulative_area += membership;
                    if cumulative_area >= target_area {
                        return Ok(set.universe[i]);
                    }
                }
                
                Ok(set.universe.last().copied().unwrap_or(0.0))
            },
        }
    }
    
    pub fn sugeno_inference(
        system: &FuzzyInferenceSystem,
        inputs: &HashMap<String, f64>
    ) -> MathResult<f64> {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        
        for rule in &system.rules {
            let mut activation_level: f64 = 1.0;
            
            // Calculate activation level
            for (var_name, term_name) in &rule.conditions {
                if let (Some(input_value), Some(variable)) = (inputs.get(var_name), system.variables.get(var_name)) {
                    if let Some(term) = variable.terms.get(term_name) {
                        let membership = term.membership(*input_value);
                        activation_level = activation_level.min(membership);
                    }
                }
            }
            
            activation_level *= rule.certainty_factor;
            
            // For simplified Sugeno, assume conclusion is a constant
            // In practice, this would be a linear function of inputs
            let conclusion_value = 1.0; // Placeholder
            
            weighted_sum += activation_level * conclusion_value;
            total_weight += activation_level;
        }
        
        if total_weight == 0.0 {
            Ok(0.0)
        } else {
            Ok(weighted_sum / total_weight)
        }
    }
    
    pub fn fuzzy_composition(relation1: &FuzzyRelation, relation2: &FuzzyRelation) -> MathResult<FuzzyRelation> {
        if relation1.domain2 != relation2.domain1 {
            return Err(MathError::InvalidArgument("Relation domains must be compatible".to_string()));
        }
        
        let rows = relation1.domain1.len();
        let cols = relation2.domain2.len();
        let intermediate = relation1.domain2.len();
        
        let mut result_matrix = vec![vec![0.0; cols]; rows];
        
        for i in 0..rows {
            for j in 0..cols {
                let mut max_val: f64 = 0.0;
                for k in 0..intermediate {
                    let min_val = relation1.relation_matrix[i][k].min(relation2.relation_matrix[k][j]);
                    max_val = max_val.max(min_val);
                }
                result_matrix[i][j] = max_val;
            }
        }
        
        Ok(FuzzyRelation {
            domain1: relation1.domain1.clone(),
            domain2: relation2.domain2.clone(),
            relation_matrix: result_matrix,
        })
    }
    
    pub fn similarity_measure(set1: &FuzzySet, set2: &FuzzySet) -> MathResult<f64> {
        if set1.universe != set2.universe {
            return Err(MathError::InvalidArgument("Fuzzy sets must have same universe".to_string()));
        }
        
        let intersection_sum: f64 = set1.membership_function.iter()
            .zip(set2.membership_function.iter())
            .map(|(&a, &b)| a.min(b))
            .sum();
        
        let union_sum: f64 = set1.membership_function.iter()
            .zip(set2.membership_function.iter())
            .map(|(&a, &b)| a.max(b))
            .sum();
        
        if union_sum == 0.0 {
            Ok(1.0) // Both sets are empty
        } else {
            Ok(intersection_sum / union_sum)
        }
    }
    
    pub fn distance_measure(set1: &FuzzySet, set2: &FuzzySet) -> MathResult<f64> {
        if set1.universe != set2.universe {
            return Err(MathError::InvalidArgument("Fuzzy sets must have same universe".to_string()));
        }
        
        let distance: f64 = set1.membership_function.iter()
            .zip(set2.membership_function.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        
        Ok(distance / set1.universe.len() as f64)
    }
}

impl MathDomain for FuzzyLogicDomain {
    fn name(&self) -> &str { "Fuzzy Logic" }
    fn description(&self) -> &str { "Fuzzy sets, fuzzy inference systems, linguistic variables, and fuzzy reasoning" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, _args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            _ => Err(MathError::InvalidOperation(format!("Operation {} not implemented in compute interface", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "create_fuzzy_set".to_string(),
            "triangular_membership".to_string(),
            "trapezoidal_membership".to_string(),
            "gaussian_membership".to_string(),
            "fuzzy_union".to_string(),
            "fuzzy_intersection".to_string(),
            "fuzzy_complement".to_string(),
            "alpha_cut".to_string(),
            "fuzzy_support".to_string(),
            "fuzzy_core".to_string(),
            "mamdani_inference".to_string(),
            "sugeno_inference".to_string(),
            "defuzzification".to_string(),
            "linguistic_variable".to_string(),
            "fuzzy_composition".to_string(),
            "similarity_measure".to_string(),
            "distance_measure".to_string(),
            "t_norm_operations".to_string(),
            "s_norm_operations".to_string(),
        ]
    }
}