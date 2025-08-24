use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct PossibilityDistribution {
    pub universe: Vec<f64>,
    pub possibility_values: Vec<f64>, // π(x) for each x in universe
    pub necessity_values: Vec<f64>,   // N(x) = 1 - π(¬x)
}

#[derive(Debug, Clone)]
pub struct PossibilityMeasure {
    pub events: HashMap<String, f64>, // Event name -> possibility value
}

#[derive(Debug, Clone)]
pub struct NecessityMeasure {
    pub events: HashMap<String, f64>, // Event name -> necessity value
}

#[derive(Debug, Clone)]
pub struct PossibilisticVariable {
    pub name: String,
    pub domain: Vec<f64>,
    pub distribution: PossibilityDistribution,
}

#[derive(Debug, Clone)]
pub struct PossibilisticConstraint {
    pub variables: Vec<String>,
    pub constraint_function: String, // Simplified as string
    pub possibility_level: f64,
}

#[derive(Debug, Clone)]
pub struct PossibilisticLogic {
    pub propositions: HashMap<String, f64>, // Proposition -> certainty level
    pub rules: Vec<PossibilisticRule>,
}

#[derive(Debug, Clone)]
pub struct PossibilisticRule {
    pub antecedent: Vec<String>, // List of proposition names
    pub consequent: String,
    pub certainty: f64,
    pub rule_type: RuleType,
}

#[derive(Debug, Clone)]
pub enum RuleType {
    Certain,      // A → B with certainty α
    Uncertain,    // A →α B (uncertain implication)
    Weighted,     // Weighted rule
}

#[derive(Debug, Clone)]
pub struct PossibilisticDecision {
    pub alternatives: Vec<String>,
    pub criteria: Vec<String>,
    pub possibility_matrix: Vec<Vec<f64>>, // alternatives × criteria
    pub weights: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct UpperLowerProbability {
    pub events: HashMap<String, (f64, f64)>, // Event -> (lower bound, upper bound)
}

#[derive(Debug, Clone)]
pub struct BeliefFunction {
    pub focal_elements: HashMap<Vec<String>, f64>, // Subset -> mass assignment
    pub universe: Vec<String>,
}

pub struct PossibilityTheoryDomain;

impl PossibilityDistribution {
    pub fn new(universe: Vec<f64>) -> Self {
        let n = universe.len();
        PossibilityDistribution {
            universe,
            possibility_values: vec![0.0; n],
            necessity_values: vec![0.0; n],
        }
    }
    
    pub fn uniform(universe: Vec<f64>) -> Self {
        let n = universe.len();
        PossibilityDistribution {
            universe,
            possibility_values: vec![1.0; n],
            necessity_values: vec![0.0; n],
        }
    }
    
    pub fn set_possibility(&mut self, index: usize, value: f64) -> MathResult<()> {
        if index >= self.universe.len() {
            return Err(MathError::InvalidArgument("Index out of bounds".to_string()));
        }
        
        if value < 0.0 || value > 1.0 {
            return Err(MathError::InvalidArgument("Possibility value must be in [0,1]".to_string()));
        }
        
        self.possibility_values[index] = value;
        self.update_necessity_values();
        Ok(())
    }
    
    fn update_necessity_values(&mut self) {
        // N(A) = 1 - Π(¬A) where Π(¬A) = max{π(x) : x ∉ A}
        for i in 0..self.universe.len() {
            let complement_max = self.possibility_values.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &val)| val)
                .fold(0.0, f64::max);
            self.necessity_values[i] = 1.0 - complement_max;
        }
    }
    
    pub fn possibility(&self, value: f64) -> f64 {
        if let Some(index) = self.universe.iter().position(|&x| (x - value).abs() < 1e-10) {
            self.possibility_values[index]
        } else {
            // Interpolate if not exact match
            for i in 0..self.universe.len() - 1 {
                if self.universe[i] <= value && value <= self.universe[i + 1] {
                    let ratio = (value - self.universe[i]) / (self.universe[i + 1] - self.universe[i]);
                    return self.possibility_values[i] + 
                           ratio * (self.possibility_values[i + 1] - self.possibility_values[i]);
                }
            }
            0.0
        }
    }
    
    pub fn necessity(&self, value: f64) -> f64 {
        if let Some(index) = self.universe.iter().position(|&x| (x - value).abs() < 1e-10) {
            self.necessity_values[index]
        } else {
            0.0 // Simplified
        }
    }
    
    pub fn alpha_cut(&self, alpha: f64) -> Vec<f64> {
        if alpha < 0.0 || alpha > 1.0 {
            return Vec::new();
        }
        
        self.universe.iter()
            .zip(self.possibility_values.iter())
            .filter_map(|(&x, &poss)| {
                if poss >= alpha {
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
        self.possibility_values.iter().fold(0.0, |max, &val| max.max(val))
    }
    
    pub fn is_normalized(&self) -> bool {
        self.height() == 1.0
    }
    
    pub fn normalize(&mut self) {
        let max_val = self.height();
        if max_val > 0.0 {
            for val in &mut self.possibility_values {
                *val /= max_val;
            }
            self.update_necessity_values();
        }
    }
}

impl PossibilityTheoryDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn union_distributions(
        dist1: &PossibilityDistribution,
        dist2: &PossibilityDistribution
    ) -> MathResult<PossibilityDistribution> {
        if dist1.universe != dist2.universe {
            return Err(MathError::InvalidArgument("Distributions must have same universe".to_string()));
        }
        
        let possibility_values: Vec<f64> = dist1.possibility_values.iter()
            .zip(dist2.possibility_values.iter())
            .map(|(&a, &b)| a.max(b))
            .collect();
        
        let mut result = PossibilityDistribution {
            universe: dist1.universe.clone(),
            possibility_values,
            necessity_values: vec![0.0; dist1.universe.len()],
        };
        
        result.update_necessity_values();
        Ok(result)
    }
    
    pub fn intersection_distributions(
        dist1: &PossibilityDistribution,
        dist2: &PossibilityDistribution
    ) -> MathResult<PossibilityDistribution> {
        if dist1.universe != dist2.universe {
            return Err(MathError::InvalidArgument("Distributions must have same universe".to_string()));
        }
        
        let possibility_values: Vec<f64> = dist1.possibility_values.iter()
            .zip(dist2.possibility_values.iter())
            .map(|(&a, &b)| a.min(b))
            .collect();
        
        let mut result = PossibilityDistribution {
            universe: dist1.universe.clone(),
            possibility_values,
            necessity_values: vec![0.0; dist1.universe.len()],
        };
        
        result.update_necessity_values();
        Ok(result)
    }
    
    pub fn complement_distribution(dist: &PossibilityDistribution) -> PossibilityDistribution {
        // Complement of possibility distribution
        let possibility_values: Vec<f64> = dist.necessity_values.clone();
        let necessity_values: Vec<f64> = dist.possibility_values.iter()
            .map(|&val| 1.0 - val)
            .collect();
        
        PossibilityDistribution {
            universe: dist.universe.clone(),
            possibility_values,
            necessity_values,
        }
    }
    
    pub fn conditional_possibility(
        joint: &PossibilityDistribution,
        condition_indices: &[usize]
    ) -> MathResult<PossibilityDistribution> {
        if condition_indices.is_empty() {
            return Ok(joint.clone());
        }
        
        // Simplified conditional possibility calculation
        let mut conditional = joint.clone();
        
        // Find maximum possibility in the conditioning set
        let condition_max = condition_indices.iter()
            .map(|&i| joint.possibility_values.get(i).copied().unwrap_or(0.0))
            .fold(0.0, f64::max);
        
        if condition_max > 0.0 {
            for i in 0..conditional.possibility_values.len() {
                if condition_indices.contains(&i) {
                    conditional.possibility_values[i] /= condition_max;
                } else {
                    conditional.possibility_values[i] = 0.0;
                }
            }
        }
        
        conditional.update_necessity_values();
        Ok(conditional)
    }
    
    pub fn possibilistic_reasoning(logic: &PossibilisticLogic, facts: &[String]) -> MathResult<HashMap<String, f64>> {
        let mut derived_certainties = HashMap::new();
        
        // Initialize with given facts
        for fact in facts {
            if let Some(&certainty) = logic.propositions.get(fact) {
                derived_certainties.insert(fact.clone(), certainty);
            }
        }
        
        // Apply rules iteratively
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;
        
        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;
            
            for rule in &logic.rules {
                // Check if all antecedents are satisfied
                let mut min_certainty: f64 = 1.0;
                let mut all_satisfied = true;
                
                for antecedent in &rule.antecedent {
                    if let Some(&certainty) = derived_certainties.get(antecedent) {
                        min_certainty = min_certainty.min(certainty);
                    } else {
                        all_satisfied = false;
                        break;
                    }
                }
                
                if all_satisfied {
                    let conclusion_certainty = match rule.rule_type {
                        RuleType::Certain => min_certainty.min(rule.certainty),
                        RuleType::Uncertain => min_certainty * rule.certainty,
                        RuleType::Weighted => min_certainty * rule.certainty,
                    };
                    
                    let current_certainty = derived_certainties.get(&rule.consequent).copied().unwrap_or(0.0);
                    let new_certainty = current_certainty.max(conclusion_certainty);
                    
                    if new_certainty > current_certainty {
                        derived_certainties.insert(rule.consequent.clone(), new_certainty);
                        changed = true;
                    }
                }
            }
        }
        
        Ok(derived_certainties)
    }
    
    pub fn possibilistic_decision_making(decision: &PossibilisticDecision) -> MathResult<String> {
        if decision.alternatives.is_empty() || decision.criteria.is_empty() {
            return Err(MathError::InvalidArgument("Empty alternatives or criteria".to_string()));
        }
        
        if decision.weights.len() != decision.criteria.len() {
            return Err(MathError::InvalidArgument("Weights must match criteria count".to_string()));
        }
        
        let mut best_alternative = 0;
        let mut best_score = f64::NEG_INFINITY;
        
        for (alt_idx, _) in decision.alternatives.iter().enumerate() {
            if alt_idx >= decision.possibility_matrix.len() {
                continue;
            }
            
            let mut weighted_score = 0.0;
            let mut total_weight = 0.0;
            
            for (crit_idx, &weight) in decision.weights.iter().enumerate() {
                if crit_idx < decision.possibility_matrix[alt_idx].len() {
                    weighted_score += weight * decision.possibility_matrix[alt_idx][crit_idx];
                    total_weight += weight;
                }
            }
            
            let final_score = if total_weight > 0.0 {
                weighted_score / total_weight
            } else {
                0.0
            };
            
            if final_score > best_score {
                best_score = final_score;
                best_alternative = alt_idx;
            }
        }
        
        Ok(decision.alternatives[best_alternative].clone())
    }
    
    pub fn dempster_shafer_combination(
        belief1: &BeliefFunction,
        belief2: &BeliefFunction
    ) -> MathResult<BeliefFunction> {
        if belief1.universe != belief2.universe {
            return Err(MathError::InvalidArgument("Belief functions must have same universe".to_string()));
        }
        
        let mut combined_masses = HashMap::new();
        let mut conflict = 0.0;
        
        // Combine focal elements
        for (set1, &mass1) in &belief1.focal_elements {
            for (set2, &mass2) in &belief2.focal_elements {
                let intersection = Self::set_intersection(set1, set2);
                
                if intersection.is_empty() {
                    conflict += mass1 * mass2;
                } else {
                    let current_mass = combined_masses.get(&intersection).copied().unwrap_or(0.0);
                    combined_masses.insert(intersection, current_mass + mass1 * mass2);
                }
            }
        }
        
        // Normalize by (1 - conflict)
        let normalization_factor = 1.0 - conflict;
        if normalization_factor <= 0.0 {
            return Err(MathError::ComputationError("Complete conflict in belief combination".to_string()));
        }
        
        for mass in combined_masses.values_mut() {
            *mass /= normalization_factor;
        }
        
        Ok(BeliefFunction {
            focal_elements: combined_masses,
            universe: belief1.universe.clone(),
        })
    }
    
    fn set_intersection(set1: &[String], set2: &[String]) -> Vec<String> {
        set1.iter()
            .filter(|item| set2.contains(item))
            .cloned()
            .collect()
    }
    
    pub fn belief_to_possibility(belief: &BeliefFunction) -> MathResult<PossibilityDistribution> {
        // Transform belief function to possibility distribution
        let universe_values: Vec<f64> = belief.universe.iter()
            .enumerate()
            .map(|(i, _)| i as f64)
            .collect();
        
        let mut possibility_values: Vec<f64> = vec![0.0; belief.universe.len()];
        
        // For each element in universe, compute its possibility
        for (i, element) in belief.universe.iter().enumerate() {
            for (focal_set, &mass) in &belief.focal_elements {
                if focal_set.contains(element) {
                    possibility_values[i] = possibility_values[i].max(mass);
                }
            }
        }
        
        let mut result = PossibilityDistribution {
            universe: universe_values,
            possibility_values,
            necessity_values: vec![0.0; belief.universe.len()],
        };
        
        result.update_necessity_values();
        Ok(result)
    }
    
    pub fn possibility_to_probability_bounds(
        dist: &PossibilityDistribution
    ) -> HashMap<String, (f64, f64)> {
        let mut bounds = HashMap::new();
        
        for (i, &poss_val) in dist.possibility_values.iter().enumerate() {
            let nec_val = dist.necessity_values[i];
            let element_name = format!("element_{}", i);
            bounds.insert(element_name, (nec_val, poss_val));
        }
        
        bounds
    }
    
    pub fn specificity_measure(dist: &PossibilityDistribution) -> f64 {
        // Measure how specific (non-uniform) the distribution is
        let n = dist.universe.len() as f64;
        if n <= 1.0 {
            return 1.0;
        }
        
        let entropy = dist.possibility_values.iter()
            .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
            .sum::<f64>();
        
        let max_entropy = (1.0 / n).ln() * (-1.0);
        
        if max_entropy != 0.0 {
            1.0 - (entropy / max_entropy)
        } else {
            1.0
        }
    }
    
    pub fn uncertainty_measure(dist: &PossibilityDistribution) -> f64 {
        // U-uncertainty measure for possibility distributions
        dist.possibility_values.iter()
            .zip(dist.necessity_values.iter())
            .map(|(&poss, &nec)| poss - nec)
            .sum::<f64>() / dist.universe.len() as f64
    }
}

impl MathDomain for PossibilityTheoryDomain {
    fn name(&self) -> &str { "Possibility Theory" }
    fn description(&self) -> &str { "Possibility distributions, fuzzy measures, possibilistic logic, and uncertainty reasoning" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, _args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            _ => Err(MathError::InvalidOperation(format!("Operation {} not implemented in compute interface", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "create_possibility_distribution".to_string(),
            "uniform_distribution".to_string(),
            "union_distributions".to_string(),
            "intersection_distributions".to_string(),
            "complement_distribution".to_string(),
            "conditional_possibility".to_string(),
            "alpha_cut".to_string(),
            "possibilistic_reasoning".to_string(),
            "possibilistic_decision_making".to_string(),
            "dempster_shafer_combination".to_string(),
            "belief_to_possibility".to_string(),
            "possibility_to_probability".to_string(),
            "specificity_measure".to_string(),
            "uncertainty_measure".to_string(),
            "necessity_measure".to_string(),
            "possibility_normalization".to_string(),
            "possibilistic_constraints".to_string(),
        ]
    }
}