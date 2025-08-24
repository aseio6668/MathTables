use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Set<T: Clone + Eq + std::hash::Hash> {
    pub elements: HashSet<T>,
    pub name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BinaryRelation<T: Clone + Eq + std::hash::Hash> {
    pub domain: Set<T>,
    pub codomain: Set<T>,
    pub pairs: HashSet<(T, T)>,
    pub properties: RelationProperties,
}

#[derive(Debug, Clone)]
pub struct RelationProperties {
    pub reflexive: Option<bool>,
    pub symmetric: Option<bool>,
    pub transitive: Option<bool>,
    pub antisymmetric: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct Axiom {
    pub name: String,
    pub statement: String,
    pub axiom_system: String,
    pub independence: bool,
}

#[derive(Debug, Clone)]
pub struct MathFunction<T: Clone + Eq + std::hash::Hash, U: Clone + Eq + std::hash::Hash> {
    pub domain: Set<T>,
    pub codomain: Set<U>,
    pub mapping: HashMap<T, U>,
}

pub struct FoundationsDomain;

impl FoundationsDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn create_set<T: Clone + Eq + std::hash::Hash>(elements: Vec<T>, name: Option<String>) -> Set<T> {
        Set {
            elements: elements.into_iter().collect(),
            name,
        }
    }
    
    pub fn set_union<T: Clone + Eq + std::hash::Hash>(a: &Set<T>, b: &Set<T>) -> Set<T> {
        let union: HashSet<T> = a.elements.union(&b.elements).cloned().collect();
        Set {
            elements: union,
            name: None,
        }
    }
    
    pub fn set_intersection<T: Clone + Eq + std::hash::Hash>(a: &Set<T>, b: &Set<T>) -> Set<T> {
        let intersection: HashSet<T> = a.elements.intersection(&b.elements).cloned().collect();
        Set {
            elements: intersection,
            name: None,
        }
    }
    
    pub fn set_difference<T: Clone + Eq + std::hash::Hash>(a: &Set<T>, b: &Set<T>) -> Set<T> {
        let difference: HashSet<T> = a.elements.difference(&b.elements).cloned().collect();
        Set {
            elements: difference,
            name: None,
        }
    }
    
    pub fn is_subset<T: Clone + Eq + std::hash::Hash>(a: &Set<T>, b: &Set<T>) -> bool {
        a.elements.is_subset(&b.elements)
    }
    
    pub fn cardinality<T: Clone + Eq + std::hash::Hash>(set: &Set<T>) -> usize {
        set.elements.len()
    }
    
    pub fn create_binary_relation<T: Clone + Eq + std::hash::Hash>(
        domain: Set<T>,
        codomain: Set<T>,
        pairs: Vec<(T, T)>,
    ) -> BinaryRelation<T> {
        BinaryRelation {
            domain,
            codomain,
            pairs: pairs.into_iter().collect(),
            properties: RelationProperties {
                reflexive: None,
                symmetric: None,
                transitive: None,
                antisymmetric: None,
            },
        }
    }
    
    pub fn check_reflexivity<T: Clone + Eq + std::hash::Hash>(relation: &BinaryRelation<T>) -> bool {
        relation.domain.elements.iter()
            .all(|x| relation.pairs.contains(&(x.clone(), x.clone())))
    }
    
    pub fn check_symmetry<T: Clone + Eq + std::hash::Hash>(relation: &BinaryRelation<T>) -> bool {
        relation.pairs.iter()
            .all(|(x, y)| relation.pairs.contains(&(y.clone(), x.clone())))
    }
    
    pub fn check_transitivity<T: Clone + Eq + std::hash::Hash>(relation: &BinaryRelation<T>) -> bool {
        for (x, y) in &relation.pairs {
            for (y2, z) in &relation.pairs {
                if y == y2 && !relation.pairs.contains(&(x.clone(), z.clone())) {
                    return false;
                }
            }
        }
        true
    }
    
    pub fn create_axiom(name: &str, statement: &str, system: &str) -> Axiom {
        Axiom {
            name: name.to_string(),
            statement: statement.to_string(),
            axiom_system: system.to_string(),
            independence: false,
        }
    }
    
    pub fn get_zfc_axioms() -> Vec<Axiom> {
        vec![
            Self::create_axiom("Extensionality", "Two sets are equal iff they have the same elements", "ZFC"),
            Self::create_axiom("Empty Set", "There exists a set with no elements", "ZFC"),
            Self::create_axiom("Pairing", "For any two sets, there exists a set containing exactly those two sets", "ZFC"),
            Self::create_axiom("Union", "For any set of sets, there exists a set containing all elements of those sets", "ZFC"),
            Self::create_axiom("Power Set", "For any set, there exists the set of all its subsets", "ZFC"),
            Self::create_axiom("Infinity", "There exists an infinite set", "ZFC"),
            Self::create_axiom("Replacement", "The image of a set under any definable function is also a set", "ZFC"),
            Self::create_axiom("Regularity", "Every non-empty set contains an element disjoint from it", "ZFC"),
            Self::create_axiom("Choice", "Every collection of non-empty sets has a choice function", "ZFC"),
        ]
    }
    
    pub fn create_function<T: Clone + Eq + std::hash::Hash, U: Clone + Eq + std::hash::Hash>(
        domain: Set<T>,
        codomain: Set<U>,
        mapping: HashMap<T, U>,
    ) -> MathResult<MathFunction<T, U>> {
        for key in mapping.keys() {
            if !domain.elements.contains(key) {
                return Err(MathError::InvalidArgument("Function domain mismatch".to_string()));
            }
        }
        
        Ok(MathFunction {
            domain,
            codomain,
            mapping,
        })
    }
    
    pub fn compose_functions<T: Clone + Eq + std::hash::Hash, U: Clone + Eq + std::hash::Hash, V: Clone + Eq + std::hash::Hash>(
        f: &MathFunction<T, U>,
        g: &MathFunction<U, V>,
    ) -> MathResult<MathFunction<T, V>> {
        let mut composition = HashMap::new();
        
        for (x, y) in &f.mapping {
            if let Some(z) = g.mapping.get(y) {
                composition.insert(x.clone(), z.clone());
            } else {
                return Err(MathError::InvalidArgument("Functions not composable".to_string()));
            }
        }
        
        Ok(MathFunction {
            domain: f.domain.clone(),
            codomain: g.codomain.clone(),
            mapping: composition,
        })
    }
}

impl MathDomain for FoundationsDomain {
    fn name(&self) -> &str { "Mathematical Foundations" }
    fn description(&self) -> &str { "Set theory, axioms, binary relations, functions, and mathematical foundations" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "get_zfc_axioms" => {
                if !args.is_empty() {
                    return Err(MathError::InvalidArgument("get_zfc_axioms requires no arguments".to_string()));
                }
                Ok(Box::new(Self::get_zfc_axioms()))
            },
            "create_axiom" => {
                if args.len() != 3 {
                    return Err(MathError::InvalidArgument("create_axiom requires 3 arguments".to_string()));
                }
                let name = args[0].downcast_ref::<String>().ok_or_else(|| MathError::InvalidArgument("First argument must be String".to_string()))?;
                let statement = args[1].downcast_ref::<String>().ok_or_else(|| MathError::InvalidArgument("Second argument must be String".to_string()))?;
                let system = args[2].downcast_ref::<String>().ok_or_else(|| MathError::InvalidArgument("Third argument must be String".to_string()))?;
                Ok(Box::new(Self::create_axiom(name, statement, system)))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "create_set".to_string(),
            "set_union".to_string(),
            "set_intersection".to_string(),
            "set_difference".to_string(),
            "is_subset".to_string(),
            "cardinality".to_string(),
            "create_binary_relation".to_string(),
            "check_reflexivity".to_string(),
            "check_symmetry".to_string(),
            "check_transitivity".to_string(),
            "create_axiom".to_string(),
            "get_zfc_axioms".to_string(),
            "create_function".to_string(),
            "compose_functions".to_string(),
        ]
    }
}