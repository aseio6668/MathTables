use crate::core::{MathDomain, MathResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalLogicDomain {
    name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemporalOperator {
    Next,        // X (next state)
    Eventually,  // F (eventually/future)
    Globally,    // G (globally/always)
    Until,       // U (until)
    Release,     // R (release/weak until)
    WeakNext,    // wX (weak next)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModalOperator {
    Necessity,   // □ (box/necessary)
    Possibility, // ◇ (diamond/possible)
    Knowledge,   // K (knowledge)
    Belief,      // B (belief)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalFormula {
    Atom(String),
    Not(Box<TemporalFormula>),
    And(Box<TemporalFormula>, Box<TemporalFormula>),
    Or(Box<TemporalFormula>, Box<TemporalFormula>),
    Implies(Box<TemporalFormula>, Box<TemporalFormula>),
    Temporal(TemporalOperator, Box<TemporalFormula>),
    Binary(TemporalOperator, Box<TemporalFormula>, Box<TemporalFormula>),
    Modal(ModalOperator, Box<TemporalFormula>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KripkeStructure {
    pub states: HashSet<String>,
    pub transitions: HashMap<String, Vec<String>>,
    pub labeling: HashMap<String, HashSet<String>>,
    pub initial_states: HashSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameStructure {
    pub players: Vec<String>,
    pub states: HashSet<String>,
    pub actions: HashMap<String, Vec<String>>,
    pub transitions: HashMap<(String, String), String>,
    pub payoffs: HashMap<String, HashMap<String, f64>>,
}

impl TemporalLogicDomain {
    pub fn new() -> Self {
        Self {
            name: "Temporal Logic".to_string(),
        }
    }

    pub fn create_atom(&self, name: &str) -> TemporalFormula {
        TemporalFormula::Atom(name.to_string())
    }

    pub fn not(&self, formula: TemporalFormula) -> TemporalFormula {
        TemporalFormula::Not(Box::new(formula))
    }

    pub fn and(&self, left: TemporalFormula, right: TemporalFormula) -> TemporalFormula {
        TemporalFormula::And(Box::new(left), Box::new(right))
    }

    pub fn or(&self, left: TemporalFormula, right: TemporalFormula) -> TemporalFormula {
        TemporalFormula::Or(Box::new(left), Box::new(right))
    }

    pub fn implies(&self, left: TemporalFormula, right: TemporalFormula) -> TemporalFormula {
        TemporalFormula::Implies(Box::new(left), Box::new(right))
    }

    pub fn next(&self, formula: TemporalFormula) -> TemporalFormula {
        TemporalFormula::Temporal(TemporalOperator::Next, Box::new(formula))
    }

    pub fn eventually(&self, formula: TemporalFormula) -> TemporalFormula {
        TemporalFormula::Temporal(TemporalOperator::Eventually, Box::new(formula))
    }

    pub fn globally(&self, formula: TemporalFormula) -> TemporalFormula {
        TemporalFormula::Temporal(TemporalOperator::Globally, Box::new(formula))
    }

    pub fn until(&self, left: TemporalFormula, right: TemporalFormula) -> TemporalFormula {
        TemporalFormula::Binary(TemporalOperator::Until, Box::new(left), Box::new(right))
    }

    pub fn release(&self, left: TemporalFormula, right: TemporalFormula) -> TemporalFormula {
        TemporalFormula::Binary(TemporalOperator::Release, Box::new(left), Box::new(right))
    }

    pub fn necessity(&self, formula: TemporalFormula) -> TemporalFormula {
        TemporalFormula::Modal(ModalOperator::Necessity, Box::new(formula))
    }

    pub fn possibility(&self, formula: TemporalFormula) -> TemporalFormula {
        TemporalFormula::Modal(ModalOperator::Possibility, Box::new(formula))
    }

    pub fn knowledge(&self, formula: TemporalFormula) -> TemporalFormula {
        TemporalFormula::Modal(ModalOperator::Knowledge, Box::new(formula))
    }

    pub fn belief(&self, formula: TemporalFormula) -> TemporalFormula {
        TemporalFormula::Modal(ModalOperator::Belief, Box::new(formula))
    }

    pub fn model_check_ltl(&self, structure: &KripkeStructure, formula: &TemporalFormula) -> bool {
        for initial_state in &structure.initial_states {
            if !self.evaluate_formula_at_state(structure, initial_state, formula) {
                return false;
            }
        }
        true
    }

    fn evaluate_formula_at_state(&self, structure: &KripkeStructure, state: &str, formula: &TemporalFormula) -> bool {
        match formula {
            TemporalFormula::Atom(prop) => {
                structure.labeling.get(state)
                    .map_or(false, |labels| labels.contains(prop))
            }
            TemporalFormula::Not(f) => {
                !self.evaluate_formula_at_state(structure, state, f)
            }
            TemporalFormula::And(left, right) => {
                self.evaluate_formula_at_state(structure, state, left) &&
                self.evaluate_formula_at_state(structure, state, right)
            }
            TemporalFormula::Or(left, right) => {
                self.evaluate_formula_at_state(structure, state, left) ||
                self.evaluate_formula_at_state(structure, state, right)
            }
            TemporalFormula::Implies(left, right) => {
                !self.evaluate_formula_at_state(structure, state, left) ||
                self.evaluate_formula_at_state(structure, state, right)
            }
            TemporalFormula::Temporal(TemporalOperator::Next, f) => {
                if let Some(successors) = structure.transitions.get(state) {
                    successors.iter().any(|next_state| 
                        self.evaluate_formula_at_state(structure, next_state, f))
                } else {
                    false
                }
            }
            TemporalFormula::Temporal(TemporalOperator::Eventually, f) => {
                self.evaluate_eventually(structure, state, f, &mut HashSet::new())
            }
            TemporalFormula::Temporal(TemporalOperator::Globally, f) => {
                self.evaluate_globally(structure, state, f, &mut HashSet::new())
            }
            TemporalFormula::Binary(TemporalOperator::Until, left, right) => {
                self.evaluate_until(structure, state, left, right, &mut HashSet::new())
            }
            TemporalFormula::Modal(ModalOperator::Necessity, f) => {
                if let Some(successors) = structure.transitions.get(state) {
                    successors.iter().all(|next_state| 
                        self.evaluate_formula_at_state(structure, next_state, f))
                } else {
                    true
                }
            }
            TemporalFormula::Modal(ModalOperator::Possibility, f) => {
                if let Some(successors) = structure.transitions.get(state) {
                    successors.iter().any(|next_state| 
                        self.evaluate_formula_at_state(structure, next_state, f))
                } else {
                    false
                }
            }
            _ => false, // Simplified for other cases
        }
    }

    fn evaluate_eventually(&self, structure: &KripkeStructure, state: &str, 
                          formula: &TemporalFormula, visited: &mut HashSet<String>) -> bool {
        if visited.contains(state) {
            return false;
        }
        visited.insert(state.to_string());

        if self.evaluate_formula_at_state(structure, state, formula) {
            return true;
        }

        if let Some(successors) = structure.transitions.get(state) {
            for next_state in successors {
                if self.evaluate_eventually(structure, next_state, formula, visited) {
                    return true;
                }
            }
        }
        false
    }

    fn evaluate_globally(&self, structure: &KripkeStructure, state: &str, 
                        formula: &TemporalFormula, visited: &mut HashSet<String>) -> bool {
        if visited.contains(state) {
            return true; // Cycle detected, assume true for global
        }
        visited.insert(state.to_string());

        if !self.evaluate_formula_at_state(structure, state, formula) {
            return false;
        }

        if let Some(successors) = structure.transitions.get(state) {
            for next_state in successors {
                if !self.evaluate_globally(structure, next_state, formula, visited) {
                    return false;
                }
            }
        }
        true
    }

    fn evaluate_until(&self, structure: &KripkeStructure, state: &str, 
                     left: &TemporalFormula, right: &TemporalFormula, 
                     visited: &mut HashSet<String>) -> bool {
        if visited.contains(state) {
            return false;
        }
        visited.insert(state.to_string());

        if self.evaluate_formula_at_state(structure, state, right) {
            return true;
        }

        if !self.evaluate_formula_at_state(structure, state, left) {
            return false;
        }

        if let Some(successors) = structure.transitions.get(state) {
            for next_state in successors {
                if self.evaluate_until(structure, next_state, left, right, visited) {
                    return true;
                }
            }
        }
        false
    }

    pub fn create_kripke_structure() -> KripkeStructure {
        KripkeStructure {
            states: HashSet::new(),
            transitions: HashMap::new(),
            labeling: HashMap::new(),
            initial_states: HashSet::new(),
        }
    }

    pub fn add_state(&self, structure: &mut KripkeStructure, state: &str) {
        structure.states.insert(state.to_string());
    }

    pub fn add_transition(&self, structure: &mut KripkeStructure, from: &str, to: &str) {
        structure.transitions
            .entry(from.to_string())
            .or_insert_with(Vec::new)
            .push(to.to_string());
    }

    pub fn add_label(&self, structure: &mut KripkeStructure, state: &str, label: &str) {
        structure.labeling
            .entry(state.to_string())
            .or_insert_with(HashSet::new)
            .insert(label.to_string());
    }

    pub fn set_initial(&self, structure: &mut KripkeStructure, state: &str) {
        structure.initial_states.insert(state.to_string());
    }

    // Game-theoretic temporal logic
    pub fn create_game_structure() -> GameStructure {
        GameStructure {
            players: Vec::new(),
            states: HashSet::new(),
            actions: HashMap::new(),
            transitions: HashMap::new(),
            payoffs: HashMap::new(),
        }
    }

    pub fn add_player(&self, game: &mut GameStructure, player: &str) {
        game.players.push(player.to_string());
    }

    pub fn add_game_state(&self, game: &mut GameStructure, state: &str) {
        game.states.insert(state.to_string());
    }

    pub fn add_action(&self, game: &mut GameStructure, player: &str, action: &str) {
        game.actions
            .entry(player.to_string())
            .or_insert_with(Vec::new)
            .push(action.to_string());
    }

    pub fn add_game_transition(&self, game: &mut GameStructure, 
                              state: &str, action: &str, next_state: &str) {
        game.transitions.insert(
            (state.to_string(), action.to_string()),
            next_state.to_string()
        );
    }

    pub fn set_payoff(&self, game: &mut GameStructure, 
                     player: &str, state: &str, payoff: f64) {
        game.payoffs
            .entry(player.to_string())
            .or_insert_with(HashMap::new)
            .insert(state.to_string(), payoff);
    }
}

impl fmt::Display for TemporalFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemporalFormula::Atom(name) => write!(f, "{}", name),
            TemporalFormula::Not(formula) => write!(f, "¬{}", formula),
            TemporalFormula::And(left, right) => write!(f, "({} ∧ {})", left, right),
            TemporalFormula::Or(left, right) => write!(f, "({} ∨ {})", left, right),
            TemporalFormula::Implies(left, right) => write!(f, "({} → {})", left, right),
            TemporalFormula::Temporal(TemporalOperator::Next, formula) => 
                write!(f, "X{}", formula),
            TemporalFormula::Temporal(TemporalOperator::Eventually, formula) => 
                write!(f, "F{}", formula),
            TemporalFormula::Temporal(TemporalOperator::Globally, formula) => 
                write!(f, "G{}", formula),
            TemporalFormula::Binary(TemporalOperator::Until, left, right) => 
                write!(f, "({} U {})", left, right),
            TemporalFormula::Binary(TemporalOperator::Release, left, right) => 
                write!(f, "({} R {})", left, right),
            TemporalFormula::Modal(ModalOperator::Necessity, formula) => 
                write!(f, "□{}", formula),
            TemporalFormula::Modal(ModalOperator::Possibility, formula) => 
                write!(f, "◇{}", formula),
            TemporalFormula::Modal(ModalOperator::Knowledge, formula) => 
                write!(f, "K{}", formula),
            TemporalFormula::Modal(ModalOperator::Belief, formula) => 
                write!(f, "B{}", formula),
            _ => write!(f, "?"),
        }
    }
}

impl MathDomain for TemporalLogicDomain {
    fn name(&self) -> &str {
        &self.name
    }

    fn supports_operation(&self, operation: &str) -> bool {
        matches!(operation, 
            "model_check" | "create_formula" | "create_structure" | 
            "temporal_next" | "temporal_eventually" | "temporal_globally" |
            "modal_necessity" | "modal_possibility" | "game_theory"
        )
    }

    fn description(&self) -> &str {
        "Temporal Logic and Modal Logic reasoning framework"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    fn compute(&self, operation: &str, _args: &[&dyn std::any::Any]) -> MathResult<Box<dyn std::any::Any>> {
        match operation {
            "create_structure" => Ok(Box::new("Kripke structure created".to_string())),
            "create_formula" => Ok(Box::new("Temporal formula created".to_string())),
            _ => Err(crate::core::MathError::NotImplemented(format!("Operation '{}' not implemented", operation))),
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "model_check".to_string(), "create_formula".to_string(),
            "create_structure".to_string(), "temporal_next".to_string(),
            "temporal_eventually".to_string(), "temporal_globally".to_string(),
            "modal_necessity".to_string(), "modal_possibility".to_string(),
            "game_theory".to_string()
        ]
    }
}

pub fn temporal_logic() -> TemporalLogicDomain {
    TemporalLogicDomain::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_formula_creation() {
        let domain = TemporalLogicDomain::new();
        let p = domain.create_atom("p");
        let q = domain.create_atom("q");
        
        let formula = domain.until(p, q);
        assert!(matches!(formula, TemporalFormula::Binary(TemporalOperator::Until, _, _)));
    }

    #[test]
    fn test_kripke_structure() {
        let domain = TemporalLogicDomain::new();
        let mut structure = TemporalLogicDomain::create_kripke_structure();
        
        domain.add_state(&mut structure, "s0");
        domain.add_state(&mut structure, "s1");
        domain.add_transition(&mut structure, "s0", "s1");
        domain.add_label(&mut structure, "s0", "p");
        domain.set_initial(&mut structure, "s0");
        
        assert!(structure.states.contains("s0"));
        assert!(structure.states.contains("s1"));
        assert!(structure.initial_states.contains("s0"));
    }

    #[test]
    fn test_modal_logic() {
        let domain = TemporalLogicDomain::new();
        let p = domain.create_atom("p");
        let necessary_p = domain.necessity(p);
        
        assert!(matches!(necessary_p, TemporalFormula::Modal(ModalOperator::Necessity, _)));
    }
}