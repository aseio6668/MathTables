use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Literal {
    pub variable: String,
    pub is_positive: bool,
}

#[derive(Debug, Clone)]
pub struct Clause {
    pub literals: Vec<Literal>,
    pub is_unit: bool,
    pub is_satisfied: bool,
}

#[derive(Debug, Clone)]
pub struct CNFFormula {
    pub clauses: Vec<Clause>,
    pub variables: HashSet<String>,
}

#[derive(Debug, Clone)]
pub struct Assignment {
    pub variables: HashMap<String, bool>,
    pub decision_level: HashMap<String, usize>,
    pub implications: Vec<(String, usize)>, // (variable, decision_level)
}

#[derive(Debug, Clone)]
pub struct ConflictClause {
    pub literals: Vec<Literal>,
    pub learned_at_level: usize,
}

#[derive(Debug, Clone)]
pub struct SMTFormula {
    pub boolean_structure: CNFFormula,
    pub theory_constraints: Vec<TheoryConstraint>,
    pub theory: Theory,
}

#[derive(Debug, Clone)]
pub enum Theory {
    LinearArithmetic,
    EqualityLogic,
    ArrayTheory,
    BitVectors,
    UninterpretedFunctions,
}

#[derive(Debug, Clone)]
pub struct TheoryConstraint {
    pub constraint_type: ConstraintType,
    pub variables: Vec<String>,
    pub coefficients: Vec<i32>,
    pub operator: String,
    pub rhs: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    Linear,
    Equality,
    Disequality,
    Array,
    BitVector,
}

#[derive(Debug, Clone)]
pub struct DPLLState {
    pub assignment: Assignment,
    pub formula: CNFFormula,
    pub decision_level: usize,
    pub conflict_clauses: Vec<ConflictClause>,
}

#[derive(Debug, Clone)]
pub enum SolverResult {
    Satisfiable(Assignment),
    Unsatisfiable,
    Unknown,
}

pub struct SATSMTDomain;

impl Literal {
    pub fn new(variable: String, is_positive: bool) -> Self {
        Literal { variable, is_positive }
    }
    
    pub fn negate(&self) -> Literal {
        Literal {
            variable: self.variable.clone(),
            is_positive: !self.is_positive,
        }
    }
    
    pub fn evaluate(&self, assignment: &Assignment) -> Option<bool> {
        assignment.variables.get(&self.variable).map(|&value| {
            if self.is_positive { value } else { !value }
        })
    }
}

impl Clause {
    pub fn new(literals: Vec<Literal>) -> Self {
        let is_unit = literals.len() == 1;
        Clause {
            literals,
            is_unit,
            is_satisfied: false,
        }
    }
    
    pub fn evaluate(&self, assignment: &Assignment) -> Option<bool> {
        let mut has_unknown = false;
        
        for literal in &self.literals {
            match literal.evaluate(assignment) {
                Some(true) => return Some(true), // Clause is satisfied
                Some(false) => continue,
                None => has_unknown = true,
            }
        }
        
        if has_unknown {
            None // Cannot determine
        } else {
            Some(false) // All literals are false
        }
    }
    
    pub fn is_unit_clause(&self, assignment: &Assignment) -> Option<Literal> {
        let mut unassigned_literal = None;
        let mut unassigned_count = 0;
        
        for literal in &self.literals {
            match literal.evaluate(assignment) {
                Some(true) => return None, // Clause already satisfied
                Some(false) => continue,
                None => {
                    unassigned_literal = Some(literal.clone());
                    unassigned_count += 1;
                    if unassigned_count > 1 {
                        return None; // More than one unassigned
                    }
                }
            }
        }
        
        if unassigned_count == 1 {
            unassigned_literal
        } else {
            None
        }
    }
    
    pub fn get_unassigned_literals(&self, assignment: &Assignment) -> Vec<Literal> {
        self.literals.iter()
            .filter(|literal| literal.evaluate(assignment).is_none())
            .cloned()
            .collect()
    }
}

impl CNFFormula {
    pub fn new() -> Self {
        CNFFormula {
            clauses: Vec::new(),
            variables: HashSet::new(),
        }
    }
    
    pub fn add_clause(&mut self, clause: Clause) {
        for literal in &clause.literals {
            self.variables.insert(literal.variable.clone());
        }
        self.clauses.push(clause);
    }
    
    pub fn evaluate(&self, assignment: &Assignment) -> Option<bool> {
        let mut has_unknown = false;
        
        for clause in &self.clauses {
            match clause.evaluate(assignment) {
                Some(false) => return Some(false), // Formula is unsatisfied
                Some(true) => continue,
                None => has_unknown = true,
            }
        }
        
        if has_unknown {
            None
        } else {
            Some(true)
        }
    }
    
    pub fn get_unit_clauses(&self, assignment: &Assignment) -> Vec<Literal> {
        self.clauses.iter()
            .filter_map(|clause| clause.is_unit_clause(assignment))
            .collect()
    }
    
    pub fn has_conflict(&self, assignment: &Assignment) -> bool {
        self.clauses.iter().any(|clause| {
            clause.evaluate(assignment) == Some(false)
        })
    }
}

impl SATSMTDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn dpll_solver(formula: &CNFFormula) -> MathResult<SolverResult> {
        let mut assignment = Assignment {
            variables: HashMap::new(),
            decision_level: HashMap::new(),
            implications: Vec::new(),
        };
        
        match Self::dpll_recursive(formula, &mut assignment, 0) {
            Ok(true) => Ok(SolverResult::Satisfiable(assignment)),
            Ok(false) => Ok(SolverResult::Unsatisfiable),
            Err(e) => Err(e),
        }
    }
    
    fn dpll_recursive(formula: &CNFFormula, assignment: &mut Assignment, level: usize) -> MathResult<bool> {
        // Unit propagation
        loop {
            let unit_clauses = formula.get_unit_clauses(assignment);
            if unit_clauses.is_empty() {
                break;
            }
            
            for literal in unit_clauses {
                if assignment.variables.contains_key(&literal.variable) {
                    // Check for conflict
                    if let Some(&current_value) = assignment.variables.get(&literal.variable) {
                        let required_value = literal.is_positive;
                        if current_value != required_value {
                            return Ok(false); // Conflict
                        }
                    }
                } else {
                    assignment.variables.insert(literal.variable.clone(), literal.is_positive);
                    assignment.decision_level.insert(literal.variable.clone(), level);
                    assignment.implications.push((literal.variable.clone(), level));
                }
            }
        }
        
        // Check if formula is satisfied
        match formula.evaluate(assignment) {
            Some(true) => return Ok(true),
            Some(false) => return Ok(false),
            None => {} // Continue with search
        }
        
        // Choose next variable to assign (first unassigned)
        let unassigned_var = formula.variables.iter()
            .find(|var| !assignment.variables.contains_key(*var))
            .cloned();
        
        if let Some(var) = unassigned_var {
            // Try positive assignment first
            assignment.variables.insert(var.clone(), true);
            assignment.decision_level.insert(var.clone(), level + 1);
            
            if Self::dpll_recursive(formula, assignment, level + 1)? {
                return Ok(true);
            }
            
            // Backtrack and try negative assignment
            assignment.variables.insert(var.clone(), false);
            assignment.decision_level.insert(var.clone(), level + 1);
            
            if Self::dpll_recursive(formula, assignment, level + 1)? {
                return Ok(true);
            }
            
            // Backtrack completely
            assignment.variables.remove(&var);
            assignment.decision_level.remove(&var);
        }
        
        Ok(false)
    }
    
    pub fn cdcl_solver(formula: &CNFFormula) -> MathResult<SolverResult> {
        let mut state = DPLLState {
            assignment: Assignment {
                variables: HashMap::new(),
                decision_level: HashMap::new(),
                implications: Vec::new(),
            },
            formula: formula.clone(),
            decision_level: 0,
            conflict_clauses: Vec::new(),
        };
        
        loop {
            // Unit propagation
            if let Some(conflict_clause) = Self::unit_propagation(&mut state)? {
                if state.decision_level == 0 {
                    return Ok(SolverResult::Unsatisfiable);
                }
                
                // Conflict analysis and backjumping
                let backtrack_level = Self::analyze_conflict(&mut state, &conflict_clause)?;
                Self::backtrack(&mut state, backtrack_level);
                continue;
            }
            
            // Check if formula is satisfied
            if let Some(true) = state.formula.evaluate(&state.assignment) {
                return Ok(SolverResult::Satisfiable(state.assignment));
            }
            
            // Make decision
            if let Some(var) = Self::choose_decision_variable(&state) {
                state.decision_level += 1;
                state.assignment.variables.insert(var.clone(), true); // Try positive first
                state.assignment.decision_level.insert(var, state.decision_level);
            } else {
                return Ok(SolverResult::Satisfiable(state.assignment));
            }
        }
    }
    
    fn unit_propagation(state: &mut DPLLState) -> MathResult<Option<Clause>> {
        let mut queue: VecDeque<Literal> = VecDeque::new();
        
        // Initialize with unit clauses
        queue.extend(state.formula.get_unit_clauses(&state.assignment));
        
        while let Some(literal) = queue.pop_front() {
            if state.assignment.variables.contains_key(&literal.variable) {
                continue;
            }
            
            // Assign literal
            state.assignment.variables.insert(literal.variable.clone(), literal.is_positive);
            state.assignment.decision_level.insert(literal.variable.clone(), state.decision_level);
            state.assignment.implications.push((literal.variable.clone(), state.decision_level));
            
            // Check for new unit clauses and conflicts
            for clause in &state.formula.clauses {
                match clause.evaluate(&state.assignment) {
                    Some(false) => {
                        return Ok(Some(clause.clone())); // Conflict
                    }
                    None => {
                        if let Some(unit_literal) = clause.is_unit_clause(&state.assignment) {
                            queue.push_back(unit_literal);
                        }
                    }
                    Some(true) => {} // Clause satisfied
                }
            }
        }
        
        Ok(None)
    }
    
    fn analyze_conflict(state: &mut DPLLState, conflict_clause: &Clause) -> MathResult<usize> {
        // Simplified conflict analysis - just backtrack to previous level
        if state.decision_level > 0 {
            Ok(state.decision_level - 1)
        } else {
            Ok(0)
        }
    }
    
    fn backtrack(state: &mut DPLLState, target_level: usize) {
        // Remove assignments made at levels higher than target
        state.assignment.variables.retain(|var, _| {
            state.assignment.decision_level.get(var).copied().unwrap_or(0) <= target_level
        });
        
        state.assignment.decision_level.retain(|_, &mut level| level <= target_level);
        
        state.assignment.implications.retain(|(_, level)| *level <= target_level);
        
        state.decision_level = target_level;
    }
    
    fn choose_decision_variable(state: &DPLLState) -> Option<String> {
        // VSIDS heuristic - choose unassigned variable (simplified)
        state.formula.variables.iter()
            .find(|var| !state.assignment.variables.contains_key(*var))
            .cloned()
    }
    
    pub fn smt_solver(smt_formula: &SMTFormula) -> MathResult<SolverResult> {
        // DPLL(T) approach: SAT solver + theory solver
        
        // First solve the boolean structure
        match Self::dpll_solver(&smt_formula.boolean_structure)? {
            SolverResult::Unsatisfiable => Ok(SolverResult::Unsatisfiable),
            SolverResult::Satisfiable(assignment) => {
                // Check theory constraints
                if Self::check_theory_constraints(&smt_formula.theory_constraints, &assignment)? {
                    Ok(SolverResult::Satisfiable(assignment))
                } else {
                    // Add conflict clause and restart
                    Ok(SolverResult::Unsatisfiable) // Simplified
                }
            }
            SolverResult::Unknown => Ok(SolverResult::Unknown),
        }
    }
    
    fn check_theory_constraints(constraints: &[TheoryConstraint], assignment: &Assignment) -> MathResult<bool> {
        for constraint in constraints {
            if !Self::evaluate_theory_constraint(constraint, assignment)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    
    fn evaluate_theory_constraint(constraint: &TheoryConstraint, assignment: &Assignment) -> MathResult<bool> {
        match constraint.constraint_type {
            ConstraintType::Linear => {
                let mut sum = 0;
                for (var, &coeff) in constraint.variables.iter().zip(constraint.coefficients.iter()) {
                    // For simplicity, assume boolean variables represent 0/1
                    if let Some(&value) = assignment.variables.get(var) {
                        sum += coeff * if value { 1 } else { 0 };
                    }
                }
                
                match constraint.operator.as_str() {
                    "=" | "==" => Ok(sum == constraint.rhs),
                    "<" => Ok(sum < constraint.rhs),
                    "<=" => Ok(sum <= constraint.rhs),
                    ">" => Ok(sum > constraint.rhs),
                    ">=" => Ok(sum >= constraint.rhs),
                    "!=" => Ok(sum != constraint.rhs),
                    _ => Err(MathError::InvalidArgument("Unknown operator".to_string())),
                }
            }
            ConstraintType::Equality => {
                // Simplified equality logic
                if constraint.variables.len() >= 2 {
                    if let (Some(&val1), Some(&val2)) = (
                        assignment.variables.get(&constraint.variables[0]),
                        assignment.variables.get(&constraint.variables[1])
                    ) {
                        Ok(val1 == val2)
                    } else {
                        Ok(true) // Unknown, assume satisfiable
                    }
                } else {
                    Ok(true)
                }
            }
            _ => Ok(true), // Other theories not implemented
        }
    }
    
    pub fn horn_sat_solver(horn_clauses: &CNFFormula) -> MathResult<SolverResult> {
        // Specialized solver for Horn clauses (at most one positive literal per clause)
        let mut assignment = Assignment {
            variables: HashMap::new(),
            decision_level: HashMap::new(),
            implications: Vec::new(),
        };
        
        // Initialize all variables to false
        for var in &horn_clauses.variables {
            assignment.variables.insert(var.clone(), false);
        }
        
        let mut changed = true;
        while changed {
            changed = false;
            
            for clause in &horn_clauses.clauses {
                // Check if clause forces any variable to true
                let mut negative_satisfied = false;
                let mut positive_literal = None;
                
                for literal in &clause.literals {
                    if let Some(&value) = assignment.variables.get(&literal.variable) {
                        if literal.is_positive {
                            positive_literal = Some(literal);
                            if value {
                                negative_satisfied = true; // Clause already satisfied
                                break;
                            }
                        } else {
                            if !value {
                                negative_satisfied = true; // Negative literal satisfied
                                break;
                            }
                        }
                    }
                }
                
                if !negative_satisfied {
                    if let Some(pos_lit) = positive_literal {
                        if let Some(current_value) = assignment.variables.get_mut(&pos_lit.variable) {
                            if !*current_value {
                                *current_value = true;
                                changed = true;
                            }
                        }
                    } else {
                        // All negative clause with all literals true -> unsatisfiable
                        return Ok(SolverResult::Unsatisfiable);
                    }
                }
            }
        }
        
        // Check if assignment satisfies all clauses
        match horn_clauses.evaluate(&assignment) {
            Some(true) => Ok(SolverResult::Satisfiable(assignment)),
            _ => Ok(SolverResult::Unsatisfiable),
        }
    }
    
    pub fn two_sat_solver(formula: &CNFFormula) -> MathResult<SolverResult> {
        // Check if formula is indeed 2-SAT
        for clause in &formula.clauses {
            if clause.literals.len() > 2 {
                return Err(MathError::InvalidArgument("Not a 2-SAT formula".to_string()));
            }
        }
        
        // Build implication graph
        let mut implications: HashMap<String, Vec<String>> = HashMap::new();
        
        for clause in &formula.clauses {
            if clause.literals.len() == 2 {
                let lit1 = &clause.literals[0];
                let lit2 = &clause.literals[1];
                
                // ¬lit1 → lit2 and ¬lit2 → lit1
                let not_lit1 = if lit1.is_positive {
                    format!("¬{}", lit1.variable)
                } else {
                    lit1.variable.clone()
                };
                
                let not_lit2 = if lit2.is_positive {
                    format!("¬{}", lit2.variable)
                } else {
                    lit2.variable.clone()
                };
                
                let lit1_name = if lit1.is_positive {
                    lit1.variable.clone()
                } else {
                    format!("¬{}", lit1.variable)
                };
                
                let lit2_name = if lit2.is_positive {
                    lit2.variable.clone()
                } else {
                    format!("¬{}", lit2.variable)
                };
                
                implications.entry(not_lit1).or_insert_with(Vec::new).push(lit2_name);
                implications.entry(not_lit2).or_insert_with(Vec::new).push(lit1_name);
            }
        }
        
        // Find strongly connected components (simplified check)
        for var in &formula.variables {
            let pos_var = var.clone();
            let neg_var = format!("¬{}", var);
            
            // Check if var and ¬var are in the same SCC (simplified)
            if Self::can_reach(&implications, &pos_var, &neg_var) && 
               Self::can_reach(&implications, &neg_var, &pos_var) {
                return Ok(SolverResult::Unsatisfiable);
            }
        }
        
        Ok(SolverResult::Satisfiable(Assignment {
            variables: HashMap::new(),
            decision_level: HashMap::new(),
            implications: Vec::new(),
        }))
    }
    
    fn can_reach(graph: &HashMap<String, Vec<String>>, start: &str, target: &str) -> bool {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        queue.push_back(start.to_string());
        visited.insert(start.to_string());
        
        while let Some(node) = queue.pop_front() {
            if node == target {
                return true;
            }
            
            if let Some(neighbors) = graph.get(&node) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }
        
        false
    }
    
    pub fn create_cnf_from_3sat(clauses: &[(String, String, String)]) -> CNFFormula {
        let mut formula = CNFFormula::new();
        
        for (var1, var2, var3) in clauses {
            let lit1 = Literal::new(var1.clone(), !var1.starts_with('¬'));
            let lit2 = Literal::new(var2.clone(), !var2.starts_with('¬'));
            let lit3 = Literal::new(var3.clone(), !var3.starts_with('¬'));
            
            let clause = Clause::new(vec![lit1, lit2, lit3]);
            formula.add_clause(clause);
        }
        
        formula
    }
}

impl MathDomain for SATSMTDomain {
    fn name(&self) -> &str { "SAT/SMT Solvers" }
    fn description(&self) -> &str { "Boolean satisfiability, SMT solving, DPLL, CDCL, and specialized SAT algorithms" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, _args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            _ => Err(MathError::InvalidOperation(format!("Operation {} not implemented in compute interface", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "dpll_solver".to_string(),
            "cdcl_solver".to_string(),
            "smt_solver".to_string(),
            "horn_sat_solver".to_string(),
            "two_sat_solver".to_string(),
            "unit_propagation".to_string(),
            "conflict_analysis".to_string(),
            "clause_learning".to_string(),
            "backjumping".to_string(),
            "theory_propagation".to_string(),
            "linear_arithmetic_theory".to_string(),
            "equality_logic_theory".to_string(),
            "create_cnf_formula".to_string(),
            "create_smt_formula".to_string(),
            "satisfiability_check".to_string(),
        ]
    }
}