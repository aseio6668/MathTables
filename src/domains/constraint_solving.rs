use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    pub name: String,
    pub domain: Vec<i32>,
    pub current_value: Option<i32>,
}

pub struct Constraint {
    pub name: String,
    pub variables: Vec<String>,
    pub constraint_type: ConstraintType,
    pub is_satisfied: Box<dyn Fn(&HashMap<String, i32>) -> bool>,
}

impl std::fmt::Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Constraint")
            .field("name", &self.name)
            .field("variables", &self.variables)
            .field("constraint_type", &self.constraint_type)
            .field("is_satisfied", &"<function>")
            .finish()
    }
}

impl Clone for Constraint {
    fn clone(&self) -> Self {
        // Note: Function cannot be cloned, so we create a dummy function
        Self {
            name: self.name.clone(),
            variables: self.variables.clone(),
            constraint_type: self.constraint_type.clone(),
            is_satisfied: Box::new(|_| true),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    Unary,
    Binary,
    NAry,
    AllDifferent,
    Linear,
    Global,
}

#[derive(Debug, Clone)]
pub struct CSP {
    pub variables: HashMap<String, Variable>,
    pub constraints: Vec<Constraint>,
}

#[derive(Debug, Clone)]
pub struct Assignment {
    pub variable_values: HashMap<String, i32>,
    pub is_complete: bool,
    pub is_consistent: bool,
}

#[derive(Debug, Clone)]
pub struct Arc {
    pub from_var: String,
    pub to_var: String,
    pub constraint_index: usize,
}

#[derive(Debug, Clone)]
pub struct SearchNode {
    pub assignment: Assignment,
    pub level: usize,
    pub variable_order: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum SearchStrategy {
    BacktrackingSearch,
    ForwardChecking,
    ArcConsistency,
    LocalSearch,
}

#[derive(Debug, Clone)]
pub enum VariableOrdering {
    LexicographicOrder,
    SmallestDomainFirst,
    MostConstrainedVariable,
    MostConstrainingVariable,
    DegreeHeuristic,
}

#[derive(Debug, Clone)]
pub enum ValueOrdering {
    LexicographicOrder,
    LeastConstrainingValue,
    RandomOrder,
}

pub struct ConstraintSolvingDomain;

impl Variable {
    pub fn new(name: String, domain: Vec<i32>) -> Self {
        Variable {
            name,
            domain,
            current_value: None,
        }
    }
    
    pub fn assign(&mut self, value: i32) -> MathResult<()> {
        if !self.domain.contains(&value) {
            return Err(MathError::InvalidArgument("Value not in domain".to_string()));
        }
        self.current_value = Some(value);
        Ok(())
    }
    
    pub fn unassign(&mut self) {
        self.current_value = None;
    }
    
    pub fn is_assigned(&self) -> bool {
        self.current_value.is_some()
    }
    
    pub fn domain_size(&self) -> usize {
        self.domain.len()
    }
    
    pub fn remove_from_domain(&mut self, value: i32) {
        self.domain.retain(|&x| x != value);
    }
    
    pub fn restore_domain(&mut self, original_domain: Vec<i32>) {
        self.domain = original_domain;
    }
}

impl CSP {
    pub fn new() -> Self {
        CSP {
            variables: HashMap::new(),
            constraints: Vec::new(),
        }
    }
    
    pub fn add_variable(&mut self, variable: Variable) {
        self.variables.insert(variable.name.clone(), variable);
    }
    
    pub fn add_constraint(&mut self, constraint: Constraint) {
        // Verify that all variables in constraint exist
        for var_name in &constraint.variables {
            if !self.variables.contains_key(var_name) {
                return; // Skip invalid constraints
            }
        }
        self.constraints.push(constraint);
    }
    
    pub fn is_consistent(&self, assignment: &Assignment) -> bool {
        for constraint in &self.constraints {
            let mut constraint_assignment = HashMap::new();
            let mut all_vars_assigned = true;
            
            for var_name in &constraint.variables {
                if let Some(&value) = assignment.variable_values.get(var_name) {
                    constraint_assignment.insert(var_name.clone(), value);
                } else {
                    all_vars_assigned = false;
                    break;
                }
            }
            
            if all_vars_assigned && !(constraint.is_satisfied)(&constraint_assignment) {
                return false;
            }
        }
        true
    }
    
    pub fn is_complete(&self, assignment: &Assignment) -> bool {
        self.variables.keys().all(|var_name| assignment.variable_values.contains_key(var_name))
    }
    
    pub fn get_unassigned_variables(&self, assignment: &Assignment) -> Vec<String> {
        self.variables.keys()
            .filter(|var_name| !assignment.variable_values.contains_key(*var_name))
            .cloned()
            .collect()
    }
}

impl ConstraintSolvingDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn backtracking_search(csp: &mut CSP) -> MathResult<Option<Assignment>> {
        let initial_assignment = Assignment {
            variable_values: HashMap::new(),
            is_complete: false,
            is_consistent: true,
        };
        
        Self::recursive_backtracking(csp, initial_assignment)
    }
    
    fn recursive_backtracking(csp: &mut CSP, assignment: Assignment) -> MathResult<Option<Assignment>> {
        if csp.is_complete(&assignment) {
            return Ok(Some(assignment));
        }
        
        let unassigned_vars = csp.get_unassigned_variables(&assignment);
        if unassigned_vars.is_empty() {
            return Ok(None);
        }
        
        let var_name = &unassigned_vars[0]; // Simple variable ordering
        
        if let Some(variable) = csp.variables.get(var_name) {
            let domain = variable.domain.clone();
            
            for &value in &domain {
                let mut new_assignment = assignment.clone();
                new_assignment.variable_values.insert(var_name.clone(), value);
                
                if csp.is_consistent(&new_assignment) {
                    if let Ok(Some(result)) = Self::recursive_backtracking(csp, new_assignment) {
                        return Ok(Some(result));
                    }
                }
            }
        }
        
        Ok(None)
    }
    
    pub fn forward_checking(csp: &mut CSP) -> MathResult<Option<Assignment>> {
        let initial_assignment = Assignment {
            variable_values: HashMap::new(),
            is_complete: false,
            is_consistent: true,
        };
        
        Self::forward_checking_recursive(csp, initial_assignment)
    }
    
    fn forward_checking_recursive(csp: &mut CSP, assignment: Assignment) -> MathResult<Option<Assignment>> {
        if csp.is_complete(&assignment) {
            return Ok(Some(assignment));
        }
        
        let unassigned_vars = csp.get_unassigned_variables(&assignment);
        if unassigned_vars.is_empty() {
            return Ok(None);
        }
        
        // Choose variable with smallest domain (MRV heuristic)
        let var_name = Self::select_variable_mrv(csp, &unassigned_vars);
        
        if let Some(variable) = csp.variables.get(&var_name) {
            let domain = variable.domain.clone();
            
            for &value in &domain {
                let mut new_assignment = assignment.clone();
                new_assignment.variable_values.insert(var_name.clone(), value);
                
                if csp.is_consistent(&new_assignment) {
                    // Forward checking: prune domains of unassigned variables
                    let mut pruned_values: HashMap<String, Vec<i32>> = HashMap::new();
                    let mut domain_wipeout = false;
                    
                    for other_var in &unassigned_vars {
                        if other_var != &var_name {
                            if let Some(other_variable) = csp.variables.get(other_var) {
                                let mut valid_values = Vec::new();
                                
                                for &other_value in &other_variable.domain {
                                    let mut test_assignment = new_assignment.clone();
                                    test_assignment.variable_values.insert(other_var.clone(), other_value);
                                    
                                    if csp.is_consistent(&test_assignment) {
                                        valid_values.push(other_value);
                                    }
                                }
                                
                                if valid_values.is_empty() {
                                    domain_wipeout = true;
                                    break;
                                }
                                
                                if valid_values.len() < other_variable.domain.len() {
                                    pruned_values.insert(other_var.clone(), valid_values);
                                }
                            }
                        }
                    }
                    
                    if !domain_wipeout {
                        // Apply domain reductions
                        let mut original_domains = HashMap::new();
                        for (var, new_domain) in &pruned_values {
                            if let Some(variable) = csp.variables.get(var) {
                                original_domains.insert(var.clone(), variable.domain.clone());
                            }
                            if let Some(variable) = csp.variables.get_mut(var) {
                                variable.domain = new_domain.clone();
                            }
                        }
                        
                        if let Ok(Some(result)) = Self::forward_checking_recursive(csp, new_assignment) {
                            return Ok(Some(result));
                        }
                        
                        // Restore domains
                        for (var, original_domain) in original_domains {
                            if let Some(variable) = csp.variables.get_mut(&var) {
                                variable.domain = original_domain;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(None)
    }
    
    fn select_variable_mrv(csp: &CSP, unassigned_vars: &[String]) -> String {
        unassigned_vars.iter()
            .min_by_key(|var_name| {
                csp.variables.get(*var_name).map_or(usize::MAX, |v| v.domain_size())
            })
            .cloned()
            .unwrap_or_else(|| unassigned_vars[0].clone())
    }
    
    pub fn arc_consistency_3(csp: &mut CSP) -> MathResult<bool> {
        let mut queue: VecDeque<Arc> = VecDeque::new();
        
        // Initialize queue with all arcs
        for (i, constraint) in csp.constraints.iter().enumerate() {
            if constraint.variables.len() == 2 {
                let var1 = &constraint.variables[0];
                let var2 = &constraint.variables[1];
                queue.push_back(Arc {
                    from_var: var1.clone(),
                    to_var: var2.clone(),
                    constraint_index: i,
                });
                queue.push_back(Arc {
                    from_var: var2.clone(),
                    to_var: var1.clone(),
                    constraint_index: i,
                });
            }
        }
        
        while let Some(arc) = queue.pop_front() {
            if Self::revise(csp, &arc)? {
                if let Some(variable) = csp.variables.get(&arc.from_var) {
                    if variable.domain.is_empty() {
                        return Ok(false);
                    }
                }
                
                // Add arcs (Xk, Xi) for each Xk ≠ Xj that shares a constraint with Xi
                for (i, constraint) in csp.constraints.iter().enumerate() {
                    if constraint.variables.contains(&arc.from_var) && i != arc.constraint_index {
                        for var in &constraint.variables {
                            if var != &arc.from_var && var != &arc.to_var {
                                queue.push_back(Arc {
                                    from_var: var.clone(),
                                    to_var: arc.from_var.clone(),
                                    constraint_index: i,
                                });
                            }
                        }
                    }
                }
            }
        }
        
        Ok(true)
    }
    
    fn revise(csp: &mut CSP, arc: &Arc) -> MathResult<bool> {
        let mut revised = false;
        
        if let Some(constraint) = csp.constraints.get(arc.constraint_index) {
            if let Some(from_var) = csp.variables.get(&arc.from_var) {
                let from_domain = from_var.domain.clone();
                let mut new_domain = Vec::new();
                
                for &x in &from_domain {
                    let mut has_support = false;
                    
                    if let Some(to_var) = csp.variables.get(&arc.to_var) {
                        for &y in &to_var.domain {
                            let mut test_assignment = HashMap::new();
                            test_assignment.insert(arc.from_var.clone(), x);
                            test_assignment.insert(arc.to_var.clone(), y);
                            
                            if (constraint.is_satisfied)(&test_assignment) {
                                has_support = true;
                                break;
                            }
                        }
                    }
                    
                    if has_support {
                        new_domain.push(x);
                    } else {
                        revised = true;
                    }
                }
                
                if let Some(from_var_mut) = csp.variables.get_mut(&arc.from_var) {
                    from_var_mut.domain = new_domain;
                }
            }
        }
        
        Ok(revised)
    }
    
    pub fn min_conflicts_local_search(csp: &mut CSP, max_steps: usize) -> MathResult<Option<Assignment>> {
        // Initialize with random complete assignment
        let mut assignment = Assignment {
            variable_values: HashMap::new(),
            is_complete: true,
            is_consistent: false,
        };
        
        for (var_name, variable) in &csp.variables {
            if !variable.domain.is_empty() {
                assignment.variable_values.insert(var_name.clone(), variable.domain[0]);
            }
        }
        
        for _ in 0..max_steps {
            if csp.is_consistent(&assignment) {
                assignment.is_consistent = true;
                return Ok(Some(assignment));
            }
            
            // Find conflicted variable
            let conflicted_vars = Self::get_conflicted_variables(csp, &assignment);
            if conflicted_vars.is_empty() {
                break;
            }
            
            let var_to_change = &conflicted_vars[0]; // Pick first conflicted variable
            
            // Find value that minimizes conflicts
            if let Some(variable) = csp.variables.get(var_to_change) {
                let mut best_value = variable.domain[0];
                let mut min_conflicts = usize::MAX;
                
                for &value in &variable.domain {
                    let mut test_assignment = assignment.clone();
                    test_assignment.variable_values.insert(var_to_change.clone(), value);
                    
                    let conflicts = Self::count_conflicts(csp, &test_assignment);
                    if conflicts < min_conflicts {
                        min_conflicts = conflicts;
                        best_value = value;
                    }
                }
                
                assignment.variable_values.insert(var_to_change.clone(), best_value);
            }
        }
        
        Ok(None)
    }
    
    fn get_conflicted_variables(csp: &CSP, assignment: &Assignment) -> Vec<String> {
        let mut conflicted = Vec::new();
        
        for constraint in &csp.constraints {
            let mut constraint_assignment = HashMap::new();
            for var_name in &constraint.variables {
                if let Some(&value) = assignment.variable_values.get(var_name) {
                    constraint_assignment.insert(var_name.clone(), value);
                }
            }
            
            if constraint_assignment.len() == constraint.variables.len() {
                if !(constraint.is_satisfied)(&constraint_assignment) {
                    for var_name in &constraint.variables {
                        if !conflicted.contains(var_name) {
                            conflicted.push(var_name.clone());
                        }
                    }
                }
            }
        }
        
        conflicted
    }
    
    fn count_conflicts(csp: &CSP, assignment: &Assignment) -> usize {
        let mut conflicts = 0;
        
        for constraint in &csp.constraints {
            let mut constraint_assignment = HashMap::new();
            for var_name in &constraint.variables {
                if let Some(&value) = assignment.variable_values.get(var_name) {
                    constraint_assignment.insert(var_name.clone(), value);
                }
            }
            
            if constraint_assignment.len() == constraint.variables.len() {
                if !(constraint.is_satisfied)(&constraint_assignment) {
                    conflicts += 1;
                }
            }
        }
        
        conflicts
    }
    
    pub fn all_different_constraint(variables: Vec<String>) -> Constraint {
        Constraint {
            name: "AllDifferent".to_string(),
            variables: variables.clone(),
            constraint_type: ConstraintType::AllDifferent,
            is_satisfied: Box::new(move |assignment| {
                let mut values = HashSet::new();
                for var in &variables {
                    if let Some(&value) = assignment.get(var) {
                        if !values.insert(value) {
                            return false; // Duplicate value found
                        }
                    }
                }
                true
            }),
        }
    }
    
    pub fn linear_constraint(
        variables: Vec<String>,
        coefficients: Vec<i32>,
        operator: String,
        rhs: i32
    ) -> MathResult<Constraint> {
        if variables.len() != coefficients.len() {
            return Err(MathError::InvalidArgument("Variables and coefficients must have same length".to_string()));
        }
        
        let vars_clone = variables.clone();
        let coeffs_clone = coefficients.clone();
        let op_clone = operator.clone();
        
        Ok(Constraint {
            name: format!("Linear_{}", operator),
            variables,
            constraint_type: ConstraintType::Linear,
            is_satisfied: Box::new(move |assignment| {
                let mut sum = 0;
                for (var, &coeff) in vars_clone.iter().zip(coeffs_clone.iter()) {
                    if let Some(&value) = assignment.get(var) {
                        sum += coeff * value;
                    } else {
                        return true; // Not all variables assigned yet
                    }
                }
                
                match op_clone.as_str() {
                    "=" | "==" => sum == rhs,
                    "<" => sum < rhs,
                    "<=" => sum <= rhs,
                    ">" => sum > rhs,
                    ">=" => sum >= rhs,
                    "!=" => sum != rhs,
                    _ => false,
                }
            }),
        })
    }
    
    pub fn n_queens_problem(n: usize) -> MathResult<CSP> {
        let mut csp = CSP::new();
        
        // Variables: one for each row, domain is column positions
        for i in 0..n {
            let domain: Vec<i32> = (0..n as i32).collect();
            let variable = Variable::new(format!("queen_{}", i), domain);
            csp.add_variable(variable);
        }
        
        // Constraints: no two queens attack each other
        for i in 0..n {
            for j in (i+1)..n {
                let var1 = format!("queen_{}", i);
                let var2 = format!("queen_{}", j);
                let row_diff = (j - i) as i32;
                
                let constraint = Constraint {
                    name: format!("NoAttack_{}_{}", i, j),
                    variables: vec![var1.clone(), var2.clone()],
                    constraint_type: ConstraintType::Binary,
                    is_satisfied: Box::new(move |assignment| {
                        if let (Some(&col1), Some(&col2)) = (assignment.get(&var1), assignment.get(&var2)) {
                            // Different columns and not on same diagonal
                            col1 != col2 && 
                            (col1 - col2).abs() != row_diff
                        } else {
                            true
                        }
                    }),
                };
                
                csp.add_constraint(constraint);
            }
        }
        
        Ok(csp)
    }
    
    pub fn graph_coloring_problem(
        vertices: Vec<String>,
        edges: Vec<(String, String)>,
        num_colors: usize
    ) -> MathResult<CSP> {
        let mut csp = CSP::new();
        
        // Variables: one for each vertex, domain is colors
        let domain: Vec<i32> = (0..num_colors as i32).collect();
        for vertex in &vertices {
            let variable = Variable::new(vertex.clone(), domain.clone());
            csp.add_variable(variable);
        }
        
        // Constraints: adjacent vertices have different colors
        for (v1, v2) in edges {
            if vertices.contains(&v1) && vertices.contains(&v2) {
                let var1 = v1.clone();
                let var2 = v2.clone();
                
                let constraint = Constraint {
                    name: format!("DifferentColors_{}_{}", v1, v2),
                    variables: vec![var1.clone(), var2.clone()],
                    constraint_type: ConstraintType::Binary,
                    is_satisfied: Box::new(move |assignment| {
                        if let (Some(&color1), Some(&color2)) = (assignment.get(&var1), assignment.get(&var2)) {
                            color1 != color2
                        } else {
                            true
                        }
                    }),
                };
                
                csp.add_constraint(constraint);
            }
        }
        
        Ok(csp)
    }
}

impl MathDomain for ConstraintSolvingDomain {
    fn name(&self) -> &str { "Constraint Solving" }
    fn description(&self) -> &str { "Constraint satisfaction problems, backtracking, arc consistency, and local search algorithms" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, _args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            _ => Err(MathError::InvalidOperation(format!("Operation {} not implemented in compute interface", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "backtracking_search".to_string(),
            "forward_checking".to_string(),
            "arc_consistency_3".to_string(),
            "min_conflicts_search".to_string(),
            "all_different_constraint".to_string(),
            "linear_constraint".to_string(),
            "n_queens_problem".to_string(),
            "graph_coloring_problem".to_string(),
            "sudoku_solver".to_string(),
            "constraint_propagation".to_string(),
            "variable_ordering_heuristics".to_string(),
            "value_ordering_heuristics".to_string(),
            "constraint_learning".to_string(),
            "restart_strategies".to_string(),
        ]
    }
}