use crate::core::{MathDomain, MathResult, MathError};
use std::collections::{HashMap, HashSet, VecDeque};
use std::any::Any;

pub struct DiscreteDomain;

impl DiscreteDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn factorial(n: u64) -> u64 {
        if n <= 1 { 1 } else { n * Self::factorial(n - 1) }
    }
    
    pub fn combination(n: u64, k: u64) -> MathResult<u64> {
        if k > n {
            return Err(MathError::InvalidArgument("k cannot be greater than n".to_string()));
        }
        Ok(Self::factorial(n) / (Self::factorial(k) * Self::factorial(n - k)))
    }
    
    pub fn permutation(n: u64, k: u64) -> MathResult<u64> {
        if k > n {
            return Err(MathError::InvalidArgument("k cannot be greater than n".to_string()));
        }
        Ok(Self::factorial(n) / Self::factorial(n - k))
    }
    
    pub fn binomial_coefficient(n: u64, k: u64) -> MathResult<u64> {
        Self::combination(n, k)
    }
    
    pub fn catalan_number(n: u64) -> u64 {
        if n == 0 { return 1; }
        Self::binomial_coefficient(2 * n, n).unwrap_or(0) / (n + 1)
    }
    
    pub fn fibonacci_sequence(n: usize) -> Vec<u64> {
        if n == 0 { return vec![]; }
        if n == 1 { return vec![0]; }
        
        let mut seq = vec![0, 1];
        for i in 2..n {
            seq.push(seq[i-1] + seq[i-2]);
        }
        seq
    }
    
    pub fn graph_shortest_path(graph: &HashMap<usize, Vec<usize>>, start: usize, end: usize) -> Option<Vec<usize>> {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent: HashMap<usize, usize> = HashMap::new();
        
        queue.push_back(start);
        visited.insert(start);
        
        while let Some(current) = queue.pop_front() {
            if current == end {
                let mut path = Vec::new();
                let mut node = end;
                path.push(node);
                
                while let Some(&p) = parent.get(&node) {
                    path.push(p);
                    node = p;
                }
                
                path.reverse();
                return Some(path);
            }
            
            if let Some(neighbors) = graph.get(&current) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        parent.insert(neighbor, current);
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        
        None
    }
    
    pub fn power_set<T: Clone>(set: &[T]) -> Vec<Vec<T>> {
        let mut result = Vec::new();
        let n = set.len();
        
        for i in 0..(1 << n) {
            let mut subset = Vec::new();
            for j in 0..n {
                if (i >> j) & 1 == 1 {
                    subset.push(set[j].clone());
                }
            }
            result.push(subset);
        }
        
        result
    }
    
    pub fn is_bipartite(graph: &HashMap<usize, Vec<usize>>) -> bool {
        let mut colors: HashMap<usize, bool> = HashMap::new();
        let mut queue = VecDeque::new();
        
        for &start in graph.keys() {
            if colors.contains_key(&start) { continue; }
            
            queue.push_back(start);
            colors.insert(start, true);
            
            while let Some(current) = queue.pop_front() {
                if let Some(neighbors) = graph.get(&current) {
                    for &neighbor in neighbors {
                        match colors.get(&neighbor) {
                            Some(&color) => {
                                if color == colors[&current] {
                                    return false;
                                }
                            },
                            None => {
                                colors.insert(neighbor, !colors[&current]);
                                queue.push_back(neighbor);
                            }
                        }
                    }
                }
            }
        }
        
        true
    }
    
    pub fn topological_sort(graph: &HashMap<usize, Vec<usize>>) -> Option<Vec<usize>> {
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        
        for (&node, _) in graph {
            in_degree.entry(node).or_insert(0);
        }
        
        for (_, neighbors) in graph {
            for &neighbor in neighbors {
                *in_degree.entry(neighbor).or_insert(0) += 1;
            }
        }
        
        for (&node, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node);
            }
        }
        
        while let Some(current) = queue.pop_front() {
            result.push(current);
            
            if let Some(neighbors) = graph.get(&current) {
                for &neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(&neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }
        
        if result.len() == in_degree.len() {
            Some(result)
        } else {
            None // Cycle detected
        }
    }
}

impl MathDomain for DiscreteDomain {
    fn name(&self) -> &str { "Discrete Mathematics" }
    fn description(&self) -> &str { "Mathematical domain for combinatorics, graph theory, and discrete structures" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "factorial" => {
                if args.len() != 1 { 
                    return Err(MathError::InvalidArgument("factorial requires 1 argument".to_string())); 
                }
                let n = args[0].downcast_ref::<u64>().ok_or_else(|| MathError::InvalidArgument("Argument must be u64".to_string()))?;
                Ok(Box::new(Self::factorial(*n)))
            },
            "combination" => {
                if args.len() != 2 { 
                    return Err(MathError::InvalidArgument("combination requires 2 arguments".to_string())); 
                }
                let n = args[0].downcast_ref::<u64>().ok_or_else(|| MathError::InvalidArgument("First argument must be u64".to_string()))?;
                let k = args[1].downcast_ref::<u64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be u64".to_string()))?;
                Ok(Box::new(Self::combination(*n, *k)?))
            },
            "permutation" => {
                if args.len() != 2 { 
                    return Err(MathError::InvalidArgument("permutation requires 2 arguments".to_string())); 
                }
                let n = args[0].downcast_ref::<u64>().ok_or_else(|| MathError::InvalidArgument("First argument must be u64".to_string()))?;
                let k = args[1].downcast_ref::<u64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be u64".to_string()))?;
                Ok(Box::new(Self::permutation(*n, *k)?))
            },
            "catalan_number" => {
                if args.len() != 1 { 
                    return Err(MathError::InvalidArgument("catalan_number requires 1 argument".to_string())); 
                }
                let n = args[0].downcast_ref::<u64>().ok_or_else(|| MathError::InvalidArgument("Argument must be u64".to_string()))?;
                Ok(Box::new(Self::catalan_number(*n)))
            },
            "fibonacci_sequence" => {
                if args.len() != 1 { 
                    return Err(MathError::InvalidArgument("fibonacci_sequence requires 1 argument".to_string())); 
                }
                let n = args[0].downcast_ref::<usize>().ok_or_else(|| MathError::InvalidArgument("Argument must be usize".to_string()))?;
                Ok(Box::new(Self::fibonacci_sequence(*n)))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "factorial".to_string(),
            "combination".to_string(),
            "permutation".to_string(),
            "binomial_coefficient".to_string(),
            "catalan_number".to_string(),
            "fibonacci_sequence".to_string(),
            "graph_shortest_path".to_string(),
            "power_set".to_string(),
            "is_bipartite".to_string(),
            "topological_sort".to_string(),
        ]
    }
}