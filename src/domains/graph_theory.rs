use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct Graph {
    pub vertices: HashSet<usize>,
    pub edges: Vec<Edge>,
    pub is_directed: bool,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub weight: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ShortestPathResult {
    pub distances: HashMap<usize, f64>,
    pub predecessors: HashMap<usize, Option<usize>>,
    pub path: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct MST {
    pub edges: Vec<Edge>,
    pub total_weight: f64,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct State {
    cost: i64,
    vertex: usize,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct GraphTheoryDomain;

impl GraphTheoryDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn create_graph(vertices: Vec<usize>, edges: Vec<Edge>, is_directed: bool) -> Graph {
        let vertex_set = vertices.into_iter().collect();
        Graph {
            vertices: vertex_set,
            edges,
            is_directed,
        }
    }
    
    pub fn add_vertex(graph: &mut Graph, vertex: usize) {
        graph.vertices.insert(vertex);
    }
    
    pub fn add_edge(graph: &mut Graph, from: usize, to: usize, weight: Option<f64>) -> MathResult<()> {
        if !graph.vertices.contains(&from) || !graph.vertices.contains(&to) {
            return Err(MathError::InvalidArgument("Vertices must exist in graph before adding edge".to_string()));
        }
        
        graph.edges.push(Edge { from, to, weight });
        Ok(())
    }
    
    pub fn breadth_first_search(graph: &Graph, start: usize) -> MathResult<Vec<usize>> {
        if !graph.vertices.contains(&start) {
            return Err(MathError::InvalidArgument("Start vertex not in graph".to_string()));
        }
        
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();
        
        queue.push_back(start);
        visited.insert(start);
        
        while let Some(vertex) = queue.pop_front() {
            result.push(vertex);
            
            let neighbors = Self::get_neighbors(graph, vertex);
            for neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
        
        Ok(result)
    }
    
    pub fn depth_first_search(graph: &Graph, start: usize) -> MathResult<Vec<usize>> {
        if !graph.vertices.contains(&start) {
            return Err(MathError::InvalidArgument("Start vertex not in graph".to_string()));
        }
        
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        
        Self::dfs_recursive(graph, start, &mut visited, &mut result);
        
        Ok(result)
    }
    
    fn dfs_recursive(graph: &Graph, vertex: usize, visited: &mut HashSet<usize>, result: &mut Vec<usize>) {
        visited.insert(vertex);
        result.push(vertex);
        
        let neighbors = Self::get_neighbors(graph, vertex);
        for neighbor in neighbors {
            if !visited.contains(&neighbor) {
                Self::dfs_recursive(graph, neighbor, visited, result);
            }
        }
    }
    
    pub fn dijkstra_shortest_path(graph: &Graph, start: usize) -> MathResult<ShortestPathResult> {
        if !graph.vertices.contains(&start) {
            return Err(MathError::InvalidArgument("Start vertex not in graph".to_string()));
        }
        
        let mut distances: HashMap<usize, f64> = HashMap::new();
        let mut predecessors: HashMap<usize, Option<usize>> = HashMap::new();
        let mut heap = BinaryHeap::new();
        
        for &vertex in &graph.vertices {
            distances.insert(vertex, f64::INFINITY);
            predecessors.insert(vertex, None);
        }
        distances.insert(start, 0.0);
        
        heap.push(State { cost: 0, vertex: start });
        
        while let Some(State { cost, vertex }) = heap.pop() {
            if cost as f64 > distances[&vertex] {
                continue;
            }
            
            for edge in &graph.edges {
                if edge.from == vertex {
                    let neighbor = edge.to;
                    let weight = edge.weight.unwrap_or(1.0);
                    let alt_distance = distances[&vertex] + weight;
                    
                    if alt_distance < distances[&neighbor] {
                        distances.insert(neighbor, alt_distance);
                        predecessors.insert(neighbor, Some(vertex));
                        heap.push(State {
                            cost: (alt_distance * 1000.0) as i64,
                            vertex: neighbor,
                        });
                    }
                }
                
                if !graph.is_directed && edge.to == vertex {
                    let neighbor = edge.from;
                    let weight = edge.weight.unwrap_or(1.0);
                    let alt_distance = distances[&vertex] + weight;
                    
                    if alt_distance < distances[&neighbor] {
                        distances.insert(neighbor, alt_distance);
                        predecessors.insert(neighbor, Some(vertex));
                        heap.push(State {
                            cost: (alt_distance * 1000.0) as i64,
                            vertex: neighbor,
                        });
                    }
                }
            }
        }
        
        Ok(ShortestPathResult {
            distances,
            predecessors,
            path: Vec::new(), // Path reconstruction would be done separately
        })
    }
    
    pub fn kruskal_mst(graph: &Graph) -> MathResult<MST> {
        if graph.is_directed {
            return Err(MathError::InvalidArgument("MST requires undirected graph".to_string()));
        }
        
        let mut edges = graph.edges.clone();
        edges.sort_by(|a, b| {
            let weight_a = a.weight.unwrap_or(1.0);
            let weight_b = b.weight.unwrap_or(1.0);
            weight_a.partial_cmp(&weight_b).unwrap_or(Ordering::Equal)
        });
        
        let mut parent: HashMap<usize, usize> = HashMap::new();
        let mut rank: HashMap<usize, usize> = HashMap::new();
        
        for &vertex in &graph.vertices {
            parent.insert(vertex, vertex);
            rank.insert(vertex, 0);
        }
        
        let mut mst_edges = Vec::new();
        let mut total_weight = 0.0;
        
        for edge in edges {
            let root_from = Self::find_root(&mut parent, edge.from);
            let root_to = Self::find_root(&mut parent, edge.to);
            
            if root_from != root_to {
                mst_edges.push(edge.clone());
                total_weight += edge.weight.unwrap_or(1.0);
                
                Self::union_sets(&mut parent, &mut rank, root_from, root_to);
            }
        }
        
        Ok(MST {
            edges: mst_edges,
            total_weight,
        })
    }
    
    fn find_root(parent: &mut HashMap<usize, usize>, vertex: usize) -> usize {
        if parent[&vertex] != vertex {
            let root = Self::find_root(parent, parent[&vertex]);
            parent.insert(vertex, root);
        }
        parent[&vertex]
    }
    
    fn union_sets(parent: &mut HashMap<usize, usize>, rank: &mut HashMap<usize, usize>, x: usize, y: usize) {
        if rank[&x] < rank[&y] {
            parent.insert(x, y);
        } else if rank[&x] > rank[&y] {
            parent.insert(y, x);
        } else {
            parent.insert(y, x);
            rank.insert(x, rank[&x] + 1);
        }
    }
    
    pub fn topological_sort(graph: &Graph) -> MathResult<Vec<usize>> {
        if !graph.is_directed {
            return Err(MathError::InvalidArgument("Topological sort requires directed graph".to_string()));
        }
        
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        for &vertex in &graph.vertices {
            in_degree.insert(vertex, 0);
        }
        
        for edge in &graph.edges {
            *in_degree.get_mut(&edge.to).unwrap() += 1;
        }
        
        let mut queue = VecDeque::new();
        for (&vertex, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(vertex);
            }
        }
        
        let mut result = Vec::new();
        
        while let Some(vertex) = queue.pop_front() {
            result.push(vertex);
            
            for edge in &graph.edges {
                if edge.from == vertex {
                    let neighbor = edge.to;
                    *in_degree.get_mut(&neighbor).unwrap() -= 1;
                    if in_degree[&neighbor] == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        
        if result.len() != graph.vertices.len() {
            return Err(MathError::ComputationError("Graph contains cycles - no topological ordering exists".to_string()));
        }
        
        Ok(result)
    }
    
    pub fn is_bipartite(graph: &Graph) -> MathResult<bool> {
        let mut colors: HashMap<usize, i32> = HashMap::new();
        
        for &vertex in &graph.vertices {
            if !colors.contains_key(&vertex) {
                if !Self::bipartite_dfs(graph, vertex, 0, &mut colors) {
                    return Ok(false);
                }
            }
        }
        
        Ok(true)
    }
    
    fn bipartite_dfs(graph: &Graph, vertex: usize, color: i32, colors: &mut HashMap<usize, i32>) -> bool {
        colors.insert(vertex, color);
        
        let neighbors = Self::get_neighbors(graph, vertex);
        for neighbor in neighbors {
            if let Some(&neighbor_color) = colors.get(&neighbor) {
                if neighbor_color == color {
                    return false;
                }
            } else {
                if !Self::bipartite_dfs(graph, neighbor, 1 - color, colors) {
                    return false;
                }
            }
        }
        
        true
    }
    
    pub fn connected_components(graph: &Graph) -> Vec<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut components = Vec::new();
        
        for &vertex in &graph.vertices {
            if !visited.contains(&vertex) {
                let mut component = Vec::new();
                Self::dfs_component(graph, vertex, &mut visited, &mut component);
                components.push(component);
            }
        }
        
        components
    }
    
    fn dfs_component(graph: &Graph, vertex: usize, visited: &mut HashSet<usize>, component: &mut Vec<usize>) {
        visited.insert(vertex);
        component.push(vertex);
        
        let neighbors = Self::get_neighbors(graph, vertex);
        for neighbor in neighbors {
            if !visited.contains(&neighbor) {
                Self::dfs_component(graph, neighbor, visited, component);
            }
        }
    }
    
    fn get_neighbors(graph: &Graph, vertex: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();
        
        for edge in &graph.edges {
            if edge.from == vertex {
                neighbors.push(edge.to);
            } else if !graph.is_directed && edge.to == vertex {
                neighbors.push(edge.from);
            }
        }
        
        neighbors
    }
    
    pub fn graph_density(graph: &Graph) -> f64 {
        let num_vertices = graph.vertices.len() as f64;
        let num_edges = graph.edges.len() as f64;
        
        if num_vertices <= 1.0 {
            return 0.0;
        }
        
        if graph.is_directed {
            num_edges / (num_vertices * (num_vertices - 1.0))
        } else {
            (2.0 * num_edges) / (num_vertices * (num_vertices - 1.0))
        }
    }
    
    pub fn clustering_coefficient(graph: &Graph, vertex: usize) -> MathResult<f64> {
        if !graph.vertices.contains(&vertex) {
            return Err(MathError::InvalidArgument("Vertex not in graph".to_string()));
        }
        
        let neighbors = Self::get_neighbors(graph, vertex);
        let degree = neighbors.len();
        
        if degree < 2 {
            return Ok(0.0);
        }
        
        let mut triangles = 0;
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                if Self::has_edge(graph, neighbors[i], neighbors[j]) {
                    triangles += 1;
                }
            }
        }
        
        let possible_edges = degree * (degree - 1) / 2;
        Ok(triangles as f64 / possible_edges as f64)
    }
    
    fn has_edge(graph: &Graph, from: usize, to: usize) -> bool {
        for edge in &graph.edges {
            if (edge.from == from && edge.to == to) || 
               (!graph.is_directed && edge.from == to && edge.to == from) {
                return true;
            }
        }
        false
    }
}

impl MathDomain for GraphTheoryDomain {
    fn name(&self) -> &str { "Graph Theory" }
    fn description(&self) -> &str { "Graph algorithms including search, shortest paths, MST, and graph analysis" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "graph_density" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("graph_density requires 1 argument".to_string()));
                }
                let graph = args[0].downcast_ref::<Graph>().ok_or_else(|| MathError::InvalidArgument("Argument must be Graph".to_string()))?;
                Ok(Box::new(Self::graph_density(graph)))
            },
            "is_bipartite" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("is_bipartite requires 1 argument".to_string()));
                }
                let graph = args[0].downcast_ref::<Graph>().ok_or_else(|| MathError::InvalidArgument("Argument must be Graph".to_string()))?;
                Ok(Box::new(Self::is_bipartite(graph)?))
            },
            "connected_components" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("connected_components requires 1 argument".to_string()));
                }
                let graph = args[0].downcast_ref::<Graph>().ok_or_else(|| MathError::InvalidArgument("Argument must be Graph".to_string()))?;
                Ok(Box::new(Self::connected_components(graph)))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "create_graph".to_string(),
            "add_vertex".to_string(),
            "add_edge".to_string(),
            "breadth_first_search".to_string(),
            "depth_first_search".to_string(),
            "dijkstra_shortest_path".to_string(),
            "kruskal_mst".to_string(),
            "topological_sort".to_string(),
            "is_bipartite".to_string(),
            "connected_components".to_string(),
            "graph_density".to_string(),
            "clustering_coefficient".to_string(),
        ]
    }
}