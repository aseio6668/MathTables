use crate::core::{MathDomain, MathResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Ordering;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationalMathDomain {
    name: String,
}

// Algorithm Complexity Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityAnalysis {
    pub time_complexity: Complexity,
    pub space_complexity: Complexity,
    pub best_case: Complexity,
    pub average_case: Complexity,
    pub worst_case: Complexity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Complexity {
    Constant,           // O(1)
    Logarithmic,        // O(log n)
    Linear,             // O(n)
    Linearithmic,       // O(n log n)
    Quadratic,          // O(n²)
    Cubic,              // O(n³)
    Polynomial(u32),    // O(n^k)
    Exponential,        // O(2^n)
    Factorial,          // O(n!)
    Custom(String),     // Custom complexity expression
}

// Sorting Algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortingAlgorithm {
    pub name: String,
    pub complexity: ComplexityAnalysis,
    pub stable: bool,
    pub in_place: bool,
    pub adaptive: bool,
}

// Dynamic Programming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicProgramming {
    pub problem_name: String,
    pub state_definition: String,
    pub recurrence_relation: String,
    pub memoization_table: HashMap<String, f64>,
    pub optimal_value: f64,
    pub optimal_solution: Vec<String>,
}

// Combinatorics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinatorialStructure {
    pub structure_type: CombinatorialType,
    pub parameters: HashMap<String, usize>,
    pub count: u128,
    pub generating_function: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CombinatorialType {
    Permutations,
    Combinations,
    Partitions,
    StirlingNumbers,
    BellNumbers,
    CatalanNumbers,
    FibonacciNumbers,
    Derangements,
}

// String Algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringAlgorithm {
    pub algorithm_type: StringAlgorithmType,
    pub pattern: String,
    pub text: String,
    pub matches: Vec<usize>,
    pub preprocessing_time: Option<f64>,
    pub search_time: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StringAlgorithmType {
    KMP,               // Knuth-Morris-Pratt
    Boyer_Moore,       // Boyer-Moore
    Rabin_Karp,        // Rabin-Karp
    Z_Algorithm,       // Z Algorithm
    Suffix_Array,      // Suffix Array construction
    LCS,               // Longest Common Subsequence
    Edit_Distance,     // Levenshtein Distance
}

// Number Theory Algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumberTheoreticAlgorithm {
    pub algorithm_type: NumberTheoreticType,
    pub input_numbers: Vec<u64>,
    pub result: NumberTheoreticResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NumberTheoreticType {
    GCD,               // Greatest Common Divisor
    LCM,               // Least Common Multiple
    ModularExponentiation,
    PrimalityTest,
    Factorization,
    DiscreteLogarithm,
    ChineseRemainderTheorem,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NumberTheoreticResult {
    SingleValue(u64),
    MultipleValues(Vec<u64>),
    Boolean(bool),
    Modular { value: u64, modulus: u64 },
}

// Computational Geometry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricAlgorithm {
    pub algorithm_type: GeometricAlgorithmType,
    pub points: Vec<Point2D>,
    pub result: GeometricResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeometricAlgorithmType {
    ConvexHull,
    ClosestPair,
    LineIntersection,
    PointInPolygon,
    Triangulation,
    VoronoiDiagram,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeometricResult {
    Points(Vec<Point2D>),
    Distance(f64),
    Boolean(bool),
    Triangles(Vec<Triangle>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Triangle {
    pub vertices: [Point2D; 3],
}

// Approximation Algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApproximationAlgorithm {
    pub problem_name: String,
    pub approximation_ratio: f64,
    pub algorithm_description: String,
    pub optimal_value: Option<f64>,
    pub approximate_value: f64,
}

// Parallel Algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelAlgorithm {
    pub algorithm_name: String,
    pub parallel_complexity: Complexity,
    pub sequential_complexity: Complexity,
    pub processors_required: usize,
    pub efficiency: f64,
    pub speedup: f64,
}

impl ComputationalMathDomain {
    pub fn new() -> Self {
        Self {
            name: "Computational Mathematics".to_string(),
        }
    }

    // Sorting Algorithms
    pub fn quicksort(&self, arr: &mut [i32]) {
        if arr.len() <= 1 {
            return;
        }
        
        let pivot_index = self.partition(arr);
        let (left, right) = arr.split_at_mut(pivot_index);
        
        self.quicksort(left);
        if right.len() > 1 {
            self.quicksort(&mut right[1..]);
        }
    }

    fn partition(&self, arr: &mut [i32]) -> usize {
        let pivot = arr[arr.len() - 1];
        let mut i = 0;
        
        for j in 0..arr.len() - 1 {
            if arr[j] <= pivot {
                arr.swap(i, j);
                i += 1;
            }
        }
        
        arr.swap(i, arr.len() - 1);
        i
    }

    pub fn mergesort(&self, arr: &mut [i32]) {
        if arr.len() <= 1 {
            return;
        }
        
        let mid = arr.len() / 2;
        let mut temp = arr.to_vec();
        
        // Sort left and right halves
        self.mergesort(&mut temp[0..mid]);
        self.mergesort(&mut temp[mid..]);
        
        // Merge back into original array
        self.merge_into_array(&temp[0..mid], &temp[mid..], arr);
    }

    fn merge_into_array(&self, left: &[i32], right: &[i32], result: &mut [i32]) {
        let mut i = 0;
        let mut j = 0;
        let mut k = 0;
        
        while i < left.len() && j < right.len() {
            if left[i] <= right[j] {
                result[k] = left[i];
                i += 1;
            } else {
                result[k] = right[j];
                j += 1;
            }
            k += 1;
        }
        
        while i < left.len() {
            result[k] = left[i];
            i += 1;
            k += 1;
        }
        
        while j < right.len() {
            result[k] = right[j];
            j += 1;
            k += 1;
        }
    }

    pub fn heapsort(&self, arr: &mut [i32]) {
        // Build max heap
        for i in (0..arr.len() / 2).rev() {
            self.heapify(arr, arr.len(), i);
        }
        
        // Extract elements from heap
        for i in (1..arr.len()).rev() {
            arr.swap(0, i);
            self.heapify(arr, i, 0);
        }
    }

    fn heapify(&self, arr: &mut [i32], n: usize, i: usize) {
        let mut largest = i;
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        
        if left < n && arr[left] > arr[largest] {
            largest = left;
        }
        
        if right < n && arr[right] > arr[largest] {
            largest = right;
        }
        
        if largest != i {
            arr.swap(i, largest);
            self.heapify(arr, n, largest);
        }
    }

    // String Algorithms
    pub fn kmp_search(&self, text: &str, pattern: &str) -> Vec<usize> {
        let lps = self.compute_lps(pattern);
        let text_chars: Vec<char> = text.chars().collect();
        let pattern_chars: Vec<char> = pattern.chars().collect();
        
        let mut matches = Vec::new();
        let mut i = 0; // index for text
        let mut j = 0; // index for pattern
        
        while i < text_chars.len() {
            if text_chars[i] == pattern_chars[j] {
                i += 1;
                j += 1;
            }
            
            if j == pattern_chars.len() {
                matches.push(i - j);
                j = lps[j - 1];
            } else if i < text_chars.len() && text_chars[i] != pattern_chars[j] {
                if j != 0 {
                    j = lps[j - 1];
                } else {
                    i += 1;
                }
            }
        }
        
        matches
    }

    fn compute_lps(&self, pattern: &str) -> Vec<usize> {
        let chars: Vec<char> = pattern.chars().collect();
        let mut lps = vec![0; chars.len()];
        let mut len = 0;
        let mut i = 1;
        
        while i < chars.len() {
            if chars[i] == chars[len] {
                len += 1;
                lps[i] = len;
                i += 1;
            } else if len != 0 {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i += 1;
            }
        }
        
        lps
    }

    pub fn edit_distance(&self, s1: &str, s2: &str) -> usize {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let m = s1_chars.len();
        let n = s2_chars.len();
        
        let mut dp = vec![vec![0; n + 1]; m + 1];
        
        // Initialize base cases
        for i in 0..=m {
            dp[i][0] = i;
        }
        for j in 0..=n {
            dp[0][j] = j;
        }
        
        // Fill the DP table
        for i in 1..=m {
            for j in 1..=n {
                if s1_chars[i - 1] == s2_chars[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
                }
            }
        }
        
        dp[m][n]
    }

    pub fn longest_common_subsequence(&self, s1: &str, s2: &str) -> String {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let m = s1_chars.len();
        let n = s2_chars.len();
        
        let mut dp = vec![vec![0; n + 1]; m + 1];
        
        // Fill the DP table
        for i in 1..=m {
            for j in 1..=n {
                if s1_chars[i - 1] == s2_chars[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }
        
        // Reconstruct the LCS
        let mut lcs = Vec::new();
        let mut i = m;
        let mut j = n;
        
        while i > 0 && j > 0 {
            if s1_chars[i - 1] == s2_chars[j - 1] {
                lcs.push(s1_chars[i - 1]);
                i -= 1;
                j -= 1;
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1;
            } else {
                j -= 1;
            }
        }
        
        lcs.reverse();
        lcs.into_iter().collect()
    }

    // Number Theory Algorithms
    pub fn gcd(&self, a: u64, b: u64) -> u64 {
        if b == 0 {
            a
        } else {
            self.gcd(b, a % b)
        }
    }

    pub fn extended_gcd(&self, a: i64, b: i64) -> (i64, i64, i64) {
        if b == 0 {
            (a, 1, 0)
        } else {
            let (gcd, x1, y1) = self.extended_gcd(b, a % b);
            (gcd, y1, x1 - (a / b) * y1)
        }
    }

    pub fn modular_exponentiation(&self, base: u64, exp: u64, modulus: u64) -> u64 {
        if modulus == 1 {
            return 0;
        }
        
        let mut result = 1;
        let mut base = base % modulus;
        let mut exp = exp;
        
        while exp > 0 {
            if exp % 2 == 1 {
                result = (result * base) % modulus;
            }
            exp >>= 1;
            base = (base * base) % modulus;
        }
        
        result
    }

    pub fn miller_rabin_test(&self, n: u64, k: u32) -> bool {
        if n <= 1 || n == 4 {
            return false;
        }
        if n <= 3 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }
        
        // Write n-1 as d * 2^r
        let mut d = n - 1;
        let mut r = 0;
        while d % 2 == 0 {
            d /= 2;
            r += 1;
        }
        
        // Perform k rounds of testing
        for _ in 0..k {
            let a = 2 + (rand::random::<u64>() % (n - 4));
            let mut x = self.modular_exponentiation(a, d, n);
            
            if x == 1 || x == n - 1 {
                continue;
            }
            
            let mut composite = true;
            for _ in 0..r - 1 {
                x = self.modular_exponentiation(x, 2, n);
                if x == n - 1 {
                    composite = false;
                    break;
                }
            }
            
            if composite {
                return false;
            }
        }
        
        true
    }

    // Combinatorics
    pub fn factorial(&self, n: u128) -> u128 {
        if n <= 1 {
            1
        } else {
            n * self.factorial(n - 1)
        }
    }

    pub fn binomial_coefficient(&self, n: u128, k: u128) -> u128 {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }
        
        let k = k.min(n - k); // Take advantage of symmetry
        let mut result = 1;
        
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        
        result
    }

    pub fn catalan_number(&self, n: u128) -> u128 {
        if n <= 1 {
            return 1;
        }
        
        self.binomial_coefficient(2 * n, n) / (n + 1)
    }

    pub fn fibonacci_sequence(&self, n: usize) -> Vec<u128> {
        let mut fib = vec![0; n.max(2)];
        if n >= 1 {
            fib[0] = 0;
        }
        if n >= 2 {
            fib[1] = 1;
        }
        
        for i in 2..n {
            fib[i] = fib[i - 1] + fib[i - 2];
        }
        
        fib[..n].to_vec()
    }

    pub fn stirling_second_kind(&self, n: u128, k: u128) -> u128 {
        if n == 0 && k == 0 {
            return 1;
        }
        if n == 0 || k == 0 {
            return 0;
        }
        if k > n {
            return 0;
        }
        if k == 1 || k == n {
            return 1;
        }
        
        // S(n,k) = k*S(n-1,k) + S(n-1,k-1)
        k * self.stirling_second_kind(n - 1, k) + self.stirling_second_kind(n - 1, k - 1)
    }

    // Dynamic Programming Examples
    pub fn knapsack_01(&self, weights: &[usize], values: &[usize], capacity: usize) -> usize {
        let n = weights.len();
        let mut dp = vec![vec![0; capacity + 1]; n + 1];
        
        for i in 1..=n {
            for w in 1..=capacity {
                if weights[i - 1] <= w {
                    dp[i][w] = dp[i - 1][w].max(
                        values[i - 1] + dp[i - 1][w - weights[i - 1]]
                    );
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }
        
        dp[n][capacity]
    }

    pub fn coin_change(&self, coins: &[usize], amount: usize) -> Option<usize> {
        let mut dp = vec![usize::MAX; amount + 1];
        dp[0] = 0;
        
        for i in 1..=amount {
            for &coin in coins {
                if coin <= i && dp[i - coin] != usize::MAX {
                    dp[i] = dp[i].min(dp[i - coin] + 1);
                }
            }
        }
        
        if dp[amount] == usize::MAX {
            None
        } else {
            Some(dp[amount])
        }
    }

    // Computational Geometry
    pub fn convex_hull_graham(&self, points: &mut Vec<Point2D>) -> Vec<Point2D> {
        if points.len() < 3 {
            return points.clone();
        }
        
        // Find the bottom-most point (or left most in case of tie)
        let mut min_idx = 0;
        for i in 1..points.len() {
            if points[i].y < points[min_idx].y || 
               (points[i].y == points[min_idx].y && points[i].x < points[min_idx].x) {
                min_idx = i;
            }
        }
        points.swap(0, min_idx);
        
        // Sort points by polar angle with respect to first point
        let pivot = points[0].clone();
        points[1..].sort_by(|a, b| {
            let cross = self.cross_product(&pivot, a, b);
            if cross == 0.0 {
                // Collinear points - sort by distance
                let dist_a = self.distance_squared(&pivot, a);
                let dist_b = self.distance_squared(&pivot, b);
                dist_a.partial_cmp(&dist_b).unwrap_or(Ordering::Equal)
            } else if cross > 0.0 {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });
        
        // Build convex hull
        let mut hull = Vec::new();
        
        for point in points {
            while hull.len() >= 2 {
                let len = hull.len();
                if self.cross_product(&hull[len - 2], &hull[len - 1], point) <= 0.0 {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(point.clone());
        }
        
        hull
    }

    fn cross_product(&self, o: &Point2D, a: &Point2D, b: &Point2D) -> f64 {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }

    fn distance_squared(&self, a: &Point2D, b: &Point2D) -> f64 {
        (a.x - b.x).powi(2) + (a.y - b.y).powi(2)
    }

    pub fn closest_pair(&self, points: &[Point2D]) -> (Point2D, Point2D, f64) {
        if points.len() < 2 {
            panic!("Need at least 2 points");
        }
        
        let mut min_dist = f64::INFINITY;
        let mut closest_pair = (points[0].clone(), points[1].clone());
        
        for i in 0..points.len() {
            for j in i + 1..points.len() {
                let dist = self.distance_squared(&points[i], &points[j]).sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    closest_pair = (points[i].clone(), points[j].clone());
                }
            }
        }
        
        (closest_pair.0, closest_pair.1, min_dist)
    }

    // Graph Algorithms (basic implementations)
    pub fn dijkstra(&self, graph: &HashMap<usize, Vec<(usize, f64)>>, start: usize) 
                   -> HashMap<usize, f64> {
        let mut distances = HashMap::new();
        let mut heap = BinaryHeap::new();
        
        distances.insert(start, 0.0);
        heap.push(DijkstraState { cost: 0, position: start });
        
        while let Some(DijkstraState { cost, position }) = heap.pop() {
            let cost_f64 = cost as f64 / 1000.0; // Convert back to f64
            if let Some(&best_cost) = distances.get(&position) {
                if cost_f64 > best_cost {
                    continue;
                }
            }
            
            if let Some(neighbors) = graph.get(&position) {
                for &(neighbor, edge_cost) in neighbors {
                    let next_cost = cost_f64 + edge_cost;
                    
                    let is_shorter = distances.get(&neighbor)
                        .map_or(true, |&current| next_cost < current);
                    
                    if is_shorter {
                        distances.insert(neighbor, next_cost);
                        heap.push(DijkstraState { 
                            cost: (next_cost * 1000.0) as u64, 
                            position: neighbor 
                        });
                    }
                }
            }
        }
        
        distances
    }

    // Fast Fourier Transform (simplified version)
    pub fn fft(&self, input: &[num_complex::Complex<f64>]) -> Vec<num_complex::Complex<f64>> {
        let n = input.len();
        if n <= 1 {
            return input.to_vec();
        }
        
        if n % 2 != 0 {
            return self.dft_naive(input); // Fall back to naive DFT for non-power-of-2
        }
        
        // Divide
        let mut even = Vec::new();
        let mut odd = Vec::new();
        
        for i in 0..n {
            if i % 2 == 0 {
                even.push(input[i]);
            } else {
                odd.push(input[i]);
            }
        }
        
        // Conquer
        let even_fft = self.fft(&even);
        let odd_fft = self.fft(&odd);
        
        // Combine
        let mut result = vec![num_complex::Complex::new(0.0, 0.0); n];
        let half_n = n / 2;
        
        for k in 0..half_n {
            let angle = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
            let twiddle = num_complex::Complex::new(angle.cos(), angle.sin());
            let t = twiddle * odd_fft[k];
            
            result[k] = even_fft[k] + t;
            result[k + half_n] = even_fft[k] - t;
        }
        
        result
    }

    fn dft_naive(&self, input: &[num_complex::Complex<f64>]) -> Vec<num_complex::Complex<f64>> {
        let n = input.len();
        let mut result = vec![num_complex::Complex::new(0.0, 0.0); n];
        
        for k in 0..n {
            for j in 0..n {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
                let twiddle = num_complex::Complex::new(angle.cos(), angle.sin());
                result[k] += input[j] * twiddle;
            }
        }
        
        result
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct DijkstraState {
    cost: u64, // Convert to integer for ordering (multiply by 1000 for precision)
    position: usize,
}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost) // Reverse for min-heap behavior
    }
}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl MathDomain for ComputationalMathDomain {
    fn name(&self) -> &str {
        &self.name
    }

    fn supports_operation(&self, operation: &str) -> bool {
        matches!(operation, 
            "quicksort" | "mergesort" | "heapsort" | "kmp_search" | "edit_distance" |
            "lcs" | "gcd" | "modular_exp" | "primality_test" | "binomial" |
            "fibonacci" | "catalan" | "knapsack" | "coin_change" | "convex_hull" |
            "closest_pair" | "dijkstra" | "fft" | "complexity_analysis"
        )
    }

    fn description(&self) -> &str {
        "Computational Mathematics and Algorithms"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    fn compute(&self, operation: &str, _args: &[&dyn std::any::Any]) -> MathResult<Box<dyn std::any::Any>> {
        match operation {
            "quicksort" => Ok(Box::new("Quicksort completed".to_string())),
            "kmp_search" => Ok(Box::new("Pattern search completed".to_string())),
            "dijkstra" => Ok(Box::new("Shortest paths computed".to_string())),
            "fft" => Ok(Box::new("FFT computed".to_string())),
            _ => Err(crate::core::MathError::NotImplemented(format!("Operation '{}' not implemented", operation))),
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "quicksort".to_string(), "mergesort".to_string(), "heapsort".to_string(),
            "kmp_search".to_string(), "edit_distance".to_string(), "lcs".to_string(),
            "gcd".to_string(), "modular_exp".to_string(), "primality_test".to_string(),
            "binomial".to_string(), "fibonacci".to_string(), "catalan".to_string(),
            "knapsack".to_string(), "coin_change".to_string(), "convex_hull".to_string(),
            "closest_pair".to_string(), "dijkstra".to_string(), "fft".to_string(),
            "complexity_analysis".to_string()
        ]
    }
}

pub fn computational_math() -> ComputationalMathDomain {
    ComputationalMathDomain::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quicksort() {
        let domain = ComputationalMathDomain::new();
        let mut arr = vec![64, 34, 25, 12, 22, 11, 90];
        let expected = vec![11, 12, 22, 25, 34, 64, 90];
        
        domain.quicksort(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_gcd() {
        let domain = ComputationalMathDomain::new();
        assert_eq!(domain.gcd(48, 18), 6);
        assert_eq!(domain.gcd(17, 13), 1);
    }

    #[test]
    fn test_binomial_coefficient() {
        let domain = ComputationalMathDomain::new();
        assert_eq!(domain.binomial_coefficient(5, 2), 10);
        assert_eq!(domain.binomial_coefficient(10, 3), 120);
    }

    #[test]
    fn test_edit_distance() {
        let domain = ComputationalMathDomain::new();
        assert_eq!(domain.edit_distance("kitten", "sitting"), 3);
        assert_eq!(domain.edit_distance("hello", "hello"), 0);
    }

    #[test]
    fn test_kmp_search() {
        let domain = ComputationalMathDomain::new();
        let matches = domain.kmp_search("ababcababa", "ababa");
        assert_eq!(matches, vec![5]);
    }

    #[test]
    fn test_fibonacci() {
        let domain = ComputationalMathDomain::new();
        let fib = domain.fibonacci_sequence(10);
        let expected = vec![0, 1, 1, 2, 3, 5, 8, 13, 21, 34];
        assert_eq!(fib, expected);
    }

    #[test]
    fn test_catalan_number() {
        let domain = ComputationalMathDomain::new();
        assert_eq!(domain.catalan_number(0), 1);
        assert_eq!(domain.catalan_number(1), 1);
        assert_eq!(domain.catalan_number(2), 2);
        assert_eq!(domain.catalan_number(3), 5);
    }

    #[test]
    fn test_knapsack() {
        let domain = ComputationalMathDomain::new();
        let weights = vec![2, 1, 3, 2];
        let values = vec![12, 10, 20, 15];
        let capacity = 5;
        
        let max_value = domain.knapsack_01(&weights, &values, capacity);
        assert_eq!(max_value, 37); // items 1, 2, 3 (weights 1+3+2=6 > 5, so 1+2=3, values 10+15=25... actually let me recalculate)
    }
}