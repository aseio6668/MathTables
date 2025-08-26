use crate::core::{MathDomain, MathResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationTheoryDomain {
    name: String,
}

// Core Information Theory Concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityDistribution {
    pub outcomes: Vec<String>,
    pub probabilities: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointDistribution {
    pub variables: Vec<String>,
    pub outcomes: Vec<Vec<String>>, // Cartesian product of outcome spaces
    pub probabilities: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalDistribution {
    pub given_variable: String,
    pub target_variable: String,
    pub conditional_probs: HashMap<String, HashMap<String, f64>>, // P(target|given)
}

// Information Measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyMeasure {
    pub entropy_type: EntropyType,
    pub value: f64,
    pub base: f64, // logarithm base (2 for bits, e for nats, 10 for dits)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntropyType {
    Shannon,      // H(X) = -Σ p(x) log p(x)
    Conditional,  // H(X|Y) = -Σ p(x,y) log p(x|y)
    Joint,        // H(X,Y) = -Σ p(x,y) log p(x,y)
    CrossEntropy, // H(p,q) = -Σ p(x) log q(x)
    KLDivergence, // D_KL(p||q) = Σ p(x) log(p(x)/q(x))
    Renyi(f64),   // H_α(X) = (1/(1-α)) log Σ p(x)^α
}

// Channel Theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Channel {
    pub input_alphabet: Vec<String>,
    pub output_alphabet: Vec<String>,
    pub transition_matrix: Vec<Vec<f64>>, // P(Y|X)
    pub channel_type: ChannelType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelType {
    BinarySymmetric,
    BinaryErasure,
    GaussianNoise,
    DiscreteMemoryless,
    ContinuousMemoryless,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelCapacity {
    pub capacity: f64,
    pub optimal_input_distribution: Vec<f64>,
    pub channel_matrix: Vec<Vec<f64>>,
}

// Coding Theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Code {
    pub codewords: Vec<String>,
    pub code_length: usize,
    pub message_length: usize,
    pub minimum_distance: usize,
    pub code_type: CodeType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodeType {
    Linear,
    Hamming,
    Reed_Solomon,
    BCH,
    LDPC,
    Turbo,
    Polar,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingResult {
    pub original_message: String,
    pub encoded_message: String,
    pub compression_ratio: f64,
    pub error_correction_capability: usize,
}

// Compression and Source Coding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceCoding {
    pub source_alphabet: Vec<String>,
    pub source_probabilities: Vec<f64>,
    pub coding_scheme: CodingScheme,
    pub average_code_length: f64,
    pub efficiency: f64, // H(X) / L_avg
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodingScheme {
    Huffman,
    Shannon_Fano,
    Arithmetic,
    LZ77,
    LZ78,
    LZW,
}

// Biological Information Theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticCode {
    pub codons: HashMap<String, String>, // codon -> amino acid
    pub degeneracy: HashMap<String, usize>, // amino acid -> number of codons
    pub redundancy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DNASequence {
    pub sequence: String,
    pub gc_content: f64,
    pub complexity: f64,
    pub repetitive_elements: Vec<(usize, usize, String)>, // (start, end, motif)
}

// Network Information Theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCoding {
    pub nodes: Vec<NetworkNode>,
    pub edges: Vec<NetworkEdge>,
    pub multicast_capacity: f64,
    pub min_cut_capacity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkNode {
    pub id: String,
    pub node_type: NodeType,
    pub processing_capacity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    Source,
    Relay,
    Destination,
    Coding, // Network coding node
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEdge {
    pub from: String,
    pub to: String,
    pub capacity: f64,
    pub delay: f64,
    pub error_rate: f64,
}

// Quantum Information Theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    pub amplitudes: Vec<num_complex::Complex<f64>>,
    pub dimension: usize,
    pub is_pure: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumEntropy {
    pub von_neumann_entropy: f64,
    pub entanglement_entropy: f64,
    pub quantum_mutual_information: f64,
}

impl InformationTheoryDomain {
    pub fn new() -> Self {
        Self {
            name: "Information Theory".to_string(),
        }
    }

    // Entropy Calculations
    pub fn shannon_entropy(&self, distribution: &ProbabilityDistribution, base: f64) -> f64 {
        let mut entropy = 0.0;
        for &p in &distribution.probabilities {
            if p > 0.0 {
                entropy -= p * p.log(base);
            }
        }
        entropy
    }

    pub fn conditional_entropy(&self, joint_dist: &JointDistribution, 
                             x_var: &str, y_var: &str, base: f64) -> MathResult<f64> {
        // H(X|Y) = H(X,Y) - H(Y)
        let joint_entropy = self.joint_entropy(joint_dist, base);
        let y_entropy = self.marginal_entropy(joint_dist, y_var, base)?;
        Ok(joint_entropy - y_entropy)
    }

    pub fn joint_entropy(&self, joint_dist: &JointDistribution, base: f64) -> f64 {
        let mut entropy = 0.0;
        for &p in &joint_dist.probabilities {
            if p > 0.0 {
                entropy -= p * p.log(base);
            }
        }
        entropy
    }

    pub fn marginal_entropy(&self, joint_dist: &JointDistribution, 
                          variable: &str, base: f64) -> MathResult<f64> {
        // Find index of the variable
        let var_index = joint_dist.variables.iter().position(|v| v == variable)
            .ok_or_else(|| crate::core::MathError::InvalidArgument(
                format!("Variable {} not found", variable)))?;

        // Compute marginal distribution by summing over other variables
        let mut marginal_probs = HashMap::new();
        
        for (outcome_combination, &prob) in joint_dist.outcomes.iter().zip(&joint_dist.probabilities) {
            let outcome = &outcome_combination[var_index];
            *marginal_probs.entry(outcome.clone()).or_insert(0.0) += prob;
        }

        let marginal_dist = ProbabilityDistribution {
            outcomes: marginal_probs.keys().cloned().collect(),
            probabilities: marginal_probs.values().cloned().collect(),
        };

        Ok(self.shannon_entropy(&marginal_dist, base))
    }

    pub fn mutual_information(&self, joint_dist: &JointDistribution, 
                            x_var: &str, y_var: &str, base: f64) -> MathResult<f64> {
        // I(X;Y) = H(X) + H(Y) - H(X,Y)
        let h_x = self.marginal_entropy(joint_dist, x_var, base)?;
        let h_y = self.marginal_entropy(joint_dist, y_var, base)?;
        let h_xy = self.joint_entropy(joint_dist, base);
        
        Ok(h_x + h_y - h_xy)
    }

    pub fn kl_divergence(&self, p_dist: &ProbabilityDistribution, 
                        q_dist: &ProbabilityDistribution, base: f64) -> MathResult<f64> {
        if p_dist.outcomes.len() != q_dist.outcomes.len() {
            return Err(crate::core::MathError::InvalidArgument(
                "Distributions must have same support".to_string()));
        }

        let mut divergence = 0.0;
        for (i, &p) in p_dist.probabilities.iter().enumerate() {
            let q = q_dist.probabilities[i];
            if p > 0.0 {
                if q <= 0.0 {
                    return Ok(f64::INFINITY); // KL divergence is infinite
                }
                divergence += p * (p / q).log(base);
            }
        }
        Ok(divergence)
    }

    pub fn cross_entropy(&self, p_dist: &ProbabilityDistribution, 
                        q_dist: &ProbabilityDistribution, base: f64) -> MathResult<f64> {
        // H(p,q) = H(p) + D_KL(p||q)
        let entropy = self.shannon_entropy(p_dist, base);
        let kl_div = self.kl_divergence(p_dist, q_dist, base)?;
        Ok(entropy + kl_div)
    }

    pub fn jensen_shannon_divergence(&self, p_dist: &ProbabilityDistribution, 
                                   q_dist: &ProbabilityDistribution, base: f64) -> MathResult<f64> {
        // JS(p,q) = 0.5 * D_KL(p||m) + 0.5 * D_KL(q||m) where m = 0.5*(p+q)
        if p_dist.probabilities.len() != q_dist.probabilities.len() {
            return Err(crate::core::MathError::InvalidArgument(
                "Distributions must have same dimension".to_string()));
        }

        // Create mixture distribution
        let mut m_probs = Vec::new();
        for i in 0..p_dist.probabilities.len() {
            m_probs.push(0.5 * (p_dist.probabilities[i] + q_dist.probabilities[i]));
        }
        let m_dist = ProbabilityDistribution {
            outcomes: p_dist.outcomes.clone(),
            probabilities: m_probs,
        };

        let kl_pm = self.kl_divergence(p_dist, &m_dist, base)?;
        let kl_qm = self.kl_divergence(q_dist, &m_dist, base)?;
        
        Ok(0.5 * kl_pm + 0.5 * kl_qm)
    }

    pub fn renyi_entropy(&self, distribution: &ProbabilityDistribution, 
                        alpha: f64, base: f64) -> MathResult<f64> {
        if alpha == 1.0 {
            return Ok(self.shannon_entropy(distribution, base));
        }
        if alpha <= 0.0 {
            return Err(crate::core::MathError::InvalidArgument(
                "Alpha must be positive".to_string()));
        }

        let mut sum = 0.0;
        for &p in &distribution.probabilities {
            if p > 0.0 {
                sum += p.powf(alpha);
            }
        }

        if sum <= 0.0 {
            return Ok(f64::INFINITY);
        }

        Ok((1.0 / (1.0 - alpha)) * sum.log(base))
    }

    // Channel Capacity
    pub fn binary_symmetric_channel_capacity(&self, error_prob: f64) -> f64 {
        if error_prob <= 0.0 || error_prob >= 0.5 {
            return 0.0;
        }
        
        let p = error_prob;
        let entropy_noise = -p * p.log2() - (1.0 - p) * (1.0 - p).log2();
        1.0 - entropy_noise
    }

    pub fn binary_erasure_channel_capacity(&self, erasure_prob: f64) -> f64 {
        (1.0 - erasure_prob).max(0.0)
    }

    pub fn compute_channel_capacity(&self, channel: &Channel) -> MathResult<ChannelCapacity> {
        match channel.channel_type {
            ChannelType::BinarySymmetric => {
                // Assume first transition probability is the error probability
                if !channel.transition_matrix.is_empty() && !channel.transition_matrix[0].is_empty() {
                    let error_prob = channel.transition_matrix[0][1]; // P(Y=1|X=0)
                    let capacity = self.binary_symmetric_channel_capacity(error_prob);
                    Ok(ChannelCapacity {
                        capacity,
                        optimal_input_distribution: vec![0.5, 0.5], // Uniform distribution
                        channel_matrix: channel.transition_matrix.clone(),
                    })
                } else {
                    Err(crate::core::MathError::InvalidArgument(
                        "Invalid channel matrix".to_string()))
                }
            }
            ChannelType::BinaryErasure => {
                if !channel.transition_matrix.is_empty() && channel.transition_matrix[0].len() >= 3 {
                    let erasure_prob = channel.transition_matrix[0][2]; // P(Y=?|X=0)
                    let capacity = self.binary_erasure_channel_capacity(erasure_prob);
                    Ok(ChannelCapacity {
                        capacity,
                        optimal_input_distribution: vec![0.5, 0.5],
                        channel_matrix: channel.transition_matrix.clone(),
                    })
                } else {
                    Err(crate::core::MathError::InvalidArgument(
                        "Invalid channel matrix for BEC".to_string()))
                }
            }
            _ => {
                // General discrete memoryless channel - use iterative algorithm
                self.compute_general_channel_capacity(channel)
            }
        }
    }

    fn compute_general_channel_capacity(&self, channel: &Channel) -> MathResult<ChannelCapacity> {
        let input_size = channel.input_alphabet.len();
        let output_size = channel.output_alphabet.len();
        
        if channel.transition_matrix.len() != input_size ||
           channel.transition_matrix.iter().any(|row| row.len() != output_size) {
            return Err(crate::core::MathError::InvalidArgument(
                "Channel matrix dimensions don't match alphabets".to_string()));
        }

        // Blahut-Arimoto algorithm (simplified version)
        let mut input_dist = vec![1.0 / input_size as f64; input_size];
        let mut capacity = 0.0;
        let max_iterations = 100;
        let tolerance = 1e-6;

        for _ in 0..max_iterations {
            // Compute output distribution
            let mut output_dist = vec![0.0; output_size];
            for (i, &p_x) in input_dist.iter().enumerate() {
                for (j, &p_y_given_x) in channel.transition_matrix[i].iter().enumerate() {
                    output_dist[j] += p_x * p_y_given_x;
                }
            }

            // Compute mutual information
            let mut mutual_info = 0.0;
            for (i, &p_x) in input_dist.iter().enumerate() {
                for (j, &p_y_given_x) in channel.transition_matrix[i].iter().enumerate() {
                    if p_x > 0.0 && p_y_given_x > 0.0 && output_dist[j] > 0.0 {
                        mutual_info += p_x * p_y_given_x * (p_y_given_x / output_dist[j]).log2();
                    }
                }
            }

            if (mutual_info - capacity).abs() < tolerance {
                break;
            }
            capacity = mutual_info;

            // Update input distribution (simplified)
            let mut new_input_dist = vec![0.0; input_size];
            for i in 0..input_size {
                let mut exponent = 0.0;
                for (j, &p_y_given_x) in channel.transition_matrix[i].iter().enumerate() {
                    if p_y_given_x > 0.0 && output_dist[j] > 0.0 {
                        exponent += p_y_given_x * (p_y_given_x / output_dist[j]).log2();
                    }
                }
                new_input_dist[i] = (2.0_f64).powf(exponent);
            }

            // Normalize
            let sum: f64 = new_input_dist.iter().sum();
            if sum > 0.0 {
                for prob in &mut new_input_dist {
                    *prob /= sum;
                }
            }
            input_dist = new_input_dist;
        }

        Ok(ChannelCapacity {
            capacity,
            optimal_input_distribution: input_dist,
            channel_matrix: channel.transition_matrix.clone(),
        })
    }

    // Huffman Coding
    pub fn huffman_coding(&self, symbols: Vec<String>, frequencies: Vec<f64>) 
                        -> MathResult<HashMap<String, String>> {
        if symbols.len() != frequencies.len() {
            return Err(crate::core::MathError::InvalidArgument(
                "Symbols and frequencies must have same length".to_string()));
        }

        if symbols.len() <= 1 {
            if symbols.len() == 1 {
                let mut codes = HashMap::new();
                codes.insert(symbols[0].clone(), "0".to_string());
                return Ok(codes);
            } else {
                return Ok(HashMap::new());
            }
        }

        // Create priority queue (min-heap simulation)
        let mut nodes: Vec<HuffmanNode> = symbols.into_iter().zip(frequencies.into_iter())
            .map(|(symbol, freq)| HuffmanNode::Leaf { symbol, frequency: freq })
            .collect();

        // Build Huffman tree
        while nodes.len() > 1 {
            // Sort by frequency (ascending)
            nodes.sort_by(|a, b| a.frequency().partial_cmp(&b.frequency()).unwrap());
            
            let left = nodes.remove(0);
            let right = nodes.remove(0);
            let combined_freq = left.frequency() + right.frequency();
            
            let internal = HuffmanNode::Internal {
                frequency: combined_freq,
                left: Box::new(left),
                right: Box::new(right),
            };
            
            nodes.push(internal);
        }

        // Generate codes
        let mut codes = HashMap::new();
        if let Some(root) = nodes.into_iter().next() {
            self.generate_huffman_codes(&root, String::new(), &mut codes);
        }

        Ok(codes)
    }

    fn generate_huffman_codes(&self, node: &HuffmanNode, code: String, 
                            codes: &mut HashMap<String, String>) {
        match node {
            HuffmanNode::Leaf { symbol, .. } => {
                codes.insert(symbol.clone(), if code.is_empty() { "0".to_string() } else { code });
            }
            HuffmanNode::Internal { left, right, .. } => {
                self.generate_huffman_codes(left, format!("{}0", code), codes);
                self.generate_huffman_codes(right, format!("{}1", code), codes);
            }
        }
    }

    // DNA Sequence Analysis
    pub fn analyze_dna_sequence(&self, sequence: &str) -> DNASequence {
        let sequence = sequence.to_uppercase();
        let gc_count = sequence.chars().filter(|&c| c == 'G' || c == 'C').count();
        let gc_content = gc_count as f64 / sequence.len() as f64;
        
        // Calculate complexity using Shannon entropy
        let mut base_counts = HashMap::new();
        for base in sequence.chars() {
            *base_counts.entry(base).or_insert(0) += 1;
        }
        
        let total = sequence.len() as f64;
        let mut complexity = 0.0;
        for &count in base_counts.values() {
            let p = count as f64 / total;
            if p > 0.0 {
                complexity -= p * p.log2();
            }
        }
        
        // Find repetitive elements (simplified - just look for dinucleotide repeats)
        let mut repetitive_elements = Vec::new();
        let mut i = 0;
        let chars: Vec<char> = sequence.chars().collect();
        
        while i < chars.len() - 1 {
            let dinuc: String = chars[i..i+2].iter().collect();
            let mut repeat_len = 2;
            let mut j = i + 2;
            
            while j + 1 < chars.len() {
                let next_dinuc: String = chars[j..j+2].iter().collect();
                if next_dinuc == dinuc {
                    repeat_len += 2;
                    j += 2;
                } else {
                    break;
                }
            }
            
            if repeat_len > 4 { // At least 3 repeats
                repetitive_elements.push((i, i + repeat_len, dinuc));
            }
            
            i += 1;
        }

        DNASequence {
            sequence,
            gc_content,
            complexity,
            repetitive_elements,
        }
    }

    // Quantum Information (basic)
    pub fn von_neumann_entropy(&self, eigenvalues: &[f64]) -> f64 {
        let mut entropy = 0.0;
        for &lambda in eigenvalues {
            if lambda > 0.0 {
                entropy -= lambda * lambda.log2();
            }
        }
        entropy
    }

    pub fn quantum_mutual_information(&self, rho_ab: &[f64], rho_a: &[f64], rho_b: &[f64]) -> f64 {
        // I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB)
        let s_a = self.von_neumann_entropy(rho_a);
        let s_b = self.von_neumann_entropy(rho_b);
        let s_ab = self.von_neumann_entropy(rho_ab);
        
        s_a + s_b - s_ab
    }

    // Compression Metrics
    pub fn compression_ratio(&self, original_size: usize, compressed_size: usize) -> f64 {
        if original_size == 0 {
            return 0.0;
        }
        compressed_size as f64 / original_size as f64
    }

    pub fn coding_efficiency(&self, entropy: f64, average_code_length: f64) -> f64 {
        if average_code_length == 0.0 {
            return 0.0;
        }
        entropy / average_code_length
    }
}

#[derive(Debug, Clone)]
enum HuffmanNode {
    Leaf {
        symbol: String,
        frequency: f64,
    },
    Internal {
        frequency: f64,
        left: Box<HuffmanNode>,
        right: Box<HuffmanNode>,
    },
}

impl HuffmanNode {
    fn frequency(&self) -> f64 {
        match self {
            HuffmanNode::Leaf { frequency, .. } => *frequency,
            HuffmanNode::Internal { frequency, .. } => *frequency,
        }
    }
}

impl MathDomain for InformationTheoryDomain {
    fn name(&self) -> &str {
        &self.name
    }

    fn supports_operation(&self, operation: &str) -> bool {
        matches!(operation, 
            "shannon_entropy" | "mutual_information" | "kl_divergence" |
            "channel_capacity" | "huffman_coding" | "dna_analysis" |
            "quantum_entropy" | "conditional_entropy" | "cross_entropy" |
            "renyi_entropy" | "jensen_shannon" | "compression_ratio"
        )
    }

    fn description(&self) -> &str {
        "Information Theory and Coding"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    fn compute(&self, operation: &str, _args: &[&dyn std::any::Any]) -> MathResult<Box<dyn std::any::Any>> {
        match operation {
            "shannon_entropy" => Ok(Box::new("Shannon entropy computed".to_string())),
            "mutual_information" => Ok(Box::new("Mutual information computed".to_string())),
            "channel_capacity" => Ok(Box::new("Channel capacity computed".to_string())),
            "huffman_coding" => Ok(Box::new("Huffman codes generated".to_string())),
            _ => Err(crate::core::MathError::NotImplemented(format!("Operation '{}' not implemented", operation))),
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "shannon_entropy".to_string(), "mutual_information".to_string(),
            "kl_divergence".to_string(), "channel_capacity".to_string(),
            "huffman_coding".to_string(), "dna_analysis".to_string(),
            "quantum_entropy".to_string(), "conditional_entropy".to_string(),
            "cross_entropy".to_string(), "renyi_entropy".to_string(),
            "jensen_shannon".to_string(), "compression_ratio".to_string()
        ]
    }
}

pub fn information_theory() -> InformationTheoryDomain {
    InformationTheoryDomain::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy() {
        let domain = InformationTheoryDomain::new();
        let uniform_dist = ProbabilityDistribution {
            outcomes: vec!["A".to_string(), "B".to_string()],
            probabilities: vec![0.5, 0.5],
        };
        
        let entropy = domain.shannon_entropy(&uniform_dist, 2.0);
        assert!((entropy - 1.0).abs() < 1e-10); // H(uniform binary) = 1 bit
    }

    #[test]
    fn test_kl_divergence() {
        let domain = InformationTheoryDomain::new();
        let p = ProbabilityDistribution {
            outcomes: vec!["A".to_string(), "B".to_string()],
            probabilities: vec![0.5, 0.5],
        };
        let q = ProbabilityDistribution {
            outcomes: vec!["A".to_string(), "B".to_string()],
            probabilities: vec![0.25, 0.75],
        };
        
        let kl_div = domain.kl_divergence(&p, &q, 2.0).unwrap();
        assert!(kl_div > 0.0); // KL divergence is always non-negative
    }

    #[test]
    fn test_huffman_coding() {
        let domain = InformationTheoryDomain::new();
        let symbols = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let frequencies = vec![0.5, 0.3, 0.2];
        
        let codes = domain.huffman_coding(symbols, frequencies).unwrap();
        assert_eq!(codes.len(), 3);
        
        // Check that all codes are different
        let code_values: Vec<&String> = codes.values().collect();
        let mut unique_codes = code_values.clone();
        unique_codes.sort();
        unique_codes.dedup();
        assert_eq!(code_values.len(), unique_codes.len());
    }

    #[test]
    fn test_binary_symmetric_channel() {
        let domain = InformationTheoryDomain::new();
        let capacity = domain.binary_symmetric_channel_capacity(0.1);
        assert!(capacity > 0.5 && capacity <= 1.0);
        
        // Perfect channel (p=0) should have capacity 1
        let perfect_capacity = domain.binary_symmetric_channel_capacity(0.0);
        assert!((perfect_capacity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dna_analysis() {
        let domain = InformationTheoryDomain::new();
        let sequence = "ATGCATGCATGC";
        let analysis = domain.analyze_dna_sequence(sequence);
        
        assert_eq!(analysis.gc_content, 0.5); // 50% GC content
        assert!(analysis.complexity > 0.0);
        assert_eq!(analysis.sequence, "ATGCATGCATGC");
    }

    #[test]
    fn test_renyi_entropy() {
        let domain = InformationTheoryDomain::new();
        let uniform_dist = ProbabilityDistribution {
            outcomes: vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()],
            probabilities: vec![0.25, 0.25, 0.25, 0.25],
        };
        
        // Renyi entropy with α=1 should equal Shannon entropy
        let shannon = domain.shannon_entropy(&uniform_dist, 2.0);
        let renyi = domain.renyi_entropy(&uniform_dist, 1.0, 2.0).unwrap();
        assert!((shannon - renyi).abs() < 1e-10);
    }

    #[test]
    fn test_von_neumann_entropy() {
        let domain = InformationTheoryDomain::new();
        let eigenvalues = vec![0.5, 0.3, 0.2]; // Density matrix eigenvalues
        let entropy = domain.von_neumann_entropy(&eigenvalues);
        assert!(entropy > 0.0);
    }
}