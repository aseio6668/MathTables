use crate::core::{
    MathDomain, MathResult, MathError, Tensor, TransformerConfig
};
use std::collections::HashMap;
use std::f64::consts::PI;

pub struct LanguageProcessingDomain;

impl LanguageProcessingDomain {
    pub fn new() -> Self {
        Self
    }
}

/// Token representation in language models
#[derive(Debug, Clone)]
pub struct Token {
    pub id: usize,
    pub text: String,
    pub position: usize,
}

/// Vocabulary for tokenization
pub struct Vocabulary {
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: HashMap<usize, String>,
    pub special_tokens: HashMap<String, usize>,
}

impl Vocabulary {
    pub fn new() -> Self {
        let mut vocab = Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            special_tokens: HashMap::new(),
        };
        
        // Add special tokens
        vocab.add_special_token("<pad>", 0);
        vocab.add_special_token("<unk>", 1);
        vocab.add_special_token("<bos>", 2);
        vocab.add_special_token("<eos>", 3);
        
        vocab
    }
    
    fn add_special_token(&mut self, token: &str, id: usize) {
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        self.special_tokens.insert(token.to_string(), id);
    }
    
    pub fn add_token(&mut self, token: &str) -> usize {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        
        let id = self.token_to_id.len();
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        id
    }
    
    pub fn get_id(&self, token: &str) -> usize {
        *self.token_to_id.get(token).unwrap_or(&1) // Return <unk> if not found
    }
    
    pub fn get_token(&self, id: usize) -> Option<&String> {
        self.id_to_token.get(&id)
    }
    
    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }
}

impl LanguageProcessingDomain {
    /// Simple tokenization by whitespace and punctuation
    pub fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        
        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current_token.is_empty() {
                    tokens.push(current_token);
                    current_token = String::new();
                }
            } else if ch.is_ascii_punctuation() {
                if !current_token.is_empty() {
                    tokens.push(current_token);
                    current_token = String::new();
                }
                tokens.push(ch.to_string());
            } else {
                current_token.push(ch.to_lowercase().next().unwrap());
            }
        }
        
        if !current_token.is_empty() {
            tokens.push(current_token);
        }
        
        tokens
    }
    
    /// Generate n-grams from token sequence
    pub fn generate_ngrams(tokens: &[String], n: usize) -> Vec<Vec<String>> {
        if n == 0 || tokens.len() < n {
            return Vec::new();
        }
        
        let mut ngrams = Vec::new();
        for i in 0..=tokens.len() - n {
            let ngram = tokens[i..i + n].to_vec();
            ngrams.push(ngram);
        }
        
        ngrams
    }
    
    /// Compute TF-IDF (Term Frequency - Inverse Document Frequency)
    pub fn compute_tf_idf(
        documents: &[Vec<String>],
        vocabulary: &[String]
    ) -> Vec<Vec<f64>> {
        let n_docs = documents.len() as f64;
        let mut tf_idf_matrix = Vec::new();
        
        // Compute document frequencies
        let mut df = HashMap::new();
        for doc in documents {
            let mut seen_terms = std::collections::HashSet::new();
            for term in doc {
                if seen_terms.insert(term) {
                    *df.entry(term.clone()).or_insert(0) += 1;
                }
            }
        }
        
        // Compute TF-IDF for each document
        for doc in documents {
            let mut tf_idf_vec = Vec::new();
            let doc_len = doc.len() as f64;
            
            for term in vocabulary {
                // Term frequency
                let tf = doc.iter().filter(|&t| t == term).count() as f64 / doc_len;
                
                // Inverse document frequency
                let doc_freq = df.get(term).copied().unwrap_or(0) as f64;
                let idf = if doc_freq > 0.0 {
                    (n_docs / doc_freq).ln()
                } else {
                    0.0
                };
                
                tf_idf_vec.push(tf * idf);
            }
            
            tf_idf_matrix.push(tf_idf_vec);
        }
        
        tf_idf_matrix
    }
    
    /// Word2Vec Skip-gram objective (simplified)
    /// Predicts context words given center word
    pub fn skip_gram_loss(
        center_embedding: &[f64],
        context_embeddings: &[Vec<f64>],
        negative_embeddings: &[Vec<f64>]
    ) -> MathResult<f64> {
        if center_embedding.is_empty() {
            return Err(MathError::InvalidArgument("Center embedding cannot be empty".to_string()));
        }
        
        let mut total_loss = 0.0;
        
        // Positive samples (context words)
        for context in context_embeddings {
            if context.len() != center_embedding.len() {
                return Err(MathError::InvalidArgument("Embedding dimensions must match".to_string()));
            }
            
            let dot_product: f64 = center_embedding.iter()
                .zip(context.iter())
                .map(|(a, b)| a * b)
                .sum();
            
            // Sigmoid loss: -log(σ(dot_product))
            let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
            total_loss -= sigmoid.ln();
        }
        
        // Negative samples
        for negative in negative_embeddings {
            if negative.len() != center_embedding.len() {
                return Err(MathError::InvalidArgument("Embedding dimensions must match".to_string()));
            }
            
            let dot_product: f64 = center_embedding.iter()
                .zip(negative.iter())
                .map(|(a, b)| a * b)
                .sum();
            
            // Negative sigmoid loss: -log(σ(-dot_product))
            let sigmoid = 1.0 / (1.0 + dot_product.exp());
            total_loss -= sigmoid.ln();
        }
        
        Ok(total_loss)
    }
    
    /// Positional encoding for transformer models
    /// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    /// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    pub fn positional_encoding(
        sequence_length: usize,
        d_model: usize
    ) -> MathResult<Vec<Vec<f64>>> {
        if d_model % 2 != 0 {
            return Err(MathError::InvalidArgument("d_model must be even for positional encoding".to_string()));
        }
        
        let mut pe = vec![vec![0.0; d_model]; sequence_length];
        
        for pos in 0..sequence_length {
            for i in (0..d_model).step_by(2) {
                let angle = pos as f64 / 10000.0_f64.powf(2.0 * i as f64 / d_model as f64);
                pe[pos][i] = angle.sin();
                pe[pos][i + 1] = angle.cos();
            }
        }
        
        Ok(pe)
    }
    
    /// Self-attention mechanism (simplified)
    pub fn self_attention(
        input_embeddings: &[Vec<f64>],
        d_k: usize
    ) -> MathResult<Vec<Vec<f64>>> {
        let seq_len = input_embeddings.len();
        let d_model = input_embeddings[0].len();
        
        if d_k == 0 {
            return Err(MathError::InvalidArgument("d_k must be positive".to_string()));
        }
        
        // Simplified: use input as Q, K, V (normally these would be linear projections)
        let mut attention_output = vec![vec![0.0; d_model]; seq_len];
        
        for i in 0..seq_len {
            let mut attention_weights = vec![0.0; seq_len];
            let mut weight_sum = 0.0;
            
            // Compute attention weights
            for j in 0..seq_len {
                // Simplified dot-product attention
                let score: f64 = input_embeddings[i].iter()
                    .zip(&input_embeddings[j])
                    .map(|(a, b)| a * b)
                    .sum();
                
                let scaled_score = score / (d_k as f64).sqrt();
                attention_weights[j] = scaled_score.exp();
                weight_sum += attention_weights[j];
            }
            
            // Normalize weights (softmax)
            for weight in &mut attention_weights {
                *weight /= weight_sum;
            }
            
            // Compute weighted sum of values
            for j in 0..d_model {
                for k in 0..seq_len {
                    attention_output[i][j] += attention_weights[k] * input_embeddings[k][j];
                }
            }
        }
        
        Ok(attention_output)
    }
    
    /// Multi-head attention (simplified)
    pub fn multi_head_attention(
        input_embeddings: &[Vec<f64>],
        num_heads: usize,
        d_k: usize
    ) -> MathResult<Vec<Vec<f64>>> {
        if num_heads == 0 {
            return Err(MathError::InvalidArgument("Number of heads must be positive".to_string()));
        }
        
        let seq_len = input_embeddings.len();
        let d_model = input_embeddings[0].len();
        
        if d_model % num_heads != 0 {
            return Err(MathError::InvalidArgument("d_model must be divisible by num_heads".to_string()));
        }
        
        let head_dim = d_model / num_heads;
        let mut multi_head_output = vec![vec![0.0; d_model]; seq_len];
        
        for head in 0..num_heads {
            let start_idx = head * head_dim;
            let end_idx = start_idx + head_dim;
            
            // Extract head-specific embeddings
            let head_embeddings: Vec<Vec<f64>> = input_embeddings.iter()
                .map(|emb| emb[start_idx..end_idx].to_vec())
                .collect();
            
            // Apply single-head attention
            let head_output = Self::self_attention(&head_embeddings, d_k)?;
            
            // Concatenate head outputs
            for i in 0..seq_len {
                for (j, &val) in head_output[i].iter().enumerate() {
                    multi_head_output[i][start_idx + j] = val;
                }
            }
        }
        
        Ok(multi_head_output)
    }
    
    /// BERT-style masked language modeling loss
    pub fn masked_language_modeling_loss(
        predictions: &[Vec<f64>],
        targets: &[usize],
        mask: &[bool]
    ) -> MathResult<f64> {
        if predictions.len() != targets.len() || predictions.len() != mask.len() {
            return Err(MathError::InvalidArgument("Predictions, targets, and mask must have same length".to_string()));
        }
        
        let mut total_loss = 0.0;
        let mut masked_positions = 0;
        
        for (i, (&target_id, &is_masked)) in targets.iter().zip(mask.iter()).enumerate() {
            if is_masked {
                if target_id >= predictions[i].len() {
                    return Err(MathError::InvalidArgument("Target ID out of vocabulary range".to_string()));
                }
                
                // Compute softmax over vocabulary
                let max_logit = predictions[i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = predictions[i].iter()
                    .map(|&x| (x - max_logit).exp())
                    .sum();
                
                let target_prob = (predictions[i][target_id] - max_logit).exp() / exp_sum;
                total_loss -= target_prob.ln();
                masked_positions += 1;
            }
        }
        
        if masked_positions == 0 {
            return Ok(0.0);
        }
        
        Ok(total_loss / masked_positions as f64)
    }
    
    /// Compute perplexity from cross-entropy loss
    /// Perplexity = exp(cross_entropy_loss)
    pub fn perplexity(cross_entropy_loss: f64) -> f64 {
        cross_entropy_loss.exp()
    }
    
    /// BLEU score computation (simplified, sentence-level)
    pub fn bleu_score(
        reference: &[String],
        candidate: &[String],
        max_n: usize
    ) -> MathResult<f64> {
        if max_n == 0 {
            return Err(MathError::InvalidArgument("max_n must be positive".to_string()));
        }
        
        let mut precision_scores = Vec::new();
        
        for n in 1..=max_n {
            let ref_ngrams = Self::generate_ngrams(reference, n);
            let cand_ngrams = Self::generate_ngrams(candidate, n);
            
            if cand_ngrams.is_empty() {
                precision_scores.push(0.0);
                continue;
            }
            
            // Count matches
            let mut ref_counts = HashMap::new();
            for ngram in ref_ngrams {
                *ref_counts.entry(ngram).or_insert(0) += 1;
            }
            
            let mut matches = 0;
            for ngram in &cand_ngrams {
                if let Some(count) = ref_counts.get_mut(ngram) {
                    if *count > 0 {
                        matches += 1;
                        *count -= 1;
                    }
                }
            }
            
            let precision = matches as f64 / cand_ngrams.len() as f64;
            precision_scores.push(precision);
        }
        
        // Geometric mean of precision scores
        let log_sum: f64 = precision_scores.iter()
            .map(|&p| if p > 0.0 { p.ln() } else { f64::NEG_INFINITY })
            .sum();
        
        if log_sum == f64::NEG_INFINITY {
            return Ok(0.0);
        }
        
        let geometric_mean = (log_sum / max_n as f64).exp();
        
        // Brevity penalty
        let ref_len = reference.len() as f64;
        let cand_len = candidate.len() as f64;
        let bp = if cand_len > ref_len {
            1.0
        } else {
            (1.0 - ref_len / cand_len).exp()
        };
        
        Ok(bp * geometric_mean)
    }
    
    /// Levenshtein distance (edit distance)
    pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
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
        
        // Fill DP table
        for i in 1..=m {
            for j in 1..=n {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }
        
        dp[m][n]
    }
    
    /// Byte Pair Encoding (BPE) - simplified training
    pub fn train_bpe(
        corpus: &[String],
        num_merges: usize
    ) -> HashMap<(String, String), String> {
        let mut merges = HashMap::new();
        let mut word_freqs = HashMap::new();
        
        // Initialize word frequencies and split into characters
        for word in corpus {
            let entry = word_freqs.entry(word.clone()).or_insert(0);
            *entry += 1;
        }
        
        // Convert words to character sequences
        let mut vocab: HashMap<String, Vec<String>> = HashMap::new();
        for word in word_freqs.keys() {
            let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            vocab.insert(word.clone(), chars);
        }
        
        // Perform merges
        for merge_num in 0..num_merges {
            let mut pair_freqs = HashMap::new();
            
            // Count adjacent pairs
            for (word, freq) in &word_freqs {
                let tokens = vocab.get(word).unwrap();
                for i in 0..tokens.len().saturating_sub(1) {
                    let pair = (tokens[i].clone(), tokens[i + 1].clone());
                    *pair_freqs.entry(pair).or_insert(0) += freq;
                }
            }
            
            if pair_freqs.is_empty() {
                break;
            }
            
            // Find most frequent pair
            let best_pair = pair_freqs.iter()
                .max_by_key(|(_, &freq)| freq)
                .map(|(pair, _)| pair.clone())
                .unwrap();
            
            // Create merged token
            let merged_token = format!("{}{}", best_pair.0, best_pair.1);
            merges.insert(best_pair.clone(), merged_token.clone());
            
            // Update vocabulary
            for tokens in vocab.values_mut() {
                let mut i = 0;
                while i < tokens.len().saturating_sub(1) {
                    if tokens[i] == best_pair.0 && tokens[i + 1] == best_pair.1 {
                        tokens[i] = merged_token.clone();
                        tokens.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }
        
        merges
    }
    
    /// Sentence similarity using embedding cosine similarity
    pub fn sentence_similarity(
        sentence1_embedding: &[f64],
        sentence2_embedding: &[f64]
    ) -> MathResult<f64> {
        if sentence1_embedding.len() != sentence2_embedding.len() {
            return Err(MathError::InvalidArgument("Sentence embeddings must have same dimension".to_string()));
        }
        
        let dot_product: f64 = sentence1_embedding.iter()
            .zip(sentence2_embedding.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        let norm1 = sentence1_embedding.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let norm2 = sentence2_embedding.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }
        
        Ok(dot_product / (norm1 * norm2))
    }
}

impl MathDomain for LanguageProcessingDomain {
    fn name(&self) -> &str {
        "Language Processing"
    }
    
    fn description(&self) -> &str {
        "Mathematical foundations for natural language processing including embeddings, transformers, attention mechanisms, and text similarity"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "tokenize".to_string(),
            "skip_gram_loss".to_string(),
            "positional_encoding".to_string(),
            "self_attention".to_string(),
            "masked_language_modeling_loss".to_string(),
            "perplexity".to_string(),
            "bleu_score".to_string(),
            "levenshtein_distance".to_string(),
            "sentence_similarity".to_string(),
        ]
    }
    
    fn compute(&self, _operation: &str, _args: &[&dyn std::any::Any]) -> MathResult<Box<dyn std::any::Any>> {
        Err(MathError::NotImplemented("Generic compute not implemented for Language Processing domain".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tokenization() {
        let text = "Hello, world! How are you?";
        let tokens = LanguageProcessingDomain::tokenize(text);
        
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"!".to_string()));
        assert!(tokens.contains(&"?".to_string()));
    }

    #[test]
    fn test_vocabulary() {
        let mut vocab = Vocabulary::new();
        
        let hello_id = vocab.add_token("hello");
        let world_id = vocab.add_token("world");
        
        assert_eq!(vocab.get_id("hello"), hello_id);
        assert_eq!(vocab.get_id("world"), world_id);
        assert_eq!(vocab.get_id("unknown"), 1); // <unk> token
    }

    #[test]
    fn test_ngrams() {
        let tokens = vec!["the".to_string(), "quick".to_string(), "brown".to_string(), "fox".to_string()];
        let bigrams = LanguageProcessingDomain::generate_ngrams(&tokens, 2);
        
        assert_eq!(bigrams.len(), 3);
        assert_eq!(bigrams[0], vec!["the", "quick"]);
        assert_eq!(bigrams[2], vec!["brown", "fox"]);
    }

    #[test]
    fn test_positional_encoding() {
        let pe = LanguageProcessingDomain::positional_encoding(4, 6).unwrap();
        
        assert_eq!(pe.len(), 4); // sequence length
        assert_eq!(pe[0].len(), 6); // d_model
        
        // First position should be [sin(0), cos(0), sin(0), cos(0), ...]
        assert_relative_eq!(pe[0][0], 0.0, epsilon = 1e-10); // sin(0)
        assert_relative_eq!(pe[0][1], 1.0, epsilon = 1e-10); // cos(0)
    }

    #[test]
    fn test_self_attention() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        
        let attention_output = LanguageProcessingDomain::self_attention(&embeddings, 3).unwrap();
        
        assert_eq!(attention_output.len(), 3);
        assert_eq!(attention_output[0].len(), 3);
    }

    #[test]
    fn test_skip_gram_loss() {
        let center = vec![1.0, 0.5, -0.5];
        let context = vec![vec![0.5, 1.0, -0.2]];
        let negative = vec![vec![-1.0, 0.2, 0.8]];
        
        let loss = LanguageProcessingDomain::skip_gram_loss(&center, &context, &negative).unwrap();
        assert!(loss > 0.0);
    }

    #[test]
    fn test_perplexity() {
        let cross_entropy = 2.0;
        let perplexity = LanguageProcessingDomain::perplexity(cross_entropy);
        assert_relative_eq!(perplexity, 7.389, epsilon = 1e-2); // e^2 ≈ 7.389
    }

    #[test]
    fn test_levenshtein_distance() {
        let dist1 = LanguageProcessingDomain::levenshtein_distance("kitten", "sitting");
        assert_eq!(dist1, 3);
        
        let dist2 = LanguageProcessingDomain::levenshtein_distance("hello", "hello");
        assert_eq!(dist2, 0);
        
        let dist3 = LanguageProcessingDomain::levenshtein_distance("", "hello");
        assert_eq!(dist3, 5);
    }

    #[test]
    fn test_bleu_score() {
        let reference = vec!["the".to_string(), "cat".to_string(), "is".to_string(), "on".to_string(), "the".to_string(), "mat".to_string()];
        let candidate1 = vec!["the".to_string(), "cat".to_string(), "is".to_string(), "on".to_string(), "the".to_string(), "mat".to_string()];
        let candidate2 = vec!["a".to_string(), "cat".to_string(), "is".to_string(), "on".to_string(), "a".to_string(), "mat".to_string()];
        
        let bleu1 = LanguageProcessingDomain::bleu_score(&reference, &candidate1, 2).unwrap();
        let bleu2 = LanguageProcessingDomain::bleu_score(&reference, &candidate2, 2).unwrap();
        
        assert!(bleu1 > bleu2); // Perfect match should have higher BLEU
        assert_relative_eq!(bleu1, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sentence_similarity() {
        let emb1 = vec![1.0, 0.0, 0.0];
        let emb2 = vec![1.0, 0.0, 0.0];
        let emb3 = vec![0.0, 1.0, 0.0];
        
        let sim1 = LanguageProcessingDomain::sentence_similarity(&emb1, &emb2).unwrap();
        let sim2 = LanguageProcessingDomain::sentence_similarity(&emb1, &emb3).unwrap();
        
        assert_relative_eq!(sim1, 1.0, epsilon = 1e-10);
        assert_relative_eq!(sim2, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tf_idf() {
        let documents = vec![
            vec!["the".to_string(), "cat".to_string(), "sat".to_string()],
            vec!["the".to_string(), "dog".to_string(), "ran".to_string()],
        ];
        let vocabulary = vec!["the".to_string(), "cat".to_string(), "dog".to_string()];
        
        let tf_idf = LanguageProcessingDomain::compute_tf_idf(&documents, &vocabulary);
        
        assert_eq!(tf_idf.len(), 2); // 2 documents
        assert_eq!(tf_idf[0].len(), 3); // 3 terms in vocabulary
        
        // "the" appears in both documents, so should have lower TF-IDF
        // "cat" and "dog" appear in only one document each, so should have higher TF-IDF
        assert!(tf_idf[0][1] > tf_idf[0][0]); // cat > the in first document
    }

    #[test]
    fn test_multi_head_attention() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.5, -0.5],
            vec![0.0, 1.0, 0.2, -0.2],
        ];
        
        let output = LanguageProcessingDomain::multi_head_attention(&embeddings, 2, 2).unwrap();
        
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 4);
    }
}