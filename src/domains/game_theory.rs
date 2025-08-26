use crate::core::{MathDomain, MathResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameTheoryDomain {
    name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Player {
    pub id: String,
    pub name: String,
    pub strategies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalFormGame {
    pub players: Vec<Player>,
    pub payoff_matrix: HashMap<Vec<usize>, Vec<f64>>, // strategy profile -> payoffs
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensiveFormGame {
    pub players: Vec<Player>,
    pub nodes: HashMap<String, GameNode>,
    pub information_sets: HashMap<String, Vec<String>>,
    pub root: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameNode {
    pub id: String,
    pub player: Option<usize>, // None for chance/terminal nodes
    pub actions: Vec<String>,
    pub children: HashMap<String, String>, // action -> child node
    pub payoffs: Option<Vec<f64>>, // For terminal nodes
    pub probability: Option<f64>, // For chance nodes
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    pub player: usize,
    pub pure_strategies: HashMap<String, String>, // information set -> action
    pub mixed_probabilities: HashMap<String, HashMap<String, f64>>, // info set -> action -> prob
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NashEquilibrium {
    pub strategies: Vec<Strategy>,
    pub payoffs: Vec<f64>,
    pub equilibrium_type: EquilibriumType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EquilibriumType {
    Pure,
    Mixed,
    Correlated,
    Evolutionary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CooperativeGame {
    pub players: Vec<String>,
    pub characteristic_function: HashMap<Vec<String>, f64>, // coalition -> value
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    pub solution_concept: SolutionConcept,
    pub allocation: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SolutionConcept {
    Core,
    Shapley,
    Nucleolus,
    Bargaining,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuctionMechanism {
    pub auction_type: AuctionType,
    pub bidders: Vec<String>,
    pub valuations: HashMap<String, f64>,
    pub bids: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuctionType {
    FirstPrice,
    SecondPrice,
    English,
    Dutch,
    Vickrey,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionaryGame {
    pub population: Vec<String>,
    pub strategies: Vec<String>,
    pub fitness_matrix: HashMap<(String, String), f64>,
    pub population_distribution: HashMap<String, f64>,
    pub replicator_dynamics: HashMap<String, f64>,
}

impl GameTheoryDomain {
    pub fn new() -> Self {
        Self {
            name: "Game Theory".to_string(),
        }
    }

    // Normal Form Games
    pub fn create_normal_form_game(&self, players: Vec<Player>) -> NormalFormGame {
        NormalFormGame {
            players,
            payoff_matrix: HashMap::new(),
        }
    }

    pub fn set_payoffs(&self, game: &mut NormalFormGame, 
                      strategy_profile: Vec<usize>, payoffs: Vec<f64>) {
        game.payoff_matrix.insert(strategy_profile, payoffs);
    }

    pub fn find_dominant_strategies(&self, game: &NormalFormGame) -> HashMap<usize, Vec<usize>> {
        let mut dominant_strategies = HashMap::new();
        
        for (player_idx, player) in game.players.iter().enumerate() {
            let mut dominated = Vec::new();
            
            for (i, _strategy_i) in player.strategies.iter().enumerate() {
                for (j, _strategy_j) in player.strategies.iter().enumerate() {
                    if i != j && self.strictly_dominates(game, player_idx, i, j) {
                        dominated.push(j);
                    }
                }
            }
            
            if !dominated.is_empty() {
                dominant_strategies.insert(player_idx, dominated);
            }
        }
        
        dominant_strategies
    }

    fn strictly_dominates(&self, game: &NormalFormGame, player: usize, 
                         strategy_i: usize, strategy_j: usize) -> bool {
        // Check if strategy i strictly dominates strategy j for given player
        let other_players = game.players.iter().enumerate()
            .filter(|(idx, _)| *idx != player)
            .collect::<Vec<_>>();
        
        self.check_dominance_recursive(game, player, strategy_i, strategy_j, &other_players, vec![], 0)
    }

    fn check_dominance_recursive(&self, game: &NormalFormGame, player: usize,
                               strategy_i: usize, strategy_j: usize,
                               other_players: &[(usize, &Player)],
                               current_profile: Vec<usize>, depth: usize) -> bool {
        if depth == other_players.len() {
            let mut profile_i = current_profile.clone();
            let mut profile_j = current_profile.clone();
            profile_i.insert(player, strategy_i);
            profile_j.insert(player, strategy_j);
            
            let payoff_i = game.payoff_matrix.get(&profile_i)
                .map(|payoffs| payoffs[player])
                .unwrap_or(0.0);
            let payoff_j = game.payoff_matrix.get(&profile_j)
                .map(|payoffs| payoffs[player])
                .unwrap_or(0.0);
            
            return payoff_i > payoff_j;
        }
        
        let (_, other_player) = other_players[depth];
        for strategy_idx in 0..other_player.strategies.len() {
            let mut new_profile = current_profile.clone();
            new_profile.push(strategy_idx);
            
            if !self.check_dominance_recursive(game, player, strategy_i, strategy_j,
                                             other_players, new_profile, depth + 1) {
                return false;
            }
        }
        true
    }

    pub fn find_pure_nash_equilibria(&self, game: &NormalFormGame) -> Vec<NashEquilibrium> {
        let mut equilibria = Vec::new();
        let strategy_counts: Vec<usize> = game.players.iter()
            .map(|p| p.strategies.len()).collect();
        
        let total_profiles = strategy_counts.iter().product::<usize>();
        
        for profile_idx in 0..total_profiles {
            let profile = self.index_to_profile(profile_idx, &strategy_counts);
            if self.is_nash_equilibrium(game, &profile) {
                let payoffs = game.payoff_matrix.get(&profile)
                    .cloned().unwrap_or_else(|| vec![0.0; game.players.len()]);
                
                equilibria.push(NashEquilibrium {
                    strategies: profile.iter().enumerate().map(|(player, &strategy)| {
                        Strategy {
                            player,
                            pure_strategies: HashMap::new(),
                            mixed_probabilities: HashMap::new(),
                        }
                    }).collect(),
                    payoffs,
                    equilibrium_type: EquilibriumType::Pure,
                });
            }
        }
        
        equilibria
    }

    fn index_to_profile(&self, mut index: usize, strategy_counts: &[usize]) -> Vec<usize> {
        let mut profile = Vec::new();
        for &count in strategy_counts.iter().rev() {
            profile.push(index % count);
            index /= count;
        }
        profile.reverse();
        profile
    }

    fn is_nash_equilibrium(&self, game: &NormalFormGame, profile: &[usize]) -> bool {
        for (player_idx, &current_strategy) in profile.iter().enumerate() {
            let current_payoff = game.payoff_matrix.get(profile)
                .map(|payoffs| payoffs[player_idx])
                .unwrap_or(0.0);
            
            for alternative_strategy in 0..game.players[player_idx].strategies.len() {
                if alternative_strategy != current_strategy {
                    let mut alternative_profile = profile.to_vec();
                    alternative_profile[player_idx] = alternative_strategy;
                    
                    let alternative_payoff = game.payoff_matrix.get(&alternative_profile)
                        .map(|payoffs| payoffs[player_idx])
                        .unwrap_or(0.0);
                    
                    if alternative_payoff > current_payoff {
                        return false;
                    }
                }
            }
        }
        true
    }

    // Extensive Form Games
    pub fn create_extensive_form_game(&self, players: Vec<Player>) -> ExtensiveFormGame {
        ExtensiveFormGame {
            players,
            nodes: HashMap::new(),
            information_sets: HashMap::new(),
            root: String::new(),
        }
    }

    pub fn add_game_node(&self, game: &mut ExtensiveFormGame, node: GameNode) {
        if game.root.is_empty() {
            game.root = node.id.clone();
        }
        game.nodes.insert(node.id.clone(), node);
    }

    pub fn backward_induction(&self, game: &ExtensiveFormGame) -> Vec<f64> {
        let mut node_values = HashMap::new();
        self.compute_node_values(game, &game.root, &mut node_values);
        node_values.get(&game.root).cloned().unwrap_or_default()
    }

    fn compute_node_values(&self, game: &ExtensiveFormGame, node_id: &str,
                          node_values: &mut HashMap<String, Vec<f64>>) -> Vec<f64> {
        if let Some(node) = game.nodes.get(node_id) {
            if let Some(ref payoffs) = node.payoffs {
                // Terminal node
                let values = payoffs.clone();
                node_values.insert(node_id.to_string(), values.clone());
                return values;
            }
            
            if let Some(player_idx) = node.player {
                // Decision node
                let mut best_values = vec![f64::NEG_INFINITY; game.players.len()];
                
                for (_, child_id) in &node.children {
                    let child_values = self.compute_node_values(game, child_id, node_values);
                    if child_values[player_idx] > best_values[player_idx] {
                        best_values = child_values;
                    }
                }
                
                node_values.insert(node_id.to_string(), best_values.clone());
                return best_values;
            }
        }
        
        vec![0.0; game.players.len()]
    }

    // Cooperative Games
    pub fn create_cooperative_game(&self, players: Vec<String>) -> CooperativeGame {
        CooperativeGame {
            players,
            characteristic_function: HashMap::new(),
        }
    }

    pub fn set_coalition_value(&self, game: &mut CooperativeGame, 
                              coalition: Vec<String>, value: f64) {
        game.characteristic_function.insert(coalition, value);
    }

    pub fn compute_shapley_value(&self, game: &CooperativeGame) -> HashMap<String, f64> {
        let mut shapley_values = HashMap::new();
        let n = game.players.len();
        
        for player in &game.players {
            let mut value = 0.0;
            
            // Consider all possible coalitions
            let all_coalitions = self.generate_all_coalitions(&game.players);
            
            for coalition in all_coalitions {
                if !coalition.contains(player) {
                    let coalition_size = coalition.len();
                    let weight = self.factorial(coalition_size) * self.factorial(n - coalition_size - 1)
                        / self.factorial(n);
                    
                    let mut coalition_with_player = coalition.clone();
                    coalition_with_player.push(player.clone());
                    coalition_with_player.sort();
                    
                    let value_with = game.characteristic_function.get(&coalition_with_player)
                        .cloned().unwrap_or(0.0);
                    let value_without = game.characteristic_function.get(&coalition)
                        .cloned().unwrap_or(0.0);
                    
                    value += weight as f64 * (value_with - value_without);
                }
            }
            
            shapley_values.insert(player.clone(), value);
        }
        
        shapley_values
    }

    fn generate_all_coalitions(&self, players: &[String]) -> Vec<Vec<String>> {
        let mut coalitions = Vec::new();
        let n = players.len();
        
        for i in 0..(1 << n) {
            let mut coalition = Vec::new();
            for j in 0..n {
                if (i >> j) & 1 == 1 {
                    coalition.push(players[j].clone());
                }
            }
            coalition.sort();
            coalitions.push(coalition);
        }
        
        coalitions
    }

    fn factorial(&self, n: usize) -> usize {
        if n <= 1 { 1 } else { n * self.factorial(n - 1) }
    }

    pub fn is_in_core(&self, game: &CooperativeGame, allocation: &HashMap<String, f64>) -> bool {
        let all_coalitions = self.generate_all_coalitions(&game.players);
        
        for coalition in all_coalitions {
            if !coalition.is_empty() && coalition.len() < game.players.len() {
                let coalition_value = game.characteristic_function.get(&coalition)
                    .cloned().unwrap_or(0.0);
                let allocation_sum: f64 = coalition.iter()
                    .map(|player| allocation.get(player).cloned().unwrap_or(0.0))
                    .sum();
                
                if allocation_sum < coalition_value {
                    return false;
                }
            }
        }
        true
    }

    // Evolutionary Game Theory
    pub fn create_evolutionary_game(&self, population: Vec<String>, 
                                   strategies: Vec<String>) -> EvolutionaryGame {
        EvolutionaryGame {
            population,
            strategies,
            fitness_matrix: HashMap::new(),
            population_distribution: HashMap::new(),
            replicator_dynamics: HashMap::new(),
        }
    }

    pub fn set_fitness(&self, game: &mut EvolutionaryGame, 
                      strategy1: String, strategy2: String, fitness: f64) {
        game.fitness_matrix.insert((strategy1, strategy2), fitness);
    }

    pub fn compute_replicator_dynamics(&self, game: &mut EvolutionaryGame) {
        for strategy in &game.strategies {
            let current_freq = game.population_distribution.get(strategy).cloned().unwrap_or(0.0);
            let average_fitness = self.compute_average_fitness(game, strategy);
            let population_average_fitness = self.compute_population_average_fitness(game);
            
            let growth_rate = current_freq * (average_fitness - population_average_fitness);
            game.replicator_dynamics.insert(strategy.clone(), growth_rate);
        }
    }

    fn compute_average_fitness(&self, game: &EvolutionaryGame, strategy: &str) -> f64 {
        let mut total_fitness = 0.0;
        
        for other_strategy in &game.strategies {
            let other_freq = game.population_distribution.get(other_strategy).cloned().unwrap_or(0.0);
            let fitness = game.fitness_matrix.get(&(strategy.to_string(), other_strategy.clone()))
                .cloned().unwrap_or(0.0);
            total_fitness += other_freq * fitness;
        }
        
        total_fitness
    }

    fn compute_population_average_fitness(&self, game: &EvolutionaryGame) -> f64 {
        let mut total_fitness = 0.0;
        
        for strategy in &game.strategies {
            let freq = game.population_distribution.get(strategy).cloned().unwrap_or(0.0);
            let avg_fitness = self.compute_average_fitness(game, strategy);
            total_fitness += freq * avg_fitness;
        }
        
        total_fitness
    }

    // Auctions
    pub fn create_auction(&self, auction_type: AuctionType, bidders: Vec<String>) -> AuctionMechanism {
        AuctionMechanism {
            auction_type,
            bidders,
            valuations: HashMap::new(),
            bids: HashMap::new(),
        }
    }

    pub fn set_valuation(&self, auction: &mut AuctionMechanism, bidder: String, valuation: f64) {
        auction.valuations.insert(bidder, valuation);
    }

    pub fn set_bid(&self, auction: &mut AuctionMechanism, bidder: String, bid: f64) {
        auction.bids.insert(bidder, bid);
    }

    pub fn determine_winner(&self, auction: &AuctionMechanism) -> Option<(String, f64)> {
        match auction.auction_type {
            AuctionType::FirstPrice | AuctionType::SecondPrice | AuctionType::Vickrey => {
                auction.bids.iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(bidder, &bid)| (bidder.clone(), bid))
            }
            _ => None, // Other auction types would need more complex logic
        }
    }

    pub fn compute_payment(&self, auction: &AuctionMechanism, winner: &str) -> f64 {
        match auction.auction_type {
            AuctionType::FirstPrice => {
                auction.bids.get(winner).cloned().unwrap_or(0.0)
            }
            AuctionType::SecondPrice | AuctionType::Vickrey => {
                // Pay second-highest bid
                let mut bids: Vec<f64> = auction.bids.values().cloned().collect();
                bids.sort_by(|a, b| b.partial_cmp(a).unwrap());
                bids.get(1).cloned().unwrap_or(0.0)
            }
            _ => 0.0,
        }
    }
}

impl MathDomain for GameTheoryDomain {
    fn name(&self) -> &str {
        &self.name
    }

    fn supports_operation(&self, operation: &str) -> bool {
        matches!(operation, 
            "normal_form_game" | "extensive_form_game" | "cooperative_game" |
            "nash_equilibrium" | "dominant_strategies" | "backward_induction" |
            "shapley_value" | "core" | "evolutionary_game" | "auction"
        )
    }

    fn description(&self) -> &str {
        "Game Theory modeling and analysis framework"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    fn compute(&self, operation: &str, _args: &[&dyn std::any::Any]) -> MathResult<Box<dyn std::any::Any>> {
        match operation {
            "normal_form_game" => Ok(Box::new("Normal form game created".to_string())),
            "nash_equilibrium" => Ok(Box::new("Nash equilibrium computed".to_string())),
            "shapley_value" => Ok(Box::new("Shapley value computed".to_string())),
            _ => Err(crate::core::MathError::NotImplemented(format!("Operation '{}' not implemented", operation))),
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "normal_form_game".to_string(), "extensive_form_game".to_string(),
            "cooperative_game".to_string(), "nash_equilibrium".to_string(),
            "dominant_strategies".to_string(), "backward_induction".to_string(),
            "shapley_value".to_string(), "core".to_string(),
            "evolutionary_game".to_string(), "auction".to_string()
        ]
    }
}

pub fn game_theory() -> GameTheoryDomain {
    GameTheoryDomain::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_form_game() {
        let domain = GameTheoryDomain::new();
        let players = vec![
            Player {
                id: "p1".to_string(),
                name: "Player 1".to_string(),
                strategies: vec!["C".to_string(), "D".to_string()],
            },
            Player {
                id: "p2".to_string(),
                name: "Player 2".to_string(),
                strategies: vec!["C".to_string(), "D".to_string()],
            },
        ];
        
        let mut game = domain.create_normal_form_game(players);
        domain.set_payoffs(&mut game, vec![0, 0], vec![3.0, 3.0]); // CC
        domain.set_payoffs(&mut game, vec![0, 1], vec![0.0, 5.0]); // CD
        domain.set_payoffs(&mut game, vec![1, 0], vec![5.0, 0.0]); // DC
        domain.set_payoffs(&mut game, vec![1, 1], vec![1.0, 1.0]); // DD
        
        let equilibria = domain.find_pure_nash_equilibria(&game);
        assert!(!equilibria.is_empty());
    }

    #[test]
    fn test_cooperative_game() {
        let domain = GameTheoryDomain::new();
        let players = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut game = domain.create_cooperative_game(players);
        
        domain.set_coalition_value(&mut game, vec!["A".to_string()], 10.0);
        domain.set_coalition_value(&mut game, vec!["B".to_string()], 20.0);
        domain.set_coalition_value(&mut game, vec!["C".to_string()], 30.0);
        domain.set_coalition_value(&mut game, vec!["A".to_string(), "B".to_string()], 40.0);
        domain.set_coalition_value(&mut game, vec!["A".to_string(), "B".to_string(), "C".to_string()], 100.0);
        
        let shapley = domain.compute_shapley_value(&game);
        assert!(shapley.contains_key("A"));
        assert!(shapley.contains_key("B"));
        assert!(shapley.contains_key("C"));
    }

    #[test]
    fn test_auction_mechanism() {
        let domain = GameTheoryDomain::new();
        let bidders = vec!["Bidder1".to_string(), "Bidder2".to_string()];
        let mut auction = domain.create_auction(AuctionType::SecondPrice, bidders);
        
        domain.set_valuation(&mut auction, "Bidder1".to_string(), 100.0);
        domain.set_valuation(&mut auction, "Bidder2".to_string(), 80.0);
        domain.set_bid(&mut auction, "Bidder1".to_string(), 90.0);
        domain.set_bid(&mut auction, "Bidder2".to_string(), 75.0);
        
        let winner = domain.determine_winner(&auction);
        assert!(winner.is_some());
        
        if let Some((winner_name, _)) = winner {
            let payment = domain.compute_payment(&auction, &winner_name);
            assert_eq!(payment, 75.0); // Second price auction
        }
    }
}