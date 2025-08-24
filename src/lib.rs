pub mod core;
pub mod domains;
pub mod plugins;
pub mod utils;

pub use core::MathTables;
pub use domains::*;

pub mod prelude {
    pub use crate::core::*;
    pub use crate::domains::number_theory::*;
    pub use crate::domains::algebra::*;
    pub use crate::domains::geometry::*;
    pub use crate::domains::calculus::*;
    pub use crate::domains::discrete::*;
    pub use crate::domains::philosophy::*;
    pub use crate::domains::physics::*;
    pub use crate::domains::foundations::*;
    pub use crate::domains::statistics::*;
    pub use crate::domains::linear_algebra::*;
    pub use crate::domains::optimization::*;
    pub use crate::domains::machine_learning::*;
    pub use crate::domains::signal_processing::*;
    pub use crate::domains::numerical_analysis::*;
    pub use crate::domains::graph_theory::*;
    pub use crate::domains::cryptography::*;
    pub use crate::domains::financial_math::*;
    pub use crate::domains::interval_math::*;
    pub use crate::domains::autodiff::*;
    pub use crate::domains::geometric_algebra::*;
    pub use crate::domains::symbolic_computation::*;
    pub use crate::domains::padic_numbers::*;
    pub use crate::domains::discrete_geometry::*;
    pub use crate::domains::topology::*;
    pub use crate::domains::modular_arithmetic::*;
    pub use crate::domains::fuzzy_logic::*;
    pub use crate::domains::possibility_theory::*;
    pub use crate::domains::constraint_solving::*;
    pub use crate::domains::sat_smt::*;
    pub use crate::domains::graphics_3d::*;
    pub use crate::domains::animation::*;
    pub use crate::domains::structural_engineering::*;
    pub use crate::domains::tpe::*;
    pub use crate::domains::biology::*;
    pub use crate::domains::silicone::*;
    pub use crate::domains::smbo_tpe::*;
    pub use crate::plugins::PluginRegistry;
    pub use crate::utils::*;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_framework_initialization() {
        let math_tables = MathTables::new();
        assert!(math_tables.is_initialized());
    }
}