use num_bigint::BigInt;
use num_rational::Rational64;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::fmt;

pub type MathResult<T> = Result<T, MathError>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MathError {
    InvalidOperation(String),
    InvalidArgument(String),
    ComputationError(String),
    DomainError(String),
    DivisionByZero,
    Overflow,
    Underflow,
    NotImplemented(String),
}

impl fmt::Display for MathError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MathError::InvalidOperation(op) => write!(f, "Invalid operation: {}", op),
            MathError::InvalidArgument(arg) => write!(f, "Invalid argument: {}", arg),
            MathError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            MathError::DomainError(domain) => write!(f, "Domain error in: {}", domain),
            MathError::DivisionByZero => write!(f, "Division by zero"),
            MathError::Overflow => write!(f, "Numerical overflow"),
            MathError::Underflow => write!(f, "Numerical underflow"),
            MathError::NotImplemented(feature) => write!(f, "Not implemented: {}", feature),
        }
    }
}

impl std::error::Error for MathError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Number {
    Integer(i64),
    BigInteger(BigInt),
    Rational(Rational64),
    Real(f64),
    Complex(Complex64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector2D {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Matrix {
    pub data: Vec<Vec<f64>>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Polynomial {
    pub coefficients: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub expression: String,
    pub variables: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Quaternion {
    pub w: f64, // scalar part
    pub x: f64, // i component
    pub y: f64, // j component
    pub z: f64, // k component
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform4x4 {
    pub matrix: [[f64; 4]; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ray3D {
    pub origin: Point3D,
    pub direction: Vector3D,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AABB {
    pub min: Point3D,
    pub max: Point3D,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OBB {
    pub center: Point3D,
    pub axes: [Vector3D; 3],
    pub extents: Vector3D,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingSphere {
    pub center: Point3D,
    pub radius: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BezierCurve2D {
    pub control_points: Vec<Point2D>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BezierCurve3D {
    pub control_points: Vec<Point3D>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatmullRomSpline2D {
    pub points: Vec<Point2D>,
    pub tension: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatmullRomSpline3D {
    pub points: Vec<Point3D>,
    pub tension: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Keyframe<T> {
    pub time: f64,
    pub value: T,
    pub easing: EasingFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EasingFunction {
    Linear,
    QuadraticIn,
    QuadraticOut,
    QuadraticInOut,
    CubicIn,
    CubicOut,
    CubicInOut,
    QuarticIn,
    QuarticOut,
    QuarticInOut,
    SineIn,
    SineOut,
    SineInOut,
    ExponentialIn,
    ExponentialOut,
    ExponentialInOut,
    CircularIn,
    CircularOut,
    CircularInOut,
    ElasticIn,
    ElasticOut,
    ElasticInOut,
    BounceIn,
    BounceOut,
    BounceInOut,
    BackIn,
    BackOut,
    BackInOut,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationCurve<T> {
    pub keyframes: Vec<Keyframe<T>>,
    pub duration: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationTrack {
    pub name: String,
    pub target_property: String,
    pub curve_f64: Option<AnimationCurve<f64>>,
    pub curve_point2d: Option<AnimationCurve<Point2D>>,
    pub curve_point3d: Option<AnimationCurve<Point3D>>,
    pub curve_vector3d: Option<AnimationCurve<Vector3D>>,
    pub curve_quaternion: Option<AnimationCurve<Quaternion>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Force {
    pub magnitude: f64,
    pub direction: Vector3D,
    pub point_of_application: Point3D,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Moment {
    pub magnitude: f64,
    pub axis: Vector3D,
    pub point: Point3D,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    pub name: String,
    pub density: f64,                    // kg/m³
    pub elastic_modulus: f64,            // Pa (Young's modulus)
    pub poisson_ratio: f64,              // dimensionless
    pub yield_strength: f64,             // Pa
    pub ultimate_strength: f64,          // Pa
    pub thermal_expansion_coefficient: f64, // 1/K
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSection {
    pub name: String,
    pub area: f64,                       // m²
    pub moment_of_inertia_y: f64,        // m⁴ (Iy)
    pub moment_of_inertia_z: f64,        // m⁴ (Iz)
    pub polar_moment_of_inertia: f64,    // m⁴ (J)
    pub section_modulus_y: f64,          // m³ (Sy)
    pub section_modulus_z: f64,          // m³ (Sz)
    pub centroid: Point2D,               // Center of mass location
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: usize,
    pub position: Point3D,
    pub constraints: NodeConstraints,
    pub applied_forces: Vec<Force>,
    pub applied_moments: Vec<Moment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConstraints {
    pub fixed_x: bool,
    pub fixed_y: bool,
    pub fixed_z: bool,
    pub fixed_rotation_x: bool,
    pub fixed_rotation_y: bool,
    pub fixed_rotation_z: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Beam {
    pub id: usize,
    pub start_node: usize,
    pub end_node: usize,
    pub material: Material,
    pub cross_section: CrossSection,
    pub length: f64,
    pub orientation_angle: f64, // rotation about beam axis
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Truss {
    pub nodes: Vec<Node>,
    pub members: Vec<TrussMember>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrussMember {
    pub id: usize,
    pub start_node: usize,
    pub end_node: usize,
    pub cross_sectional_area: f64,
    pub elastic_modulus: f64,
    pub length: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadCase {
    pub name: String,
    pub load_factor: f64,
    pub forces: Vec<Force>,
    pub moments: Vec<Moment>,
    pub distributed_loads: Vec<DistributedLoad>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedLoad {
    pub beam_id: usize,
    pub start_position: f64,    // Along beam length (0.0 to 1.0)
    pub end_position: f64,      // Along beam length (0.0 to 1.0)
    pub start_magnitude: f64,   // Force per unit length
    pub end_magnitude: f64,     // Force per unit length
    pub direction: Vector3D,    // Direction of load
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressState {
    pub normal_stress_x: f64,
    pub normal_stress_y: f64,
    pub normal_stress_z: f64,
    pub shear_stress_xy: f64,
    pub shear_stress_yz: f64,
    pub shear_stress_zx: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrainState {
    pub normal_strain_x: f64,
    pub normal_strain_y: f64,
    pub normal_strain_z: f64,
    pub shear_strain_xy: f64,
    pub shear_strain_yz: f64,
    pub shear_strain_zx: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamAnalysisResult {
    pub beam_id: usize,
    pub max_bending_moment: f64,
    pub max_shear_force: f64,
    pub max_deflection: f64,
    pub max_stress: f64,
    pub safety_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalAnalysisResult {
    pub natural_frequencies: Vec<f64>,   // Hz
    pub mode_shapes: Vec<Vec<f64>>,      // Normalized eigenvectors
    pub participation_factors: Vec<f64>, // For each mode
}

// ===== TPE (Thermoplastic Elastomer) Domain Types =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TPEMaterial {
    pub name: String,
    pub hardness_shore_a: f64,           // Shore A hardness (0-100)
    pub tensile_strength: f64,           // Pa
    pub elongation_at_break: f64,        // % (0-1000)
    pub elastic_modulus: f64,            // Pa at specific temperature
    pub density: f64,                    // kg/m³
    pub glass_transition_temp: f64,      // K - Tg
    pub melting_temp: f64,               // K - Tm
    pub thermal_expansion_coeff: f64,    // 1/K
    pub thermal_conductivity: f64,       // W/(m·K)
    pub specific_heat: f64,              // J/(kg·K)
    pub melt_flow_index: f64,            // g/10min at specified conditions
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockCopolymer {
    pub hard_block_fraction: f64,        // Volume fraction (0-1)
    pub soft_block_fraction: f64,        // Volume fraction (0-1)
    pub domain_spacing: f64,             // nm - characteristic length scale
    pub hard_block_modulus: f64,         // Pa
    pub soft_block_modulus: f64,         // Pa
    pub interface_thickness: f64,        // nm
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalTransition {
    pub temperature: f64,                // K
    pub transition_type: TransitionType,
    pub enthalpy_change: f64,            // J/kg
    pub specific_heat_change: f64,       // J/(kg·K)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionType {
    GlassTransition,
    Melting,
    Crystallization,
    CrossLinking,
    Degradation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConditions {
    pub melt_temperature: f64,           // K
    pub injection_pressure: f64,        // Pa
    pub cooling_rate: f64,               // K/s
    pub residence_time: f64,             // s
    pub shear_rate: f64,                 // 1/s
}

// ===== Biology Domain Types =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    pub id: usize,
    pub position: Point3D,
    pub cell_type: CellType,
    pub state: CellState,
    pub metabolic_rate: f64,             // Arbitrary units
    pub growth_rate: f64,                // 1/s
    pub neighbors: Vec<usize>,           // Cell IDs
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CellType {
    Stem,
    Differentiated(String),              // Tissue-specific name
    Cancer,
    Immune,
    Neural,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CellState {
    Quiescent,
    Growing,
    Dividing,
    Dying,
    Differentiated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub id: usize,
    pub position: Point3D,
    pub membrane_potential: f64,         // mV
    pub threshold_potential: f64,        // mV
    pub resting_potential: f64,          // mV
    pub refractory_period: f64,          // ms
    pub axon_length: f64,                // µm
    pub dendrite_count: usize,
    pub synapses: Vec<Synapse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    pub pre_neuron_id: usize,
    pub post_neuron_id: usize,
    pub weight: f64,                     // Synaptic strength
    pub delay: f64,                      // ms
    pub neurotransmitter_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiochemicalReaction {
    pub id: String,
    pub reactants: Vec<(String, f64)>,   // (molecule_name, stoichiometry)
    pub products: Vec<(String, f64)>,    // (molecule_name, stoichiometry)
    pub rate_constant: f64,              // Units depend on reaction order
    pub activation_energy: f64,          // J/mol
    pub enzyme: Option<String>,          // Optional enzyme catalyst
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Population {
    pub species: String,
    pub size: f64,                       // Number of individuals
    pub growth_rate: f64,                // Intrinsic growth rate
    pub carrying_capacity: f64,          // Maximum sustainable population
    pub mortality_rate: f64,             // Natural death rate
    pub migration_rate: f64,             // Net immigration rate
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSystem {
    pub axiom: String,                   // Starting string
    pub rules: Vec<(char, String)>,      // Production rules
    pub angle: f64,                      // Degrees for turtle graphics
    pub generations: usize,              // Number of iterations
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuringPattern {
    pub activator_diffusion: f64,        // Diffusion coefficient
    pub inhibitor_diffusion: f64,        // Diffusion coefficient
    pub reaction_rate: f64,              // Rate constant
    pub pattern_wavelength: f64,         // Characteristic length
}

// ===== Silicone Domain Types =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiloxaneChain {
    pub chain_length: usize,             // Number of Si-O units
    pub side_groups: Vec<SideGroup>,     // Attached organic groups
    pub crosslink_density: f64,          // mol/m³
    pub molecular_weight: f64,           // g/mol
    pub branching_factor: f64,           // Average branches per chain
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SideGroup {
    Methyl,                              // -CH3
    Phenyl,                              // -C6H5
    Vinyl,                               // -CH=CH2
    Trifluoropropyl,                     // -C3H4F3
    Hydroxy,                             // -OH
    Custom(String),                      // User-defined group
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiliconeMaterial {
    pub name: String,
    pub base_polymer: SiloxaneChain,
    pub hardness_shore_a: f64,           // Shore A hardness (0-100)
    pub tensile_strength: f64,           // Pa
    pub elongation_at_break: f64,        // % (0-1000)
    pub tear_strength: f64,              // N/mm
    pub temperature_range: (f64, f64),   // (min_K, max_K) operating range
    pub thermal_conductivity: f64,       // W/(m·K)
    pub electrical_resistivity: f64,     // Ω·m
    pub dielectric_strength: f64,        // V/m
    pub gas_permeability: f64,           // cm³·mm/(m²·day·atm)
    pub biocompatibility: BioCompatibility,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BioCompatibility {
    Medical,                             // USP Class VI
    Food,                                // FDA approved
    Industrial,                          // Standard grade
    Implantable,                         // Long-term biocompatible
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RheologicalData {
    pub viscosity: f64,                  // Pa·s
    pub shear_rate: f64,                 // 1/s
    pub temperature: f64,                // K
    pub cure_time: f64,                  // s
    pub gel_time: f64,                   // s
    pub pot_life: f64,                   // s
}

// ===== Advanced Algorithm Types =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TPENode {
    pub parameter_name: String,
    pub value: f64,
    pub left_child: Option<Box<TPENode>>,
    pub right_child: Option<Box<TPENode>>,
    pub split_value: f64,
    pub samples: Vec<(f64, f64)>,        // (parameter_value, objective_value)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SMBOConfig {
    pub n_initial_points: usize,         // Random sampling before modeling
    pub n_ei_candidates: usize,          // Expected improvement candidates
    pub acquisition_function: AcquisitionFunction,
    pub surrogate_model: SurrogateModel,
    pub optimization_budget: usize,      // Total function evaluations
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AcquisitionFunction {
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound(f64),           // beta parameter
    TreeParzenEstimator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SurrogateModel {
    GaussianProcess,
    RandomForest,
    TreeParzenEstimator,
    NeuralNetwork,
}

// ===== Estrogen Receptor Modulator Types =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriphenylethyleneCompound {
    pub name: String,
    pub molecular_formula: String,
    pub molecular_weight: f64,           // g/mol
    pub binding_affinity_er_alpha: f64,  // IC50 in nM
    pub binding_affinity_er_beta: f64,   // IC50 in nM
    pub selectivity_ratio: f64,          // ERα/ERβ selectivity
    pub agonist_activity: f64,           // % relative to estradiol
    pub antagonist_activity: f64,        // % inhibition
    pub tissue_selectivity: Vec<(String, f64)>, // (tissue, activity)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceptorBinding {
    pub ligand: String,
    pub receptor: String,
    pub kd: f64,                         // Dissociation constant (M)
    pub bmax: f64,                       // Maximum binding capacity
    pub hill_coefficient: f64,           // Cooperativity
    pub ec50: f64,                       // Half-maximum effective concentration
}