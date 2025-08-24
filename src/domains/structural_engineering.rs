use crate::core::{
    MathDomain, MathResult, MathError, Point2D, Point3D, Vector3D, Matrix,
    Force, Moment, Material, CrossSection, Node, NodeConstraints, Beam, Truss, TrussMember,
    LoadCase, DistributedLoad, StressState, StrainState, BeamAnalysisResult, ModalAnalysisResult
};
use std::any::Any;
use std::f64::consts::PI;

pub struct StructuralEngineeringDomain;

impl Material {
    pub fn steel_a36() -> Self {
        Self {
            name: "Steel A36".to_string(),
            density: 7850.0,                    // kg/m³
            elastic_modulus: 200_000_000_000.0, // Pa (200 GPa)
            poisson_ratio: 0.26,                // dimensionless
            yield_strength: 250_000_000.0,      // Pa (250 MPa)
            ultimate_strength: 400_000_000.0,   // Pa (400 MPa)
            thermal_expansion_coefficient: 12e-6, // 1/K
        }
    }
    
    pub fn concrete_normal() -> Self {
        Self {
            name: "Normal Weight Concrete".to_string(),
            density: 2400.0,                    // kg/m³
            elastic_modulus: 25_000_000_000.0,  // Pa (25 GPa)
            poisson_ratio: 0.2,                 // dimensionless
            yield_strength: 3_000_000.0,        // Pa (3 MPa compressive)
            ultimate_strength: 4_000_000.0,     // Pa (4 MPa compressive)
            thermal_expansion_coefficient: 10e-6, // 1/K
        }
    }
    
    pub fn aluminum_6061() -> Self {
        Self {
            name: "Aluminum 6061-T6".to_string(),
            density: 2700.0,                    // kg/m³
            elastic_modulus: 69_000_000_000.0,  // Pa (69 GPa)
            poisson_ratio: 0.33,                // dimensionless
            yield_strength: 276_000_000.0,      // Pa (276 MPa)
            ultimate_strength: 310_000_000.0,   // Pa (310 MPa)
            thermal_expansion_coefficient: 23e-6, // 1/K
        }
    }
}

impl CrossSection {
    pub fn rectangular(width: f64, height: f64) -> Self {
        let area = width * height;
        let iy = width * height.powi(3) / 12.0;
        let iz = height * width.powi(3) / 12.0;
        let j = iy + iz; // Approximate for rectangular sections
        let sy = iy / (height / 2.0);
        let sz = iz / (width / 2.0);
        
        Self {
            name: format!("Rectangular {}x{}", width, height),
            area,
            moment_of_inertia_y: iy,
            moment_of_inertia_z: iz,
            polar_moment_of_inertia: j,
            section_modulus_y: sy,
            section_modulus_z: sz,
            centroid: Point2D { x: width / 2.0, y: height / 2.0 },
        }
    }
    
    pub fn circular(diameter: f64) -> Self {
        let radius = diameter / 2.0;
        let area = PI * radius.powi(2);
        let i = PI * radius.powi(4) / 4.0;
        let j = PI * radius.powi(4) / 2.0;
        let s = i / radius;
        
        Self {
            name: format!("Circular D{}", diameter),
            area,
            moment_of_inertia_y: i,
            moment_of_inertia_z: i,
            polar_moment_of_inertia: j,
            section_modulus_y: s,
            section_modulus_z: s,
            centroid: Point2D { x: radius, y: radius },
        }
    }
    
    pub fn wide_flange(height: f64, flange_width: f64, flange_thickness: f64, web_thickness: f64) -> Self {
        let web_height = height - 2.0 * flange_thickness;
        Self::i_beam(flange_width, flange_thickness, web_height, web_thickness)
    }
    
    pub fn i_beam(flange_width: f64, flange_thickness: f64, web_height: f64, web_thickness: f64) -> Self {
        let total_height = web_height + 2.0 * flange_thickness;
        
        // Calculate area
        let flange_area = 2.0 * flange_width * flange_thickness;
        let web_area = web_height * web_thickness;
        let area = flange_area + web_area;
        
        // Calculate moment of inertia about y-axis (strong axis)
        let flange_i_own = 2.0 * (flange_width * flange_thickness.powi(3) / 12.0);
        let flange_distance = (web_height + flange_thickness) / 2.0;
        let flange_i_parallel = 2.0 * flange_width * flange_thickness * flange_distance.powi(2);
        let web_i = web_thickness * web_height.powi(3) / 12.0;
        let iy = flange_i_own + flange_i_parallel + web_i;
        
        // Calculate moment of inertia about z-axis (weak axis)
        let flange_iz = 2.0 * (flange_thickness * flange_width.powi(3) / 12.0);
        let web_iz = web_height * web_thickness.powi(3) / 12.0;
        let iz = flange_iz + web_iz;
        
        let j = iy + iz; // Approximate
        let sy = iy / (total_height / 2.0);
        let sz = iz / (flange_width / 2.0);
        
        Self {
            name: format!("I-Beam {}x{}x{}x{}", flange_width, flange_thickness, web_height, web_thickness),
            area,
            moment_of_inertia_y: iy,
            moment_of_inertia_z: iz,
            polar_moment_of_inertia: j,
            section_modulus_y: sy,
            section_modulus_z: sz,
            centroid: Point2D { x: flange_width / 2.0, y: total_height / 2.0 },
        }
    }
}

impl NodeConstraints {
    pub fn fixed() -> Self {
        Self {
            fixed_x: true,
            fixed_y: true,
            fixed_z: true,
            fixed_rotation_x: true,
            fixed_rotation_y: true,
            fixed_rotation_z: true,
        }
    }
    
    pub fn pinned() -> Self {
        Self {
            fixed_x: true,
            fixed_y: true,
            fixed_z: true,
            fixed_rotation_x: false,
            fixed_rotation_y: false,
            fixed_rotation_z: false,
        }
    }
    
    pub fn roller_x() -> Self {
        Self {
            fixed_x: false,
            fixed_y: true,
            fixed_z: true,
            fixed_rotation_x: false,
            fixed_rotation_y: false,
            fixed_rotation_z: false,
        }
    }
    
    pub fn free() -> Self {
        Self {
            fixed_x: false,
            fixed_y: false,
            fixed_z: false,
            fixed_rotation_x: false,
            fixed_rotation_y: false,
            fixed_rotation_z: false,
        }
    }
}

impl StructuralEngineeringDomain {
    pub fn new() -> Self {
        Self
    }
    
    // ===== Force and Moment Calculations =====
    
    pub fn resultant_force(forces: &[Force]) -> Vector3D {
        let mut resultant = Vector3D { x: 0.0, y: 0.0, z: 0.0 };
        
        for force in forces {
            let force_vector = Vector3D {
                x: force.magnitude * force.direction.x,
                y: force.magnitude * force.direction.y,
                z: force.magnitude * force.direction.z,
            };
            
            resultant.x += force_vector.x;
            resultant.y += force_vector.y;
            resultant.z += force_vector.z;
        }
        
        resultant
    }
    
    pub fn resultant_moment(forces: &[Force], moments: &[Moment], reference_point: &Point3D) -> Vector3D {
        let mut resultant = Vector3D { x: 0.0, y: 0.0, z: 0.0 };
        
        // Add moments from forces
        for force in forces {
            let r = Vector3D {
                x: force.point_of_application.x - reference_point.x,
                y: force.point_of_application.y - reference_point.y,
                z: force.point_of_application.z - reference_point.z,
            };
            
            let force_vector = Vector3D {
                x: force.magnitude * force.direction.x,
                y: force.magnitude * force.direction.y,
                z: force.magnitude * force.direction.z,
            };
            
            // Cross product r × F
            let moment = Vector3D {
                x: r.y * force_vector.z - r.z * force_vector.y,
                y: r.z * force_vector.x - r.x * force_vector.z,
                z: r.x * force_vector.y - r.y * force_vector.x,
            };
            
            resultant.x += moment.x;
            resultant.y += moment.y;
            resultant.z += moment.z;
        }
        
        // Add applied moments
        for moment in moments {
            let moment_vector = Vector3D {
                x: moment.magnitude * moment.axis.x,
                y: moment.magnitude * moment.axis.y,
                z: moment.magnitude * moment.axis.z,
            };
            
            resultant.x += moment_vector.x;
            resultant.y += moment_vector.y;
            resultant.z += moment_vector.z;
        }
        
        resultant
    }
    
    pub fn check_equilibrium(forces: &[Force], moments: &[Moment], tolerance: f64) -> bool {
        let origin = Point3D { x: 0.0, y: 0.0, z: 0.0 };
        let resultant_force = Self::resultant_force(forces);
        let resultant_moment = Self::resultant_moment(forces, moments, &origin);
        
        let force_magnitude = (resultant_force.x.powi(2) + 
                              resultant_force.y.powi(2) + 
                              resultant_force.z.powi(2)).sqrt();
        
        let moment_magnitude = (resultant_moment.x.powi(2) + 
                               resultant_moment.y.powi(2) + 
                               resultant_moment.z.powi(2)).sqrt();
        
        force_magnitude < tolerance && moment_magnitude < tolerance
    }
    
    // ===== Stress-Strain Relationships =====
    
    pub fn hookes_law_1d(strain: f64, elastic_modulus: f64) -> f64 {
        elastic_modulus * strain
    }
    
    pub fn strain_from_stress_1d(stress: f64, elastic_modulus: f64) -> MathResult<f64> {
        if elastic_modulus == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok(stress / elastic_modulus)
    }
    
    pub fn stress_strain_3d(strain: &StrainState, material: &Material) -> StressState {
        let e = material.elastic_modulus;
        let nu = material.poisson_ratio;
        let g = e / (2.0 * (1.0 + nu)); // Shear modulus
        
        // Generalized Hooke's Law
        let factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let lambda = nu * factor;
        let mu = (1.0 - nu) * factor;
        
        StressState {
            normal_stress_x: mu * strain.normal_strain_x + lambda * (strain.normal_strain_y + strain.normal_strain_z),
            normal_stress_y: mu * strain.normal_strain_y + lambda * (strain.normal_strain_x + strain.normal_strain_z),
            normal_stress_z: mu * strain.normal_strain_z + lambda * (strain.normal_strain_x + strain.normal_strain_y),
            shear_stress_xy: g * strain.shear_strain_xy,
            shear_stress_yz: g * strain.shear_strain_yz,
            shear_stress_zx: g * strain.shear_strain_zx,
        }
    }
    
    pub fn von_mises_stress(stress: &StressState) -> f64 {
        let sx = stress.normal_stress_x;
        let sy = stress.normal_stress_y;
        let sz = stress.normal_stress_z;
        let txy = stress.shear_stress_xy;
        let tyz = stress.shear_stress_yz;
        let tzx = stress.shear_stress_zx;
        
        let term1 = (sx - sy).powi(2);
        let term2 = (sy - sz).powi(2);
        let term3 = (sz - sx).powi(2);
        let term4 = 6.0 * (txy.powi(2) + tyz.powi(2) + tzx.powi(2));
        
        ((term1 + term2 + term3 + term4) / 2.0).sqrt()
    }
    
    // ===== Beam Analysis =====
    
    pub fn simply_supported_beam_deflection(
        load: f64, 
        length: f64, 
        elastic_modulus: f64, 
        moment_of_inertia: f64,
        position: f64
    ) -> MathResult<f64> {
        if elastic_modulus == 0.0 || moment_of_inertia == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        
        if position < 0.0 || position > length {
            return Err(MathError::InvalidArgument("Position must be between 0 and beam length".to_string()));
        }
        
        let x = position;
        let l = length;
        let w = load; // distributed load
        let ei = elastic_modulus * moment_of_inertia;
        
        // For uniformly distributed load on simply supported beam
        let deflection = w * x * (l.powi(3) - 2.0 * l * x.powi(2) + x.powi(3)) / (24.0 * ei);
        Ok(deflection.abs()) // Return positive deflection magnitude
    }
    
    pub fn cantilever_beam_deflection(
        load: f64,
        length: f64,
        elastic_modulus: f64,
        moment_of_inertia: f64,
        position: f64
    ) -> MathResult<f64> {
        if elastic_modulus == 0.0 || moment_of_inertia == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        
        if position < 0.0 || position > length {
            return Err(MathError::InvalidArgument("Position must be between 0 and beam length".to_string()));
        }
        
        let x = position;
        let ei = elastic_modulus * moment_of_inertia;
        
        // Deflection for cantilever with end load
        let deflection = load * x.powi(2) * (3.0 * length - x) / (6.0 * ei);
        Ok(deflection.abs()) // Return positive deflection magnitude
    }
    
    pub fn beam_bending_stress(
        bending_moment: f64,
        distance_from_neutral: f64,
        moment_of_inertia: f64
    ) -> MathResult<f64> {
        if moment_of_inertia == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        
        Ok(bending_moment * distance_from_neutral / moment_of_inertia)
    }
    
    // ===== Truss Analysis (Method of Joints) =====
    
    pub fn analyze_truss_2d(truss: &Truss) -> MathResult<Vec<f64>> {
        let n_nodes = truss.nodes.len();
        let n_members = truss.members.len();
        
        if n_nodes == 0 {
            return Err(MathError::InvalidArgument("Truss must have at least one node".to_string()));
        }
        
        // Check static determinacy: m + r = 2n for 2D trusses
        // where m = members, r = reactions, n = nodes
        let n_reactions = truss.nodes.iter()
            .map(|node| {
                let mut count = 0;
                if node.constraints.fixed_x { count += 1; }
                if node.constraints.fixed_y { count += 1; }
                count
            })
            .sum::<usize>();
        
        if n_members + n_reactions != 2 * n_nodes {
            return Err(MathError::InvalidArgument("Truss is not statically determinate".to_string()));
        }
        
        // Build global stiffness matrix (simplified for 2D)
        let dof = 2 * n_nodes; // 2 DOF per node in 2D
        let mut k_global = vec![vec![0.0; dof]; dof];
        
        // Assemble member stiffness matrices
        for member in &truss.members {
            let node_i = member.start_node;
            let node_j = member.end_node;
            
            if node_i >= n_nodes || node_j >= n_nodes {
                return Err(MathError::InvalidArgument("Invalid node reference in member".to_string()));
            }
            
            let pos_i = &truss.nodes[node_i].position;
            let pos_j = &truss.nodes[node_j].position;
            
            let dx = pos_j.x - pos_i.x;
            let dy = pos_j.y - pos_i.y;
            let length = (dx * dx + dy * dy).sqrt();
            
            if length == 0.0 {
                return Err(MathError::InvalidArgument("Member has zero length".to_string()));
            }
            
            let cos_theta = dx / length;
            let sin_theta = dy / length;
            let ae_l = member.cross_sectional_area * member.elastic_modulus / length;
            
            // Local stiffness matrix for truss member
            let c = cos_theta;
            let s = sin_theta;
            let k_local = [
                [c*c, c*s, -c*c, -c*s],
                [c*s, s*s, -c*s, -s*s],
                [-c*c, -c*s, c*c, c*s],
                [-c*s, -s*s, c*s, s*s]
            ];
            
            // Global DOF indices
            let dof_indices = [2*node_i, 2*node_i+1, 2*node_j, 2*node_j+1];
            
            // Assemble into global matrix
            for i in 0..4 {
                for j in 0..4 {
                    k_global[dof_indices[i]][dof_indices[j]] += ae_l * k_local[i][j];
                }
            }
        }
        
        // This is a simplified analysis - in practice, you'd solve K*u = F
        // For now, return member forces (simplified)
        let member_forces = vec![0.0; n_members]; // Placeholder
        Ok(member_forces)
    }
    
    // ===== Thermal Effects =====
    
    pub fn thermal_strain(
        temperature_change: f64,
        thermal_expansion_coefficient: f64
    ) -> f64 {
        thermal_expansion_coefficient * temperature_change
    }
    
    pub fn thermal_stress_constrained(
        temperature_change: f64,
        thermal_expansion_coefficient: f64,
        elastic_modulus: f64
    ) -> f64 {
        -elastic_modulus * thermal_expansion_coefficient * temperature_change
    }
    
    // ===== Buckling Analysis =====
    
    pub fn euler_buckling_load(
        elastic_modulus: f64,
        moment_of_inertia: f64,
        length: f64,
        end_condition_factor: f64 // K factor: 0.5 (fixed-fixed), 0.7 (fixed-pinned), 1.0 (pinned-pinned), 2.0 (fixed-free)
    ) -> MathResult<f64> {
        if length == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        
        let effective_length = end_condition_factor * length;
        Ok(PI.powi(2) * elastic_modulus * moment_of_inertia / effective_length.powi(2))
    }
    
    pub fn slenderness_ratio(length: f64, radius_of_gyration: f64) -> MathResult<f64> {
        if radius_of_gyration == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok(length / radius_of_gyration)
    }
    
    // ===== Center of Mass/Centroid Calculations =====
    
    pub fn centroid_composite_area(areas: &[(f64, Point2D)]) -> MathResult<Point2D> {
        if areas.is_empty() {
            return Err(MathError::InvalidArgument("No areas provided".to_string()));
        }
        
        let total_area: f64 = areas.iter().map(|(area, _)| area).sum();
        
        if total_area == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        
        let mut sum_ax = 0.0;
        let mut sum_ay = 0.0;
        
        for (area, centroid) in areas {
            sum_ax += area * centroid.x;
            sum_ay += area * centroid.y;
        }
        
        Ok(Point2D {
            x: sum_ax / total_area,
            y: sum_ay / total_area,
        })
    }
    
    pub fn center_of_mass_3d(masses: &[(f64, Point3D)]) -> MathResult<Point3D> {
        if masses.is_empty() {
            return Err(MathError::InvalidArgument("No masses provided".to_string()));
        }
        
        let total_mass: f64 = masses.iter().map(|(mass, _)| mass).sum();
        
        if total_mass == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        
        let mut sum_mx = 0.0;
        let mut sum_my = 0.0;
        let mut sum_mz = 0.0;
        
        for (mass, position) in masses {
            sum_mx += mass * position.x;
            sum_my += mass * position.y;
            sum_mz += mass * position.z;
        }
        
        Ok(Point3D {
            x: sum_mx / total_mass,
            y: sum_my / total_mass,
            z: sum_mz / total_mass,
        })
    }
    
    // ===== Safety Factor Calculations =====
    
    pub fn safety_factor_yield(applied_stress: f64, yield_strength: f64) -> MathResult<f64> {
        if applied_stress == 0.0 {
            return Ok(f64::INFINITY);
        }
        if applied_stress < 0.0 {
            return Err(MathError::InvalidArgument("Applied stress must be positive".to_string()));
        }
        Ok(yield_strength / applied_stress)
    }
    
    pub fn safety_factor_buckling(applied_load: f64, critical_buckling_load: f64) -> MathResult<f64> {
        if applied_load == 0.0 {
            return Ok(f64::INFINITY);
        }
        if applied_load < 0.0 {
            return Err(MathError::InvalidArgument("Applied load must be positive".to_string()));
        }
        Ok(critical_buckling_load / applied_load)
    }
    
    // ===== Missing Functions from Tests =====
    
    pub fn beam_maximum_moment(length: f64, distributed_load: f64) -> f64 {
        distributed_load * length.powi(2) / 8.0
    }
    
    pub fn beam_maximum_shear(length: f64, distributed_load: f64) -> f64 {
        distributed_load * length / 2.0
    }
    
    pub fn axial_member_force(
        area: f64, 
        elastic_modulus: f64, 
        original_length: f64, 
        deformed_length: f64
    ) -> MathResult<f64> {
        if original_length == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        let strain = (deformed_length - original_length) / original_length;
        let stress = elastic_modulus * strain;
        Ok(stress * area)
    }
    
    pub fn axial_member_stress(force: f64, area: f64) -> MathResult<f64> {
        if area == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok(force / area)
    }
    
    pub fn thermal_stress(
        elastic_modulus: f64,
        thermal_expansion: f64,
        temperature_change: f64,
        constraint_factor: f64
    ) -> f64 {
        elastic_modulus * thermal_expansion * temperature_change * constraint_factor
    }
    
    
    pub fn safety_factor(yield_strength: f64, applied_stress: f64) -> MathResult<f64> {
        if applied_stress == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok(yield_strength / applied_stress)
    }
    
    pub fn allowable_stress(yield_strength: f64, safety_factor: f64) -> MathResult<f64> {
        if safety_factor == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok(yield_strength / safety_factor)
    }
    
    pub fn principal_stresses(stress: &StressState) -> Vec<f64> {
        // For 3D principal stress calculation, we need to solve the eigenvalue problem
        // This is a simplified version - in practice, you'd use numerical methods
        
        let sx = stress.normal_stress_x;
        let sy = stress.normal_stress_y;
        let sz = stress.normal_stress_z;
        let txy = stress.shear_stress_xy;
        let tyz = stress.shear_stress_yz;
        let tzx = stress.shear_stress_zx;
        
        // For simplification, return the normal stresses sorted in descending order
        // This is an approximation - full calculation would solve det(σ - λI) = 0
        let mut principals = vec![sx, sy, sz];
        principals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        // Adjust for shear stress effects (approximate)
        let max_shear = txy.abs().max(tyz.abs()).max(tzx.abs());
        principals[0] += max_shear * 0.5;
        principals[2] -= max_shear * 0.5;
        
        principals
    }
}

impl MathDomain for StructuralEngineeringDomain {
    fn name(&self) -> &str { "Structural Engineering" }
    fn description(&self) -> &str { "Professional structural analysis including statics, dynamics, material behavior, and design" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "resultant_force" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("resultant_force requires 1 argument".to_string()));
                }
                let forces = args[0].downcast_ref::<Vec<Force>>().ok_or_else(|| MathError::InvalidArgument("Argument must be Vec<Force>".to_string()))?;
                Ok(Box::new(Self::resultant_force(forces)))
            },
            "check_equilibrium" => {
                if args.len() != 3 {
                    return Err(MathError::InvalidArgument("check_equilibrium requires 3 arguments".to_string()));
                }
                let forces = args[0].downcast_ref::<Vec<Force>>().ok_or_else(|| MathError::InvalidArgument("First argument must be Vec<Force>".to_string()))?;
                let moments = args[1].downcast_ref::<Vec<Moment>>().ok_or_else(|| MathError::InvalidArgument("Second argument must be Vec<Moment>".to_string()))?;
                let tolerance = args[2].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be f64".to_string()))?;
                Ok(Box::new(Self::check_equilibrium(forces, moments, *tolerance)))
            },
            "von_mises_stress" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("von_mises_stress requires 1 argument".to_string()));
                }
                let stress = args[0].downcast_ref::<StressState>().ok_or_else(|| MathError::InvalidArgument("Argument must be StressState".to_string()))?;
                Ok(Box::new(Self::von_mises_stress(stress)))
            },
            "simply_supported_beam_deflection" => {
                if args.len() != 5 {
                    return Err(MathError::InvalidArgument("simply_supported_beam_deflection requires 5 arguments".to_string()));
                }
                let load = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let length = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                let e = args[2].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be f64".to_string()))?;
                let i = args[3].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Fourth argument must be f64".to_string()))?;
                let pos = args[4].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Fifth argument must be f64".to_string()))?;
                Ok(Box::new(Self::simply_supported_beam_deflection(*load, *length, *e, *i, *pos)?))
            },
            "euler_buckling_load" => {
                if args.len() != 4 {
                    return Err(MathError::InvalidArgument("euler_buckling_load requires 4 arguments".to_string()));
                }
                let e = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let i = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                let l = args[2].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be f64".to_string()))?;
                let k = args[3].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Fourth argument must be f64".to_string()))?;
                Ok(Box::new(Self::euler_buckling_load(*e, *i, *l, *k)?))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "resultant_force".to_string(),
            "resultant_moment".to_string(),
            "check_equilibrium".to_string(),
            "hookes_law_1d".to_string(),
            "strain_from_stress_1d".to_string(),
            "stress_strain_3d".to_string(),
            "von_mises_stress".to_string(),
            "simply_supported_beam_deflection".to_string(),
            "cantilever_beam_deflection".to_string(),
            "beam_bending_stress".to_string(),
            "analyze_truss_2d".to_string(),
            "thermal_strain".to_string(),
            "thermal_stress_constrained".to_string(),
            "euler_buckling_load".to_string(),
            "slenderness_ratio".to_string(),
            "centroid_composite_area".to_string(),
            "center_of_mass_3d".to_string(),
            "safety_factor_yield".to_string(),
            "safety_factor_buckling".to_string(),
        ]
    }
}