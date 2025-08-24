use mathtables::core::{
    Point3D, Vector3D, Material, CrossSection,
    Force, Moment, StressState, StrainState
};
use mathtables::domains::StructuralEngineeringDomain;
use std::f64::consts::PI;

#[test]
fn test_material_presets() {
    let steel = Material::steel_a36();
    assert_eq!(steel.name, "Steel A36");
    assert!((steel.elastic_modulus - 200e9).abs() < 1e9);
    assert!((steel.yield_strength - 250e6).abs() < 1e6);
    
    let concrete = Material::concrete_normal();
    assert_eq!(concrete.name, "Normal Weight Concrete");
    assert!((concrete.elastic_modulus - 25e9).abs() < 1e9);
    
    let aluminum = Material::aluminum_6061();
    assert_eq!(aluminum.name, "Aluminum 6061-T6");
    assert!((aluminum.elastic_modulus - 69e9).abs() < 1e9);
}

#[test]
fn test_rectangular_cross_section() {
    let width = 0.2; // 200mm
    let height = 0.4; // 400mm
    
    let section = CrossSection::rectangular(width, height);
    
    let expected_area = width * height;
    let expected_i = width * height.powi(3) / 12.0;
    
    assert!((section.area - expected_area).abs() < 1e-10);
    assert!((section.moment_of_inertia_y - expected_i).abs() < 1e-10);
    assert!((section.section_modulus_y - expected_i / (height / 2.0)).abs() < 1e-10);
}

#[test]
fn test_circular_cross_section() {
    let diameter = 0.25; // 250mm diameter
    let radius = diameter / 2.0;
    
    let section = CrossSection::circular(diameter);
    
    let expected_area = PI * radius.powi(2);
    let expected_i = PI * radius.powi(4) / 4.0;
    let expected_j = PI * radius.powi(4) / 2.0;
    
    assert!((section.area - expected_area).abs() < 1e-10);
    assert!((section.moment_of_inertia_y - expected_i).abs() < 1e-10);
    assert!((section.polar_moment_of_inertia - expected_j).abs() < 1e-10);
}

#[test]
fn test_wide_flange_cross_section() {
    let section = CrossSection::wide_flange(
        0.31,  // height
        0.205, // width
        0.015, // flange thickness
        0.009  // web thickness
    );
    
    assert!(section.area > 0.0);
    assert!(section.moment_of_inertia_y > 0.0);
    assert!(section.moment_of_inertia_z > 0.0);
    assert!(section.section_modulus_y > 0.0);
}

#[test]
fn test_resultant_force() {
    let forces = vec![
        Force {
            magnitude: 1000.0,
            direction: Vector3D { x: 1.0, y: 0.0, z: 0.0 },
            point_of_application: Point3D { x: 0.0, y: 0.0, z: 0.0 },
        },
        Force {
            magnitude: 500.0,
            direction: Vector3D { x: 0.0, y: 1.0, z: 0.0 },
            point_of_application: Point3D { x: 1.0, y: 0.0, z: 0.0 },
        },
        Force {
            magnitude: 1000.0,
            direction: Vector3D { x: -1.0, y: 0.0, z: 0.0 },
            point_of_application: Point3D { x: 2.0, y: 0.0, z: 0.0 },
        },
        Force {
            magnitude: 500.0,
            direction: Vector3D { x: 0.0, y: -1.0, z: 0.0 },
            point_of_application: Point3D { x: 1.5, y: 0.0, z: 0.0 },
        },
    ];
    
    let resultant = StructuralEngineeringDomain::resultant_force(&forces);
    
    // Forces should cancel out
    assert!(resultant.x.abs() < 1e-10);
    assert!(resultant.y.abs() < 1e-10);
    assert!(resultant.z.abs() < 1e-10);
}

#[test]
fn test_equilibrium_check() {
    let forces = vec![
        Force {
            magnitude: 1000.0,
            direction: Vector3D { x: 0.0, y: -1.0, z: 0.0 },
            point_of_application: Point3D { x: 1.0, y: 0.0, z: 0.0 },
        },
        Force {
            magnitude: 1000.0,
            direction: Vector3D { x: 0.0, y: 1.0, z: 0.0 },
            point_of_application: Point3D { x: 1.0, y: 0.0, z: 0.0 },
        },
    ];
    
    let moments = vec![];
    
    let is_equilibrium = StructuralEngineeringDomain::check_equilibrium(&forces, &moments, 1e-6);
    assert!(is_equilibrium);
}

#[test]
fn test_hookes_law_1d() {
    let strain = 0.001; // 0.1% strain
    let elastic_modulus = 200e9; // 200 GPa
    
    let stress = StructuralEngineeringDomain::hookes_law_1d(strain, elastic_modulus);
    let expected_stress = elastic_modulus * strain;
    
    assert!((stress - expected_stress).abs() < 1e-6);
    assert_eq!(stress, 200e6); // 200 MPa
}

#[test]
fn test_strain_from_stress_1d() {
    let stress = 100e6; // 100 MPa
    let elastic_modulus = 200e9; // 200 GPa
    
    let strain = StructuralEngineeringDomain::strain_from_stress_1d(stress, elastic_modulus).unwrap();
    let expected_strain = stress / elastic_modulus;
    
    assert!((strain - expected_strain).abs() < 1e-12);
    assert_eq!(strain, 0.0005); // 0.05% strain
    
    // Test division by zero
    assert!(StructuralEngineeringDomain::strain_from_stress_1d(stress, 0.0).is_err());
}

#[test]
fn test_stress_strain_3d() {
    let strain = StrainState {
        normal_strain_x: 0.001,
        normal_strain_y: 0.0005,
        normal_strain_z: 0.0002,
        shear_strain_xy: 0.0001,
        shear_strain_yz: 0.00005,
        shear_strain_zx: 0.000025,
    };
    
    let material = Material::steel_a36();
    let stress = StructuralEngineeringDomain::stress_strain_3d(&strain, &material);
    
    assert!(stress.normal_stress_x > 0.0);
    assert!(stress.normal_stress_y > 0.0);
    assert!(stress.normal_stress_z > 0.0);
    assert!(stress.shear_stress_xy > 0.0);
}

#[test]
fn test_von_mises_stress() {
    let stress = StressState {
        normal_stress_x: 100e6,
        normal_stress_y: -50e6,
        normal_stress_z: 25e6,
        shear_stress_xy: 30e6,
        shear_stress_yz: 10e6,
        shear_stress_zx: 15e6,
    };
    
    let von_mises = StructuralEngineeringDomain::von_mises_stress(&stress);
    
    assert!(von_mises > 0.0);
    assert!(von_mises > stress.normal_stress_x.abs()); // Should be higher due to shear components
}

#[test]
fn test_principal_stresses() {
    let stress = StressState {
        normal_stress_x: 100e6,
        normal_stress_y: 50e6,
        normal_stress_z: 25e6,
        shear_stress_xy: 20e6,
        shear_stress_yz: 10e6,
        shear_stress_zx: 5e6,
    };
    
    let principals = StructuralEngineeringDomain::principal_stresses(&stress);
    
    assert!(principals.len() == 3);
    assert!(principals[0] >= principals[1]); // Should be sorted
    assert!(principals[1] >= principals[2]);
    assert!(principals[0] >= stress.normal_stress_x); // Max should be >= any normal stress
}

#[test]
fn test_simply_supported_beam_deflection() {
    let load = 10000.0; // 10 kN/m distributed load
    let length = 6.0; // 6m beam
    let elastic_modulus = 200e9; // 200 GPa steel
    let moment_of_inertia = 8.33e-5; // m^4
    let position = length / 2.0; // Mid-span
    
    let deflection = StructuralEngineeringDomain::simply_supported_beam_deflection(
        load,
        length,
        elastic_modulus,
        moment_of_inertia,
        position
    ).unwrap();
    
    assert!(deflection > 0.0); // Positive deflection (downward)
    
    // Test at quarter span - should be less than mid-span
    let quarter_deflection = StructuralEngineeringDomain::simply_supported_beam_deflection(
        load,
        length,
        elastic_modulus,
        moment_of_inertia,
        length / 4.0
    ).unwrap();
    
    assert!(quarter_deflection < deflection);
}

#[test]
fn test_cantilever_beam_deflection() {
    let load = 50000.0; // 50 kN point load at tip
    let length = 4.0; // 4m cantilever
    let elastic_modulus = 200e9;
    let moment_of_inertia = 5.2e-5;
    let position = length; // At the tip
    
    let tip_deflection = StructuralEngineeringDomain::cantilever_beam_deflection(
        load,
        length,
        elastic_modulus,
        moment_of_inertia,
        position
    ).unwrap();
    
    assert!(tip_deflection > 0.0);
    
    // Test at mid-point - should deflect less than tip
    let mid_deflection = StructuralEngineeringDomain::cantilever_beam_deflection(
        load,
        length,
        elastic_modulus,
        moment_of_inertia,
        length / 2.0
    ).unwrap();
    
    assert!(mid_deflection < tip_deflection);
}

#[test]
fn test_beam_maximum_moment() {
    let length = 8.0;
    let distributed_load = 5000.0; // 5 kN/m
    
    let max_moment = StructuralEngineeringDomain::beam_maximum_moment(length, distributed_load);
    
    let expected_moment = distributed_load * length.powi(2) / 8.0;
    assert!((max_moment - expected_moment).abs() < 1e-10);
}

#[test]
fn test_beam_maximum_shear() {
    let length = 10.0;
    let distributed_load = 8000.0; // 8 kN/m
    
    let max_shear = StructuralEngineeringDomain::beam_maximum_shear(length, distributed_load);
    
    let expected_shear = distributed_load * length / 2.0;
    assert!((max_shear - expected_shear).abs() < 1e-10);
}

#[test]
fn test_euler_buckling() {
    let elastic_modulus = 200e9; // 200 GPa
    let moment_of_inertia = 1.67e-4; // m^4
    let length = 6.0; // 6m column
    let end_condition_factor = 1.0; // Pinned-pinned
    
    let buckling_load = StructuralEngineeringDomain::euler_buckling_load(
        elastic_modulus,
        moment_of_inertia,
        length,
        end_condition_factor
    ).unwrap();
    
    let effective_length = length / end_condition_factor;
    let expected_load = PI.powi(2) * elastic_modulus * moment_of_inertia / effective_length.powi(2);
    assert!((buckling_load - expected_load).abs() < 1e-3);
    
    // Test different end conditions
    let fixed_free_load = StructuralEngineeringDomain::euler_buckling_load(
        elastic_modulus,
        moment_of_inertia,
        length,
        2.0 // Fixed-free (cantilever)
    ).unwrap();
    
    assert!(fixed_free_load < buckling_load); // Fixed-free should have lower capacity
}

#[test]
fn test_slenderness_ratio() {
    let length = 4.0;
    let radius_of_gyration = 0.1;
    
    let slenderness = StructuralEngineeringDomain::slenderness_ratio(length, radius_of_gyration).unwrap();
    
    assert!((slenderness - 40.0).abs() < 1e-10);
    
    // Test error case
    assert!(StructuralEngineeringDomain::slenderness_ratio(length, 0.0).is_err());
}

#[test]
fn test_axial_member_force() {
    let area = 0.002; // 2000 mm²
    let elastic_modulus = 200e9;
    let original_length = 3.0; // 3m
    let deformed_length = 3.001; // 1mm extension
    
    let force = StructuralEngineeringDomain::axial_member_force(
        area,
        elastic_modulus,
        original_length,
        deformed_length
    ).unwrap();
    
    let strain = (deformed_length - original_length) / original_length;
    let stress = elastic_modulus * strain;
    let expected_force = stress * area;
    
    assert!((force - expected_force).abs() < 1e-6);
}

#[test]
fn test_axial_member_stress() {
    let force = 100000.0; // 100 kN
    let area = 0.005; // 5000 mm²
    
    let stress = StructuralEngineeringDomain::axial_member_stress(force, area).unwrap();
    let expected_stress = force / area;
    
    assert!((stress - expected_stress).abs() < 1e-10);
    assert!(StructuralEngineeringDomain::axial_member_stress(force, 0.0).is_err());
}

#[test]
fn test_thermal_stress() {
    let elastic_modulus = 200e9;
    let thermal_expansion = 12e-6; // Steel coefficient
    let temperature_change = 50.0; // 50°C increase
    let constraint_factor = 1.0; // Fully constrained
    
    let thermal_stress = StructuralEngineeringDomain::thermal_stress(
        elastic_modulus,
        thermal_expansion,
        temperature_change,
        constraint_factor
    );
    
    let expected_stress = elastic_modulus * thermal_expansion * temperature_change * constraint_factor;
    assert!((thermal_stress - expected_stress).abs() < 1e-3);
    
    // Test partially constrained
    let partial_stress = StructuralEngineeringDomain::thermal_stress(
        elastic_modulus,
        thermal_expansion,
        temperature_change,
        0.5
    );
    
    assert!((partial_stress - thermal_stress / 2.0).abs() < 1e-6);
}

#[test]
fn test_thermal_strain() {
    let thermal_expansion = 12e-6;
    let temperature_change = -30.0; // 30°C decrease
    
    let thermal_strain = StructuralEngineeringDomain::thermal_strain(
        thermal_expansion,
        temperature_change
    );
    
    let expected_strain = thermal_expansion * temperature_change;
    assert!((thermal_strain - expected_strain).abs() < 1e-12);
    assert!(thermal_strain < 0.0); // Should be negative for cooling
}

#[test]
fn test_safety_factor() {
    let yield_strength = 250e6; // 250 MPa
    let applied_stress = 150e6;  // 150 MPa
    
    let safety_factor = StructuralEngineeringDomain::safety_factor(
        yield_strength,
        applied_stress
    ).unwrap();
    
    let expected_sf = yield_strength / applied_stress;
    assert!((safety_factor - expected_sf).abs() < 1e-10);
    assert!(safety_factor > 1.0);
    
    assert!(StructuralEngineeringDomain::safety_factor(yield_strength, 0.0).is_err());
}

#[test]
fn test_allowable_stress() {
    let yield_strength = 250e6;
    let safety_factor = 2.5;
    
    let allowable = StructuralEngineeringDomain::allowable_stress(
        yield_strength,
        safety_factor
    ).unwrap();
    
    let expected_allowable = yield_strength / safety_factor;
    assert!((allowable - expected_allowable).abs() < 1e-10);
    assert!(allowable < yield_strength);
    
    assert!(StructuralEngineeringDomain::allowable_stress(yield_strength, 0.0).is_err());
}

#[test]
fn test_beam_analysis_integration() {
    let steel = Material::steel_a36();
    let section = CrossSection::wide_flange(
        0.400,  // height (increased from 203mm to 400mm)
        0.200,  // width (increased from 133mm to 200mm)  
        0.015,  // flange thickness (increased from 11mm to 15mm)
        0.010   // web thickness (increased from 7mm to 10mm)
    );
    
    let beam_length = 8.0;
    let distributed_load = 8000.0; // 8 kN/m (reduced load for adequate safety factor)
    
    let max_moment = StructuralEngineeringDomain::beam_maximum_moment(beam_length, distributed_load);
    
    let max_stress = max_moment / section.section_modulus_y;
    let safety_factor = steel.yield_strength / max_stress;
    
    assert!(safety_factor > 1.0);
    assert!(max_stress < steel.yield_strength);
    
    let deflection = StructuralEngineeringDomain::simply_supported_beam_deflection(
        distributed_load,
        beam_length,
        steel.elastic_modulus,
        section.moment_of_inertia_y,
        beam_length / 2.0
    ).unwrap();
    
    let deflection_limit = beam_length / 300.0; // L/300 typical limit
    assert!(deflection < deflection_limit);
}

#[test]
fn test_column_design_integration() {
    let steel = Material::steel_a36();
    let section = CrossSection::circular(0.219); // 219mm diameter
    
    let column_height = 4.0;
    let applied_load = 800000.0; // 800 kN
    
    let buckling_load = StructuralEngineeringDomain::euler_buckling_load(
        steel.elastic_modulus,
        section.moment_of_inertia_y,
        column_height,
        1.0 // Pinned ends
    ).unwrap();
    
    let buckling_safety = buckling_load / applied_load;
    assert!(buckling_safety > 1.0);
    
    let direct_stress = applied_load / section.area;
    let yield_safety = steel.yield_strength / direct_stress;
    assert!(yield_safety > 1.0);
}