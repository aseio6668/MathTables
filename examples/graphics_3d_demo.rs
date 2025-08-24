use mathtables::core::{Point3D, Vector3D, Quaternion, Transform4x4, Ray3D, AABB, BoundingSphere};
use mathtables::domains::Graphics3DDomain;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎮 3D Graphics and Animation Demo");
    println!("=================================\n");

    // ===== Quaternion Rotation Demo =====
    println!("🔄 Quaternion Rotations:");
    
    // Create a quaternion for 90-degree rotation around Z-axis
    let z_axis = Vector3D { x: 0.0, y: 0.0, z: 1.0 };
    let rotation_90 = Quaternion::from_axis_angle(&z_axis, PI / 2.0);
    
    // Rotate a point
    let original_point = Point3D { x: 1.0, y: 0.0, z: 0.0 };
    let rotated_point = rotation_90.rotate_point(&original_point);
    
    println!("  Original point: ({:.2}, {:.2}, {:.2})", 
             original_point.x, original_point.y, original_point.z);
    println!("  After 90° Z rotation: ({:.2}, {:.2}, {:.2})", 
             rotated_point.x, rotated_point.y, rotated_point.z);
    
    // ===== SLERP Animation Demo =====
    println!("\n🎬 SLERP Animation (Quaternion Interpolation):");
    
    let start_rotation = Quaternion::identity();  // No rotation
    let end_rotation = Quaternion::from_axis_angle(&z_axis, PI);  // 180-degree rotation
    
    println!("  Animating from 0° to 180° rotation:");
    for i in 0..=5 {
        let t = i as f64 / 5.0;
        let interpolated = start_rotation.slerp(&end_rotation, t)?;
        let animated_point = interpolated.rotate_point(&original_point);
        
        println!("    t={:.1}: ({:.2}, {:.2}, {:.2})", 
                 t, animated_point.x, animated_point.y, animated_point.z);
    }
    
    // ===== 4x4 Transformation Matrices =====
    println!("\n📐 4x4 Transformation Matrices:");
    
    // Create a complex transformation: translate, then scale, then rotate
    let translation = Transform4x4::translation(2.0, 3.0, 1.0);
    let scaling = Transform4x4::scaling(2.0, 1.5, 0.8);
    let rotation = Transform4x4::from_quaternion(&rotation_90);
    
    // Combine transformations (note: order matters!)
    let combined = translation.multiply(&scaling).multiply(&rotation);
    
    let test_point = Point3D { x: 1.0, y: 1.0, z: 1.0 };
    let transformed = combined.transform_point(&test_point);
    
    println!("  Original point: ({:.2}, {:.2}, {:.2})", 
             test_point.x, test_point.y, test_point.z);
    println!("  After combined transform: ({:.2}, {:.2}, {:.2})", 
             transformed.x, transformed.y, transformed.z);
    
    // ===== Perspective Projection =====
    println!("\n🎭 Perspective Projection:");
    
    let fov = PI / 4.0;  // 45 degrees
    let aspect = 16.0 / 9.0;
    let near = 0.1;
    let far = 100.0;
    
    let perspective = Transform4x4::perspective(fov, aspect, near, far)?;
    
    let world_point = Point3D { x: 1.0, y: 1.0, z: -5.0 };  // Behind camera
    let projected = perspective.transform_point(&world_point);
    
    println!("  3D world point: ({:.2}, {:.2}, {:.2})", 
             world_point.x, world_point.y, world_point.z);
    println!("  Projected to screen: ({:.2}, {:.2}, {:.2})", 
             projected.x, projected.y, projected.z);
    
    // ===== Raycasting Demo =====
    println!("\n🎯 Raycasting:");
    
    // Create a ray from the camera towards the scene
    let camera_pos = Point3D { x: 0.0, y: 0.0, z: 0.0 };
    let ray_direction = Vector3D { x: 0.0, y: 0.0, z: -1.0 };  // Looking down -Z
    let ray = Ray3D::new(camera_pos, ray_direction);
    
    // Test intersection with a sphere
    let sphere_center = Point3D { x: 0.0, y: 0.0, z: -5.0 };
    let sphere = BoundingSphere::new(sphere_center, 2.0);
    
    if let Some((t1, t2)) = ray.intersect_sphere(&sphere) {
        let hit_point_near = ray.point_at_parameter(t1);
        let hit_point_far = ray.point_at_parameter(t2);
        
        println!("  Ray hits sphere!");
        println!("    Near intersection at t={:.2}: ({:.2}, {:.2}, {:.2})", 
                 t1, hit_point_near.x, hit_point_near.y, hit_point_near.z);
        println!("    Far intersection at t={:.2}: ({:.2}, {:.2}, {:.2})", 
                 t2, hit_point_far.x, hit_point_far.y, hit_point_far.z);
    }
    
    // ===== Bounding Volume Collision Detection =====
    println!("\n💥 Collision Detection:");
    
    // Create two AABBs
    let box1 = AABB::new(
        Point3D { x: 0.0, y: 0.0, z: 0.0 },
        Point3D { x: 2.0, y: 2.0, z: 2.0 }
    );
    
    let box2 = AABB::new(
        Point3D { x: 1.0, y: 1.0, z: 1.0 },
        Point3D { x: 3.0, y: 3.0, z: 3.0 }
    );
    
    let box3 = AABB::new(
        Point3D { x: 5.0, y: 5.0, z: 5.0 },
        Point3D { x: 7.0, y: 7.0, z: 7.0 }
    );
    
    println!("  Box1 intersects Box2: {}", box1.intersects_aabb(&box2));
    println!("  Box1 intersects Box3: {}", box1.intersects_aabb(&box3));
    
    // ===== Linear Interpolation Demo =====
    println!("\n📈 Linear Interpolation:");
    
    let start_pos = Point3D { x: 0.0, y: 0.0, z: 0.0 };
    let end_pos = Point3D { x: 10.0, y: 5.0, z: -3.0 };
    
    println!("  Animating position from start to end:");
    for i in 0..=4 {
        let t = i as f64 / 4.0;
        let lerped_pos = Graphics3DDomain::lerp_point3d(&start_pos, &end_pos, t)?;
        
        println!("    t={:.2}: ({:.1}, {:.1}, {:.1})", 
                 t, lerped_pos.x, lerped_pos.y, lerped_pos.z);
    }
    
    // ===== Practical Example: Animated Camera =====
    println!("\n🎥 Animated Camera System:");
    
    // Define camera positions
    let camera_start = Point3D { x: 10.0, y: 5.0, z: 10.0 };
    let camera_end = Point3D { x: -5.0, y: 8.0, z: -2.0 };
    
    // Define rotations (looking at origin)
    let look_start = Quaternion::from_euler_angles(0.0, -PI/6.0, 0.0);
    let look_end = Quaternion::from_euler_angles(0.0, PI/4.0, 0.0);
    
    println!("  Camera animation keyframes:");
    for frame in 0..=3 {
        let t = frame as f64 / 3.0;
        
        // Interpolate position and rotation
        let cam_pos = Graphics3DDomain::lerp_point3d(&camera_start, &camera_end, t)?;
        let cam_rot = look_start.slerp(&look_end, t)?;
        
        // Create view matrix (simplified - normally you'd do more complex calculations)
        let translation = Transform4x4::translation(-cam_pos.x, -cam_pos.y, -cam_pos.z);
        let rotation = Transform4x4::from_quaternion(&cam_rot);
        let view_matrix = rotation.multiply(&translation);
        
        println!("    Frame {}: Camera at ({:.1}, {:.1}, {:.1})", 
                 frame, cam_pos.x, cam_pos.y, cam_pos.z);
        
        // Test transforming a world point
        let world_object = Point3D { x: 0.0, y: 0.0, z: 0.0 };
        let view_space = view_matrix.transform_point(&world_object);
        println!("      World origin in view space: ({:.2}, {:.2}, {:.2})", 
                 view_space.x, view_space.y, view_space.z);
    }
    
    println!("\n✨ 3D Graphics Demo Complete! Your framework now supports:");
    println!("   • Quaternion rotations with SLERP interpolation");
    println!("   • 4x4 transformation matrices (translate, scale, rotate, project)");
    println!("   • Raycasting with sphere/AABB/plane intersections");
    println!("   • Bounding volume collision detection");
    println!("   • Linear interpolation for smooth animations");
    println!("   • Professional camera systems and view matrices");
    
    Ok(())
}