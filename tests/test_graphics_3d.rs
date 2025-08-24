use mathtables::core::{Point3D, Vector3D, Quaternion, Transform4x4, Ray3D, AABB, BoundingSphere};
use mathtables::domains::Graphics3DDomain;
use std::f64::consts::PI;

#[test]
fn test_quaternion_creation() {
    let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    assert_eq!(q.w, 1.0);
    assert_eq!(q.x, 0.0);
    assert_eq!(q.y, 0.0);
    assert_eq!(q.z, 0.0);
    
    let identity = Quaternion::identity();
    assert_eq!(identity.w, 1.0);
    assert_eq!(identity.x, 0.0);
    assert_eq!(identity.y, 0.0);
    assert_eq!(identity.z, 0.0);
}

#[test]
fn test_quaternion_from_axis_angle() {
    let axis = Vector3D { x: 0.0, y: 0.0, z: 1.0 };
    let angle = PI / 2.0; // 90 degrees
    let q = Quaternion::from_axis_angle(&axis, angle);
    
    assert!((q.w - (PI / 4.0).cos()).abs() < 1e-10);
    assert!((q.x - 0.0).abs() < 1e-10);
    assert!((q.y - 0.0).abs() < 1e-10);
    assert!((q.z - (PI / 4.0).sin()).abs() < 1e-10);
}

#[test]
fn test_quaternion_from_euler_angles() {
    let q = Quaternion::from_euler_angles(0.0, 0.0, PI / 2.0); // 90-degree yaw
    
    // Should be similar to axis-angle result for Z rotation
    assert!((q.w - (PI / 4.0).cos()).abs() < 1e-10);
    assert!((q.x - 0.0).abs() < 1e-10);
    assert!((q.y - 0.0).abs() < 1e-10);
    assert!((q.z - (PI / 4.0).sin()).abs() < 1e-10);
}

#[test]
fn test_quaternion_magnitude_and_normalize() {
    let q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
    let magnitude = q.magnitude();
    assert_eq!(magnitude, 2.0);
    
    let normalized = q.normalize().unwrap();
    assert!((normalized.magnitude() - 1.0).abs() < 1e-10);
    assert_eq!(normalized.w, 0.5);
    assert_eq!(normalized.x, 0.5);
    assert_eq!(normalized.y, 0.5);
    assert_eq!(normalized.z, 0.5);
}

#[test]
fn test_quaternion_conjugate() {
    let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    let conj = q.conjugate();
    
    assert_eq!(conj.w, 1.0);
    assert_eq!(conj.x, -2.0);
    assert_eq!(conj.y, -3.0);
    assert_eq!(conj.z, -4.0);
}

#[test]
fn test_quaternion_multiply() {
    let q1 = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    let q2 = Quaternion::new(0.0, 1.0, 0.0, 0.0);
    let result = q1.multiply(&q2);
    
    assert_eq!(result.w, 0.0);
    assert_eq!(result.x, 1.0);
    assert_eq!(result.y, 0.0);
    assert_eq!(result.z, 0.0);
}

#[test]
fn test_quaternion_slerp() {
    let q1 = Quaternion::identity();
    let axis = Vector3D { x: 0.0, y: 0.0, z: 1.0 };
    let q2 = Quaternion::from_axis_angle(&axis, PI / 2.0);
    
    let half = q1.slerp(&q2, 0.5).unwrap();
    let expected = Quaternion::from_axis_angle(&axis, PI / 4.0);
    
    assert!((half.w - expected.w).abs() < 1e-10);
    assert!((half.x - expected.x).abs() < 1e-10);
    assert!((half.y - expected.y).abs() < 1e-10);
    assert!((half.z - expected.z).abs() < 1e-10);
}

#[test]
fn test_quaternion_rotate_point() {
    let axis = Vector3D { x: 0.0, y: 0.0, z: 1.0 };
    let q = Quaternion::from_axis_angle(&axis, PI / 2.0); // 90-degree Z rotation
    let point = Point3D { x: 1.0, y: 0.0, z: 0.0 };
    
    let rotated = q.rotate_point(&point);
    
    assert!((rotated.x - 0.0).abs() < 1e-10);
    assert!((rotated.y - 1.0).abs() < 1e-10);
    assert!((rotated.z - 0.0).abs() < 1e-10);
}

#[test]
fn test_transform4x4_identity() {
    let identity = Transform4x4::identity();
    
    for i in 0..4 {
        for j in 0..4 {
            if i == j {
                assert_eq!(identity.matrix[i][j], 1.0);
            } else {
                assert_eq!(identity.matrix[i][j], 0.0);
            }
        }
    }
}

#[test]
fn test_transform4x4_translation() {
    let translation = Transform4x4::translation(5.0, 3.0, -2.0);
    let point = Point3D { x: 1.0, y: 2.0, z: 3.0 };
    let transformed = translation.transform_point(&point);
    
    assert_eq!(transformed.x, 6.0);
    assert_eq!(transformed.y, 5.0);
    assert_eq!(transformed.z, 1.0);
}

#[test]
fn test_transform4x4_scaling() {
    let scaling = Transform4x4::scaling(2.0, 3.0, 0.5);
    let point = Point3D { x: 4.0, y: 2.0, z: 6.0 };
    let transformed = scaling.transform_point(&point);
    
    assert_eq!(transformed.x, 8.0);
    assert_eq!(transformed.y, 6.0);
    assert_eq!(transformed.z, 3.0);
}

#[test]
fn test_transform4x4_rotation_x() {
    let rotation = Transform4x4::rotation_x(PI / 2.0); // 90 degrees
    let point = Point3D { x: 0.0, y: 1.0, z: 0.0 };
    let transformed = rotation.transform_point(&point);
    
    assert!((transformed.x - 0.0).abs() < 1e-10);
    assert!((transformed.y - 0.0).abs() < 1e-10);
    assert!((transformed.z - 1.0).abs() < 1e-10);
}

#[test]
fn test_transform4x4_from_quaternion() {
    let axis = Vector3D { x: 0.0, y: 0.0, z: 1.0 };
    let q = Quaternion::from_axis_angle(&axis, PI / 2.0);
    let transform = Transform4x4::from_quaternion(&q);
    
    let point = Point3D { x: 1.0, y: 0.0, z: 0.0 };
    let transformed = transform.transform_point(&point);
    
    assert!((transformed.x - 0.0).abs() < 1e-10);
    assert!((transformed.y - 1.0).abs() < 1e-10);
    assert!((transformed.z - 0.0).abs() < 1e-10);
}

#[test]
fn test_transform4x4_multiply() {
    let translation = Transform4x4::translation(1.0, 2.0, 3.0);
    let scaling = Transform4x4::scaling(2.0, 2.0, 2.0);
    let combined = translation.multiply(&scaling);
    
    let point = Point3D { x: 1.0, y: 1.0, z: 1.0 };
    let transformed = combined.transform_point(&point);
    
    assert_eq!(transformed.x, 3.0); // (1 * 2) + 1
    assert_eq!(transformed.y, 4.0); // (1 * 2) + 2
    assert_eq!(transformed.z, 5.0); // (1 * 2) + 3
}

#[test]
fn test_transform4x4_perspective() {
    let perspective = Transform4x4::perspective(PI / 4.0, 16.0 / 9.0, 0.1, 100.0).unwrap();
    
    // Test that it's valid (non-zero determinant implied by successful creation)
    let point = Point3D { x: 0.0, y: 0.0, z: -1.0 };
    let projected = perspective.transform_point(&point);
    
    // Should project properly (Z becomes positive after perspective divide)
    assert!(projected.z > 0.0);
}

#[test]
fn test_transform4x4_orthographic() {
    let ortho = Transform4x4::orthographic(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0).unwrap();
    
    let point = Point3D { x: 5.0, y: 5.0, z: -50.0 };
    let projected = ortho.transform_point(&point);
    
    // Orthographic projection should preserve ratios
    assert_eq!(projected.x, 0.5);  // 5/10 = 0.5 in normalized coordinates
    assert_eq!(projected.y, 0.5);  // 5/10 = 0.5 in normalized coordinates
}

#[test]
fn test_ray3d_creation_and_point_at_parameter() {
    let origin = Point3D { x: 0.0, y: 0.0, z: 0.0 };
    let direction = Vector3D { x: 1.0, y: 0.0, z: 0.0 };
    let ray = Ray3D::new(origin, direction);
    
    let point_at_5 = ray.point_at_parameter(5.0);
    assert_eq!(point_at_5.x, 5.0);
    assert_eq!(point_at_5.y, 0.0);
    assert_eq!(point_at_5.z, 0.0);
}

#[test]
fn test_ray_intersect_plane() {
    let origin = Point3D { x: 0.0, y: 0.0, z: 0.0 };
    let direction = Vector3D { x: 0.0, y: 0.0, z: 1.0 };
    let ray = Ray3D::new(origin, direction);
    
    let plane_point = Point3D { x: 0.0, y: 0.0, z: 5.0 };
    let plane_normal = Vector3D { x: 0.0, y: 0.0, z: -1.0 };
    
    let intersection = ray.intersect_plane(&plane_point, &plane_normal);
    assert!(intersection.is_some());
    assert_eq!(intersection.unwrap(), 5.0);
}

#[test]
fn test_ray_intersect_sphere() {
    let origin = Point3D { x: -10.0, y: 0.0, z: 0.0 };
    let direction = Vector3D { x: 1.0, y: 0.0, z: 0.0 };
    let ray = Ray3D::new(origin, direction);
    
    let sphere_center = Point3D { x: 0.0, y: 0.0, z: 0.0 };
    let sphere = BoundingSphere::new(sphere_center, 2.0);
    
    let intersection = ray.intersect_sphere(&sphere);
    assert!(intersection.is_some());
    
    let (t1, t2) = intersection.unwrap();
    assert_eq!(t1, 8.0);  // First intersection at x = -2
    assert_eq!(t2, 12.0); // Second intersection at x = 2
}

#[test]
fn test_ray_intersect_aabb() {
    let origin = Point3D { x: -5.0, y: 0.0, z: 0.0 };
    let direction = Vector3D { x: 1.0, y: 0.0, z: 0.0 };
    let ray = Ray3D::new(origin, direction);
    
    let aabb_min = Point3D { x: -1.0, y: -1.0, z: -1.0 };
    let aabb_max = Point3D { x: 1.0, y: 1.0, z: 1.0 };
    let aabb = AABB::new(aabb_min, aabb_max);
    
    let intersection = ray.intersect_aabb(&aabb);
    assert!(intersection.is_some());
    
    let (t_min, t_max) = intersection.unwrap();
    assert_eq!(t_min, 4.0);  // Entry at x = -1
    assert_eq!(t_max, 6.0);  // Exit at x = 1
}

#[test]
fn test_aabb_creation_and_properties() {
    let min = Point3D { x: -2.0, y: -3.0, z: -4.0 };
    let max = Point3D { x: 2.0, y: 3.0, z: 4.0 };
    let aabb = AABB::new(min, max);
    
    let center = aabb.center();
    assert_eq!(center.x, 0.0);
    assert_eq!(center.y, 0.0);
    assert_eq!(center.z, 0.0);
    
    let extents = aabb.extents();
    assert_eq!(extents.x, 2.0);
    assert_eq!(extents.y, 3.0);
    assert_eq!(extents.z, 4.0);
    
    let volume = aabb.volume();
    assert_eq!(volume, 4.0 * 6.0 * 8.0); // width * height * depth
}

#[test]
fn test_aabb_contains_point() {
    let min = Point3D { x: -1.0, y: -1.0, z: -1.0 };
    let max = Point3D { x: 1.0, y: 1.0, z: 1.0 };
    let aabb = AABB::new(min, max);
    
    let inside_point = Point3D { x: 0.0, y: 0.0, z: 0.0 };
    let outside_point = Point3D { x: 2.0, y: 0.0, z: 0.0 };
    
    assert!(aabb.contains_point(&inside_point));
    assert!(!aabb.contains_point(&outside_point));
}

#[test]
fn test_aabb_intersects_aabb() {
    let aabb1 = AABB::new(
        Point3D { x: 0.0, y: 0.0, z: 0.0 },
        Point3D { x: 2.0, y: 2.0, z: 2.0 }
    );
    
    let aabb2_intersecting = AABB::new(
        Point3D { x: 1.0, y: 1.0, z: 1.0 },
        Point3D { x: 3.0, y: 3.0, z: 3.0 }
    );
    
    let aabb2_non_intersecting = AABB::new(
        Point3D { x: 3.0, y: 3.0, z: 3.0 },
        Point3D { x: 4.0, y: 4.0, z: 4.0 }
    );
    
    assert!(aabb1.intersects_aabb(&aabb2_intersecting));
    assert!(!aabb1.intersects_aabb(&aabb2_non_intersecting));
}

#[test]
fn test_bounding_sphere_creation_and_properties() {
    let center = Point3D { x: 1.0, y: 2.0, z: 3.0 };
    let sphere = BoundingSphere::new(center, 5.0);
    
    assert_eq!(sphere.center.x, 1.0);
    assert_eq!(sphere.center.y, 2.0);
    assert_eq!(sphere.center.z, 3.0);
    assert_eq!(sphere.radius, 5.0);
    
    let volume = sphere.volume();
    let expected_volume = 4.0 / 3.0 * PI * 5.0_f64.powi(3);
    assert!((volume - expected_volume).abs() < 1e-10);
}

#[test]
fn test_bounding_sphere_contains_point() {
    let center = Point3D { x: 0.0, y: 0.0, z: 0.0 };
    let sphere = BoundingSphere::new(center, 3.0);
    
    let inside_point = Point3D { x: 1.0, y: 1.0, z: 1.0 };
    let outside_point = Point3D { x: 3.0, y: 3.0, z: 3.0 };
    
    assert!(sphere.contains_point(&inside_point));
    assert!(!sphere.contains_point(&outside_point));
}

#[test]
fn test_bounding_sphere_intersects_sphere() {
    let sphere1 = BoundingSphere::new(
        Point3D { x: 0.0, y: 0.0, z: 0.0 }, 
        2.0
    );
    
    let sphere2_intersecting = BoundingSphere::new(
        Point3D { x: 3.0, y: 0.0, z: 0.0 }, 
        2.0
    );
    
    let sphere2_non_intersecting = BoundingSphere::new(
        Point3D { x: 5.0, y: 0.0, z: 0.0 }, 
        2.0
    );
    
    assert!(sphere1.intersects_sphere(&sphere2_intersecting));
    assert!(!sphere1.intersects_sphere(&sphere2_non_intersecting));
}

#[test]
fn test_graphics3d_domain_lerp() {
    let result = Graphics3DDomain::lerp(0.0, 10.0, 0.5).unwrap();
    assert_eq!(result, 5.0);
    
    let result_start = Graphics3DDomain::lerp(0.0, 10.0, 0.0).unwrap();
    assert_eq!(result_start, 0.0);
    
    let result_end = Graphics3DDomain::lerp(0.0, 10.0, 1.0).unwrap();
    assert_eq!(result_end, 10.0);
}

#[test]
fn test_graphics3d_domain_lerp_vector3d() {
    let v1 = Vector3D { x: 0.0, y: 0.0, z: 0.0 };
    let v2 = Vector3D { x: 10.0, y: 20.0, z: 30.0 };
    
    let result = Graphics3DDomain::lerp_vector3d(&v1, &v2, 0.5).unwrap();
    assert_eq!(result.x, 5.0);
    assert_eq!(result.y, 10.0);
    assert_eq!(result.z, 15.0);
}

#[test]
fn test_graphics3d_domain_lerp_point3d() {
    let p1 = Point3D { x: 1.0, y: 2.0, z: 3.0 };
    let p2 = Point3D { x: 5.0, y: 8.0, z: 15.0 };
    
    let result = Graphics3DDomain::lerp_point3d(&p1, &p2, 0.25).unwrap();
    assert_eq!(result.x, 2.0);  // 1 + 0.25 * (5-1) = 2
    assert_eq!(result.y, 3.5);  // 2 + 0.25 * (8-2) = 3.5
    assert_eq!(result.z, 6.0);  // 3 + 0.25 * (15-3) = 6
}

#[test]
fn test_invalid_lerp_parameter() {
    let result = Graphics3DDomain::lerp(0.0, 10.0, 1.5);
    assert!(result.is_err());
    
    let result = Graphics3DDomain::lerp(0.0, 10.0, -0.5);
    assert!(result.is_err());
}

#[test]
fn test_quaternion_slerp_edge_cases() {
    let q1 = Quaternion::identity();
    let q2 = Quaternion::identity();
    
    // SLERP between identical quaternions should return the original
    let result = q1.slerp(&q2, 0.5).unwrap();
    assert_eq!(result.w, q1.w);
    assert_eq!(result.x, q1.x);
    assert_eq!(result.y, q1.y);
    assert_eq!(result.z, q1.z);
}

#[test]
fn test_zero_magnitude_quaternion() {
    let zero_q = Quaternion::new(0.0, 0.0, 0.0, 0.0);
    let normalize_result = zero_q.normalize();
    assert!(normalize_result.is_err());
}