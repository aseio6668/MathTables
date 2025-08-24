use mathtables::core::{
    Point2D, Point3D, Vector3D, 
    BezierCurve2D, BezierCurve3D, CatmullRomSpline2D, 
    Keyframe, EasingFunction, AnimationCurve
};
use mathtables::domains::AnimationDomain;

#[test]
fn test_linear_easing() {
    let easing = EasingFunction::Linear;
    
    assert_eq!(easing.evaluate(0.0).unwrap(), 0.0);
    assert_eq!(easing.evaluate(0.5).unwrap(), 0.5);
    assert_eq!(easing.evaluate(1.0).unwrap(), 1.0);
}

#[test]
fn test_quadratic_easing() {
    let ease_in = EasingFunction::QuadraticIn;
    let ease_out = EasingFunction::QuadraticOut;
    let ease_in_out = EasingFunction::QuadraticInOut;
    
    // Quadratic In should start slow
    assert_eq!(ease_in.evaluate(0.0).unwrap(), 0.0);
    assert_eq!(ease_in.evaluate(0.5).unwrap(), 0.25);
    assert_eq!(ease_in.evaluate(1.0).unwrap(), 1.0);
    
    // Quadratic Out should end slow
    assert_eq!(ease_out.evaluate(0.0).unwrap(), 0.0);
    assert_eq!(ease_out.evaluate(0.5).unwrap(), 0.75);
    assert_eq!(ease_out.evaluate(1.0).unwrap(), 1.0);
    
    // Quadratic InOut should be symmetric
    assert_eq!(ease_in_out.evaluate(0.0).unwrap(), 0.0);
    assert_eq!(ease_in_out.evaluate(0.5).unwrap(), 0.5);
    assert_eq!(ease_in_out.evaluate(1.0).unwrap(), 1.0);
}

#[test]
fn test_cubic_easing() {
    let ease_in = EasingFunction::CubicIn;
    let ease_out = EasingFunction::CubicOut;
    
    // Cubic In should start very slow
    assert_eq!(ease_in.evaluate(0.0).unwrap(), 0.0);
    assert_eq!(ease_in.evaluate(0.5).unwrap(), 0.125);
    assert_eq!(ease_in.evaluate(1.0).unwrap(), 1.0);
    
    // Cubic Out should end very slow
    assert_eq!(ease_out.evaluate(0.0).unwrap(), 0.0);
    assert_eq!(ease_out.evaluate(0.5).unwrap(), 0.875);
    assert_eq!(ease_out.evaluate(1.0).unwrap(), 1.0);
}

#[test]
fn test_sine_easing() {
    let ease_in = EasingFunction::SineIn;
    let ease_out = EasingFunction::SineOut;
    
    assert_eq!(ease_in.evaluate(0.0).unwrap(), 0.0);
    assert!((ease_in.evaluate(1.0).unwrap() - 1.0).abs() < 1e-10);
    
    assert_eq!(ease_out.evaluate(0.0).unwrap(), 0.0);
    assert!((ease_out.evaluate(1.0).unwrap() - 1.0).abs() < 1e-10);
    
    // Sine easing should be smooth
    let mid_in = ease_in.evaluate(0.5).unwrap();
    let mid_out = ease_out.evaluate(0.5).unwrap();
    
    assert!(mid_in > 0.0 && mid_in < 1.0);
    assert!(mid_out > 0.0 && mid_out < 1.0);
}

#[test]
fn test_bounce_easing() {
    let bounce_out = EasingFunction::BounceOut;
    
    assert_eq!(bounce_out.evaluate(0.0).unwrap(), 0.0);
    assert_eq!(bounce_out.evaluate(1.0).unwrap(), 1.0);
    
    // Bounce should create multiple peaks
    let values: Vec<f64> = (0..=10)
        .map(|i| bounce_out.evaluate(i as f64 / 10.0).unwrap())
        .collect();
    
    // Should have some variation (bouncing effect)
    let has_variation = values.windows(2).any(|w| (w[1] - w[0]).abs() > 0.1);
    assert!(has_variation);
}

#[test]
fn test_elastic_easing() {
    let elastic_out = EasingFunction::ElasticOut;
    
    assert_eq!(elastic_out.evaluate(0.0).unwrap(), 0.0);
    assert_eq!(elastic_out.evaluate(1.0).unwrap(), 1.0);
    
    // Elastic should overshoot and oscillate
    let mid_value = elastic_out.evaluate(0.8).unwrap();
    assert!(mid_value > 0.0);  // Should be positive at this point
}

#[test]
fn test_back_easing() {
    let back_in = EasingFunction::BackIn;
    let back_out = EasingFunction::BackOut;
    
    assert!((back_in.evaluate(0.0).unwrap() - 0.0).abs() < 1e-10);
    assert!((back_in.evaluate(1.0).unwrap() - 1.0).abs() < 1e-10);
    
    assert!((back_out.evaluate(0.0).unwrap() - 0.0).abs() < 1e-10);
    assert!((back_out.evaluate(1.0).unwrap() - 1.0).abs() < 1e-10);
    
    // Back easing should have characteristic curve behavior
    let back_in_quarter = back_in.evaluate(0.25).unwrap();
    let back_out_quarter = back_out.evaluate(0.25).unwrap();
    
    // Test that they produce reasonable values
    assert!(back_in_quarter.is_finite());
    assert!(back_out_quarter.is_finite());
}

#[test]
fn test_easing_parameter_validation() {
    let easing = EasingFunction::Linear;
    
    assert!(easing.evaluate(-0.1).is_err());
    assert!(easing.evaluate(1.1).is_err());
    assert!(easing.evaluate(0.5).is_ok());
}

#[test]
fn test_bezier_curve_2d_linear() {
    let p0 = Point2D { x: 0.0, y: 0.0 };
    let p1 = Point2D { x: 10.0, y: 5.0 };
    let curve = BezierCurve2D::linear(p0, p1);
    
    let start = curve.evaluate(0.0).unwrap();
    let middle = curve.evaluate(0.5).unwrap();
    let end = curve.evaluate(1.0).unwrap();
    
    assert_eq!(start.x, 0.0);
    assert_eq!(start.y, 0.0);
    
    assert_eq!(middle.x, 5.0);
    assert_eq!(middle.y, 2.5);
    
    assert_eq!(end.x, 10.0);
    assert_eq!(end.y, 5.0);
}

#[test]
fn test_bezier_curve_2d_quadratic() {
    let p0 = Point2D { x: 0.0, y: 0.0 };
    let p1 = Point2D { x: 5.0, y: 10.0 }; // Control point
    let p2 = Point2D { x: 10.0, y: 0.0 };
    let curve = BezierCurve2D::quadratic(p0, p1, p2);
    
    let start = curve.evaluate(0.0).unwrap();
    let middle = curve.evaluate(0.5).unwrap();
    let end = curve.evaluate(1.0).unwrap();
    
    assert_eq!(start.x, 0.0);
    assert_eq!(start.y, 0.0);
    
    assert_eq!(middle.x, 5.0);
    assert_eq!(middle.y, 5.0); // Should be influenced by control point
    
    assert_eq!(end.x, 10.0);
    assert_eq!(end.y, 0.0);
}

#[test]
fn test_bezier_curve_2d_cubic() {
    let p0 = Point2D { x: 0.0, y: 0.0 };
    let p1 = Point2D { x: 2.0, y: 8.0 };
    let p2 = Point2D { x: 8.0, y: 8.0 };
    let p3 = Point2D { x: 10.0, y: 0.0 };
    let curve = BezierCurve2D::cubic(p0, p1, p2, p3);
    
    let start = curve.evaluate(0.0).unwrap();
    let end = curve.evaluate(1.0).unwrap();
    
    assert_eq!(start.x, 0.0);
    assert_eq!(start.y, 0.0);
    assert_eq!(end.x, 10.0);
    assert_eq!(end.y, 0.0);
    
    // Middle point should be influenced by control points
    let middle = curve.evaluate(0.5).unwrap();
    assert_eq!(middle.x, 5.0);
    assert!(middle.y > 0.0); // Should be above the line connecting start and end
}

#[test]
fn test_bezier_curve_2d_derivative() {
    let p0 = Point2D { x: 0.0, y: 0.0 };
    let p1 = Point2D { x: 10.0, y: 5.0 };
    let curve = BezierCurve2D::linear(p0, p1);
    
    let derivative = curve.derivative(0.5).unwrap();
    
    // For linear curve, derivative should be constant
    assert_eq!(derivative.x, 10.0); // dx/dt
    assert_eq!(derivative.y, 5.0);  // dy/dt
}

#[test]
fn test_bezier_curve_2d_length() {
    let p0 = Point2D { x: 0.0, y: 0.0 };
    let p1 = Point2D { x: 3.0, y: 4.0 };
    let curve = BezierCurve2D::linear(p0, p1);
    
    let length = curve.length(100);
    
    // Length of straight line from (0,0) to (3,4) should be 5
    assert!((length - 5.0).abs() < 0.01);
}

#[test]
fn test_bezier_curve_3d() {
    let p0 = Point3D { x: 0.0, y: 0.0, z: 0.0 };
    let p1 = Point3D { x: 10.0, y: 5.0, z: 2.0 };
    let curve = BezierCurve3D::linear(p0, p1);
    
    let start = curve.evaluate(0.0).unwrap();
    let middle = curve.evaluate(0.5).unwrap();
    let end = curve.evaluate(1.0).unwrap();
    
    assert_eq!(start.x, 0.0);
    assert_eq!(start.y, 0.0);
    assert_eq!(start.z, 0.0);
    
    assert_eq!(middle.x, 5.0);
    assert_eq!(middle.y, 2.5);
    assert_eq!(middle.z, 1.0);
    
    assert_eq!(end.x, 10.0);
    assert_eq!(end.y, 5.0);
    assert_eq!(end.z, 2.0);
}

#[test]
fn test_bezier_curve_3d_derivative() {
    let p0 = Point3D { x: 0.0, y: 0.0, z: 0.0 };
    let p1 = Point3D { x: 6.0, y: 3.0, z: 9.0 };
    let curve = BezierCurve3D::linear(p0, p1);
    
    let derivative = curve.derivative(0.5).unwrap();
    
    assert_eq!(derivative.x, 6.0);
    assert_eq!(derivative.y, 3.0);
    assert_eq!(derivative.z, 9.0);
}

#[test]
fn test_catmull_rom_spline_2d() {
    let points = vec![
        Point2D { x: 0.0, y: 0.0 },
        Point2D { x: 1.0, y: 2.0 },
        Point2D { x: 2.0, y: 1.0 },
        Point2D { x: 3.0, y: 3.0 },
    ];
    
    let spline = CatmullRomSpline2D::new(points, 1.0);
    
    // Should pass through the middle points
    let result = spline.evaluate(0.5).unwrap(); // Should be close to second point
    
    assert!(result.x > 0.5 && result.x < 2.5);
    assert!(result.y > 0.5 && result.y < 2.5);
}

#[test]
fn test_animation_curve_single_keyframe() {
    let keyframes = vec![
        Keyframe {
            time: 5.0,
            value: 42.0,
            easing: EasingFunction::Linear,
        }
    ];
    
    let curve = AnimationCurve::new(keyframes, 10.0);
    
    // Should return the single value regardless of time
    assert_eq!(curve.evaluate(0.0).unwrap(), 42.0);
    assert_eq!(curve.evaluate(5.0).unwrap(), 42.0);
    assert_eq!(curve.evaluate(10.0).unwrap(), 42.0);
}

#[test]
fn test_animation_curve_interpolation() {
    let keyframes = vec![
        Keyframe {
            time: 0.0,
            value: 0.0,
            easing: EasingFunction::Linear,
        },
        Keyframe {
            time: 10.0,
            value: 100.0,
            easing: EasingFunction::Linear,
        }
    ];
    
    let curve = AnimationCurve::new(keyframes, 10.0);
    
    assert_eq!(curve.evaluate(0.0).unwrap(), 0.0);
    assert_eq!(curve.evaluate(5.0).unwrap(), 50.0);
    assert_eq!(curve.evaluate(10.0).unwrap(), 100.0);
}

#[test]
fn test_animation_curve_with_easing() {
    let keyframes = vec![
        Keyframe {
            time: 0.0,
            value: 0.0,
            easing: EasingFunction::Linear,
        },
        Keyframe {
            time: 10.0,
            value: 100.0,
            easing: EasingFunction::QuadraticIn,
        }
    ];
    
    let curve = AnimationCurve::new(keyframes, 10.0);
    
    let mid_value = curve.evaluate(5.0).unwrap();
    
    // With QuadraticIn easing, should be less than linear interpolation
    assert!(mid_value < 50.0);
    assert!(mid_value > 0.0);
}

#[test]
fn test_animation_curve_multiple_keyframes() {
    let keyframes = vec![
        Keyframe { time: 0.0, value: 0.0, easing: EasingFunction::Linear },
        Keyframe { time: 5.0, value: 50.0, easing: EasingFunction::Linear },
        Keyframe { time: 10.0, value: 20.0, easing: EasingFunction::Linear },
    ];
    
    let curve = AnimationCurve::new(keyframes, 10.0);
    
    assert_eq!(curve.evaluate(0.0).unwrap(), 0.0);
    assert_eq!(curve.evaluate(2.5).unwrap(), 25.0); // Half way to first keyframe
    assert_eq!(curve.evaluate(5.0).unwrap(), 50.0);
    assert_eq!(curve.evaluate(7.5).unwrap(), 35.0); // Half way from 50 to 20
    assert_eq!(curve.evaluate(10.0).unwrap(), 20.0);
}

#[test]
fn test_animation_domain_interpolate_with_easing() {
    let result = AnimationDomain::interpolate_with_easing(
        0.0, 100.0, 0.5, &EasingFunction::Linear
    ).unwrap();
    
    assert_eq!(result, 50.0);
    
    let eased_result = AnimationDomain::interpolate_with_easing(
        0.0, 100.0, 0.5, &EasingFunction::QuadraticIn
    ).unwrap();
    
    // Should be less than linear interpolation
    assert!(eased_result < 50.0);
}

#[test]
fn test_animation_domain_interpolate_point2d_with_easing() {
    let start = Point2D { x: 0.0, y: 0.0 };
    let end = Point2D { x: 10.0, y: 20.0 };
    
    let result = AnimationDomain::interpolate_point2d_with_easing(
        &start, &end, 0.5, &EasingFunction::Linear
    ).unwrap();
    
    assert_eq!(result.x, 5.0);
    assert_eq!(result.y, 10.0);
}

#[test]
fn test_animation_domain_interpolate_point3d_with_easing() {
    let start = Point3D { x: 0.0, y: 0.0, z: 0.0 };
    let end = Point3D { x: 6.0, y: 9.0, z: 12.0 };
    
    let result = AnimationDomain::interpolate_point3d_with_easing(
        &start, &end, 0.5, &EasingFunction::Linear
    ).unwrap();
    
    assert_eq!(result.x, 3.0);
    assert_eq!(result.y, 4.5);
    assert_eq!(result.z, 6.0);
}

#[test]
fn test_time_utilities() {
    // Test time_to_normalized
    let normalized = AnimationDomain::time_to_normalized(5.0, 0.0, 10.0).unwrap();
    assert_eq!(normalized, 0.5);
    
    // Test clamping
    let clamped = AnimationDomain::time_to_normalized(-5.0, 0.0, 10.0).unwrap();
    assert_eq!(clamped, 0.0);
    
    // Test ping pong
    let ping_pong = AnimationDomain::ping_pong_time(5.0, 10.0).unwrap(); // First half of cycle
    assert_eq!(ping_pong, 0.5);
    
    let ping_pong2 = AnimationDomain::ping_pong_time(15.0, 10.0).unwrap(); // Second half of cycle
    assert_eq!(ping_pong2, 0.5);
    
    // Test loop
    let looped = AnimationDomain::loop_time(15.0, 10.0).unwrap();
    assert_eq!(looped, 0.5); // 15 % 10 = 5, 5/10 = 0.5
}

#[test]
fn test_bezier_parameter_validation() {
    let curve = BezierCurve2D::linear(
        Point2D { x: 0.0, y: 0.0 },
        Point2D { x: 1.0, y: 1.0 }
    );
    
    assert!(curve.evaluate(-0.1).is_err());
    assert!(curve.evaluate(1.1).is_err());
    assert!(curve.evaluate(0.5).is_ok());
}

#[test]
fn test_animation_curve_empty_keyframes() {
    let curve: AnimationCurve<f64> = AnimationCurve::new(vec![], 10.0);
    
    assert!(curve.evaluate(5.0).is_err());
}

#[test]
fn test_catmull_rom_insufficient_points() {
    let points = vec![
        Point2D { x: 0.0, y: 0.0 },
        Point2D { x: 1.0, y: 1.0 },
    ]; // Only 2 points, need 4
    
    let spline = CatmullRomSpline2D::new(points, 1.0);
    assert!(spline.evaluate(0.5).is_err());
}

#[test]
fn test_domain_convenience_methods() {
    // Test some of the convenience methods
    assert_eq!(AnimationDomain::ease_in_out_cubic(0.0).unwrap(), 0.0);
    assert_eq!(AnimationDomain::ease_in_out_cubic(1.0).unwrap(), 1.0);
    
    assert_eq!(AnimationDomain::ease_bounce_out(0.0).unwrap(), 0.0);
    assert_eq!(AnimationDomain::ease_bounce_out(1.0).unwrap(), 1.0);
    
    assert_eq!(AnimationDomain::ease_elastic_out(0.0).unwrap(), 0.0);
    assert_eq!(AnimationDomain::ease_elastic_out(1.0).unwrap(), 1.0);
}