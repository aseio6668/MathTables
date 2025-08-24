use mathtables::core::{
    Point2D, Point3D, Vector2D, 
    BezierCurve2D, BezierCurve3D, CatmullRomSpline2D,
    Keyframe, EasingFunction, AnimationCurve
};
use mathtables::domains::AnimationDomain;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎬 Animation & Easing Domain Demo");
    println!("=================================\n");

    // ===== Easing Functions Showcase =====
    println!("🎭 Easing Functions Showcase:");
    
    let easing_functions = vec![
        ("Linear", EasingFunction::Linear),
        ("Quad In", EasingFunction::QuadraticIn),
        ("Quad Out", EasingFunction::QuadraticOut),
        ("Cubic InOut", EasingFunction::CubicInOut),
        ("Bounce Out", EasingFunction::BounceOut),
        ("Elastic Out", EasingFunction::ElasticOut),
        ("Back InOut", EasingFunction::BackInOut),
    ];
    
    for (name, easing) in &easing_functions {
        println!("  {} easing curve:", name);
        print!("    ");
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let value = easing.evaluate(t)?;
            print!("{:.2} ", value);
        }
        println!();
    }
    
    // ===== 2D Bezier Curve Demo =====
    println!("\n📐 2D Bezier Curve Demonstration:");
    
    // Quadratic Bezier - Creating a parabolic arc
    let p0 = Point2D { x: 0.0, y: 0.0 };
    let p1 = Point2D { x: 5.0, y: 10.0 }; // Control point creates the arch
    let p2 = Point2D { x: 10.0, y: 0.0 };
    let quadratic_curve = BezierCurve2D::quadratic(p0, p1, p2);
    
    println!("  Quadratic Bezier curve points:");
    for i in 0..=5 {
        let t = i as f64 / 5.0;
        let point = quadratic_curve.evaluate(t)?;
        println!("    t={:.1}: ({:.2}, {:.2})", t, point.x, point.y);
    }
    
    let curve_length = quadratic_curve.length(100);
    println!("    Curve length: {:.2} units", curve_length);
    
    // Cubic Bezier - S-shaped curve
    println!("\n  Cubic Bezier (S-curve):");
    let cubic = BezierCurve2D::cubic(
        Point2D { x: 0.0, y: 0.0 },   // Start
        Point2D { x: 3.0, y: 8.0 },   // Control point 1
        Point2D { x: 7.0, y: -2.0 },  // Control point 2
        Point2D { x: 10.0, y: 5.0 }   // End
    );
    
    for i in 0..=4 {
        let t = i as f64 / 4.0;
        let point = cubic.evaluate(t)?;
        let derivative = cubic.derivative(t)?;
        println!("    t={:.2}: pos({:.1}, {:.1}), velocity({:.1}, {:.1})", 
                 t, point.x, point.y, derivative.x, derivative.y);
    }
    
    // ===== 3D Bezier Curve Demo =====
    println!("\n🌎 3D Bezier Curve (Flight Path):");
    
    let flight_path = BezierCurve3D::cubic(
        Point3D { x: 0.0, y: 0.0, z: 1000.0 },    // Takeoff
        Point3D { x: 2000.0, y: 500.0, z: 8000.0 }, // Climb control
        Point3D { x: 6000.0, y: -300.0, z: 9000.0 }, // Cruise control
        Point3D { x: 10000.0, y: 0.0, z: 1000.0 }   // Landing
    );
    
    println!("  Aircraft flight path:");
    for i in 0..=4 {
        let t = i as f64 / 4.0;
        let position = flight_path.evaluate(t)?;
        let velocity = flight_path.derivative(t)?;
        let speed = (velocity.x.powi(2) + velocity.y.powi(2) + velocity.z.powi(2)).sqrt();
        
        println!("    Phase {}: pos({:.0}, {:.0}, {:.0})ft, speed: {:.0} ft/s", 
                 i, position.x, position.y, position.z, speed);
    }
    
    // ===== Catmull-Rom Spline Demo =====
    println!("\n🎢 Catmull-Rom Spline (Smooth Path Through Points):");
    
    let waypoints = vec![
        Point2D { x: 0.0, y: 2.0 },
        Point2D { x: 2.0, y: 1.0 },
        Point2D { x: 4.0, y: 4.0 },
        Point2D { x: 6.0, y: 2.0 },
        Point2D { x: 8.0, y: 3.0 },
    ];
    
    let spline = CatmullRomSpline2D::new(waypoints, 1.0);
    
    println!("  Smooth spline through waypoints:");
    for i in 0..=8 {
        let t = i as f64 / 8.0;
        let point = spline.evaluate(t)?;
        println!("    t={:.2}: ({:.2}, {:.2})", t, point.x, point.y);
    }
    
    // ===== Animation Curve System =====
    println!("\n⏱️ Animation Curve System:");
    
    // Create an animation curve for a bouncing ball's height
    let ball_height_keyframes = vec![
        Keyframe { time: 0.0, value: 0.0, easing: EasingFunction::QuadraticOut }, // Ground
        Keyframe { time: 0.5, value: 8.0, easing: EasingFunction::QuadraticIn },  // Peak
        Keyframe { time: 1.0, value: 0.0, easing: EasingFunction::BounceOut },    // Bounce back
        Keyframe { time: 1.3, value: 4.0, easing: EasingFunction::QuadraticIn },  // Smaller bounce
        Keyframe { time: 1.6, value: 0.0, easing: EasingFunction::Linear },       // Settle
    ];
    
    let ball_animation = AnimationCurve::new(ball_height_keyframes, 2.0);
    
    println!("  Bouncing ball animation (height over time):");
    for frame in 0..=16 {
        let time = frame as f64 * 0.1;
        let height = ball_animation.evaluate(time)?;
        println!("    Frame {:2} (t={:.1}s): height = {:.2} units", 
                 frame, time, height);
    }
    
    // ===== Advanced Easing Interpolation =====
    println!("\n🎨 Advanced Easing Interpolation:");
    
    let start_color = Point3D { x: 255.0, y: 0.0, z: 0.0 };    // Red
    let end_color = Point3D { x: 0.0, y: 0.0, z: 255.0 };      // Blue
    
    println!("  Color transition with different easings:");
    
    let easings = vec![
        ("Linear", EasingFunction::Linear),
        ("Ease In", EasingFunction::CubicIn),
        ("Elastic", EasingFunction::ElasticOut),
    ];
    
    for (name, easing) in &easings {
        println!("    {} transition:", name);
        for i in 0..=4 {
            let t = i as f64 / 4.0;
            let color = AnimationDomain::interpolate_point3d_with_easing(
                &start_color, &end_color, t, easing
            )?;
            println!("      t={:.2}: RGB({:.0}, {:.0}, {:.0})", 
                     t, color.x, color.y, color.z);
        }
    }
    
    // ===== Time Utilities Demo =====
    println!("\n⏰ Time Utilities:");
    
    // Loop animation
    println!("  Loop animation (5-second cycle):");
    for second in 0..12 {
        let loop_t = AnimationDomain::loop_time(second as f64, 5.0)?;
        println!("    Second {:2}: loop_t = {:.2}", second, loop_t);
    }
    
    // Ping-pong animation
    println!("\n  Ping-pong animation (3-second cycle):");
    for frame in 0..15 {
        let time = frame as f64 * 0.5;
        let ping_pong_t = AnimationDomain::ping_pong_time(time, 3.0)?;
        println!("    Frame {:2} (t={:.1}s): ping_pong_t = {:.2}", 
                 frame, time, ping_pong_t);
    }
    
    // ===== Practical Example: UI Animation =====
    println!("\n📱 Practical Example: UI Menu Slide Animation");
    
    let menu_start = Point2D { x: -300.0, y: 100.0 }; // Off-screen
    let menu_end = Point2D { x: 0.0, y: 100.0 };      // On-screen
    
    println!("  Menu sliding in with Back easing:");
    
    for frame in 0..=8 {
        let t = frame as f64 / 8.0;
        let position = AnimationDomain::interpolate_point2d_with_easing(
            &menu_start, &menu_end, t, &EasingFunction::BackOut
        )?;
        
        let progress = (t * 100.0) as u32;
        let bar = "█".repeat((progress / 10) as usize);
        println!("    Frame {}: x={:6.1} [{}{}] {}%", 
                 frame, position.x, bar, " ".repeat(10 - (progress / 10) as usize), progress);
    }
    
    // ===== Performance Timing Example =====
    println!("\n⚡ Performance: Evaluating Complex Animation Chain");
    
    let start_time = std::time::Instant::now();
    
    let complex_curve = BezierCurve3D::cubic(
        Point3D { x: 0.0, y: 0.0, z: 0.0 },
        Point3D { x: 100.0, y: 200.0, z: 50.0 },
        Point3D { x: 300.0, y: -100.0, z: 150.0 },
        Point3D { x: 500.0, y: 100.0, z: 0.0 }
    );
    
    let mut total_distance = 0.0;
    let samples = 1000;
    
    for i in 0..samples {
        let t = i as f64 / (samples - 1) as f64;
        let eased_t = EasingFunction::ElasticOut.evaluate(t)?;
        let point = complex_curve.evaluate(eased_t)?;
        
        if i > 0 {
            let prev_t = (i - 1) as f64 / (samples - 1) as f64;
            let prev_eased_t = EasingFunction::ElasticOut.evaluate(prev_t)?;
            let prev_point = complex_curve.evaluate(prev_eased_t)?;
            
            let dx = point.x - prev_point.x;
            let dy = point.y - prev_point.y;
            let dz = point.z - prev_point.z;
            total_distance += (dx*dx + dy*dy + dz*dz).sqrt();
        }
    }
    
    let elapsed = start_time.elapsed();
    
    println!("  Computed {} curve samples with easing in {:?}", samples, elapsed);
    println!("  Total path length: {:.2} units", total_distance);
    println!("  Performance: {:.0} evaluations/second", 
             samples as f64 / elapsed.as_secs_f64());
    
    println!("\n✨ Animation Demo Complete! Your framework now supports:");
    println!("   • 15+ professional easing functions (cubic, bounce, elastic, back, etc.)");
    println!("   • Bezier curves (linear, quadratic, cubic) in 2D and 3D");
    println!("   • Catmull-Rom splines for smooth interpolation through points");
    println!("   • Keyframe animation system with per-keyframe easing");
    println!("   • Advanced interpolation with custom easing functions");
    println!("   • Time utilities (looping, ping-pong, normalization)");
    println!("   • Curve derivatives and length calculations");
    println!("   • High-performance evaluation suitable for real-time use");
    
    Ok(())
}