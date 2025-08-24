use crate::core::{
    MathDomain, MathResult, MathError, Point2D, Point3D, Vector2D, Vector3D,
    BezierCurve2D, BezierCurve3D, CatmullRomSpline2D, CatmullRomSpline3D,
    Keyframe, EasingFunction, AnimationCurve
};
use std::any::Any;
use std::f64::consts::PI;

pub struct AnimationDomain;

impl EasingFunction {
    pub fn evaluate(&self, t: f64) -> MathResult<f64> {
        if t < 0.0 || t > 1.0 {
            return Err(MathError::InvalidArgument("Easing parameter t must be between 0 and 1".to_string()));
        }
        
        let result = match self {
            EasingFunction::Linear => t,
            
            // Quadratic easing
            EasingFunction::QuadraticIn => t * t,
            EasingFunction::QuadraticOut => t * (2.0 - t),
            EasingFunction::QuadraticInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            },
            
            // Cubic easing
            EasingFunction::CubicIn => t * t * t,
            EasingFunction::CubicOut => {
                let t1 = t - 1.0;
                1.0 + t1 * t1 * t1
            },
            EasingFunction::CubicInOut => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    let t1 = 2.0 * t - 2.0;
                    1.0 + t1 * t1 * t1 / 2.0
                }
            },
            
            // Quartic easing
            EasingFunction::QuarticIn => t * t * t * t,
            EasingFunction::QuarticOut => {
                let t1 = t - 1.0;
                1.0 - t1 * t1 * t1 * t1
            },
            EasingFunction::QuarticInOut => {
                if t < 0.5 {
                    8.0 * t * t * t * t
                } else {
                    let t1 = t - 1.0;
                    1.0 - 8.0 * t1 * t1 * t1 * t1
                }
            },
            
            // Sine easing
            EasingFunction::SineIn => 1.0 - (t * PI / 2.0).cos(),
            EasingFunction::SineOut => (t * PI / 2.0).sin(),
            EasingFunction::SineInOut => -(((PI * t).cos() - 1.0) / 2.0),
            
            // Exponential easing
            EasingFunction::ExponentialIn => {
                if t == 0.0 { 0.0 } else { (2.0_f64).powf(10.0 * (t - 1.0)) }
            },
            EasingFunction::ExponentialOut => {
                if t == 1.0 { 1.0 } else { 1.0 - (2.0_f64).powf(-10.0 * t) }
            },
            EasingFunction::ExponentialInOut => {
                if t == 0.0 { return Ok(0.0); }
                if t == 1.0 { return Ok(1.0); }
                
                if t < 0.5 {
                    (2.0_f64).powf(20.0 * t - 10.0) / 2.0
                } else {
                    (2.0 - (2.0_f64).powf(-20.0 * t + 10.0)) / 2.0
                }
            },
            
            // Circular easing
            EasingFunction::CircularIn => 1.0 - (1.0 - t * t).sqrt(),
            EasingFunction::CircularOut => (1.0 - (t - 1.0) * (t - 1.0)).sqrt(),
            EasingFunction::CircularInOut => {
                if t < 0.5 {
                    (1.0 - (1.0 - (2.0 * t).powi(2)).sqrt()) / 2.0
                } else {
                    ((1.0 - (-2.0 * t + 2.0).powi(2)).sqrt() + 1.0) / 2.0
                }
            },
            
            // Elastic easing
            EasingFunction::ElasticIn => {
                if t == 0.0 { return Ok(0.0); }
                if t == 1.0 { return Ok(1.0); }
                
                let c4 = (2.0 * PI) / 3.0;
                -(2.0_f64).powf(10.0 * t - 10.0) * ((t * 10.0 - 10.75) * c4).sin()
            },
            EasingFunction::ElasticOut => {
                if t == 0.0 { return Ok(0.0); }
                if t == 1.0 { return Ok(1.0); }
                
                let c4 = (2.0 * PI) / 3.0;
                (2.0_f64).powf(-10.0 * t) * ((t * 10.0 - 0.75) * c4).sin() + 1.0
            },
            EasingFunction::ElasticInOut => {
                if t == 0.0 { return Ok(0.0); }
                if t == 1.0 { return Ok(1.0); }
                
                let c5 = (2.0 * PI) / 4.5;
                
                if t < 0.5 {
                    -((2.0_f64).powf(20.0 * t - 10.0) * ((20.0 * t - 11.125) * c5).sin()) / 2.0
                } else {
                    ((2.0_f64).powf(-20.0 * t + 10.0) * ((20.0 * t - 11.125) * c5).sin()) / 2.0 + 1.0
                }
            },
            
            // Bounce easing
            EasingFunction::BounceIn => 1.0 - Self::bounce_out(1.0 - t),
            EasingFunction::BounceOut => Self::bounce_out(t),
            EasingFunction::BounceInOut => {
                if t < 0.5 {
                    (1.0 - Self::bounce_out(1.0 - 2.0 * t)) / 2.0
                } else {
                    (1.0 + Self::bounce_out(2.0 * t - 1.0)) / 2.0
                }
            },
            
            // Back easing (overshoots then returns)
            EasingFunction::BackIn => {
                let c1 = 1.70158;
                let c3 = c1 + 1.0;
                c3 * t * t * t - c1 * t * t
            },
            EasingFunction::BackOut => {
                let c1 = 1.70158;
                let c3 = c1 + 1.0;
                1.0 + c3 * (t - 1.0).powi(3) + c1 * (t - 1.0).powi(2)
            },
            EasingFunction::BackInOut => {
                let c1 = 1.70158;
                let c2 = c1 * 1.525;
                
                if t < 0.5 {
                    ((2.0 * t).powi(2) * ((c2 + 1.0) * 2.0 * t - c2)) / 2.0
                } else {
                    ((2.0 * t - 2.0).powi(2) * ((c2 + 1.0) * (t * 2.0 - 2.0) + c2) + 2.0) / 2.0
                }
            },
        };
        
        Ok(result.clamp(0.0, 1.0))
    }
    
    fn bounce_out(t: f64) -> f64 {
        let n1 = 7.5625;
        let d1 = 2.75;
        
        if t < 1.0 / d1 {
            n1 * t * t
        } else if t < 2.0 / d1 {
            let t_adj = t - 1.5 / d1;
            n1 * t_adj * t_adj + 0.75
        } else if t < 2.5 / d1 {
            let t_adj = t - 2.25 / d1;
            n1 * t_adj * t_adj + 0.9375
        } else {
            let t_adj = t - 2.625 / d1;
            n1 * t_adj * t_adj + 0.984375
        }
    }
}

impl BezierCurve2D {
    pub fn new(control_points: Vec<Point2D>) -> Self {
        Self { control_points }
    }
    
    pub fn linear(p0: Point2D, p1: Point2D) -> Self {
        Self::new(vec![p0, p1])
    }
    
    pub fn quadratic(p0: Point2D, p1: Point2D, p2: Point2D) -> Self {
        Self::new(vec![p0, p1, p2])
    }
    
    pub fn cubic(p0: Point2D, p1: Point2D, p2: Point2D, p3: Point2D) -> Self {
        Self::new(vec![p0, p1, p2, p3])
    }
    
    pub fn evaluate(&self, t: f64) -> MathResult<Point2D> {
        if t < 0.0 || t > 1.0 {
            return Err(MathError::InvalidArgument("Bezier parameter t must be between 0 and 1".to_string()));
        }
        
        if self.control_points.is_empty() {
            return Err(MathError::InvalidArgument("Bezier curve must have at least one control point".to_string()));
        }
        
        if self.control_points.len() == 1 {
            return Ok(self.control_points[0].clone());
        }
        
        // De Casteljau's algorithm
        let mut points = self.control_points.clone();
        let n = points.len();
        
        for layer in 0..(n - 1) {
            for i in 0..(n - layer - 1) {
                points[i] = Point2D {
                    x: (1.0 - t) * points[i].x + t * points[i + 1].x,
                    y: (1.0 - t) * points[i].y + t * points[i + 1].y,
                };
            }
        }
        
        Ok(points[0].clone())
    }
    
    pub fn derivative(&self, t: f64) -> MathResult<Vector2D> {
        if t < 0.0 || t > 1.0 {
            return Err(MathError::InvalidArgument("Parameter t must be between 0 and 1".to_string()));
        }
        
        if self.control_points.len() < 2 {
            return Ok(Vector2D { x: 0.0, y: 0.0 });
        }
        
        // Create derivative curve (one degree lower)
        let mut derivative_points = Vec::new();
        let n = self.control_points.len() - 1;
        
        for i in 0..n {
            derivative_points.push(Point2D {
                x: n as f64 * (self.control_points[i + 1].x - self.control_points[i].x),
                y: n as f64 * (self.control_points[i + 1].y - self.control_points[i].y),
            });
        }
        
        let derivative_curve = BezierCurve2D::new(derivative_points);
        let derivative_point = derivative_curve.evaluate(t)?;
        
        Ok(Vector2D {
            x: derivative_point.x,
            y: derivative_point.y,
        })
    }
    
    pub fn length(&self, samples: usize) -> f64 {
        if samples < 2 {
            return 0.0;
        }
        
        let mut total_length = 0.0;
        let mut prev_point = match self.evaluate(0.0) {
            Ok(p) => p,
            Err(_) => return 0.0,
        };
        
        for i in 1..=samples {
            let t = i as f64 / samples as f64;
            let current_point = match self.evaluate(t) {
                Ok(p) => p,
                Err(_) => continue,
            };
            
            let dx = current_point.x - prev_point.x;
            let dy = current_point.y - prev_point.y;
            total_length += (dx * dx + dy * dy).sqrt();
            
            prev_point = current_point;
        }
        
        total_length
    }
}

impl BezierCurve3D {
    pub fn new(control_points: Vec<Point3D>) -> Self {
        Self { control_points }
    }
    
    pub fn linear(p0: Point3D, p1: Point3D) -> Self {
        Self::new(vec![p0, p1])
    }
    
    pub fn quadratic(p0: Point3D, p1: Point3D, p2: Point3D) -> Self {
        Self::new(vec![p0, p1, p2])
    }
    
    pub fn cubic(p0: Point3D, p1: Point3D, p2: Point3D, p3: Point3D) -> Self {
        Self::new(vec![p0, p1, p2, p3])
    }
    
    pub fn evaluate(&self, t: f64) -> MathResult<Point3D> {
        if t < 0.0 || t > 1.0 {
            return Err(MathError::InvalidArgument("Bezier parameter t must be between 0 and 1".to_string()));
        }
        
        if self.control_points.is_empty() {
            return Err(MathError::InvalidArgument("Bezier curve must have at least one control point".to_string()));
        }
        
        if self.control_points.len() == 1 {
            return Ok(self.control_points[0].clone());
        }
        
        // De Casteljau's algorithm for 3D
        let mut points = self.control_points.clone();
        let n = points.len();
        
        for layer in 0..(n - 1) {
            for i in 0..(n - layer - 1) {
                points[i] = Point3D {
                    x: (1.0 - t) * points[i].x + t * points[i + 1].x,
                    y: (1.0 - t) * points[i].y + t * points[i + 1].y,
                    z: (1.0 - t) * points[i].z + t * points[i + 1].z,
                };
            }
        }
        
        Ok(points[0].clone())
    }
    
    pub fn derivative(&self, t: f64) -> MathResult<Vector3D> {
        if t < 0.0 || t > 1.0 {
            return Err(MathError::InvalidArgument("Parameter t must be between 0 and 1".to_string()));
        }
        
        if self.control_points.len() < 2 {
            return Ok(Vector3D { x: 0.0, y: 0.0, z: 0.0 });
        }
        
        let mut derivative_points = Vec::new();
        let n = self.control_points.len() - 1;
        
        for i in 0..n {
            derivative_points.push(Point3D {
                x: n as f64 * (self.control_points[i + 1].x - self.control_points[i].x),
                y: n as f64 * (self.control_points[i + 1].y - self.control_points[i].y),
                z: n as f64 * (self.control_points[i + 1].z - self.control_points[i].z),
            });
        }
        
        let derivative_curve = BezierCurve3D::new(derivative_points);
        let derivative_point = derivative_curve.evaluate(t)?;
        
        Ok(Vector3D {
            x: derivative_point.x,
            y: derivative_point.y,
            z: derivative_point.z,
        })
    }
}

impl CatmullRomSpline2D {
    pub fn new(points: Vec<Point2D>, tension: f64) -> Self {
        Self { points, tension }
    }
    
    pub fn evaluate(&self, t: f64) -> MathResult<Point2D> {
        if self.points.len() < 4 {
            return Err(MathError::InvalidArgument("Catmull-Rom spline requires at least 4 points".to_string()));
        }
        
        let segment_count = self.points.len() - 3;
        let segment_t = t * segment_count as f64;
        let segment_index = segment_t.floor() as usize;
        let local_t = segment_t - segment_index as f64;
        
        if segment_index >= segment_count {
            return Ok(self.points[self.points.len() - 2].clone());
        }
        
        let p0 = &self.points[segment_index];
        let p1 = &self.points[segment_index + 1];
        let p2 = &self.points[segment_index + 2];
        let p3 = &self.points[segment_index + 3];
        
        let t2 = local_t * local_t;
        let t3 = t2 * local_t;
        
        let tension = self.tension;
        
        let x = tension * (
            (-t3 + 2.0 * t2 - local_t) * p0.x +
            (3.0 * t3 - 5.0 * t2 + 2.0) * p1.x +
            (-3.0 * t3 + 4.0 * t2 + local_t) * p2.x +
            (t3 - t2) * p3.x
        ) * 0.5;
        
        let y = tension * (
            (-t3 + 2.0 * t2 - local_t) * p0.y +
            (3.0 * t3 - 5.0 * t2 + 2.0) * p1.y +
            (-3.0 * t3 + 4.0 * t2 + local_t) * p2.y +
            (t3 - t2) * p3.y
        ) * 0.5;
        
        Ok(Point2D { x, y })
    }
}

impl<T: Clone> AnimationCurve<T> {
    pub fn new(keyframes: Vec<Keyframe<T>>, duration: f64) -> Self {
        Self { keyframes, duration }
    }
    
    pub fn add_keyframe(&mut self, keyframe: Keyframe<T>) {
        self.keyframes.push(keyframe);
        self.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }
    
    pub fn get_keyframe_indices_at_time(&self, time: f64) -> (Option<usize>, Option<usize>) {
        if self.keyframes.is_empty() {
            return (None, None);
        }
        
        // Find keyframes surrounding the time
        let mut prev_index = None;
        let mut next_index = None;
        
        for (i, keyframe) in self.keyframes.iter().enumerate() {
            if keyframe.time <= time {
                prev_index = Some(i);
            }
            if keyframe.time >= time && next_index.is_none() {
                next_index = Some(i);
                break;
            }
        }
        
        (prev_index, next_index)
    }
}

impl AnimationCurve<f64> {
    pub fn evaluate(&self, time: f64) -> MathResult<f64> {
        if self.keyframes.is_empty() {
            return Err(MathError::InvalidArgument("Animation curve has no keyframes".to_string()));
        }
        
        let clamped_time = time.clamp(0.0, self.duration);
        
        let (prev_idx, next_idx) = self.get_keyframe_indices_at_time(clamped_time);
        
        match (prev_idx, next_idx) {
            (Some(prev), Some(next)) if prev == next => {
                // Exact keyframe match
                Ok(self.keyframes[prev].value)
            },
            (Some(prev), Some(next)) => {
                // Interpolate between keyframes
                let prev_kf = &self.keyframes[prev];
                let next_kf = &self.keyframes[next];
                
                let time_range = next_kf.time - prev_kf.time;
                if time_range == 0.0 {
                    return Ok(prev_kf.value);
                }
                
                let local_t = (clamped_time - prev_kf.time) / time_range;
                let eased_t = next_kf.easing.evaluate(local_t)?;
                
                Ok(prev_kf.value + eased_t * (next_kf.value - prev_kf.value))
            },
            (Some(prev), None) => {
                // After last keyframe
                Ok(self.keyframes[prev].value)
            },
            (None, Some(next)) => {
                // Before first keyframe
                Ok(self.keyframes[next].value)
            },
            (None, None) => {
                // Should not happen if keyframes is not empty
                Err(MathError::ComputationError("No valid keyframes found".to_string()))
            }
        }
    }
}

impl AnimationDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn ease_in_out_cubic(t: f64) -> MathResult<f64> {
        EasingFunction::CubicInOut.evaluate(t)
    }
    
    pub fn ease_bounce_out(t: f64) -> MathResult<f64> {
        EasingFunction::BounceOut.evaluate(t)
    }
    
    pub fn ease_elastic_out(t: f64) -> MathResult<f64> {
        EasingFunction::ElasticOut.evaluate(t)
    }
    
    pub fn ease_back_in_out(t: f64) -> MathResult<f64> {
        EasingFunction::BackInOut.evaluate(t)
    }
    
    pub fn interpolate_with_easing(start: f64, end: f64, t: f64, easing: &EasingFunction) -> MathResult<f64> {
        if t < 0.0 || t > 1.0 {
            return Err(MathError::InvalidArgument("Parameter t must be between 0 and 1".to_string()));
        }
        
        let eased_t = easing.evaluate(t)?;
        Ok(start + eased_t * (end - start))
    }
    
    pub fn interpolate_point2d_with_easing(start: &Point2D, end: &Point2D, t: f64, easing: &EasingFunction) -> MathResult<Point2D> {
        let eased_t = easing.evaluate(t)?;
        
        Ok(Point2D {
            x: start.x + eased_t * (end.x - start.x),
            y: start.y + eased_t * (end.y - start.y),
        })
    }
    
    pub fn interpolate_point3d_with_easing(start: &Point3D, end: &Point3D, t: f64, easing: &EasingFunction) -> MathResult<Point3D> {
        let eased_t = easing.evaluate(t)?;
        
        Ok(Point3D {
            x: start.x + eased_t * (end.x - start.x),
            y: start.y + eased_t * (end.y - start.y),
            z: start.z + eased_t * (end.z - start.z),
        })
    }
    
    pub fn time_to_normalized(current_time: f64, start_time: f64, duration: f64) -> MathResult<f64> {
        if duration <= 0.0 {
            return Err(MathError::InvalidArgument("Duration must be positive".to_string()));
        }
        
        let elapsed = current_time - start_time;
        Ok((elapsed / duration).clamp(0.0, 1.0))
    }
    
    pub fn ping_pong_time(time: f64, duration: f64) -> MathResult<f64> {
        if duration <= 0.0 {
            return Err(MathError::InvalidArgument("Duration must be positive".to_string()));
        }
        
        let cycle_time = time % (2.0 * duration);
        if cycle_time <= duration {
            Ok(cycle_time / duration)
        } else {
            Ok((2.0 * duration - cycle_time) / duration)
        }
    }
    
    pub fn loop_time(time: f64, duration: f64) -> MathResult<f64> {
        if duration <= 0.0 {
            return Err(MathError::InvalidArgument("Duration must be positive".to_string()));
        }
        
        Ok((time % duration) / duration)
    }
}

impl MathDomain for AnimationDomain {
    fn name(&self) -> &str { "Animation and Easing" }
    fn description(&self) -> &str { "Animation curves, easing functions, Bezier curves, and timeline systems" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "ease_cubic_in_out" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("ease_cubic_in_out requires 1 argument".to_string()));
                }
                let t = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Argument must be f64".to_string()))?;
                Ok(Box::new(Self::ease_in_out_cubic(*t)?))
            },
            "ease_bounce_out" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("ease_bounce_out requires 1 argument".to_string()));
                }
                let t = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Argument must be f64".to_string()))?;
                Ok(Box::new(Self::ease_bounce_out(*t)?))
            },
            "bezier_2d_evaluate" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("bezier_2d_evaluate requires 2 arguments".to_string()));
                }
                let curve = args[0].downcast_ref::<BezierCurve2D>().ok_or_else(|| MathError::InvalidArgument("First argument must be BezierCurve2D".to_string()))?;
                let t = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                Ok(Box::new(curve.evaluate(*t)?))
            },
            "bezier_3d_evaluate" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("bezier_3d_evaluate requires 2 arguments".to_string()));
                }
                let curve = args[0].downcast_ref::<BezierCurve3D>().ok_or_else(|| MathError::InvalidArgument("First argument must be BezierCurve3D".to_string()))?;
                let t = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                Ok(Box::new(curve.evaluate(*t)?))
            },
            "animation_curve_evaluate" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("animation_curve_evaluate requires 2 arguments".to_string()));
                }
                let curve = args[0].downcast_ref::<AnimationCurve<f64>>().ok_or_else(|| MathError::InvalidArgument("First argument must be AnimationCurve<f64>".to_string()))?;
                let time = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                Ok(Box::new(curve.evaluate(*time)?))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "ease_linear".to_string(),
            "ease_quadratic_in".to_string(),
            "ease_quadratic_out".to_string(),
            "ease_quadratic_in_out".to_string(),
            "ease_cubic_in".to_string(),
            "ease_cubic_out".to_string(),
            "ease_cubic_in_out".to_string(),
            "ease_sine_in".to_string(),
            "ease_sine_out".to_string(),
            "ease_sine_in_out".to_string(),
            "ease_exponential_in".to_string(),
            "ease_exponential_out".to_string(),
            "ease_exponential_in_out".to_string(),
            "ease_elastic_in".to_string(),
            "ease_elastic_out".to_string(),
            "ease_elastic_in_out".to_string(),
            "ease_bounce_in".to_string(),
            "ease_bounce_out".to_string(),
            "ease_bounce_in_out".to_string(),
            "ease_back_in".to_string(),
            "ease_back_out".to_string(),
            "ease_back_in_out".to_string(),
            "bezier_2d_evaluate".to_string(),
            "bezier_2d_derivative".to_string(),
            "bezier_3d_evaluate".to_string(),
            "bezier_3d_derivative".to_string(),
            "catmull_rom_2d_evaluate".to_string(),
            "animation_curve_evaluate".to_string(),
            "interpolate_with_easing".to_string(),
            "time_to_normalized".to_string(),
            "ping_pong_time".to_string(),
            "loop_time".to_string(),
        ]
    }
}