use crate::core::{MathDomain, MathResult, MathError, Point2D, Point3D, Vector2D, Vector3D};
use crate::utils::PI;
use std::any::Any;

pub struct GeometryDomain;

impl GeometryDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn distance_2d(p1: &Point2D, p2: &Point2D) -> f64 {
        ((p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2)).sqrt()
    }
    
    pub fn distance_3d(p1: &Point3D, p2: &Point3D) -> f64 {
        ((p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2) + (p2.z - p1.z).powi(2)).sqrt()
    }
    
    pub fn vector_magnitude_2d(v: &Vector2D) -> f64 {
        (v.x.powi(2) + v.y.powi(2)).sqrt()
    }
    
    pub fn vector_magnitude_3d(v: &Vector3D) -> f64 {
        (v.x.powi(2) + v.y.powi(2) + v.z.powi(2)).sqrt()
    }
    
    pub fn dot_product_2d(v1: &Vector2D, v2: &Vector2D) -> f64 {
        v1.x * v2.x + v1.y * v2.y
    }
    
    pub fn dot_product_3d(v1: &Vector3D, v2: &Vector3D) -> f64 {
        v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
    }
    
    pub fn cross_product_3d(v1: &Vector3D, v2: &Vector3D) -> Vector3D {
        Vector3D {
            x: v1.y * v2.z - v1.z * v2.y,
            y: v1.z * v2.x - v1.x * v2.z,
            z: v1.x * v2.y - v1.y * v2.x,
        }
    }
    
    pub fn angle_between_vectors_2d(v1: &Vector2D, v2: &Vector2D) -> f64 {
        let dot = Self::dot_product_2d(v1, v2);
        let mag1 = Self::vector_magnitude_2d(v1);
        let mag2 = Self::vector_magnitude_2d(v2);
        
        if mag1 == 0.0 || mag2 == 0.0 { return 0.0; }
        
        (dot / (mag1 * mag2)).acos()
    }
    
    pub fn circle_area(radius: f64) -> MathResult<f64> {
        if radius < 0.0 {
            return Err(MathError::InvalidArgument("Radius cannot be negative".to_string()));
        }
        Ok(PI * radius.powi(2))
    }
    
    pub fn circle_circumference(radius: f64) -> MathResult<f64> {
        if radius < 0.0 {
            return Err(MathError::InvalidArgument("Radius cannot be negative".to_string()));
        }
        Ok(2.0 * PI * radius)
    }
    
    pub fn sphere_volume(radius: f64) -> MathResult<f64> {
        if radius < 0.0 {
            return Err(MathError::InvalidArgument("Radius cannot be negative".to_string()));
        }
        Ok(4.0 / 3.0 * PI * radius.powi(3))
    }
    
    pub fn sphere_surface_area(radius: f64) -> MathResult<f64> {
        if radius < 0.0 {
            return Err(MathError::InvalidArgument("Radius cannot be negative".to_string()));
        }
        Ok(4.0 * PI * radius.powi(2))
    }
    
    pub fn triangle_area(base: f64, height: f64) -> MathResult<f64> {
        if base < 0.0 || height < 0.0 {
            return Err(MathError::InvalidArgument("Base and height must be non-negative".to_string()));
        }
        Ok(0.5 * base * height)
    }
    
    pub fn triangle_area_heron(a: f64, b: f64, c: f64) -> MathResult<f64> {
        if a <= 0.0 || b <= 0.0 || c <= 0.0 {
            return Err(MathError::InvalidArgument("All sides must be positive".to_string()));
        }
        
        if a + b <= c || a + c <= b || b + c <= a {
            return Err(MathError::InvalidArgument("Invalid triangle: triangle inequality violated".to_string()));
        }
        
        let s = (a + b + c) / 2.0;
        Ok((s * (s - a) * (s - b) * (s - c)).sqrt())
    }
    
    pub fn rectangle_area(width: f64, height: f64) -> MathResult<f64> {
        if width < 0.0 || height < 0.0 {
            return Err(MathError::InvalidArgument("Width and height must be non-negative".to_string()));
        }
        Ok(width * height)
    }
    
    pub fn rectangle_perimeter(width: f64, height: f64) -> MathResult<f64> {
        if width < 0.0 || height < 0.0 {
            return Err(MathError::InvalidArgument("Width and height must be non-negative".to_string()));
        }
        Ok(2.0 * (width + height))
    }
}

impl MathDomain for GeometryDomain {
    fn name(&self) -> &str { "Geometry" }
    fn description(&self) -> &str { "Mathematical domain for geometric calculations and spatial relationships" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "distance_2d" => {
                if args.len() != 2 { 
                    return Err(MathError::InvalidArgument("distance_2d requires 2 arguments".to_string())); 
                }
                let p1 = args[0].downcast_ref::<Point2D>().ok_or_else(|| MathError::InvalidArgument("First argument must be Point2D".to_string()))?;
                let p2 = args[1].downcast_ref::<Point2D>().ok_or_else(|| MathError::InvalidArgument("Second argument must be Point2D".to_string()))?;
                Ok(Box::new(Self::distance_2d(p1, p2)))
            },
            "distance_3d" => {
                if args.len() != 2 { 
                    return Err(MathError::InvalidArgument("distance_3d requires 2 arguments".to_string())); 
                }
                let p1 = args[0].downcast_ref::<Point3D>().ok_or_else(|| MathError::InvalidArgument("First argument must be Point3D".to_string()))?;
                let p2 = args[1].downcast_ref::<Point3D>().ok_or_else(|| MathError::InvalidArgument("Second argument must be Point3D".to_string()))?;
                Ok(Box::new(Self::distance_3d(p1, p2)))
            },
            "circle_area" => {
                if args.len() != 1 { 
                    return Err(MathError::InvalidArgument("circle_area requires 1 argument".to_string())); 
                }
                let radius = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Argument must be f64".to_string()))?;
                Ok(Box::new(Self::circle_area(*radius)?))
            },
            "circle_circumference" => {
                if args.len() != 1 { 
                    return Err(MathError::InvalidArgument("circle_circumference requires 1 argument".to_string())); 
                }
                let radius = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Argument must be f64".to_string()))?;
                Ok(Box::new(Self::circle_circumference(*radius)?))
            },
            "sphere_volume" => {
                if args.len() != 1 { 
                    return Err(MathError::InvalidArgument("sphere_volume requires 1 argument".to_string())); 
                }
                let radius = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Argument must be f64".to_string()))?;
                Ok(Box::new(Self::sphere_volume(*radius)?))
            },
            "triangle_area_heron" => {
                if args.len() != 3 { 
                    return Err(MathError::InvalidArgument("triangle_area_heron requires 3 arguments".to_string())); 
                }
                let a = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let b = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                let c = args[2].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be f64".to_string()))?;
                Ok(Box::new(Self::triangle_area_heron(*a, *b, *c)?))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "distance_2d".to_string(),
            "distance_3d".to_string(),
            "vector_magnitude_2d".to_string(),
            "vector_magnitude_3d".to_string(),
            "dot_product_2d".to_string(),
            "dot_product_3d".to_string(),
            "cross_product_3d".to_string(),
            "angle_between_vectors_2d".to_string(),
            "circle_area".to_string(),
            "circle_circumference".to_string(),
            "sphere_volume".to_string(),
            "sphere_surface_area".to_string(),
            "triangle_area".to_string(),
            "triangle_area_heron".to_string(),
            "rectangle_area".to_string(),
            "rectangle_perimeter".to_string(),
        ]
    }
}