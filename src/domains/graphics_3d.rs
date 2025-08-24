use crate::core::{MathDomain, MathResult, MathError, Point3D, Vector3D, Quaternion, Transform4x4, Ray3D, AABB, BoundingSphere};
use std::any::Any;

pub struct Graphics3DDomain;

impl Quaternion {
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }
    
    pub fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0)
    }
    
    pub fn from_axis_angle(axis: &Vector3D, angle: f64) -> Self {
        let half_angle = angle * 0.5;
        let sin_half = half_angle.sin();
        let cos_half = half_angle.cos();
        
        let magnitude = (axis.x * axis.x + axis.y * axis.y + axis.z * axis.z).sqrt();
        if magnitude == 0.0 {
            return Self::identity();
        }
        
        let normalized_x = axis.x / magnitude;
        let normalized_y = axis.y / magnitude;
        let normalized_z = axis.z / magnitude;
        
        Self::new(
            cos_half,
            normalized_x * sin_half,
            normalized_y * sin_half,
            normalized_z * sin_half,
        )
    }
    
    pub fn from_euler_angles(roll: f64, pitch: f64, yaw: f64) -> Self {
        let (sr, cr) = (roll * 0.5).sin_cos();
        let (sp, cp) = (pitch * 0.5).sin_cos();
        let (sy, cy) = (yaw * 0.5).sin_cos();
        
        Self::new(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    }
    
    pub fn magnitude(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    
    pub fn normalize(&self) -> MathResult<Self> {
        let mag = self.magnitude();
        if mag == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        
        Ok(Self::new(
            self.w / mag,
            self.x / mag,
            self.y / mag,
            self.z / mag,
        ))
    }
    
    pub fn conjugate(&self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }
    
    pub fn multiply(&self, other: &Self) -> Self {
        Self::new(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )
    }
    
    pub fn rotate_point(&self, point: &Point3D) -> Point3D {
        let q_point = Quaternion::new(0.0, point.x, point.y, point.z);
        let rotated = self.multiply(&q_point).multiply(&self.conjugate());
        Point3D {
            x: rotated.x,
            y: rotated.y,
            z: rotated.z,
        }
    }
    
    pub fn slerp(&self, other: &Self, t: f64) -> MathResult<Self> {
        if t < 0.0 || t > 1.0 {
            return Err(MathError::InvalidArgument("t must be between 0 and 1".to_string()));
        }
        
        if t == 0.0 {
            return Ok(self.clone());
        }
        if t == 1.0 {
            return Ok(other.clone());
        }
        
        let dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;
        let abs_dot = dot.abs();
        
        if abs_dot >= 1.0 {
            return Ok(self.clone());
        }
        
        let theta = abs_dot.acos();
        let sin_theta = theta.sin();
        
        if sin_theta.abs() < 1e-10 {
            let lerp_w = self.w * (1.0 - t) + other.w * t;
            let lerp_x = self.x * (1.0 - t) + other.x * t;
            let lerp_y = self.y * (1.0 - t) + other.y * t;
            let lerp_z = self.z * (1.0 - t) + other.z * t;
            return Ok(Self::new(lerp_w, lerp_x, lerp_y, lerp_z));
        }
        
        let scale_a = ((1.0 - t) * theta).sin() / sin_theta;
        let scale_b = (t * theta).sin() / sin_theta;
        
        if dot < 0.0 {
            Ok(Self::new(
                self.w * scale_a - other.w * scale_b,
                self.x * scale_a - other.x * scale_b,
                self.y * scale_a - other.y * scale_b,
                self.z * scale_a - other.z * scale_b,
            ))
        } else {
            Ok(Self::new(
                self.w * scale_a + other.w * scale_b,
                self.x * scale_a + other.x * scale_b,
                self.y * scale_a + other.y * scale_b,
                self.z * scale_a + other.z * scale_b,
            ))
        }
    }
}

impl Transform4x4 {
    pub fn identity() -> Self {
        Self {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
    
    pub fn translation(x: f64, y: f64, z: f64) -> Self {
        Self {
            matrix: [
                [1.0, 0.0, 0.0, x],
                [0.0, 1.0, 0.0, y],
                [0.0, 0.0, 1.0, z],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
    
    pub fn scaling(x: f64, y: f64, z: f64) -> Self {
        Self {
            matrix: [
                [x, 0.0, 0.0, 0.0],
                [0.0, y, 0.0, 0.0],
                [0.0, 0.0, z, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
    
    pub fn rotation_x(angle: f64) -> Self {
        let (sin_a, cos_a) = angle.sin_cos();
        Self {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos_a, -sin_a, 0.0],
                [0.0, sin_a, cos_a, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
    
    pub fn rotation_y(angle: f64) -> Self {
        let (sin_a, cos_a) = angle.sin_cos();
        Self {
            matrix: [
                [cos_a, 0.0, sin_a, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-sin_a, 0.0, cos_a, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
    
    pub fn rotation_z(angle: f64) -> Self {
        let (sin_a, cos_a) = angle.sin_cos();
        Self {
            matrix: [
                [cos_a, -sin_a, 0.0, 0.0],
                [sin_a, cos_a, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
    
    pub fn from_quaternion(q: &Quaternion) -> Self {
        let norm = q.normalize().unwrap_or_else(|_| Quaternion::identity());
        let (w, x, y, z) = (norm.w, norm.x, norm.y, norm.z);
        
        Self {
            matrix: [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w), 0.0],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w), 0.0],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
    
    pub fn perspective(fov_y: f64, aspect: f64, near: f64, far: f64) -> MathResult<Self> {
        if near <= 0.0 || far <= near || aspect <= 0.0 || fov_y <= 0.0 {
            return Err(MathError::InvalidArgument("Invalid perspective parameters".to_string()));
        }
        
        let f = 1.0 / (fov_y * 0.5).tan();
        let z_range = far - near;
        
        Ok(Self {
            matrix: [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, -(far + near) / z_range, -2.0 * far * near / z_range],
                [0.0, 0.0, -1.0, 0.0],
            ],
        })
    }
    
    pub fn orthographic(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> MathResult<Self> {
        if left >= right || bottom >= top || near >= far {
            return Err(MathError::InvalidArgument("Invalid orthographic parameters".to_string()));
        }
        
        let width = right - left;
        let height = top - bottom;
        let depth = far - near;
        
        Ok(Self {
            matrix: [
                [2.0 / width, 0.0, 0.0, -(right + left) / width],
                [0.0, 2.0 / height, 0.0, -(top + bottom) / height],
                [0.0, 0.0, -2.0 / depth, -(far + near) / depth],
                [0.0, 0.0, 0.0, 1.0],
            ],
        })
    }
    
    pub fn multiply(&self, other: &Self) -> Self {
        let mut result = Self::identity();
        
        for i in 0..4 {
            for j in 0..4 {
                result.matrix[i][j] = 0.0;
                for k in 0..4 {
                    result.matrix[i][j] += self.matrix[i][k] * other.matrix[k][j];
                }
            }
        }
        
        result
    }
    
    pub fn transform_point(&self, point: &Point3D) -> Point3D {
        let x = self.matrix[0][0] * point.x + self.matrix[0][1] * point.y + self.matrix[0][2] * point.z + self.matrix[0][3];
        let y = self.matrix[1][0] * point.x + self.matrix[1][1] * point.y + self.matrix[1][2] * point.z + self.matrix[1][3];
        let z = self.matrix[2][0] * point.x + self.matrix[2][1] * point.y + self.matrix[2][2] * point.z + self.matrix[2][3];
        let w = self.matrix[3][0] * point.x + self.matrix[3][1] * point.y + self.matrix[3][2] * point.z + self.matrix[3][3];
        
        if w != 0.0 {
            Point3D { x: x / w, y: y / w, z: z / w }
        } else {
            Point3D { x, y, z }
        }
    }
    
    pub fn transform_vector(&self, vector: &Vector3D) -> Vector3D {
        Vector3D {
            x: self.matrix[0][0] * vector.x + self.matrix[0][1] * vector.y + self.matrix[0][2] * vector.z,
            y: self.matrix[1][0] * vector.x + self.matrix[1][1] * vector.y + self.matrix[1][2] * vector.z,
            z: self.matrix[2][0] * vector.x + self.matrix[2][1] * vector.y + self.matrix[2][2] * vector.z,
        }
    }
}

impl Ray3D {
    pub fn new(origin: Point3D, direction: Vector3D) -> Self {
        Self { origin, direction }
    }
    
    pub fn point_at_parameter(&self, t: f64) -> Point3D {
        Point3D {
            x: self.origin.x + t * self.direction.x,
            y: self.origin.y + t * self.direction.y,
            z: self.origin.z + t * self.direction.z,
        }
    }
    
    pub fn intersect_plane(&self, plane_point: &Point3D, plane_normal: &Vector3D) -> Option<f64> {
        let normal_mag = (plane_normal.x * plane_normal.x + 
                         plane_normal.y * plane_normal.y + 
                         plane_normal.z * plane_normal.z).sqrt();
        
        if normal_mag == 0.0 {
            return None;
        }
        
        let norm_x = plane_normal.x / normal_mag;
        let norm_y = plane_normal.y / normal_mag;
        let norm_z = plane_normal.z / normal_mag;
        
        let denominator = self.direction.x * norm_x + 
                         self.direction.y * norm_y + 
                         self.direction.z * norm_z;
        
        if denominator.abs() < 1e-10 {
            return None;
        }
        
        let numerator = (plane_point.x - self.origin.x) * norm_x +
                       (plane_point.y - self.origin.y) * norm_y +
                       (plane_point.z - self.origin.z) * norm_z;
        
        Some(numerator / denominator)
    }
    
    pub fn intersect_sphere(&self, sphere: &BoundingSphere) -> Option<(f64, f64)> {
        let oc_x = self.origin.x - sphere.center.x;
        let oc_y = self.origin.y - sphere.center.y;
        let oc_z = self.origin.z - sphere.center.z;
        
        let a = self.direction.x * self.direction.x + 
                self.direction.y * self.direction.y + 
                self.direction.z * self.direction.z;
        
        let b = 2.0 * (oc_x * self.direction.x + 
                      oc_y * self.direction.y + 
                      oc_z * self.direction.z);
        
        let c = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z - sphere.radius * sphere.radius;
        
        let discriminant = b * b - 4.0 * a * c;
        
        if discriminant < 0.0 {
            return None;
        }
        
        let sqrt_discriminant = discriminant.sqrt();
        let t1 = (-b - sqrt_discriminant) / (2.0 * a);
        let t2 = (-b + sqrt_discriminant) / (2.0 * a);
        
        Some((t1, t2))
    }
    
    pub fn intersect_aabb(&self, aabb: &AABB) -> Option<(f64, f64)> {
        let mut t_min = (aabb.min.x - self.origin.x) / self.direction.x;
        let mut t_max = (aabb.max.x - self.origin.x) / self.direction.x;
        
        if t_min > t_max {
            std::mem::swap(&mut t_min, &mut t_max);
        }
        
        let ty_min = (aabb.min.y - self.origin.y) / self.direction.y;
        let ty_max = (aabb.max.y - self.origin.y) / self.direction.y;
        
        let (ty_min, ty_max) = if ty_min > ty_max { (ty_max, ty_min) } else { (ty_min, ty_max) };
        
        if t_min > ty_max || ty_min > t_max {
            return None;
        }
        
        t_min = t_min.max(ty_min);
        t_max = t_max.min(ty_max);
        
        let tz_min = (aabb.min.z - self.origin.z) / self.direction.z;
        let tz_max = (aabb.max.z - self.origin.z) / self.direction.z;
        
        let (tz_min, tz_max) = if tz_min > tz_max { (tz_max, tz_min) } else { (tz_min, tz_max) };
        
        if t_min > tz_max || tz_min > t_max {
            return None;
        }
        
        t_min = t_min.max(tz_min);
        t_max = t_max.min(tz_max);
        
        Some((t_min, t_max))
    }
}

impl AABB {
    pub fn new(min: Point3D, max: Point3D) -> Self {
        Self { min, max }
    }
    
    pub fn from_center_extents(center: &Point3D, extents: &Vector3D) -> Self {
        Self {
            min: Point3D {
                x: center.x - extents.x,
                y: center.y - extents.y,
                z: center.z - extents.z,
            },
            max: Point3D {
                x: center.x + extents.x,
                y: center.y + extents.y,
                z: center.z + extents.z,
            },
        }
    }
    
    pub fn contains_point(&self, point: &Point3D) -> bool {
        point.x >= self.min.x && point.x <= self.max.x &&
        point.y >= self.min.y && point.y <= self.max.y &&
        point.z >= self.min.z && point.z <= self.max.z
    }
    
    pub fn intersects_aabb(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x &&
        self.min.y <= other.max.y && self.max.y >= other.min.y &&
        self.min.z <= other.max.z && self.max.z >= other.min.z
    }
    
    pub fn center(&self) -> Point3D {
        Point3D {
            x: (self.min.x + self.max.x) * 0.5,
            y: (self.min.y + self.max.y) * 0.5,
            z: (self.min.z + self.max.z) * 0.5,
        }
    }
    
    pub fn extents(&self) -> Vector3D {
        Vector3D {
            x: (self.max.x - self.min.x) * 0.5,
            y: (self.max.y - self.min.y) * 0.5,
            z: (self.max.z - self.min.z) * 0.5,
        }
    }
    
    pub fn volume(&self) -> f64 {
        let width = self.max.x - self.min.x;
        let height = self.max.y - self.min.y;
        let depth = self.max.z - self.min.z;
        width * height * depth
    }
}

impl BoundingSphere {
    pub fn new(center: Point3D, radius: f64) -> Self {
        Self { center, radius }
    }
    
    pub fn contains_point(&self, point: &Point3D) -> bool {
        let dx = point.x - self.center.x;
        let dy = point.y - self.center.y;
        let dz = point.z - self.center.z;
        let distance_squared = dx * dx + dy * dy + dz * dz;
        distance_squared <= self.radius * self.radius
    }
    
    pub fn intersects_sphere(&self, other: &BoundingSphere) -> bool {
        let dx = self.center.x - other.center.x;
        let dy = self.center.y - other.center.y;
        let dz = self.center.z - other.center.z;
        let distance_squared = dx * dx + dy * dy + dz * dz;
        let radius_sum = self.radius + other.radius;
        distance_squared <= radius_sum * radius_sum
    }
    
    pub fn volume(&self) -> f64 {
        4.0 / 3.0 * std::f64::consts::PI * self.radius.powi(3)
    }
}

impl Graphics3DDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn lerp(a: f64, b: f64, t: f64) -> MathResult<f64> {
        if t < 0.0 || t > 1.0 {
            return Err(MathError::InvalidArgument("t must be between 0 and 1".to_string()));
        }
        Ok(a + t * (b - a))
    }
    
    pub fn lerp_vector3d(a: &Vector3D, b: &Vector3D, t: f64) -> MathResult<Vector3D> {
        if t < 0.0 || t > 1.0 {
            return Err(MathError::InvalidArgument("t must be between 0 and 1".to_string()));
        }
        
        Ok(Vector3D {
            x: a.x + t * (b.x - a.x),
            y: a.y + t * (b.y - a.y),
            z: a.z + t * (b.z - a.z),
        })
    }
    
    pub fn lerp_point3d(a: &Point3D, b: &Point3D, t: f64) -> MathResult<Point3D> {
        if t < 0.0 || t > 1.0 {
            return Err(MathError::InvalidArgument("t must be between 0 and 1".to_string()));
        }
        
        Ok(Point3D {
            x: a.x + t * (b.x - a.x),
            y: a.y + t * (b.y - a.y),
            z: a.z + t * (b.z - a.z),
        })
    }
}

impl MathDomain for Graphics3DDomain {
    fn name(&self) -> &str { "3D Graphics and Animation" }
    fn description(&self) -> &str { "Advanced 3D mathematics including quaternions, transformations, raycasting, and animation" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "quaternion_slerp" => {
                if args.len() != 3 {
                    return Err(MathError::InvalidArgument("quaternion_slerp requires 3 arguments".to_string()));
                }
                let q1 = args[0].downcast_ref::<Quaternion>().ok_or_else(|| MathError::InvalidArgument("First argument must be Quaternion".to_string()))?;
                let q2 = args[1].downcast_ref::<Quaternion>().ok_or_else(|| MathError::InvalidArgument("Second argument must be Quaternion".to_string()))?;
                let t = args[2].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be f64".to_string()))?;
                Ok(Box::new(q1.slerp(q2, *t)?))
            },
            "lerp_vector3d" => {
                if args.len() != 3 {
                    return Err(MathError::InvalidArgument("lerp_vector3d requires 3 arguments".to_string()));
                }
                let v1 = args[0].downcast_ref::<Vector3D>().ok_or_else(|| MathError::InvalidArgument("First argument must be Vector3D".to_string()))?;
                let v2 = args[1].downcast_ref::<Vector3D>().ok_or_else(|| MathError::InvalidArgument("Second argument must be Vector3D".to_string()))?;
                let t = args[2].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be f64".to_string()))?;
                Ok(Box::new(Self::lerp_vector3d(v1, v2, *t)?))
            },
            "lerp_point3d" => {
                if args.len() != 3 {
                    return Err(MathError::InvalidArgument("lerp_point3d requires 3 arguments".to_string()));
                }
                let p1 = args[0].downcast_ref::<Point3D>().ok_or_else(|| MathError::InvalidArgument("First argument must be Point3D".to_string()))?;
                let p2 = args[1].downcast_ref::<Point3D>().ok_or_else(|| MathError::InvalidArgument("Second argument must be Point3D".to_string()))?;
                let t = args[2].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be f64".to_string()))?;
                Ok(Box::new(Self::lerp_point3d(p1, p2, *t)?))
            },
            "ray_intersect_sphere" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("ray_intersect_sphere requires 2 arguments".to_string()));
                }
                let ray = args[0].downcast_ref::<Ray3D>().ok_or_else(|| MathError::InvalidArgument("First argument must be Ray3D".to_string()))?;
                let sphere = args[1].downcast_ref::<BoundingSphere>().ok_or_else(|| MathError::InvalidArgument("Second argument must be BoundingSphere".to_string()))?;
                Ok(Box::new(ray.intersect_sphere(sphere)))
            },
            "ray_intersect_aabb" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("ray_intersect_aabb requires 2 arguments".to_string()));
                }
                let ray = args[0].downcast_ref::<Ray3D>().ok_or_else(|| MathError::InvalidArgument("First argument must be Ray3D".to_string()))?;
                let aabb = args[1].downcast_ref::<AABB>().ok_or_else(|| MathError::InvalidArgument("Second argument must be AABB".to_string()))?;
                Ok(Box::new(ray.intersect_aabb(aabb)))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "quaternion_slerp".to_string(),
            "quaternion_from_axis_angle".to_string(),
            "quaternion_from_euler".to_string(),
            "quaternion_multiply".to_string(),
            "quaternion_normalize".to_string(),
            "transform_translation".to_string(),
            "transform_rotation".to_string(),
            "transform_scaling".to_string(),
            "transform_perspective".to_string(),
            "transform_orthographic".to_string(),
            "transform_multiply".to_string(),
            "ray_intersect_plane".to_string(),
            "ray_intersect_sphere".to_string(),
            "ray_intersect_aabb".to_string(),
            "aabb_contains_point".to_string(),
            "aabb_intersects_aabb".to_string(),
            "sphere_contains_point".to_string(),
            "sphere_intersects_sphere".to_string(),
            "lerp".to_string(),
            "lerp_vector3d".to_string(),
            "lerp_point3d".to_string(),
        ]
    }
}