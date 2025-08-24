use crate::core::types::{Number, Point2D, Point3D, Vector2D, Vector3D};
use crate::utils::{DEGREES_TO_RADIANS, RADIANS_TO_DEGREES};
use num_bigint::BigInt;
use num_rational::Rational64;
use num_complex::Complex64;

pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * DEGREES_TO_RADIANS
}

pub fn radians_to_degrees(radians: f64) -> f64 {
    radians * RADIANS_TO_DEGREES
}

pub fn celsius_to_fahrenheit(celsius: f64) -> f64 {
    celsius * 9.0 / 5.0 + 32.0
}

pub fn fahrenheit_to_celsius(fahrenheit: f64) -> f64 {
    (fahrenheit - 32.0) * 5.0 / 9.0
}

pub fn celsius_to_kelvin(celsius: f64) -> f64 {
    celsius + 273.15
}

pub fn kelvin_to_celsius(kelvin: f64) -> f64 {
    kelvin - 273.15
}

pub fn point_2d_to_3d(point: &Point2D, z: f64) -> Point3D {
    Point3D {
        x: point.x,
        y: point.y,
        z,
    }
}

pub fn point_3d_to_2d(point: &Point3D) -> Point2D {
    Point2D {
        x: point.x,
        y: point.y,
    }
}

pub fn vector_2d_to_3d(vector: &Vector2D, z: f64) -> Vector3D {
    Vector3D {
        x: vector.x,
        y: vector.y,
        z,
    }
}

pub fn vector_3d_to_2d(vector: &Vector3D) -> Vector2D {
    Vector2D {
        x: vector.x,
        y: vector.y,
    }
}

pub fn cartesian_to_polar(point: &Point2D) -> (f64, f64) {
    let r = (point.x * point.x + point.y * point.y).sqrt();
    let theta = point.y.atan2(point.x);
    (r, theta)
}

pub fn polar_to_cartesian(r: f64, theta: f64) -> Point2D {
    Point2D {
        x: r * theta.cos(),
        y: r * theta.sin(),
    }
}

pub fn cartesian_to_spherical(point: &Point3D) -> (f64, f64, f64) {
    let r = (point.x * point.x + point.y * point.y + point.z * point.z).sqrt();
    let theta = (point.y / point.x).atan();
    let phi = (point.z / r).acos();
    (r, theta, phi)
}

pub fn spherical_to_cartesian(r: f64, theta: f64, phi: f64) -> Point3D {
    Point3D {
        x: r * phi.sin() * theta.cos(),
        y: r * phi.sin() * theta.sin(),
        z: r * phi.cos(),
    }
}

impl From<i64> for Number {
    fn from(value: i64) -> Self {
        Number::Integer(value)
    }
}

impl From<BigInt> for Number {
    fn from(value: BigInt) -> Self {
        Number::BigInteger(value)
    }
}

impl From<f64> for Number {
    fn from(value: f64) -> Self {
        Number::Real(value)
    }
}

impl From<Complex64> for Number {
    fn from(value: Complex64) -> Self {
        Number::Complex(value)
    }
}

impl From<Rational64> for Number {
    fn from(value: Rational64) -> Self {
        Number::Rational(value)
    }
}