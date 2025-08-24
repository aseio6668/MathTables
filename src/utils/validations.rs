use crate::core::types::{Point2D, Point3D, Vector2D, Vector3D, Matrix, Polynomial};

pub fn is_finite_number(x: f64) -> bool {
    x.is_finite()
}

pub fn is_positive(x: f64) -> bool {
    x > 0.0
}

pub fn is_non_negative(x: f64) -> bool {
    x >= 0.0
}

pub fn is_in_range(x: f64, min: f64, max: f64) -> bool {
    x >= min && x <= max
}

pub fn is_integer(x: f64) -> bool {
    x.fract() == 0.0
}

pub fn is_prime_candidate(n: u64) -> bool {
    n >= 2
}

pub fn is_valid_matrix(matrix: &Matrix) -> bool {
    if matrix.data.is_empty() {
        return false;
    }
    
    let expected_cols = matrix.data[0].len();
    if expected_cols == 0 {
        return false;
    }
    
    matrix.rows == matrix.data.len() &&
    matrix.cols == expected_cols &&
    matrix.data.iter().all(|row| row.len() == expected_cols) &&
    matrix.data.iter().all(|row| row.iter().all(|&x| x.is_finite()))
}

pub fn is_square_matrix(matrix: &Matrix) -> bool {
    is_valid_matrix(matrix) && matrix.rows == matrix.cols
}

pub fn is_valid_point_2d(point: &Point2D) -> bool {
    point.x.is_finite() && point.y.is_finite()
}

pub fn is_valid_point_3d(point: &Point3D) -> bool {
    point.x.is_finite() && point.y.is_finite() && point.z.is_finite()
}

pub fn is_valid_vector_2d(vector: &Vector2D) -> bool {
    vector.x.is_finite() && vector.y.is_finite()
}

pub fn is_valid_vector_3d(vector: &Vector3D) -> bool {
    vector.x.is_finite() && vector.y.is_finite() && vector.z.is_finite()
}

pub fn is_zero_vector_2d(vector: &Vector2D) -> bool {
    vector.x == 0.0 && vector.y == 0.0
}

pub fn is_zero_vector_3d(vector: &Vector3D) -> bool {
    vector.x == 0.0 && vector.y == 0.0 && vector.z == 0.0
}

pub fn is_valid_polynomial(poly: &Polynomial) -> bool {
    !poly.coefficients.is_empty() && 
    poly.coefficients.iter().all(|&x| x.is_finite())
}

pub fn is_constant_polynomial(poly: &Polynomial) -> bool {
    poly.coefficients.len() == 1
}

pub fn is_linear_polynomial(poly: &Polynomial) -> bool {
    poly.coefficients.len() == 2 && poly.coefficients[1] != 0.0
}

pub fn is_quadratic_polynomial(poly: &Polynomial) -> bool {
    poly.coefficients.len() == 3 && poly.coefficients[2] != 0.0
}

pub fn is_valid_triangle(a: f64, b: f64, c: f64) -> bool {
    a > 0.0 && b > 0.0 && c > 0.0 &&
    a + b > c && a + c > b && b + c > a
}

pub fn is_right_triangle(a: f64, b: f64, c: f64) -> bool {
    if !is_valid_triangle(a, b, c) {
        return false;
    }
    
    let mut sides = vec![a, b, c];
    sides.sort_by(|x, y| x.partial_cmp(y).unwrap());
    
    let epsilon = 1e-10;
    (sides[0] * sides[0] + sides[1] * sides[1] - sides[2] * sides[2]).abs() < epsilon
}

pub fn are_collinear_points_2d(p1: &Point2D, p2: &Point2D, p3: &Point2D) -> bool {
    let epsilon = 1e-10;
    let area = (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y);
    area.abs() < epsilon
}

pub fn are_parallel_vectors_2d(v1: &Vector2D, v2: &Vector2D) -> bool {
    if is_zero_vector_2d(v1) || is_zero_vector_2d(v2) {
        return true;
    }
    
    let epsilon = 1e-10;
    (v1.x * v2.y - v1.y * v2.x).abs() < epsilon
}

pub fn are_perpendicular_vectors_2d(v1: &Vector2D, v2: &Vector2D) -> bool {
    let epsilon = 1e-10;
    (v1.x * v2.x + v1.y * v2.y).abs() < epsilon
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finite_number_validation() {
        assert!(is_finite_number(5.0));
        assert!(is_finite_number(-3.14));
        assert!(!is_finite_number(f64::INFINITY));
        assert!(!is_finite_number(f64::NEG_INFINITY));
        assert!(!is_finite_number(f64::NAN));
    }

    #[test]
    fn test_triangle_validation() {
        assert!(is_valid_triangle(3.0, 4.0, 5.0));
        assert!(!is_valid_triangle(1.0, 2.0, 5.0));
        assert!(is_right_triangle(3.0, 4.0, 5.0));
        assert!(!is_right_triangle(2.0, 3.0, 4.0));
    }

    #[test]
    fn test_matrix_validation() {
        let valid_matrix = Matrix {
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            rows: 2,
            cols: 2,
        };
        assert!(is_valid_matrix(&valid_matrix));
        assert!(is_square_matrix(&valid_matrix));

        let invalid_matrix = Matrix {
            data: vec![vec![1.0, 2.0], vec![3.0]],
            rows: 2,
            cols: 2,
        };
        assert!(!is_valid_matrix(&invalid_matrix));
    }
}