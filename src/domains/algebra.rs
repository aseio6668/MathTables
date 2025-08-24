use crate::core::{MathDomain, MathResult, MathError, Matrix, Polynomial};
use nalgebra::{DMatrix, DVector};
use std::any::Any;

pub struct AlgebraDomain;

impl AlgebraDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn solve_linear_system(matrix: &Matrix, constants: &[f64]) -> MathResult<Vec<f64>> {
        if matrix.rows != constants.len() {
            return Err(MathError::InvalidArgument("Matrix rows must match constants length".to_string()));
        }
        
        let a = DMatrix::from_row_slice(matrix.rows, matrix.cols, &matrix.data.concat());
        let b = DVector::from_column_slice(constants);
        
        match a.lu().solve(&b) {
            Some(solution) => Ok(solution.as_slice().to_vec()),
            None => Err(MathError::ComputationError("No unique solution exists".to_string()))
        }
    }
    
    pub fn matrix_multiply(a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
        if a.cols != b.rows {
            return Err(MathError::InvalidArgument("Matrix dimensions incompatible for multiplication".to_string()));
        }
        
        let mat_a = DMatrix::from_row_slice(a.rows, a.cols, &a.data.concat());
        let mat_b = DMatrix::from_row_slice(b.rows, b.cols, &b.data.concat());
        let result = mat_a * mat_b;
        
        let result_data: Vec<Vec<f64>> = (0..result.nrows())
            .map(|i| (0..result.ncols()).map(|j| result[(i, j)]).collect())
            .collect();
            
        Ok(Matrix {
            data: result_data,
            rows: result.nrows(),
            cols: result.ncols(),
        })
    }
    
    pub fn matrix_determinant(matrix: &Matrix) -> MathResult<f64> {
        if matrix.rows != matrix.cols {
            return Err(MathError::InvalidArgument("Determinant requires square matrix".to_string()));
        }
        
        let mat = DMatrix::from_row_slice(matrix.rows, matrix.cols, &matrix.data.concat());
        Ok(mat.determinant())
    }
    
    pub fn polynomial_add(a: &Polynomial, b: &Polynomial) -> Polynomial {
        let max_len = a.coefficients.len().max(b.coefficients.len());
        let mut result = vec![0.0; max_len];
        
        for (i, &coeff) in a.coefficients.iter().enumerate() {
            result[i] += coeff;
        }
        for (i, &coeff) in b.coefficients.iter().enumerate() {
            result[i] += coeff;
        }
        
        Polynomial { coefficients: result }
    }
    
    pub fn polynomial_multiply(a: &Polynomial, b: &Polynomial) -> Polynomial {
        let result_len = a.coefficients.len() + b.coefficients.len() - 1;
        let mut result = vec![0.0; result_len];
        
        for (i, &a_coeff) in a.coefficients.iter().enumerate() {
            for (j, &b_coeff) in b.coefficients.iter().enumerate() {
                result[i + j] += a_coeff * b_coeff;
            }
        }
        
        Polynomial { coefficients: result }
    }
    
    pub fn polynomial_evaluate(poly: &Polynomial, x: f64) -> f64 {
        poly.coefficients.iter()
            .enumerate()
            .fold(0.0, |acc, (i, &coeff)| acc + coeff * x.powi(i as i32))
    }
    
    pub fn quadratic_roots(a: f64, b: f64, c: f64) -> MathResult<(Option<f64>, Option<f64>)> {
        if a == 0.0 {
            return Err(MathError::InvalidArgument("Coefficient 'a' cannot be zero for quadratic equation".to_string()));
        }
        
        let discriminant = b * b - 4.0 * a * c;
        
        if discriminant < 0.0 {
            Ok((None, None)) // Complex roots
        } else if discriminant == 0.0 {
            let root = -b / (2.0 * a);
            Ok((Some(root), None))
        } else {
            let sqrt_discriminant = discriminant.sqrt();
            let root1 = (-b + sqrt_discriminant) / (2.0 * a);
            let root2 = (-b - sqrt_discriminant) / (2.0 * a);
            Ok((Some(root1), Some(root2)))
        }
    }
}

impl MathDomain for AlgebraDomain {
    fn name(&self) -> &str { "Algebra" }
    fn description(&self) -> &str { "Mathematical domain for algebraic operations, matrices, and polynomials" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "matrix_determinant" => {
                if args.len() != 1 { 
                    return Err(MathError::InvalidArgument("matrix_determinant requires 1 argument".to_string())); 
                }
                let matrix = args[0].downcast_ref::<Matrix>().ok_or_else(|| MathError::InvalidArgument("Argument must be Matrix".to_string()))?;
                Ok(Box::new(Self::matrix_determinant(matrix)?))
            },
            "polynomial_evaluate" => {
                if args.len() != 2 { 
                    return Err(MathError::InvalidArgument("polynomial_evaluate requires 2 arguments".to_string())); 
                }
                let poly = args[0].downcast_ref::<Polynomial>().ok_or_else(|| MathError::InvalidArgument("First argument must be Polynomial".to_string()))?;
                let x = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                Ok(Box::new(Self::polynomial_evaluate(poly, *x)))
            },
            "quadratic_roots" => {
                if args.len() != 3 { 
                    return Err(MathError::InvalidArgument("quadratic_roots requires 3 arguments".to_string())); 
                }
                let a = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let b = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                let c = args[2].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be f64".to_string()))?;
                Ok(Box::new(Self::quadratic_roots(*a, *b, *c)?))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "solve_linear_system".to_string(),
            "matrix_multiply".to_string(),
            "matrix_determinant".to_string(),
            "polynomial_add".to_string(),
            "polynomial_multiply".to_string(),
            "polynomial_evaluate".to_string(),
            "quadratic_roots".to_string(),
        ]
    }
}