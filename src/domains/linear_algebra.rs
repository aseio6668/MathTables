use crate::core::{MathDomain, MathResult, MathError, Matrix};
use nalgebra::{DMatrix, SVD};
use std::any::Any;

#[derive(Debug, Clone)]
pub struct EigenDecomposition {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Matrix,
}

#[derive(Debug, Clone)]
pub struct SVDResult {
    pub u: Matrix,
    pub singular_values: Vec<f64>,
    pub v_transpose: Matrix,
}

#[derive(Debug, Clone)]
pub struct QRDecomposition {
    pub q: Matrix,
    pub r: Matrix,
}

#[derive(Debug, Clone)]
pub struct LUDecomposition {
    pub l: Matrix,
    pub u: Matrix,
    pub p: Matrix, // permutation matrix
}

pub struct LinearAlgebraDomain;

impl LinearAlgebraDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn singular_value_decomposition(matrix: &Matrix) -> MathResult<SVDResult> {
        let mat = DMatrix::from_row_slice(matrix.rows, matrix.cols, &matrix.data.concat());
        
        let svd = SVD::new(mat.clone(), true, true);
        let u_mat = svd.u.ok_or_else(|| MathError::ComputationError("U matrix not computed".to_string()))?;
        let v_t_mat = svd.v_t.ok_or_else(|| MathError::ComputationError("V^T matrix not computed".to_string()))?;
        let singular_vals = svd.singular_values.as_slice().to_vec();
                
        let u_data: Vec<Vec<f64>> = (0..u_mat.nrows())
            .map(|i| (0..u_mat.ncols()).map(|j| u_mat[(i, j)]).collect())
            .collect();
            
        let v_t_data: Vec<Vec<f64>> = (0..v_t_mat.nrows())
            .map(|i| (0..v_t_mat.ncols()).map(|j| v_t_mat[(i, j)]).collect())
            .collect();
        
        Ok(SVDResult {
            u: Matrix {
                data: u_data,
                rows: u_mat.nrows(),
                cols: u_mat.ncols(),
            },
            singular_values: singular_vals,
            v_transpose: Matrix {
                data: v_t_data,
                rows: v_t_mat.nrows(),
                cols: v_t_mat.ncols(),
            },
        })
    }
    
    pub fn qr_decomposition(matrix: &Matrix) -> MathResult<QRDecomposition> {
        let mat = DMatrix::from_row_slice(matrix.rows, matrix.cols, &matrix.data.concat());
        let qr = mat.qr();
        
        let q_mat = qr.q();
        let r_mat = qr.r();
        
        let q_data: Vec<Vec<f64>> = (0..q_mat.nrows())
            .map(|i| (0..q_mat.ncols()).map(|j| q_mat[(i, j)]).collect())
            .collect();
            
        let r_data: Vec<Vec<f64>> = (0..r_mat.nrows())
            .map(|i| (0..r_mat.ncols()).map(|j| r_mat[(i, j)]).collect())
            .collect();
        
        Ok(QRDecomposition {
            q: Matrix {
                data: q_data,
                rows: q_mat.nrows(),
                cols: q_mat.ncols(),
            },
            r: Matrix {
                data: r_data,
                rows: r_mat.nrows(),
                cols: r_mat.ncols(),
            },
        })
    }
    
    pub fn lu_decomposition(matrix: &Matrix) -> MathResult<LUDecomposition> {
        if matrix.rows != matrix.cols {
            return Err(MathError::InvalidArgument("LU decomposition requires square matrix".to_string()));
        }
        
        let mat = DMatrix::from_row_slice(matrix.rows, matrix.cols, &matrix.data.concat());
        let lu = mat.lu();
        
        let l_mat = lu.l();
        let u_mat = lu.u();
        
        // Create identity matrix for permutation
        let mut p_data = vec![vec![0.0; matrix.cols]; matrix.rows];
        for i in 0..matrix.rows {
            p_data[i][i] = 1.0;
        }
        
        let l_data: Vec<Vec<f64>> = (0..l_mat.nrows())
            .map(|i| (0..l_mat.ncols()).map(|j| l_mat[(i, j)]).collect())
            .collect();
            
        let u_data: Vec<Vec<f64>> = (0..u_mat.nrows())
            .map(|i| (0..u_mat.ncols()).map(|j| u_mat[(i, j)]).collect())
            .collect();
        
        Ok(LUDecomposition {
            l: Matrix {
                data: l_data,
                rows: l_mat.nrows(),
                cols: l_mat.ncols(),
            },
            u: Matrix {
                data: u_data,
                rows: u_mat.nrows(),
                cols: u_mat.ncols(),
            },
            p: Matrix {
                data: p_data,
                rows: matrix.rows,
                cols: matrix.cols,
            },
        })
    }
    
    pub fn matrix_rank(matrix: &Matrix) -> MathResult<usize> {
        let mat = DMatrix::from_row_slice(matrix.rows, matrix.cols, &matrix.data.concat());
        Ok(mat.rank(1e-10))
    }
    
    pub fn matrix_trace(matrix: &Matrix) -> MathResult<f64> {
        if matrix.rows != matrix.cols {
            return Err(MathError::InvalidArgument("Trace requires square matrix".to_string()));
        }
        
        let trace = matrix.data.iter()
            .enumerate()
            .filter_map(|(i, row)| row.get(i))
            .sum();
        
        Ok(trace)
    }
    
    pub fn matrix_condition_number(matrix: &Matrix) -> MathResult<f64> {
        let svd_result = Self::singular_value_decomposition(matrix)?;
        
        if svd_result.singular_values.is_empty() {
            return Err(MathError::ComputationError("No singular values found".to_string()));
        }
        
        let max_singular = svd_result.singular_values.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_singular = svd_result.singular_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if min_singular == 0.0 {
            Ok(f64::INFINITY)
        } else {
            Ok(max_singular / min_singular)
        }
    }
    
    pub fn matrix_frobenius_norm(matrix: &Matrix) -> f64 {
        matrix.data.iter()
            .flat_map(|row| row.iter())
            .map(|&x| x * x)
            .sum::<f64>()
            .sqrt()
    }
    
    pub fn matrix_transpose(matrix: &Matrix) -> Matrix {
        let mut transposed_data = vec![vec![0.0; matrix.rows]; matrix.cols];
        
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                transposed_data[j][i] = matrix.data[i][j];
            }
        }
        
        Matrix {
            data: transposed_data,
            rows: matrix.cols,
            cols: matrix.rows,
        }
    }
    
    pub fn matrix_pseudoinverse(matrix: &Matrix) -> MathResult<Matrix> {
        let svd_result = Self::singular_value_decomposition(matrix)?;
        let tolerance = 1e-10;
        
        let u_mat = DMatrix::from_row_slice(
            svd_result.u.rows, 
            svd_result.u.cols, 
            &svd_result.u.data.concat()
        );
        let v_t_mat = DMatrix::from_row_slice(
            svd_result.v_transpose.rows,
            svd_result.v_transpose.cols,
            &svd_result.v_transpose.data.concat()
        );
        
        let mut sigma_inv = DMatrix::zeros(matrix.cols, matrix.rows);
        for (i, &s) in svd_result.singular_values.iter().enumerate() {
            if s > tolerance {
                sigma_inv[(i, i)] = 1.0 / s;
            }
        }
        
        let pinv = v_t_mat.transpose() * sigma_inv * u_mat.transpose();
        
        let pinv_data: Vec<Vec<f64>> = (0..pinv.nrows())
            .map(|i| (0..pinv.ncols()).map(|j| pinv[(i, j)]).collect())
            .collect();
        
        Ok(Matrix {
            data: pinv_data,
            rows: pinv.nrows(),
            cols: pinv.ncols(),
        })
    }
}

impl MathDomain for LinearAlgebraDomain {
    fn name(&self) -> &str { "Advanced Linear Algebra" }
    fn description(&self) -> &str { "Advanced linear algebra operations including decompositions and matrix analysis" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "singular_value_decomposition" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("SVD requires 1 argument".to_string()));
                }
                let matrix = args[0].downcast_ref::<Matrix>().ok_or_else(|| MathError::InvalidArgument("Argument must be Matrix".to_string()))?;
                Ok(Box::new(Self::singular_value_decomposition(matrix)?))
            },
            "qr_decomposition" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("QR decomposition requires 1 argument".to_string()));
                }
                let matrix = args[0].downcast_ref::<Matrix>().ok_or_else(|| MathError::InvalidArgument("Argument must be Matrix".to_string()))?;
                Ok(Box::new(Self::qr_decomposition(matrix)?))
            },
            "lu_decomposition" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("LU decomposition requires 1 argument".to_string()));
                }
                let matrix = args[0].downcast_ref::<Matrix>().ok_or_else(|| MathError::InvalidArgument("Argument must be Matrix".to_string()))?;
                Ok(Box::new(Self::lu_decomposition(matrix)?))
            },
            "matrix_rank" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("Matrix rank requires 1 argument".to_string()));
                }
                let matrix = args[0].downcast_ref::<Matrix>().ok_or_else(|| MathError::InvalidArgument("Argument must be Matrix".to_string()))?;
                Ok(Box::new(Self::matrix_rank(matrix)?))
            },
            "matrix_trace" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("Matrix trace requires 1 argument".to_string()));
                }
                let matrix = args[0].downcast_ref::<Matrix>().ok_or_else(|| MathError::InvalidArgument("Argument must be Matrix".to_string()))?;
                Ok(Box::new(Self::matrix_trace(matrix)?))
            },
            "matrix_condition_number" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("Condition number requires 1 argument".to_string()));
                }
                let matrix = args[0].downcast_ref::<Matrix>().ok_or_else(|| MathError::InvalidArgument("Argument must be Matrix".to_string()))?;
                Ok(Box::new(Self::matrix_condition_number(matrix)?))
            },
            "matrix_frobenius_norm" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("Frobenius norm requires 1 argument".to_string()));
                }
                let matrix = args[0].downcast_ref::<Matrix>().ok_or_else(|| MathError::InvalidArgument("Argument must be Matrix".to_string()))?;
                Ok(Box::new(Self::matrix_frobenius_norm(matrix)))
            },
            "matrix_transpose" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("Matrix transpose requires 1 argument".to_string()));
                }
                let matrix = args[0].downcast_ref::<Matrix>().ok_or_else(|| MathError::InvalidArgument("Argument must be Matrix".to_string()))?;
                Ok(Box::new(Self::matrix_transpose(matrix)))
            },
            "matrix_pseudoinverse" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("Pseudoinverse requires 1 argument".to_string()));
                }
                let matrix = args[0].downcast_ref::<Matrix>().ok_or_else(|| MathError::InvalidArgument("Argument must be Matrix".to_string()))?;
                Ok(Box::new(Self::matrix_pseudoinverse(matrix)?))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "singular_value_decomposition".to_string(),
            "qr_decomposition".to_string(),
            "lu_decomposition".to_string(),
            "matrix_rank".to_string(),
            "matrix_trace".to_string(),
            "matrix_condition_number".to_string(),
            "matrix_frobenius_norm".to_string(),
            "matrix_transpose".to_string(),
            "matrix_pseudoinverse".to_string(),
        ]
    }
}