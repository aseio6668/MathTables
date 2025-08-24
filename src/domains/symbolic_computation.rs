use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum SymbolicExpression {
    Symbol(String),
    Number(f64),
    Rational(i64, i64),
    Add(Box<SymbolicExpression>, Box<SymbolicExpression>),
    Mul(Box<SymbolicExpression>, Box<SymbolicExpression>),
    Sub(Box<SymbolicExpression>, Box<SymbolicExpression>),
    Div(Box<SymbolicExpression>, Box<SymbolicExpression>),
    Pow(Box<SymbolicExpression>, Box<SymbolicExpression>),
    Sin(Box<SymbolicExpression>),
    Cos(Box<SymbolicExpression>),
    Exp(Box<SymbolicExpression>),
    Ln(Box<SymbolicExpression>),
    Sqrt(Box<SymbolicExpression>),
    Abs(Box<SymbolicExpression>),
}

#[derive(Debug, Clone)]
pub struct SymbolicMatrix {
    pub data: Vec<Vec<SymbolicExpression>>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone)]
pub struct SymbolicPolynomial {
    pub coefficients: HashMap<Vec<usize>, SymbolicExpression>, // Multi-variable polynomial
    pub variables: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AlgebraicNumber {
    pub minimal_polynomial: SymbolicPolynomial,
    pub approximate_value: f64,
}

#[derive(Debug, Clone)]
pub struct SymbolicLimit {
    pub expression: SymbolicExpression,
    pub variable: String,
    pub approach: LimitApproach,
}

#[derive(Debug, Clone)]
pub enum LimitApproach {
    Finite(f64),
    PosInfinity,
    NegInfinity,
    LeftSide(f64),
    RightSide(f64),
}

pub struct SymbolicComputationDomain;

impl SymbolicExpression {
    pub fn symbol(name: &str) -> Self {
        SymbolicExpression::Symbol(name.to_string())
    }
    
    pub fn number(value: f64) -> Self {
        SymbolicExpression::Number(value)
    }
    
    pub fn rational(numerator: i64, denominator: i64) -> Self {
        SymbolicExpression::Rational(numerator, denominator)
    }
    
    pub fn zero() -> Self {
        SymbolicExpression::Number(0.0)
    }
    
    pub fn one() -> Self {
        SymbolicExpression::Number(1.0)
    }
    
    pub fn add(left: SymbolicExpression, right: SymbolicExpression) -> Self {
        SymbolicExpression::Add(Box::new(left), Box::new(right))
    }
    
    pub fn mul(left: SymbolicExpression, right: SymbolicExpression) -> Self {
        SymbolicExpression::Mul(Box::new(left), Box::new(right))
    }
    
    pub fn pow(base: SymbolicExpression, exp: SymbolicExpression) -> Self {
        SymbolicExpression::Pow(Box::new(base), Box::new(exp))
    }
    
    pub fn sin(expr: SymbolicExpression) -> Self {
        SymbolicExpression::Sin(Box::new(expr))
    }
    
    pub fn substitute(&self, variable: &str, value: &SymbolicExpression) -> SymbolicExpression {
        match self {
            SymbolicExpression::Symbol(name) => {
                if name == variable {
                    value.clone()
                } else {
                    self.clone()
                }
            },
            SymbolicExpression::Number(_) | SymbolicExpression::Rational(_, _) => self.clone(),
            SymbolicExpression::Add(left, right) => {
                SymbolicExpression::Add(
                    Box::new(left.substitute(variable, value)),
                    Box::new(right.substitute(variable, value))
                )
            },
            SymbolicExpression::Mul(left, right) => {
                SymbolicExpression::Mul(
                    Box::new(left.substitute(variable, value)),
                    Box::new(right.substitute(variable, value))
                )
            },
            SymbolicExpression::Sub(left, right) => {
                SymbolicExpression::Sub(
                    Box::new(left.substitute(variable, value)),
                    Box::new(right.substitute(variable, value))
                )
            },
            SymbolicExpression::Div(left, right) => {
                SymbolicExpression::Div(
                    Box::new(left.substitute(variable, value)),
                    Box::new(right.substitute(variable, value))
                )
            },
            SymbolicExpression::Pow(base, exp) => {
                SymbolicExpression::Pow(
                    Box::new(base.substitute(variable, value)),
                    Box::new(exp.substitute(variable, value))
                )
            },
            SymbolicExpression::Sin(expr) => {
                SymbolicExpression::Sin(Box::new(expr.substitute(variable, value)))
            },
            SymbolicExpression::Cos(expr) => {
                SymbolicExpression::Cos(Box::new(expr.substitute(variable, value)))
            },
            SymbolicExpression::Exp(expr) => {
                SymbolicExpression::Exp(Box::new(expr.substitute(variable, value)))
            },
            SymbolicExpression::Ln(expr) => {
                SymbolicExpression::Ln(Box::new(expr.substitute(variable, value)))
            },
            SymbolicExpression::Sqrt(expr) => {
                SymbolicExpression::Sqrt(Box::new(expr.substitute(variable, value)))
            },
            SymbolicExpression::Abs(expr) => {
                SymbolicExpression::Abs(Box::new(expr.substitute(variable, value)))
            },
        }
    }
    
    pub fn differentiate(&self, variable: &str) -> SymbolicExpression {
        match self {
            SymbolicExpression::Symbol(name) => {
                if name == variable {
                    SymbolicExpression::one()
                } else {
                    SymbolicExpression::zero()
                }
            },
            SymbolicExpression::Number(_) | SymbolicExpression::Rational(_, _) => {
                SymbolicExpression::zero()
            },
            SymbolicExpression::Add(left, right) => {
                SymbolicExpression::Add(
                    Box::new(left.differentiate(variable)),
                    Box::new(right.differentiate(variable))
                )
            },
            SymbolicExpression::Sub(left, right) => {
                SymbolicExpression::Sub(
                    Box::new(left.differentiate(variable)),
                    Box::new(right.differentiate(variable))
                )
            },
            SymbolicExpression::Mul(left, right) => {
                // Product rule: (uv)' = u'v + uv'
                SymbolicExpression::Add(
                    Box::new(SymbolicExpression::Mul(
                        Box::new(left.differentiate(variable)),
                        right.clone()
                    )),
                    Box::new(SymbolicExpression::Mul(
                        left.clone(),
                        Box::new(right.differentiate(variable))
                    ))
                )
            },
            SymbolicExpression::Div(left, right) => {
                // Quotient rule: (u/v)' = (u'v - uv')/v²
                let numerator = SymbolicExpression::Sub(
                    Box::new(SymbolicExpression::Mul(
                        Box::new(left.differentiate(variable)),
                        right.clone()
                    )),
                    Box::new(SymbolicExpression::Mul(
                        left.clone(),
                        Box::new(right.differentiate(variable))
                    ))
                );
                SymbolicExpression::Div(
                    Box::new(numerator),
                    Box::new(SymbolicExpression::Pow(
                        right.clone(),
                        Box::new(SymbolicExpression::number(2.0))
                    ))
                )
            },
            SymbolicExpression::Pow(base, exp) => {
                // Power rule with chain rule: (u^v)' = u^v * (v' * ln(u) + v * u'/u)
                let term1 = SymbolicExpression::Mul(
                    Box::new(exp.differentiate(variable)),
                    Box::new(SymbolicExpression::Ln(base.clone()))
                );
                let term2 = SymbolicExpression::Mul(
                    exp.clone(),
                    Box::new(SymbolicExpression::Div(
                        Box::new(base.differentiate(variable)),
                        base.clone()
                    ))
                );
                SymbolicExpression::Mul(
                    Box::new(self.clone()),
                    Box::new(SymbolicExpression::Add(Box::new(term1), Box::new(term2)))
                )
            },
            SymbolicExpression::Sin(expr) => {
                // Chain rule: (sin(u))' = cos(u) * u'
                SymbolicExpression::Mul(
                    Box::new(SymbolicExpression::Cos(expr.clone())),
                    Box::new(expr.differentiate(variable))
                )
            },
            SymbolicExpression::Cos(expr) => {
                // Chain rule: (cos(u))' = -sin(u) * u'
                SymbolicExpression::Mul(
                    Box::new(SymbolicExpression::Mul(
                        Box::new(SymbolicExpression::number(-1.0)),
                        Box::new(SymbolicExpression::Sin(expr.clone()))
                    )),
                    Box::new(expr.differentiate(variable))
                )
            },
            SymbolicExpression::Exp(expr) => {
                // Chain rule: (e^u)' = e^u * u'
                SymbolicExpression::Mul(
                    Box::new(self.clone()),
                    Box::new(expr.differentiate(variable))
                )
            },
            SymbolicExpression::Ln(expr) => {
                // Chain rule: (ln(u))' = u'/u
                SymbolicExpression::Div(
                    Box::new(expr.differentiate(variable)),
                    expr.clone()
                )
            },
            SymbolicExpression::Sqrt(expr) => {
                // Chain rule: (√u)' = u'/(2√u)
                SymbolicExpression::Div(
                    Box::new(expr.differentiate(variable)),
                    Box::new(SymbolicExpression::Mul(
                        Box::new(SymbolicExpression::number(2.0)),
                        Box::new(SymbolicExpression::Sqrt(expr.clone()))
                    ))
                )
            },
            SymbolicExpression::Abs(expr) => {
                // |u|' = u'/|u| * u (for u ≠ 0)
                SymbolicExpression::Mul(
                    Box::new(SymbolicExpression::Div(
                        expr.clone(),
                        Box::new(SymbolicExpression::Abs(expr.clone()))
                    )),
                    Box::new(expr.differentiate(variable))
                )
            },
        }
    }
    
    pub fn simplify(&self) -> SymbolicExpression {
        match self {
            SymbolicExpression::Add(left, right) => {
                let left_simplified = left.simplify();
                let right_simplified = right.simplify();
                
                match (&left_simplified, &right_simplified) {
                    (SymbolicExpression::Number(0.0), _) => right_simplified,
                    (_, SymbolicExpression::Number(0.0)) => left_simplified,
                    (SymbolicExpression::Number(a), SymbolicExpression::Number(b)) => {
                        SymbolicExpression::Number(a + b)
                    },
                    _ => SymbolicExpression::Add(
                        Box::new(left_simplified),
                        Box::new(right_simplified)
                    ),
                }
            },
            SymbolicExpression::Mul(left, right) => {
                let left_simplified = left.simplify();
                let right_simplified = right.simplify();
                
                match (&left_simplified, &right_simplified) {
                    (SymbolicExpression::Number(0.0), _) | (_, SymbolicExpression::Number(0.0)) => {
                        SymbolicExpression::Number(0.0)
                    },
                    (SymbolicExpression::Number(1.0), _) => right_simplified,
                    (_, SymbolicExpression::Number(1.0)) => left_simplified,
                    (SymbolicExpression::Number(a), SymbolicExpression::Number(b)) => {
                        SymbolicExpression::Number(a * b)
                    },
                    _ => SymbolicExpression::Mul(
                        Box::new(left_simplified),
                        Box::new(right_simplified)
                    ),
                }
            },
            SymbolicExpression::Sub(left, right) => {
                let left_simplified = left.simplify();
                let right_simplified = right.simplify();
                
                match (&left_simplified, &right_simplified) {
                    (_, SymbolicExpression::Number(0.0)) => left_simplified,
                    (SymbolicExpression::Number(a), SymbolicExpression::Number(b)) => {
                        SymbolicExpression::Number(a - b)
                    },
                    _ if left_simplified == right_simplified => SymbolicExpression::Number(0.0),
                    _ => SymbolicExpression::Sub(
                        Box::new(left_simplified),
                        Box::new(right_simplified)
                    ),
                }
            },
            SymbolicExpression::Div(left, right) => {
                let left_simplified = left.simplify();
                let right_simplified = right.simplify();
                
                match (&left_simplified, &right_simplified) {
                    (SymbolicExpression::Number(0.0), _) => SymbolicExpression::Number(0.0),
                    (_, SymbolicExpression::Number(1.0)) => left_simplified,
                    (SymbolicExpression::Number(a), SymbolicExpression::Number(b)) if *b != 0.0 => {
                        SymbolicExpression::Number(a / b)
                    },
                    _ if left_simplified == right_simplified => SymbolicExpression::Number(1.0),
                    _ => SymbolicExpression::Div(
                        Box::new(left_simplified),
                        Box::new(right_simplified)
                    ),
                }
            },
            SymbolicExpression::Pow(base, exp) => {
                let base_simplified = base.simplify();
                let exp_simplified = exp.simplify();
                
                match (&base_simplified, &exp_simplified) {
                    (_, SymbolicExpression::Number(0.0)) => SymbolicExpression::Number(1.0),
                    (_, SymbolicExpression::Number(1.0)) => base_simplified,
                    (SymbolicExpression::Number(1.0), _) => SymbolicExpression::Number(1.0),
                    (SymbolicExpression::Number(a), SymbolicExpression::Number(b)) => {
                        SymbolicExpression::Number(a.powf(*b))
                    },
                    _ => SymbolicExpression::Pow(
                        Box::new(base_simplified),
                        Box::new(exp_simplified)
                    ),
                }
            },
            _ => self.clone(),
        }
    }
    
    pub fn evaluate(&self, variables: &HashMap<String, f64>) -> MathResult<f64> {
        match self {
            SymbolicExpression::Symbol(name) => {
                variables.get(name).copied()
                    .ok_or_else(|| MathError::InvalidArgument(format!("Undefined variable: {}", name)))
            },
            SymbolicExpression::Number(value) => Ok(*value),
            SymbolicExpression::Rational(num, den) => {
                if *den == 0 {
                    Err(MathError::DivisionByZero)
                } else {
                    Ok(*num as f64 / *den as f64)
                }
            },
            SymbolicExpression::Add(left, right) => {
                Ok(left.evaluate(variables)? + right.evaluate(variables)?)
            },
            SymbolicExpression::Sub(left, right) => {
                Ok(left.evaluate(variables)? - right.evaluate(variables)?)
            },
            SymbolicExpression::Mul(left, right) => {
                Ok(left.evaluate(variables)? * right.evaluate(variables)?)
            },
            SymbolicExpression::Div(left, right) => {
                let right_val = right.evaluate(variables)?;
                if right_val == 0.0 {
                    Err(MathError::DivisionByZero)
                } else {
                    Ok(left.evaluate(variables)? / right_val)
                }
            },
            SymbolicExpression::Pow(base, exp) => {
                Ok(base.evaluate(variables)?.powf(exp.evaluate(variables)?))
            },
            SymbolicExpression::Sin(expr) => {
                Ok(expr.evaluate(variables)?.sin())
            },
            SymbolicExpression::Cos(expr) => {
                Ok(expr.evaluate(variables)?.cos())
            },
            SymbolicExpression::Exp(expr) => {
                Ok(expr.evaluate(variables)?.exp())
            },
            SymbolicExpression::Ln(expr) => {
                let val = expr.evaluate(variables)?;
                if val <= 0.0 {
                    Err(MathError::DomainError("Logarithm of non-positive number".to_string()))
                } else {
                    Ok(val.ln())
                }
            },
            SymbolicExpression::Sqrt(expr) => {
                let val = expr.evaluate(variables)?;
                if val < 0.0 {
                    Err(MathError::DomainError("Square root of negative number".to_string()))
                } else {
                    Ok(val.sqrt())
                }
            },
            SymbolicExpression::Abs(expr) => {
                Ok(expr.evaluate(variables)?.abs())
            },
        }
    }
    
    pub fn expand(&self) -> SymbolicExpression {
        match self {
            SymbolicExpression::Mul(left, right) => {
                let left_expanded = left.expand();
                let right_expanded = right.expand();
                
                // Distribute multiplication over addition
                match (&left_expanded, &right_expanded) {
                    (SymbolicExpression::Add(a, b), c) => {
                        SymbolicExpression::Add(
                            Box::new(SymbolicExpression::Mul(a.clone(), Box::new(c.clone())).expand()),
                            Box::new(SymbolicExpression::Mul(b.clone(), Box::new(c.clone())).expand())
                        )
                    },
                    (a, SymbolicExpression::Add(b, c)) => {
                        SymbolicExpression::Add(
                            Box::new(SymbolicExpression::Mul(Box::new(a.clone()), b.clone()).expand()),
                            Box::new(SymbolicExpression::Mul(Box::new(a.clone()), c.clone()).expand())
                        )
                    },
                    _ => SymbolicExpression::Mul(Box::new(left_expanded), Box::new(right_expanded)),
                }
            },
            SymbolicExpression::Add(left, right) => {
                SymbolicExpression::Add(
                    Box::new(left.expand()),
                    Box::new(right.expand())
                )
            },
            SymbolicExpression::Sub(left, right) => {
                SymbolicExpression::Sub(
                    Box::new(left.expand()),
                    Box::new(right.expand())
                )
            },
            SymbolicExpression::Pow(base, exp) => {
                // Expand (a+b)^n for small integer n
                if let SymbolicExpression::Number(n) = **exp {
                    if n >= 0.0 && n == n.floor() && n <= 10.0 {
                        return Self::binomial_expansion(&base.expand(), n as u32);
                    }
                }
                SymbolicExpression::Pow(Box::new(base.expand()), exp.clone())
            },
            _ => self.clone(),
        }
    }
    
    fn binomial_expansion(base: &SymbolicExpression, n: u32) -> SymbolicExpression {
        if n == 0 {
            return SymbolicExpression::one();
        }
        if n == 1 {
            return base.clone();
        }
        
        if let SymbolicExpression::Add(a, b) = base {
            let mut result = SymbolicExpression::zero();
            
            for k in 0..=n {
                let binomial_coeff = Self::binomial_coefficient(n, k);
                let a_power = if k == 0 {
                    SymbolicExpression::one()
                } else {
                    SymbolicExpression::Pow(a.clone(), Box::new(SymbolicExpression::number(k as f64)))
                };
                let b_power = if n - k == 0 {
                    SymbolicExpression::one()
                } else {
                    SymbolicExpression::Pow(b.clone(), Box::new(SymbolicExpression::number((n - k) as f64)))
                };
                
                let term = SymbolicExpression::Mul(
                    Box::new(SymbolicExpression::Mul(
                        Box::new(SymbolicExpression::number(binomial_coeff as f64)),
                        Box::new(a_power)
                    )),
                    Box::new(b_power)
                );
                
                result = SymbolicExpression::Add(Box::new(result), Box::new(term));
            }
            
            result
        } else {
            SymbolicExpression::Pow(Box::new(base.clone()), Box::new(SymbolicExpression::number(n as f64)))
        }
    }
    
    fn binomial_coefficient(n: u32, k: u32) -> u32 {
        if k > n {
            return 0;
        }
        
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }
}

impl fmt::Display for SymbolicExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolicExpression::Symbol(name) => write!(f, "{}", name),
            SymbolicExpression::Number(value) => write!(f, "{}", value),
            SymbolicExpression::Rational(num, den) => write!(f, "{}/{}", num, den),
            SymbolicExpression::Add(left, right) => write!(f, "({} + {})", left, right),
            SymbolicExpression::Sub(left, right) => write!(f, "({} - {})", left, right),
            SymbolicExpression::Mul(left, right) => write!(f, "({} * {})", left, right),
            SymbolicExpression::Div(left, right) => write!(f, "({} / {})", left, right),
            SymbolicExpression::Pow(base, exp) => write!(f, "({})^({})", base, exp),
            SymbolicExpression::Sin(expr) => write!(f, "sin({})", expr),
            SymbolicExpression::Cos(expr) => write!(f, "cos({})", expr),
            SymbolicExpression::Exp(expr) => write!(f, "exp({})", expr),
            SymbolicExpression::Ln(expr) => write!(f, "ln({})", expr),
            SymbolicExpression::Sqrt(expr) => write!(f, "sqrt({})", expr),
            SymbolicExpression::Abs(expr) => write!(f, "|{}|", expr),
        }
    }
}

impl SymbolicComputationDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn symbolic_differentiate(expr: &SymbolicExpression, variable: &str) -> SymbolicExpression {
        expr.differentiate(variable).simplify()
    }
    
    pub fn symbolic_integrate_polynomial(
        expr: &SymbolicExpression,
        variable: &str,
    ) -> MathResult<SymbolicExpression> {
        // Basic polynomial integration
        match expr {
            SymbolicExpression::Symbol(name) if name == variable => {
                // ∫x dx = x²/2
                Ok(SymbolicExpression::Div(
                    Box::new(SymbolicExpression::Pow(
                        Box::new(SymbolicExpression::symbol(variable)),
                        Box::new(SymbolicExpression::number(2.0))
                    )),
                    Box::new(SymbolicExpression::number(2.0))
                ))
            },
            SymbolicExpression::Number(c) => {
                // ∫c dx = cx
                Ok(SymbolicExpression::Mul(
                    Box::new(SymbolicExpression::number(*c)),
                    Box::new(SymbolicExpression::symbol(variable))
                ))
            },
            SymbolicExpression::Pow(base, exp) if **base == SymbolicExpression::symbol(variable) => {
                if let SymbolicExpression::Number(n) = **exp {
                    if n != -1.0 {
                        // ∫x^n dx = x^(n+1)/(n+1)
                        Ok(SymbolicExpression::Div(
                            Box::new(SymbolicExpression::Pow(
                                base.clone(),
                                Box::new(SymbolicExpression::number(n + 1.0))
                            )),
                            Box::new(SymbolicExpression::number(n + 1.0))
                        ))
                    } else {
                        // ∫x^(-1) dx = ln|x|
                        Ok(SymbolicExpression::Ln(
                            Box::new(SymbolicExpression::Abs(base.clone()))
                        ))
                    }
                } else {
                    Err(MathError::NotImplemented("Integration of non-constant exponents".to_string()))
                }
            },
            SymbolicExpression::Add(left, right) => {
                // ∫(f + g) dx = ∫f dx + ∫g dx
                let left_integral = Self::symbolic_integrate_polynomial(left, variable)?;
                let right_integral = Self::symbolic_integrate_polynomial(right, variable)?;
                Ok(SymbolicExpression::Add(
                    Box::new(left_integral),
                    Box::new(right_integral)
                ))
            },
            SymbolicExpression::Mul(left, right) => {
                // Check if one factor is constant
                if Self::is_constant_wrt(left, variable) {
                    let integral = Self::symbolic_integrate_polynomial(right, variable)?;
                    Ok(SymbolicExpression::Mul(left.clone(), Box::new(integral)))
                } else if Self::is_constant_wrt(right, variable) {
                    let integral = Self::symbolic_integrate_polynomial(left, variable)?;
                    Ok(SymbolicExpression::Mul(right.clone(), Box::new(integral)))
                } else {
                    Err(MathError::NotImplemented("Integration by parts not implemented".to_string()))
                }
            },
            _ => Err(MathError::NotImplemented("Complex symbolic integration not implemented".to_string())),
        }
    }
    
    fn is_constant_wrt(expr: &SymbolicExpression, variable: &str) -> bool {
        match expr {
            SymbolicExpression::Symbol(name) => name != variable,
            SymbolicExpression::Number(_) | SymbolicExpression::Rational(_, _) => true,
            SymbolicExpression::Add(left, right) |
            SymbolicExpression::Sub(left, right) |
            SymbolicExpression::Mul(left, right) |
            SymbolicExpression::Div(left, right) |
            SymbolicExpression::Pow(left, right) => {
                Self::is_constant_wrt(left, variable) && Self::is_constant_wrt(right, variable)
            },
            SymbolicExpression::Sin(expr) |
            SymbolicExpression::Cos(expr) |
            SymbolicExpression::Exp(expr) |
            SymbolicExpression::Ln(expr) |
            SymbolicExpression::Sqrt(expr) |
            SymbolicExpression::Abs(expr) => {
                Self::is_constant_wrt(expr, variable)
            },
        }
    }
    
    pub fn solve_linear_equation(
        equation: &SymbolicExpression,
        variable: &str,
    ) -> MathResult<SymbolicExpression> {
        // Solve ax + b = 0 for x
        // This is a simplified implementation
        match equation {
            SymbolicExpression::Add(left, right) => {
                // ax + b = 0 -> x = -b/a
                if Self::contains_variable(left, variable) && Self::is_constant_wrt(right, variable) {
                    // Assume left is ax, solve for x
                    if let SymbolicExpression::Mul(a, x) = left.as_ref() {
                        if **x == SymbolicExpression::symbol(variable) && Self::is_constant_wrt(a, variable) {
                            return Ok(SymbolicExpression::Div(
                                Box::new(SymbolicExpression::Mul(
                                    Box::new(SymbolicExpression::number(-1.0)),
                                    right.clone()
                                )),
                                a.clone()
                            ));
                        }
                    }
                }
                Err(MathError::NotImplemented("Complex equation solving not implemented".to_string()))
            },
            _ => Err(MathError::NotImplemented("Non-linear equation solving not implemented".to_string())),
        }
    }
    
    fn contains_variable(expr: &SymbolicExpression, variable: &str) -> bool {
        match expr {
            SymbolicExpression::Symbol(name) => name == variable,
            SymbolicExpression::Number(_) | SymbolicExpression::Rational(_, _) => false,
            SymbolicExpression::Add(left, right) |
            SymbolicExpression::Sub(left, right) |
            SymbolicExpression::Mul(left, right) |
            SymbolicExpression::Div(left, right) |
            SymbolicExpression::Pow(left, right) => {
                Self::contains_variable(left, variable) || Self::contains_variable(right, variable)
            },
            SymbolicExpression::Sin(expr) |
            SymbolicExpression::Cos(expr) |
            SymbolicExpression::Exp(expr) |
            SymbolicExpression::Ln(expr) |
            SymbolicExpression::Sqrt(expr) |
            SymbolicExpression::Abs(expr) => {
                Self::contains_variable(expr, variable)
            },
        }
    }
    
    pub fn symbolic_limit(
        expr: &SymbolicExpression,
        variable: &str,
        approach: &LimitApproach,
    ) -> MathResult<SymbolicExpression> {
        match approach {
            LimitApproach::Finite(value) => {
                // Try direct substitution first
                let substituted = expr.substitute(variable, &SymbolicExpression::number(*value));
                let simplified = substituted.simplify();
                
                // Check for indeterminate forms
                match &simplified {
                    SymbolicExpression::Div(num, den) => {
                        let num_val = if let SymbolicExpression::Number(n) = num.as_ref() { *n } else { return Ok(simplified); };
                        let den_val = if let SymbolicExpression::Number(d) = den.as_ref() { *d } else { return Ok(simplified); };
                        
                        if num_val == 0.0 && den_val == 0.0 {
                            // 0/0 form - try L'Hôpital's rule
                            let num_deriv = num.differentiate(variable);
                            let den_deriv = den.differentiate(variable);
                            let lhopital = SymbolicExpression::Div(
                                Box::new(num_deriv),
                                Box::new(den_deriv)
                            );
                            Self::symbolic_limit(&lhopital, variable, approach)
                        } else {
                            Ok(simplified)
                        }
                    },
                    _ => Ok(simplified),
                }
            },
            LimitApproach::PosInfinity => {
                // Analyze behavior as variable approaches +∞
                Self::analyze_infinity_limit(expr, variable, true)
            },
            LimitApproach::NegInfinity => {
                // Analyze behavior as variable approaches -∞
                Self::analyze_infinity_limit(expr, variable, false)
            },
            _ => Err(MathError::NotImplemented("One-sided limits not fully implemented".to_string())),
        }
    }
    
    fn analyze_infinity_limit(
        expr: &SymbolicExpression,
        variable: &str,
        positive: bool,
    ) -> MathResult<SymbolicExpression> {
        match expr {
            SymbolicExpression::Symbol(name) if name == variable => {
                if positive {
                    Ok(SymbolicExpression::symbol("∞"))
                } else {
                    Ok(SymbolicExpression::symbol("-∞"))
                }
            },
            SymbolicExpression::Number(n) => Ok(SymbolicExpression::number(*n)),
            SymbolicExpression::Div(num, den) => {
                // Analyze rational functions
                let num_degree = Self::polynomial_degree(num, variable);
                let den_degree = Self::polynomial_degree(den, variable);
                
                if num_degree > den_degree {
                    if positive {
                        Ok(SymbolicExpression::symbol("∞"))
                    } else {
                        Ok(SymbolicExpression::symbol("-∞"))
                    }
                } else if num_degree < den_degree {
                    Ok(SymbolicExpression::number(0.0))
                } else {
                    // Same degree - ratio of leading coefficients
                    Ok(SymbolicExpression::symbol("finite_limit"))
                }
            },
            _ => Err(MathError::NotImplemented("Complex infinity limit analysis not implemented".to_string())),
        }
    }
    
    fn polynomial_degree(expr: &SymbolicExpression, variable: &str) -> i32 {
        match expr {
            SymbolicExpression::Symbol(name) if name == variable => 1,
            SymbolicExpression::Number(_) => 0,
            SymbolicExpression::Pow(base, exp) => {
                if **base == SymbolicExpression::symbol(variable) {
                    if let SymbolicExpression::Number(n) = **exp {
                        n as i32
                    } else {
                        -1 // Non-polynomial
                    }
                } else {
                    0
                }
            },
            SymbolicExpression::Add(left, right) | SymbolicExpression::Sub(left, right) => {
                Self::polynomial_degree(left, variable).max(Self::polynomial_degree(right, variable))
            },
            SymbolicExpression::Mul(left, right) => {
                Self::polynomial_degree(left, variable) + Self::polynomial_degree(right, variable)
            },
            _ => -1, // Non-polynomial
        }
    }
    
    pub fn taylor_series(
        expr: &SymbolicExpression,
        variable: &str,
        center: f64,
        order: usize,
    ) -> MathResult<SymbolicExpression> {
        let mut series = SymbolicExpression::zero();
        let mut current_expr = expr.clone();
        let center_expr = SymbolicExpression::number(center);
        let x_minus_a = SymbolicExpression::Sub(
            Box::new(SymbolicExpression::symbol(variable)),
            Box::new(center_expr.clone())
        );
        
        for n in 0..=order {
            // Evaluate nth derivative at center
            let derivative_at_center = current_expr.substitute(variable, &center_expr);
            
            // Create term: f^(n)(a) * (x-a)^n / n!
            let factorial = (1..=n).fold(1, |acc, x| acc * x) as f64;
            let power_term = if n == 0 {
                SymbolicExpression::one()
            } else {
                SymbolicExpression::Pow(
                    Box::new(x_minus_a.clone()),
                    Box::new(SymbolicExpression::number(n as f64))
                )
            };
            
            let term = SymbolicExpression::Div(
                Box::new(SymbolicExpression::Mul(
                    Box::new(derivative_at_center),
                    Box::new(power_term)
                )),
                Box::new(SymbolicExpression::number(factorial))
            );
            
            series = SymbolicExpression::Add(Box::new(series), Box::new(term));
            
            // Compute next derivative
            if n < order {
                current_expr = current_expr.differentiate(variable);
            }
        }
        
        Ok(series.simplify())
    }
}

impl MathDomain for SymbolicComputationDomain {
    fn name(&self) -> &str { "Symbolic Computation" }
    fn description(&self) -> &str { "Symbolic mathematics including differentiation, integration, equation solving, and series expansion" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, _args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            _ => Err(MathError::InvalidOperation(format!("Operation {} not implemented in compute interface", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "symbolic_differentiate".to_string(),
            "symbolic_integrate".to_string(),
            "symbolic_simplify".to_string(),
            "symbolic_expand".to_string(),
            "symbolic_factor".to_string(),
            "solve_equation".to_string(),
            "symbolic_limit".to_string(),
            "taylor_series".to_string(),
            "partial_fraction_decomposition".to_string(),
            "symbolic_substitution".to_string(),
        ]
    }
}