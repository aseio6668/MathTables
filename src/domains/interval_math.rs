use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;
use std::ops::{Add, Sub, Mul, Div, Neg};

#[derive(Debug, Clone, PartialEq)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone)]
pub struct IntervalVector {
    pub intervals: Vec<Interval>,
}

#[derive(Debug, Clone)]
pub struct IntervalMatrix {
    pub data: Vec<Vec<Interval>>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone)]
pub struct IntervalFunction {
    pub domain: Interval,
    pub range: Option<Interval>,
}

pub struct IntervalMathDomain;

impl Interval {
    pub fn new(lower: f64, upper: f64) -> MathResult<Self> {
        if lower > upper {
            return Err(MathError::InvalidArgument("Lower bound cannot exceed upper bound".to_string()));
        }
        if lower.is_nan() || upper.is_nan() {
            return Err(MathError::InvalidArgument("Interval bounds cannot be NaN".to_string()));
        }
        Ok(Interval { lower, upper })
    }
    
    pub fn point(value: f64) -> MathResult<Self> {
        if value.is_nan() {
            return Err(MathError::InvalidArgument("Point value cannot be NaN".to_string()));
        }
        Ok(Interval { lower: value, upper: value })
    }
    
    pub fn empty() -> Self {
        Interval { lower: f64::INFINITY, upper: f64::NEG_INFINITY }
    }
    
    pub fn entire() -> Self {
        Interval { lower: f64::NEG_INFINITY, upper: f64::INFINITY }
    }
    
    pub fn is_empty(&self) -> bool {
        self.lower > self.upper
    }
    
    pub fn is_point(&self) -> bool {
        self.lower == self.upper
    }
    
    pub fn width(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.upper - self.lower
        }
    }
    
    pub fn midpoint(&self) -> f64 {
        if self.is_empty() {
            f64::NAN
        } else {
            (self.lower + self.upper) / 2.0
        }
    }
    
    pub fn radius(&self) -> f64 {
        self.width() / 2.0
    }
    
    pub fn contains(&self, value: f64) -> bool {
        !self.is_empty() && self.lower <= value && value <= self.upper
    }
    
    pub fn contains_interval(&self, other: &Interval) -> bool {
        if other.is_empty() {
            true
        } else if self.is_empty() {
            false
        } else {
            self.lower <= other.lower && other.upper <= self.upper
        }
    }
    
    pub fn intersects(&self, other: &Interval) -> bool {
        !self.is_empty() && !other.is_empty() && 
        self.lower <= other.upper && other.lower <= self.upper
    }
    
    pub fn intersection(&self, other: &Interval) -> Interval {
        if !self.intersects(other) {
            Interval::empty()
        } else {
            Interval {
                lower: self.lower.max(other.lower),
                upper: self.upper.min(other.upper),
            }
        }
    }
    
    pub fn hull(&self, other: &Interval) -> Interval {
        if self.is_empty() {
            other.clone()
        } else if other.is_empty() {
            self.clone()
        } else {
            Interval {
                lower: self.lower.min(other.lower),
                upper: self.upper.max(other.upper),
            }
        }
    }
    
    pub fn abs(&self) -> Interval {
        if self.is_empty() {
            Interval::empty()
        } else if self.lower >= 0.0 {
            self.clone()
        } else if self.upper <= 0.0 {
            Interval { lower: -self.upper, upper: -self.lower }
        } else {
            Interval { lower: 0.0, upper: (-self.lower).max(self.upper) }
        }
    }
    
    pub fn sqrt(&self) -> MathResult<Interval> {
        if self.is_empty() {
            return Ok(Interval::empty());
        }
        if self.upper < 0.0 {
            return Err(MathError::DomainError("Square root of negative interval".to_string()));
        }
        
        let lower = if self.lower < 0.0 { 0.0 } else { self.lower.sqrt() };
        let upper = self.upper.sqrt();
        
        Ok(Interval { lower, upper })
    }
    
    pub fn powi(&self, n: i32) -> Interval {
        if self.is_empty() {
            return Interval::empty();
        }
        
        if n == 0 {
            return Interval { lower: 1.0, upper: 1.0 };
        }
        
        if n % 2 == 0 {
            // Even power
            if self.contains(0.0) {
                let abs_interval = self.abs();
                Interval { lower: 0.0, upper: abs_interval.upper.powi(n) }
            } else {
                let a = self.lower.powi(n);
                let b = self.upper.powi(n);
                Interval { lower: a.min(b), upper: a.max(b) }
            }
        } else {
            // Odd power
            Interval { lower: self.lower.powi(n), upper: self.upper.powi(n) }
        }
    }
    
    pub fn exp(&self) -> Interval {
        if self.is_empty() {
            Interval::empty()
        } else {
            Interval { lower: self.lower.exp(), upper: self.upper.exp() }
        }
    }
    
    pub fn ln(&self) -> MathResult<Interval> {
        if self.is_empty() {
            return Ok(Interval::empty());
        }
        if self.upper <= 0.0 {
            return Err(MathError::DomainError("Logarithm of non-positive interval".to_string()));
        }
        
        let lower = if self.lower <= 0.0 { f64::NEG_INFINITY } else { self.lower.ln() };
        let upper = self.upper.ln();
        
        Ok(Interval { lower, upper })
    }
    
    pub fn sin(&self) -> Interval {
        if self.is_empty() {
            return Interval::empty();
        }
        
        let period = 2.0 * std::f64::consts::PI;
        if self.width() >= period {
            return Interval { lower: -1.0, upper: 1.0 };
        }
        
        let lower_sin = self.lower.sin();
        let upper_sin = self.upper.sin();
        
        let pi_2 = std::f64::consts::PI / 2.0;
        let pi_3_2 = 3.0 * std::f64::consts::PI / 2.0;
        
        let mut min_val = lower_sin.min(upper_sin);
        let mut max_val = lower_sin.max(upper_sin);
        
        // Check for extrema within the interval
        let k_min = (self.lower / period).floor() as i32;
        let k_max = (self.upper / period).ceil() as i32;
        
        for k in k_min..=k_max {
            let max_point = k as f64 * period + pi_2;
            let min_point = k as f64 * period + pi_3_2;
            
            if self.contains(max_point) {
                max_val = 1.0;
            }
            if self.contains(min_point) {
                min_val = -1.0;
            }
        }
        
        Interval { lower: min_val, upper: max_val }
    }
    
    pub fn cos(&self) -> Interval {
        if self.is_empty() {
            return Interval::empty();
        }
        
        let period = 2.0 * std::f64::consts::PI;
        if self.width() >= period {
            return Interval { lower: -1.0, upper: 1.0 };
        }
        
        let lower_cos = self.lower.cos();
        let upper_cos = self.upper.cos();
        
        let mut min_val = lower_cos.min(upper_cos);
        let mut max_val = lower_cos.max(upper_cos);
        
        // Check for extrema within the interval
        let k_min = (self.lower / period).floor() as i32;
        let k_max = (self.upper / period).ceil() as i32;
        
        for k in k_min..=k_max {
            let max_point = k as f64 * period; // cos(0) = 1
            let min_point = k as f64 * period + std::f64::consts::PI; // cos(π) = -1
            
            if self.contains(max_point) {
                max_val = 1.0;
            }
            if self.contains(min_point) {
                min_val = -1.0;
            }
        }
        
        Interval { lower: min_val, upper: max_val }
    }
}

impl Add for &Interval {
    type Output = Interval;
    
    fn add(self, other: &Interval) -> Interval {
        if self.is_empty() || other.is_empty() {
            Interval::empty()
        } else {
            Interval {
                lower: self.lower + other.lower,
                upper: self.upper + other.upper,
            }
        }
    }
}

impl Sub for &Interval {
    type Output = Interval;
    
    fn sub(self, other: &Interval) -> Interval {
        if self.is_empty() || other.is_empty() {
            Interval::empty()
        } else {
            Interval {
                lower: self.lower - other.upper,
                upper: self.upper - other.lower,
            }
        }
    }
}

impl Mul for &Interval {
    type Output = Interval;
    
    fn mul(self, other: &Interval) -> Interval {
        if self.is_empty() || other.is_empty() {
            return Interval::empty();
        }
        
        let products = [
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        ];
        
        let min_val = products.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = products.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        Interval { lower: min_val, upper: max_val }
    }
}

impl Div for &Interval {
    type Output = MathResult<Interval>;
    
    fn div(self, other: &Interval) -> MathResult<Interval> {
        if self.is_empty() || other.is_empty() {
            return Ok(Interval::empty());
        }
        
        if other.contains(0.0) {
            if other.is_point() && other.lower == 0.0 {
                return Err(MathError::DivisionByZero);
            }
            // Division by interval containing zero - can result in infinite interval
            return Ok(Interval::entire());
        }
        
        let quotients = [
            self.lower / other.lower,
            self.lower / other.upper,
            self.upper / other.lower,
            self.upper / other.upper,
        ];
        
        let min_val = quotients.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = quotients.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        Ok(Interval { lower: min_val, upper: max_val })
    }
}

impl Neg for &Interval {
    type Output = Interval;
    
    fn neg(self) -> Interval {
        if self.is_empty() {
            Interval::empty()
        } else {
            Interval { lower: -self.upper, upper: -self.lower }
        }
    }
}

impl IntervalMathDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn create_interval(lower: f64, upper: f64) -> MathResult<Interval> {
        Interval::new(lower, upper)
    }
    
    pub fn interval_arithmetic_add(a: &Interval, b: &Interval) -> Interval {
        a + b
    }
    
    pub fn interval_arithmetic_sub(a: &Interval, b: &Interval) -> Interval {
        a - b
    }
    
    pub fn interval_arithmetic_mul(a: &Interval, b: &Interval) -> Interval {
        a * b
    }
    
    pub fn interval_arithmetic_div(a: &Interval, b: &Interval) -> MathResult<Interval> {
        a / b
    }
    
    pub fn interval_function_evaluation<F>(interval: &Interval, f: F) -> Interval 
    where 
        F: Fn(f64) -> f64,
    {
        if interval.is_empty() {
            return Interval::empty();
        }
        
        // Simple evaluation at endpoints (can be improved with more sophisticated methods)
        let f_lower = f(interval.lower);
        let f_upper = f(interval.upper);
        
        Interval {
            lower: f_lower.min(f_upper),
            upper: f_lower.max(f_upper),
        }
    }
    
    pub fn newton_interval_method<F, DF>(
        f: F, 
        df: DF, 
        initial: &Interval, 
        tolerance: f64, 
        max_iterations: usize
    ) -> MathResult<Vec<Interval>>
    where
        F: Fn(f64) -> f64,
        DF: Fn(f64) -> f64,
    {
        let mut intervals = vec![initial.clone()];
        let mut result = Vec::new();
        
        for _ in 0..max_iterations {
            let mut new_intervals = Vec::new();
            
            for interval in &intervals {
                if interval.width() < tolerance {
                    result.push(interval.clone());
                    continue;
                }
                
                let mid = interval.midpoint();
                let f_mid = f(mid);
                let df_mid = df(mid);
                
                if df_mid.abs() < 1e-12 {
                    // Split interval if derivative is too small
                    let left = Interval::new(interval.lower, mid)?;
                    let right = Interval::new(mid, interval.upper)?;
                    new_intervals.push(left);
                    new_intervals.push(right);
                } else {
                    // Newton step
                    let newton_point = mid - f_mid / df_mid;
                    let newton_interval = Interval::point(newton_point)?;
                    let intersection = interval.intersection(&newton_interval);
                    
                    if !intersection.is_empty() {
                        new_intervals.push(intersection);
                    }
                }
            }
            
            intervals = new_intervals;
            if intervals.is_empty() {
                break;
            }
        }
        
        result.extend(intervals);
        Ok(result)
    }
    
    pub fn interval_constraint_propagation(
        variables: &mut Vec<Interval>,
        constraints: &[Box<dyn Fn(&[Interval]) -> Vec<Interval>>]
    ) -> MathResult<bool> {
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;
        
        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;
            
            for constraint in constraints {
                let new_intervals = constraint(variables);
                
                for (i, new_interval) in new_intervals.iter().enumerate() {
                    if i < variables.len() {
                        let intersection = variables[i].intersection(new_interval);
                        if intersection != variables[i] {
                            variables[i] = intersection;
                            changed = true;
                        }
                        
                        if variables[i].is_empty() {
                            return Ok(false); // Inconsistent system
                        }
                    }
                }
            }
        }
        
        Ok(true)
    }
    
    pub fn interval_bisection_optimization<F>(
        f: F,
        domain: &Interval,
        tolerance: f64,
        maximize: bool
    ) -> MathResult<(Interval, f64)>
    where
        F: Fn(&Interval) -> Interval,
    {
        let mut current_interval = domain.clone();
        let mut best_value = if maximize { f64::NEG_INFINITY } else { f64::INFINITY };
        
        while current_interval.width() > tolerance {
            let mid = current_interval.midpoint();
            let left = Interval::new(current_interval.lower, mid)?;
            let right = Interval::new(mid, current_interval.upper)?;
            
            let left_range = f(&left);
            let right_range = f(&right);
            
            let left_bound = if maximize { left_range.upper } else { left_range.lower };
            let right_bound = if maximize { right_range.upper } else { right_range.lower };
            
            if (maximize && left_bound >= right_bound) || (!maximize && left_bound <= right_bound) {
                current_interval = left;
                best_value = left_bound;
            } else {
                current_interval = right;
                best_value = right_bound;
            }
        }
        
        Ok((current_interval, best_value))
    }
    
    pub fn interval_integration_midpoint(f: &dyn Fn(f64) -> f64, interval: &Interval, n: usize) -> MathResult<Interval> {
        if interval.is_empty() {
            return Ok(Interval::empty());
        }
        
        let width = interval.width();
        let h = width / n as f64;
        let mut sum_lower = 0.0;
        let mut sum_upper = 0.0;
        
        for i in 0..n {
            let x_left = interval.lower + i as f64 * h;
            let x_right = x_left + h;
            let sub_interval = Interval::new(x_left, x_right)?;
            
            // Evaluate function over subinterval (simplified)
            let f_left = f(x_left);
            let f_right = f(x_right);
            let f_mid = f(sub_interval.midpoint());
            
            let min_f = f_left.min(f_right).min(f_mid);
            let max_f = f_left.max(f_right).max(f_mid);
            
            sum_lower += min_f * h;
            sum_upper += max_f * h;
        }
        
        Ok(Interval::new(sum_lower, sum_upper)?)
    }
}

impl MathDomain for IntervalMathDomain {
    fn name(&self) -> &str { "Interval Mathematics" }
    fn description(&self) -> &str { "Interval arithmetic, constraint propagation, and uncertainty quantification" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "create_interval" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("create_interval requires 2 arguments".to_string()));
                }
                let lower = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let upper = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                Ok(Box::new(Self::create_interval(*lower, *upper)?))
            },
            "interval_add" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("interval_add requires 2 arguments".to_string()));
                }
                let a = args[0].downcast_ref::<Interval>().ok_or_else(|| MathError::InvalidArgument("First argument must be Interval".to_string()))?;
                let b = args[1].downcast_ref::<Interval>().ok_or_else(|| MathError::InvalidArgument("Second argument must be Interval".to_string()))?;
                Ok(Box::new(Self::interval_arithmetic_add(a, b)))
            },
            "interval_mul" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("interval_mul requires 2 arguments".to_string()));
                }
                let a = args[0].downcast_ref::<Interval>().ok_or_else(|| MathError::InvalidArgument("First argument must be Interval".to_string()))?;
                let b = args[1].downcast_ref::<Interval>().ok_or_else(|| MathError::InvalidArgument("Second argument must be Interval".to_string()))?;
                Ok(Box::new(Self::interval_arithmetic_mul(a, b)))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "create_interval".to_string(),
            "interval_add".to_string(),
            "interval_sub".to_string(),
            "interval_mul".to_string(),
            "interval_div".to_string(),
            "interval_abs".to_string(),
            "interval_sqrt".to_string(),
            "interval_exp".to_string(),
            "interval_ln".to_string(),
            "interval_sin".to_string(),
            "interval_cos".to_string(),
            "interval_intersection".to_string(),
            "interval_hull".to_string(),
            "newton_interval_method".to_string(),
            "interval_constraint_propagation".to_string(),
            "interval_optimization".to_string(),
            "interval_integration".to_string(),
        ]
    }
}