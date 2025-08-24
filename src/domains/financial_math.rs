use crate::core::{MathDomain, MathResult, MathError};
use std::any::Any;

#[derive(Debug, Clone)]
pub struct CashFlow {
    pub amount: f64,
    pub time: f64, // Time in years
}

#[derive(Debug, Clone)]
pub struct Bond {
    pub face_value: f64,
    pub coupon_rate: f64,
    pub maturity: f64, // Years
    pub payment_frequency: f64, // Payments per year
}

#[derive(Debug, Clone)]
pub struct OptionPricing {
    pub option_price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

#[derive(Debug, Clone)]
pub struct MortgagePayment {
    pub monthly_payment: f64,
    pub total_interest: f64,
    pub payment_schedule: Vec<PaymentDetail>,
}

#[derive(Debug, Clone)]
pub struct PaymentDetail {
    pub payment_number: usize,
    pub payment_amount: f64,
    pub principal_amount: f64,
    pub interest_amount: f64,
    pub remaining_balance: f64,
}

#[derive(Debug, Clone)]
pub struct PortfolioMetrics {
    pub expected_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub var_95: f64, // Value at Risk at 95% confidence
    pub max_drawdown: f64,
}

pub struct FinancialMathDomain;

impl FinancialMathDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn present_value(future_value: f64, rate: f64, periods: f64) -> MathResult<f64> {
        if rate < -1.0 {
            return Err(MathError::InvalidArgument("Interest rate cannot be less than -100%".to_string()));
        }
        
        let pv = future_value / (1.0 + rate).powf(periods);
        Ok(pv)
    }
    
    pub fn future_value(present_value: f64, rate: f64, periods: f64) -> MathResult<f64> {
        if rate < -1.0 {
            return Err(MathError::InvalidArgument("Interest rate cannot be less than -100%".to_string()));
        }
        
        let fv = present_value * (1.0 + rate).powf(periods);
        Ok(fv)
    }
    
    pub fn net_present_value(cash_flows: &[CashFlow], discount_rate: f64) -> MathResult<f64> {
        if discount_rate < -1.0 {
            return Err(MathError::InvalidArgument("Discount rate cannot be less than -100%".to_string()));
        }
        
        let mut npv = 0.0;
        for cash_flow in cash_flows {
            npv += cash_flow.amount / (1.0 + discount_rate).powf(cash_flow.time);
        }
        
        Ok(npv)
    }
    
    pub fn internal_rate_of_return(cash_flows: &[CashFlow], initial_guess: f64) -> MathResult<f64> {
        if cash_flows.is_empty() {
            return Err(MathError::InvalidArgument("Cash flows cannot be empty".to_string()));
        }
        
        let mut rate = initial_guess;
        let tolerance = 1e-6;
        let max_iterations = 100;
        
        for _ in 0..max_iterations {
            let mut npv = 0.0;
            let mut dnpv = 0.0;
            
            for cash_flow in cash_flows {
                let discount_factor = (1.0 + rate).powf(cash_flow.time);
                npv += cash_flow.amount / discount_factor;
                dnpv -= cash_flow.amount * cash_flow.time / (discount_factor * (1.0 + rate));
            }
            
            if npv.abs() < tolerance {
                return Ok(rate);
            }
            
            if dnpv.abs() < 1e-12 {
                return Err(MathError::ComputationError("IRR calculation failed - derivative too small".to_string()));
            }
            
            rate -= npv / dnpv;
        }
        
        Err(MathError::ComputationError("IRR calculation did not converge".to_string()))
    }
    
    pub fn bond_price(bond: &Bond, yield_rate: f64) -> MathResult<f64> {
        if yield_rate < 0.0 || bond.payment_frequency <= 0.0 {
            return Err(MathError::InvalidArgument("Invalid bond parameters".to_string()));
        }
        
        let coupon_payment = bond.face_value * bond.coupon_rate / bond.payment_frequency;
        let total_payments = bond.maturity * bond.payment_frequency;
        let period_yield = yield_rate / bond.payment_frequency;
        
        let mut price = 0.0;
        
        // Present value of coupon payments
        for i in 1..=(total_payments as i32) {
            price += coupon_payment / (1.0 + period_yield).powi(i);
        }
        
        // Present value of face value
        price += bond.face_value / (1.0 + period_yield).powf(total_payments);
        
        Ok(price)
    }
    
    pub fn bond_duration(bond: &Bond, yield_rate: f64) -> MathResult<f64> {
        if yield_rate < 0.0 || bond.payment_frequency <= 0.0 {
            return Err(MathError::InvalidArgument("Invalid bond parameters".to_string()));
        }
        
        let coupon_payment = bond.face_value * bond.coupon_rate / bond.payment_frequency;
        let total_payments = bond.maturity * bond.payment_frequency;
        let period_yield = yield_rate / bond.payment_frequency;
        
        let mut weighted_cash_flows = 0.0;
        let mut total_present_value = 0.0;
        
        for i in 1..=(total_payments as i32) {
            let pv = coupon_payment / (1.0 + period_yield).powi(i);
            let time_in_years = i as f64 / bond.payment_frequency;
            weighted_cash_flows += pv * time_in_years;
            total_present_value += pv;
        }
        
        // Face value
        let face_pv = bond.face_value / (1.0 + period_yield).powf(total_payments);
        weighted_cash_flows += face_pv * bond.maturity;
        total_present_value += face_pv;
        
        let duration = weighted_cash_flows / total_present_value;
        Ok(duration)
    }
    
    pub fn black_scholes_call(
        spot_price: f64,
        strike_price: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        volatility: f64,
    ) -> MathResult<f64> {
        if spot_price <= 0.0 || strike_price <= 0.0 || time_to_expiry <= 0.0 || volatility <= 0.0 {
            return Err(MathError::InvalidArgument("Black-Scholes parameters must be positive".to_string()));
        }
        
        let d1 = (spot_price / strike_price).ln() + (risk_free_rate + 0.5 * volatility * volatility) * time_to_expiry;
        let d1 = d1 / (volatility * time_to_expiry.sqrt());
        
        let d2 = d1 - volatility * time_to_expiry.sqrt();
        
        let call_price = spot_price * Self::standard_normal_cdf(d1) 
            - strike_price * (-risk_free_rate * time_to_expiry).exp() * Self::standard_normal_cdf(d2);
        
        Ok(call_price)
    }
    
    pub fn black_scholes_put(
        spot_price: f64,
        strike_price: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        volatility: f64,
    ) -> MathResult<f64> {
        if spot_price <= 0.0 || strike_price <= 0.0 || time_to_expiry <= 0.0 || volatility <= 0.0 {
            return Err(MathError::InvalidArgument("Black-Scholes parameters must be positive".to_string()));
        }
        
        let d1 = (spot_price / strike_price).ln() + (risk_free_rate + 0.5 * volatility * volatility) * time_to_expiry;
        let d1 = d1 / (volatility * time_to_expiry.sqrt());
        
        let d2 = d1 - volatility * time_to_expiry.sqrt();
        
        let put_price = strike_price * (-risk_free_rate * time_to_expiry).exp() * Self::standard_normal_cdf(-d2)
            - spot_price * Self::standard_normal_cdf(-d1);
        
        Ok(put_price)
    }
    
    fn standard_normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / 2_f64.sqrt()))
    }
    
    fn erf(x: f64) -> f64 {
        // Approximation of error function using Abramowitz and Stegun formula
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;
        
        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let x = x.abs();
        
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        
        sign * y
    }
    
    pub fn mortgage_calculation(
        principal: f64,
        annual_rate: f64,
        years: usize,
    ) -> MathResult<MortgagePayment> {
        if principal <= 0.0 || annual_rate < 0.0 || years == 0 {
            return Err(MathError::InvalidArgument("Invalid mortgage parameters".to_string()));
        }
        
        let monthly_rate = annual_rate / 12.0;
        let num_payments = years * 12;
        
        let monthly_payment = if monthly_rate == 0.0 {
            principal / num_payments as f64
        } else {
            principal * (monthly_rate * (1.0 + monthly_rate).powi(num_payments as i32)) /
            ((1.0 + monthly_rate).powi(num_payments as i32) - 1.0)
        };
        
        let mut payment_schedule = Vec::new();
        let mut remaining_balance = principal;
        let mut total_interest = 0.0;
        
        for payment_num in 1..=num_payments {
            let interest_amount = remaining_balance * monthly_rate;
            let principal_amount = monthly_payment - interest_amount;
            remaining_balance -= principal_amount;
            total_interest += interest_amount;
            
            payment_schedule.push(PaymentDetail {
                payment_number: payment_num,
                payment_amount: monthly_payment,
                principal_amount,
                interest_amount,
                remaining_balance,
            });
        }
        
        Ok(MortgagePayment {
            monthly_payment,
            total_interest,
            payment_schedule,
        })
    }
    
    pub fn compound_annual_growth_rate(
        beginning_value: f64,
        ending_value: f64,
        years: f64,
    ) -> MathResult<f64> {
        if beginning_value <= 0.0 || ending_value <= 0.0 || years <= 0.0 {
            return Err(MathError::InvalidArgument("CAGR requires positive values and time period".to_string()));
        }
        
        let cagr = (ending_value / beginning_value).powf(1.0 / years) - 1.0;
        Ok(cagr)
    }
    
    pub fn portfolio_expected_return(returns: &[f64], weights: &[f64]) -> MathResult<f64> {
        if returns.len() != weights.len() || returns.is_empty() {
            return Err(MathError::InvalidArgument("Returns and weights must have same non-zero length".to_string()));
        }
        
        let weight_sum: f64 = weights.iter().sum();
        if (weight_sum - 1.0).abs() > 1e-6 {
            return Err(MathError::InvalidArgument("Weights must sum to 1.0".to_string()));
        }
        
        let expected_return: f64 = returns.iter()
            .zip(weights.iter())
            .map(|(r, w)| r * w)
            .sum();
        
        Ok(expected_return)
    }
    
    pub fn portfolio_variance(returns: &[Vec<f64>], weights: &[f64]) -> MathResult<f64> {
        if returns.is_empty() || returns.len() != weights.len() {
            return Err(MathError::InvalidArgument("Invalid portfolio parameters".to_string()));
        }
        
        let n = returns.len();
        let mut covariance_matrix = vec![vec![0.0; n]; n];
        
        // Calculate means
        let means: Vec<f64> = returns.iter()
            .map(|asset_returns| asset_returns.iter().sum::<f64>() / asset_returns.len() as f64)
            .collect();
        
        // Calculate covariance matrix
        for i in 0..n {
            for j in 0..n {
                let cov = returns[i].iter()
                    .zip(returns[j].iter())
                    .map(|(ri, rj)| (ri - means[i]) * (rj - means[j]))
                    .sum::<f64>() / (returns[i].len() - 1) as f64;
                
                covariance_matrix[i][j] = cov;
            }
        }
        
        // Calculate portfolio variance: w^T * Cov * w
        let mut variance = 0.0;
        for i in 0..n {
            for j in 0..n {
                variance += weights[i] * weights[j] * covariance_matrix[i][j];
            }
        }
        
        Ok(variance)
    }
    
    pub fn value_at_risk(returns: &[f64], confidence_level: f64) -> MathResult<f64> {
        if returns.is_empty() || confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(MathError::InvalidArgument("Invalid VaR parameters".to_string()));
        }
        
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let index = ((1.0 - confidence_level) * returns.len() as f64) as usize;
        let var = -sorted_returns[index.min(returns.len() - 1)];
        
        Ok(var)
    }
}

impl MathDomain for FinancialMathDomain {
    fn name(&self) -> &str { "Financial Mathematics" }
    fn description(&self) -> &str { "Financial mathematics including time value of money, bond pricing, options, and portfolio analysis" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "present_value" => {
                if args.len() != 3 {
                    return Err(MathError::InvalidArgument("present_value requires 3 arguments".to_string()));
                }
                let fv = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let rate = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                let periods = args[2].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be f64".to_string()))?;
                Ok(Box::new(Self::present_value(*fv, *rate, *periods)?))
            },
            "future_value" => {
                if args.len() != 3 {
                    return Err(MathError::InvalidArgument("future_value requires 3 arguments".to_string()));
                }
                let pv = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let rate = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                let periods = args[2].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be f64".to_string()))?;
                Ok(Box::new(Self::future_value(*pv, *rate, *periods)?))
            },
            "compound_annual_growth_rate" => {
                if args.len() != 3 {
                    return Err(MathError::InvalidArgument("CAGR requires 3 arguments".to_string()));
                }
                let beginning = args[0].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("First argument must be f64".to_string()))?;
                let ending = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                let years = args[2].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Third argument must be f64".to_string()))?;
                Ok(Box::new(Self::compound_annual_growth_rate(*beginning, *ending, *years)?))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "present_value".to_string(),
            "future_value".to_string(),
            "net_present_value".to_string(),
            "internal_rate_of_return".to_string(),
            "bond_price".to_string(),
            "bond_duration".to_string(),
            "black_scholes_call".to_string(),
            "black_scholes_put".to_string(),
            "mortgage_calculation".to_string(),
            "compound_annual_growth_rate".to_string(),
            "portfolio_expected_return".to_string(),
            "portfolio_variance".to_string(),
            "value_at_risk".to_string(),
        ]
    }
}