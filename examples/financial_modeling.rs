use mathtables::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MathTables Financial Mathematics Demo ===\n");

    // 1. Time Value of Money Calculations
    println!("1. Time Value of Money...");
    
    // Present Value calculations
    let future_value = 10000.0;
    let interest_rate = 0.05; // 5% annual
    let periods = 10.0; // 10 years
    
    let pv = FinancialMathDomain::present_value(future_value, interest_rate, periods)?;
    println!("   Present Value of ${:.2} in {} years at {:.1}%: ${:.2}", 
             future_value, periods, interest_rate * 100.0, pv);
    
    // Future Value calculations
    let present_value = 5000.0;
    let fv = FinancialMathDomain::future_value(present_value, interest_rate, periods)?;
    println!("   Future Value of ${:.2} in {} years at {:.1}%: ${:.2}", 
             present_value, periods, interest_rate * 100.0, fv);
    
    // Compound Annual Growth Rate
    let beginning_value = 1000.0;
    let ending_value = 2500.0;
    let years = 8.0;
    let cagr = FinancialMathDomain::compound_annual_growth_rate(beginning_value, ending_value, years)?;
    println!("   CAGR from ${:.2} to ${:.2} over {} years: {:.2}%", 
             beginning_value, ending_value, years, cagr * 100.0);
    
    // 2. Net Present Value and Internal Rate of Return
    println!("\n2. Investment Analysis...");
    
    // Define cash flows for an investment project
    let cash_flows = vec![
        CashFlow { amount: -50000.0, time: 0.0 }, // Initial investment
        CashFlow { amount: 15000.0, time: 1.0 },  // Year 1 return
        CashFlow { amount: 20000.0, time: 2.0 },  // Year 2 return
        CashFlow { amount: 25000.0, time: 3.0 },  // Year 3 return
        CashFlow { amount: 18000.0, time: 4.0 },  // Year 4 return
    ];
    
    let discount_rate = 0.08; // 8% cost of capital
    let npv = FinancialMathDomain::net_present_value(&cash_flows, discount_rate)?;
    println!("   Project NPV at {:.1}% discount rate: ${:.2}", discount_rate * 100.0, npv);
    
    if npv > 0.0 {
        println!("   ✓ Project is profitable (NPV > 0)");
    } else {
        println!("   ✗ Project is not profitable (NPV < 0)");
    }
    
    // Calculate IRR
    let irr = FinancialMathDomain::internal_rate_of_return(&cash_flows, 0.1)?;
    println!("   Project IRR: {:.2}%", irr * 100.0);
    
    if irr > discount_rate {
        println!("   ✓ IRR exceeds cost of capital - project is attractive");
    } else {
        println!("   ✗ IRR below cost of capital - project may not be attractive");
    }
    
    // 3. Bond Pricing and Analysis
    println!("\n3. Bond Analysis...");
    
    // Corporate bond example
    let corporate_bond = Bond {
        face_value: 1000.0,
        coupon_rate: 0.06,    // 6% annual coupon
        maturity: 5.0,        // 5 years to maturity
        payment_frequency: 2.0, // Semi-annual payments
    };
    
    let market_yields = vec![0.04, 0.05, 0.06, 0.07, 0.08];
    
    println!("   Bond Details:");
    println!("   - Face Value: ${:.2}", corporate_bond.face_value);
    println!("   - Coupon Rate: {:.1}%", corporate_bond.coupon_rate * 100.0);
    println!("   - Maturity: {:.1} years", corporate_bond.maturity);
    println!("   - Payment Frequency: {:.0} per year", corporate_bond.payment_frequency);
    
    println!("\n   Bond Pricing at Different Market Yields:");
    println!("   Yield   Price    Duration  Premium/Discount");
    println!("   ----    -----    --------  ----------------");
    
    for &yield_rate in &market_yields {
        let bond_price = FinancialMathDomain::bond_price(&corporate_bond, yield_rate)?;
        let duration = FinancialMathDomain::bond_duration(&corporate_bond, yield_rate)?;
        
        let premium_discount = if bond_price > corporate_bond.face_value {
            "Premium"
        } else if bond_price < corporate_bond.face_value {
            "Discount"
        } else {
            "Par"
        };
        
        println!("   {:.1}%   ${:.2}   {:.2} years  {}", 
                 yield_rate * 100.0, bond_price, duration, premium_discount);
    }
    
    // 4. Options Pricing (Black-Scholes)
    println!("\n4. Options Pricing...");
    
    let stock_price = 100.0;
    let strike_prices = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let time_to_expiry = 0.25; // 3 months
    let risk_free_rate = 0.03; // 3% risk-free rate
    let volatility = 0.25;     // 25% volatility
    
    println!("   Black-Scholes Options Pricing:");
    println!("   Stock Price: ${:.2}", stock_price);
    println!("   Time to Expiry: {:.2} years", time_to_expiry);
    println!("   Risk-free Rate: {:.1}%", risk_free_rate * 100.0);
    println!("   Volatility: {:.1}%", volatility * 100.0);
    
    println!("\n   Strike   Call Price   Put Price");
    println!("   ------   ----------   ---------");
    
    for &strike in &strike_prices {
        let call_price = FinancialMathDomain::black_scholes_call(
            stock_price, strike, time_to_expiry, risk_free_rate, volatility
        )?;
        
        let put_price = FinancialMathDomain::black_scholes_put(
            stock_price, strike, time_to_expiry, risk_free_rate, volatility
        )?;
        
        println!("   ${:.2}    ${:.2}       ${:.2}", strike, call_price, put_price);
    }
    
    // 5. Mortgage Calculations
    println!("\n5. Mortgage Analysis...");
    
    let loan_amount = 300000.0;
    let annual_rate = 0.045; // 4.5% APR
    let loan_years = 30;
    
    let mortgage = FinancialMathDomain::mortgage_calculation(loan_amount, annual_rate, loan_years)?;
    
    println!("   Mortgage Details:");
    println!("   - Loan Amount: ${:.2}", loan_amount);
    println!("   - Annual Rate: {:.2}%", annual_rate * 100.0);
    println!("   - Loan Term: {} years", loan_years);
    
    println!("\n   Payment Analysis:");
    println!("   - Monthly Payment: ${:.2}", mortgage.monthly_payment);
    println!("   - Total Interest: ${:.2}", mortgage.total_interest);
    println!("   - Total Payments: ${:.2}", mortgage.monthly_payment * (loan_years * 12) as f64);
    
    // Show first few payments
    println!("\n   First 6 Payment Details:");
    println!("   Payment#  Amount    Principal  Interest   Balance");
    println!("   --------  ------    ---------  --------   -------");
    
    for payment in mortgage.payment_schedule.iter().take(6) {
        println!("   {:>8}  ${:.2}   ${:.2}    ${:.2}   ${:.2}",
                 payment.payment_number,
                 payment.payment_amount,
                 payment.principal_amount,
                 payment.interest_amount,
                 payment.remaining_balance);
    }
    
    // 6. Portfolio Analysis
    println!("\n6. Portfolio Analysis...");
    
    // Portfolio with three assets
    let asset_returns = vec![0.08, 0.12, 0.06]; // Expected annual returns
    let portfolio_weights = vec![0.4, 0.3, 0.3]; // Portfolio allocation
    
    let portfolio_return = FinancialMathDomain::portfolio_expected_return(&asset_returns, &portfolio_weights)?;
    println!("   Portfolio Expected Return: {:.2}%", portfolio_return * 100.0);
    
    // Historical returns for variance calculation (simulated)
    let historical_returns = vec![
        vec![0.05, 0.08, 0.12, -0.02, 0.15, 0.06, 0.09], // Asset 1
        vec![0.18, 0.05, 0.20, -0.10, 0.25, 0.08, 0.14], // Asset 2
        vec![0.04, 0.06, 0.08, 0.02, 0.07, 0.05, 0.06],  // Asset 3
    ];
    
    let portfolio_variance = FinancialMathDomain::portfolio_variance(&historical_returns, &portfolio_weights)?;
    let portfolio_volatility = portfolio_variance.sqrt();
    
    println!("   Portfolio Variance: {:.6}", portfolio_variance);
    println!("   Portfolio Volatility: {:.2}%", portfolio_volatility * 100.0);
    
    // Sharpe ratio (assuming risk-free rate of 2%)
    let risk_free_rate_annual = 0.02;
    let sharpe_ratio = (portfolio_return - risk_free_rate_annual) / portfolio_volatility;
    println!("   Sharpe Ratio: {:.3}", sharpe_ratio);
    
    // Value at Risk (VaR)
    let daily_returns = vec![
        0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03,
        0.01, 0.02, -0.04, 0.03, -0.01, 0.02, -0.02,
        0.01, 0.03, -0.01, 0.02, -0.03, 0.04
    ];
    
    let var_95 = FinancialMathDomain::value_at_risk(&daily_returns, 0.95)?;
    println!("   Value at Risk (95% confidence): {:.2}%", var_95 * 100.0);
    
    // 7. Financial Planning Scenarios
    println!("\n7. Financial Planning Scenarios...");
    
    // Retirement planning
    let current_age = 30;
    let retirement_age = 65;
    let years_to_retirement = retirement_age - current_age;
    let annual_contribution = 12000.0;
    let expected_return = 0.07;
    
    // Future value of annuity (approximation using single payment)
    let retirement_savings = FinancialMathDomain::future_value(
        annual_contribution * years_to_retirement as f64, 
        expected_return, 
        years_to_retirement as f64
    )?;
    
    println!("   Retirement Planning:");
    println!("   - Current Age: {}", current_age);
    println!("   - Retirement Age: {}", retirement_age);
    println!("   - Annual Contribution: ${:.2}", annual_contribution);
    println!("   - Expected Return: {:.1}%", expected_return * 100.0);
    println!("   - Estimated Retirement Savings: ${:.2}", retirement_savings);
    
    // Required monthly withdrawal for 25 years in retirement
    let retirement_duration = 25.0;
    let withdrawal_pv = FinancialMathDomain::present_value(
        retirement_savings, 
        0.04, // Lower return in retirement
        retirement_duration
    )?;
    let monthly_withdrawal = withdrawal_pv / (retirement_duration * 12.0);
    
    println!("   - Estimated Monthly Withdrawal Capacity: ${:.2}", monthly_withdrawal);
    
    // 8. Risk Analysis Summary
    println!("\n8. Risk Analysis Summary...");
    
    // Compare different investment scenarios
    let scenarios = vec![
        ("Conservative", 0.04, 0.05),
        ("Moderate", 0.07, 0.12),
        ("Aggressive", 0.10, 0.20),
    ];
    
    let investment_amount = 10000.0;
    let investment_period = 10.0;
    
    println!("   Investment Scenarios (${:.0} for {} years):", investment_amount, investment_period);
    println!("   Strategy      Return  Risk   Future Value  Risk-Adj Return");
    println!("   -----------   ------  ----   ------------  ---------------");
    
    for (strategy, expected_return, volatility) in scenarios {
        let future_val = FinancialMathDomain::future_value(investment_amount, expected_return, investment_period)?;
        let risk_adjusted_return = expected_return / volatility; // Simple risk-adjusted metric
        
        println!("   {:11}   {:.1}%   {:.1}%   ${:.2}       {:.3}",
                 strategy, 
                 expected_return * 100.0,
                 volatility * 100.0,
                 future_val,
                 risk_adjusted_return);
    }
    
    println!("\n=== Financial Mathematics Demo Complete ===");
    println!("This demo showcased:");
    println!("• Time value of money calculations (PV, FV, CAGR)");
    println!("• Investment analysis (NPV, IRR)");
    println!("• Bond pricing and duration analysis");
    println!("• Options pricing with Black-Scholes model");
    println!("• Mortgage calculations and amortization");
    println!("• Portfolio optimization and risk analysis");
    println!("• Value at Risk (VaR) calculations");
    println!("• Financial planning and scenario analysis");
    
    Ok(())
}