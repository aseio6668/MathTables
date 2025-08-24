use mathtables::prelude::*;
use mathtables::plugins::{Plugin, PluginFunction};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MathTables Plugin System Demo ===\n");
    
    let mut math_framework = MathTables::new();
    
    // Create a statistics plugin
    let mut stats_plugin = Plugin::new(
        "statistics".to_string(),
        "1.0.0".to_string(),
        "Statistical analysis functions".to_string(),
    );
    
    // Add mean function
    let mean_function: PluginFunction = Box::new(|args| {
        if let Some(data) = args.get(0).and_then(|arg| arg.downcast_ref::<Vec<f64>>()) {
            let sum: f64 = data.iter().sum();
            let mean = sum / data.len() as f64;
            Box::new(mean)
        } else {
            Box::new(0.0f64)
        }
    });
    
    // Add variance function
    let variance_function: PluginFunction = Box::new(|args| {
        if let Some(data) = args.get(0).and_then(|arg| arg.downcast_ref::<Vec<f64>>()) {
            let sum: f64 = data.iter().sum();
            let mean = sum / data.len() as f64;
            let variance: f64 = data.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / data.len() as f64;
            Box::new(variance)
        } else {
            Box::new(0.0f64)
        }
    });
    
    // Add standard deviation function
    let std_dev_function: PluginFunction = Box::new(|args| {
        if let Some(data) = args.get(0).and_then(|arg| arg.downcast_ref::<Vec<f64>>()) {
            let sum: f64 = data.iter().sum();
            let mean = sum / data.len() as f64;
            let variance: f64 = data.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / data.len() as f64;
            let std_dev = variance.sqrt();
            Box::new(std_dev)
        } else {
            Box::new(0.0f64)
        }
    });
    
    stats_plugin.add_function("mean".to_string(), mean_function);
    stats_plugin.add_function("variance".to_string(), variance_function);
    stats_plugin.add_function("std_dev".to_string(), std_dev_function);
    
    // Register the plugin
    math_framework.plugin_registry().register_plugin(stats_plugin);
    
    // Create a linear algebra plugin
    let mut linalg_plugin = Plugin::new(
        "linear_algebra".to_string(),
        "1.0.0".to_string(),
        "Extended linear algebra operations".to_string(),
    );
    
    // Add vector dot product function
    let dot_product_function: PluginFunction = Box::new(|args| {
        if let (Some(v1), Some(v2)) = (
            args.get(0).and_then(|arg| arg.downcast_ref::<Vec<f64>>()),
            args.get(1).and_then(|arg| arg.downcast_ref::<Vec<f64>>()),
        ) {
            if v1.len() == v2.len() {
                let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
                Box::new(dot_product)
            } else {
                Box::new(0.0f64)
            }
        } else {
            Box::new(0.0f64)
        }
    });
    
    // Add vector magnitude function
    let magnitude_function: PluginFunction = Box::new(|args| {
        if let Some(vector) = args.get(0).and_then(|arg| arg.downcast_ref::<Vec<f64>>()) {
            let magnitude = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
            Box::new(magnitude)
        } else {
            Box::new(0.0f64)
        }
    });
    
    linalg_plugin.add_function("dot_product".to_string(), dot_product_function);
    linalg_plugin.add_function("magnitude".to_string(), magnitude_function);
    
    math_framework.plugin_registry().register_plugin(linalg_plugin);
    
    // Create a financial mathematics plugin
    let mut finance_plugin = Plugin::new(
        "finance".to_string(),
        "1.0.0".to_string(),
        "Financial mathematics calculations".to_string(),
    );
    
    // Add compound interest function
    let compound_interest_function: PluginFunction = Box::new(|args| {
        if let (Some(&principal), Some(&rate), Some(&time), Some(&n)) = (
            args.get(0).and_then(|arg| arg.downcast_ref::<f64>()),
            args.get(1).and_then(|arg| arg.downcast_ref::<f64>()),
            args.get(2).and_then(|arg| arg.downcast_ref::<f64>()),
            args.get(3).and_then(|arg| arg.downcast_ref::<f64>()),
        ) {
            let amount = principal * (1.0 + rate / n).powf(n * time);
            Box::new(amount)
        } else {
            Box::new(0.0f64)
        }
    });
    
    // Add present value function
    let present_value_function: PluginFunction = Box::new(|args| {
        if let (Some(&future_value), Some(&rate), Some(&time)) = (
            args.get(0).and_then(|arg| arg.downcast_ref::<f64>()),
            args.get(1).and_then(|arg| arg.downcast_ref::<f64>()),
            args.get(2).and_then(|arg| arg.downcast_ref::<f64>()),
        ) {
            let present_value = future_value / (1.0 + rate).powf(time);
            Box::new(present_value)
        } else {
            Box::new(0.0f64)
        }
    });
    
    finance_plugin.add_function("compound_interest".to_string(), compound_interest_function);
    finance_plugin.add_function("present_value".to_string(), present_value_function);
    
    math_framework.plugin_registry().register_plugin(finance_plugin);
    
    println!("Registered plugins: {:?}\n", math_framework.plugin_registry().list_plugins());
    
    // Test statistics plugin
    println!("=== Statistics Plugin ===");
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    println!("Data: {:?}", data);
    
    if let Some(mean_result) = math_framework.plugin_registry().call_plugin_function("statistics", "mean", &[&data]) {
        println!("Mean: {:?}", mean_result.downcast_ref::<f64>().unwrap());
    }
    
    if let Some(variance_result) = math_framework.plugin_registry().call_plugin_function("statistics", "variance", &[&data]) {
        println!("Variance: {:?}", variance_result.downcast_ref::<f64>().unwrap());
    }
    
    if let Some(std_dev_result) = math_framework.plugin_registry().call_plugin_function("statistics", "std_dev", &[&data]) {
        println!("Standard Deviation: {:?}", std_dev_result.downcast_ref::<f64>().unwrap());
    }
    
    // Test linear algebra plugin
    println!("\n=== Linear Algebra Plugin ===");
    let vector1 = vec![1.0, 2.0, 3.0];
    let vector2 = vec![4.0, 5.0, 6.0];
    
    println!("Vector 1: {:?}", vector1);
    println!("Vector 2: {:?}", vector2);
    
    if let Some(dot_result) = math_framework.plugin_registry().call_plugin_function("linear_algebra", "dot_product", &[&vector1, &vector2]) {
        println!("Dot Product: {:?}", dot_result.downcast_ref::<f64>().unwrap());
    }
    
    if let Some(mag_result) = math_framework.plugin_registry().call_plugin_function("linear_algebra", "magnitude", &[&vector1]) {
        println!("Vector 1 Magnitude: {:?}", mag_result.downcast_ref::<f64>().unwrap());
    }
    
    // Test finance plugin
    println!("\n=== Finance Plugin ===");
    let principal = 1000.0f64;
    let rate = 0.05f64; // 5% annual rate
    let time = 10.0f64; // 10 years
    let n = 12.0f64; // Compounded monthly
    
    println!("Principal: ${:.2}", principal);
    println!("Annual Rate: {:.1}%", rate * 100.0);
    println!("Time: {:.0} years", time);
    println!("Compounding: {:.0} times per year", n);
    
    if let Some(compound_result) = math_framework.plugin_registry().call_plugin_function("finance", "compound_interest", &[&principal, &rate, &time, &n]) {
        println!("Compound Interest Amount: ${:.2}", compound_result.downcast_ref::<f64>().unwrap());
    }
    
    let future_value = 1628.89f64;
    if let Some(pv_result) = math_framework.plugin_registry().call_plugin_function("finance", "present_value", &[&future_value, &rate, &time]) {
        println!("Present Value of ${:.2}: ${:.2}", future_value, pv_result.downcast_ref::<f64>().unwrap());
    }
    
    Ok(())
}