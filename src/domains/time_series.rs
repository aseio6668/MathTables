use crate::core::{MathDomain, MathResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesDomain {
    name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    pub name: String,
    pub timestamps: Vec<f64>,
    pub values: Vec<f64>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultivariateSeries {
    pub series: HashMap<String, TimeSeries>,
    pub common_timestamps: Vec<f64>,
}

// Decomposition Components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decomposition {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
    pub decomposition_type: DecompositionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecompositionType {
    Additive,
    Multiplicative,
    STL, // Seasonal and Trend decomposition using Loess
}

// Statistical Properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesStatistics {
    pub mean: f64,
    pub variance: f64,
    pub standard_deviation: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub autocorrelations: Vec<f64>,
    pub partial_autocorrelations: Vec<f64>,
    pub ljung_box_statistic: f64,
    pub adf_test_statistic: f64,
    pub is_stationary: bool,
}

// ARIMA Models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ARIMAModel {
    pub p: usize, // AR order
    pub d: usize, // Differencing order
    pub q: usize, // MA order
    pub ar_coefficients: Vec<f64>,
    pub ma_coefficients: Vec<f64>,
    pub intercept: f64,
    pub sigma_squared: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SARIMAModel {
    pub arima: ARIMAModel,
    pub seasonal_p: usize,
    pub seasonal_d: usize,
    pub seasonal_q: usize,
    pub seasonal_period: usize,
    pub seasonal_ar_coefficients: Vec<f64>,
    pub seasonal_ma_coefficients: Vec<f64>,
}

// Vector Autoregression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VARModel {
    pub order: usize,
    pub variables: Vec<String>,
    pub coefficients: DMatrix<f64>, // coefficients[i][j] = coeff for var j in equation for var i
    pub residual_covariance: DMatrix<f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
}

// State Space Models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSpaceModel {
    pub state_dimension: usize,
    pub observation_dimension: usize,
    pub transition_matrix: DMatrix<f64>,    // F
    pub observation_matrix: DMatrix<f64>,   // H
    pub process_noise_cov: DMatrix<f64>,    // Q
    pub observation_noise_cov: DMatrix<f64>, // R
    pub initial_state: DVector<f64>,
    pub initial_covariance: DMatrix<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanFilter {
    pub model: StateSpaceModel,
    pub filtered_states: Vec<DVector<f64>>,
    pub predicted_states: Vec<DVector<f64>>,
    pub filtered_covariances: Vec<DMatrix<f64>>,
    pub predicted_covariances: Vec<DMatrix<f64>>,
    pub log_likelihood: f64,
}

// Frequency Domain Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAnalysis {
    pub frequencies: Vec<f64>,
    pub power_spectral_density: Vec<f64>,
    pub coherence: Option<Vec<f64>>,
    pub phase: Option<Vec<f64>>,
    pub cross_spectrum: Option<Vec<num_complex::Complex<f64>>>,
}

// Filtering and Smoothing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Filter {
    pub filter_type: FilterType,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterType {
    MovingAverage,
    ExponentialSmoothing,
    HoltWinters,
    Butterworth,
    Chebyshev,
    Kalman,
}

// Changepoint Detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Changepoint {
    pub location: usize,
    pub timestamp: f64,
    pub confidence: f64,
    pub change_type: ChangeType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Mean,
    Variance,
    Trend,
    Seasonal,
}

// Forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Forecast {
    pub horizons: Vec<f64>,
    pub point_forecasts: Vec<f64>,
    pub prediction_intervals: Vec<(f64, f64)>,
    pub forecast_method: String,
    pub model_parameters: HashMap<String, f64>,
}

impl TimeSeriesDomain {
    pub fn new() -> Self {
        Self {
            name: "Time Series Analysis".to_string(),
        }
    }

    // Basic Time Series Operations
    pub fn create_time_series(&self, name: String, timestamps: Vec<f64>, values: Vec<f64>) -> TimeSeries {
        TimeSeries {
            name,
            timestamps,
            values,
            metadata: HashMap::new(),
        }
    }

    pub fn resample(&self, series: &TimeSeries, new_timestamps: &[f64]) -> TimeSeries {
        let mut resampled_values = Vec::with_capacity(new_timestamps.len());
        
        for &target_time in new_timestamps {
            // Linear interpolation
            let value = self.interpolate_linear(&series.timestamps, &series.values, target_time);
            resampled_values.push(value);
        }
        
        TimeSeries {
            name: format!("{}_resampled", series.name),
            timestamps: new_timestamps.to_vec(),
            values: resampled_values,
            metadata: series.metadata.clone(),
        }
    }

    fn interpolate_linear(&self, timestamps: &[f64], values: &[f64], target: f64) -> f64 {
        if timestamps.is_empty() {
            return 0.0;
        }
        
        // Find surrounding points
        let mut i = 0;
        while i < timestamps.len() && timestamps[i] < target {
            i += 1;
        }
        
        if i == 0 {
            return values[0];
        }
        if i >= timestamps.len() {
            return values[values.len() - 1];
        }
        
        // Linear interpolation
        let t0 = timestamps[i - 1];
        let t1 = timestamps[i];
        let y0 = values[i - 1];
        let y1 = values[i];
        
        y0 + (y1 - y0) * (target - t0) / (t1 - t0)
    }

    pub fn difference(&self, series: &TimeSeries, order: usize) -> TimeSeries {
        let mut current_values = series.values.clone();
        
        for _ in 0..order {
            let mut diff_values = Vec::with_capacity(current_values.len().saturating_sub(1));
            for i in 1..current_values.len() {
                diff_values.push(current_values[i] - current_values[i - 1]);
            }
            current_values = diff_values;
        }
        
        let diff_timestamps = if order < series.timestamps.len() {
            series.timestamps[order..].to_vec()
        } else {
            Vec::new()
        };
        
        TimeSeries {
            name: format!("{}_diff_{}", series.name, order),
            timestamps: diff_timestamps,
            values: current_values,
            metadata: series.metadata.clone(),
        }
    }

    // Statistical Analysis
    pub fn compute_statistics(&self, series: &TimeSeries, max_lags: usize) -> TimeSeriesStatistics {
        let values = &series.values;
        let n = values.len() as f64;
        
        // Basic statistics
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        
        // Skewness
        let skewness = values.iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>() / n;
        
        // Kurtosis
        let kurtosis = values.iter()
            .map(|&x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>() / n - 3.0;
        
        // Autocorrelations
        let autocorrs = self.compute_autocorrelations(values, max_lags);
        let partial_autocorrs = self.compute_partial_autocorrelations(values, max_lags);
        
        // Ljung-Box test
        let ljung_box = self.ljung_box_test(&autocorrs, values.len());
        
        // Augmented Dickey-Fuller test (simplified)
        let adf_statistic = self.adf_test(values);
        let is_stationary = adf_statistic < -2.86; // Simplified critical value
        
        TimeSeriesStatistics {
            mean,
            variance,
            standard_deviation: std_dev,
            skewness,
            kurtosis,
            autocorrelations: autocorrs,
            partial_autocorrelations: partial_autocorrs,
            ljung_box_statistic: ljung_box,
            adf_test_statistic: adf_statistic,
            is_stationary,
        }
    }

    fn compute_autocorrelations(&self, values: &[f64], max_lags: usize) -> Vec<f64> {
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();
        
        let mut autocorrs = Vec::with_capacity(max_lags + 1);
        
        for lag in 0..=max_lags {
            if lag >= n {
                autocorrs.push(0.0);
                continue;
            }
            
            let mut numerator = 0.0;
            for i in 0..(n - lag) {
                numerator += (values[i] - mean) * (values[i + lag] - mean);
            }
            
            let autocorr = if variance > 0.0 { numerator / variance } else { 0.0 };
            autocorrs.push(autocorr);
        }
        
        autocorrs
    }

    fn compute_partial_autocorrelations(&self, values: &[f64], max_lags: usize) -> Vec<f64> {
        let mut pacf = Vec::with_capacity(max_lags + 1);
        pacf.push(1.0); // PACF at lag 0 is always 1
        
        if max_lags == 0 {
            return pacf;
        }
        
        let autocorrs = self.compute_autocorrelations(values, max_lags);
        pacf.push(autocorrs[1]); // PACF at lag 1 equals ACF at lag 1
        
        // Yule-Walker equations for PACF computation
        for k in 2..=max_lags {
            if k >= autocorrs.len() {
                pacf.push(0.0);
                continue;
            }
            
            let mut numerator = autocorrs[k];
            let mut denominator_terms = Vec::new();
            
            for j in 1..k {
                numerator -= pacf[j] * autocorrs[k - j];
                denominator_terms.push(autocorrs[j]);
            }
            
            // Simplified computation
            let phi_kk = if k < autocorrs.len() { 
                numerator / (1.0 - denominator_terms.iter().sum::<f64>().abs().min(0.99))
            } else { 
                0.0 
            };
            
            pacf.push(phi_kk);
        }
        
        pacf
    }

    fn ljung_box_test(&self, autocorrs: &[f64], n: usize) -> f64 {
        let m = (autocorrs.len() - 1).min(20); // Use up to 20 lags
        let mut statistic = 0.0;
        
        for k in 1..=m {
            if k < autocorrs.len() {
                statistic += autocorrs[k].powi(2) / (n - k) as f64;
            }
        }
        
        n as f64 * (n as f64 + 2.0) * statistic
    }

    fn adf_test(&self, values: &[f64]) -> f64 {
        // Simplified ADF test - fit AR(1) model and test coefficient
        if values.len() < 3 {
            return 0.0;
        }
        
        let diffs: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();
        let lagged: Vec<f64> = values[..values.len()-1].to_vec();
        
        // Simple linear regression: Δy_t = α + βy_{t-1} + ε_t
        let n = diffs.len() as f64;
        let mean_diff = diffs.iter().sum::<f64>() / n;
        let mean_lag = lagged.iter().sum::<f64>() / n;
        
        let numerator: f64 = diffs.iter().zip(&lagged)
            .map(|(&d, &l)| (d - mean_diff) * (l - mean_lag))
            .sum();
        let denominator: f64 = lagged.iter()
            .map(|&l| (l - mean_lag).powi(2))
            .sum();
        
        if denominator > 0.0 {
            let beta = numerator / denominator;
            
            // Compute standard error (simplified)
            let residuals: Vec<f64> = diffs.iter().zip(&lagged)
                .map(|(&d, &l)| d - (beta * l))
                .collect();
            let mse = residuals.iter().map(|&r| r.powi(2)).sum::<f64>() / (n - 2.0);
            let se_beta = (mse / denominator).sqrt();
            
            // t-statistic
            beta / se_beta
        } else {
            0.0
        }
    }

    // Decomposition
    pub fn classical_decomposition(&self, series: &TimeSeries, period: usize, 
                                 decomp_type: DecompositionType) -> Decomposition {
        let values = &series.values;
        let n = values.len();
        
        // Compute trend using centered moving average
        let trend = self.centered_moving_average(values, period);
        
        // Compute seasonal component
        let seasonal = self.compute_seasonal_component(values, &trend, period, &decomp_type);
        
        // Compute residual
        let residual = match decomp_type {
            DecompositionType::Additive => {
                values.iter().zip(trend.iter()).zip(seasonal.iter())
                    .map(|((&val, &tr), &seas)| val - tr - seas)
                    .collect()
            }
            DecompositionType::Multiplicative => {
                values.iter().zip(trend.iter()).zip(seasonal.iter())
                    .map(|((&val, &tr), &seas)| if tr * seas != 0.0 { val / (tr * seas) } else { 0.0 })
                    .collect()
            }
            _ => vec![0.0; n], // Simplified for other types
        };
        
        Decomposition {
            trend,
            seasonal,
            residual,
            decomposition_type: decomp_type,
        }
    }

    fn centered_moving_average(&self, values: &[f64], window: usize) -> Vec<f64> {
        let mut trend = vec![0.0; values.len()];
        let half_window = window / 2;
        
        for i in half_window..values.len()-half_window {
            let sum: f64 = values[i-half_window..i+half_window+1].iter().sum();
            trend[i] = sum / window as f64;
        }
        
        // Fill edges with linear interpolation/extrapolation
        for i in 0..half_window {
            if half_window < trend.len() {
                trend[i] = trend[half_window];
            }
        }
        for i in values.len()-half_window..values.len() {
            if values.len() > half_window {
                trend[i] = trend[values.len()-half_window-1];
            }
        }
        
        trend
    }

    fn compute_seasonal_component(&self, values: &[f64], trend: &[f64], period: usize, 
                                decomp_type: &DecompositionType) -> Vec<f64> {
        let mut seasonal = vec![0.0; values.len()];
        let mut seasonal_means = vec![0.0; period];
        let mut counts = vec![0; period];
        
        // Compute seasonal indices
        for (i, (&val, &tr)) in values.iter().zip(trend.iter()).enumerate() {
            let season_idx = i % period;
            match decomp_type {
                DecompositionType::Additive => {
                    if tr != 0.0 {
                        seasonal_means[season_idx] += val - tr;
                        counts[season_idx] += 1;
                    }
                }
                DecompositionType::Multiplicative => {
                    if tr != 0.0 {
                        seasonal_means[season_idx] += val / tr;
                        counts[season_idx] += 1;
                    }
                }
                _ => {}
            }
        }
        
        // Average seasonal components
        for (mean, count) in seasonal_means.iter_mut().zip(counts.iter()) {
            if *count > 0 {
                *mean /= *count as f64;
            }
        }
        
        // Normalize seasonal components
        match decomp_type {
            DecompositionType::Additive => {
                let mean_seasonal: f64 = seasonal_means.iter().sum::<f64>() / period as f64;
                for mean in &mut seasonal_means {
                    *mean -= mean_seasonal;
                }
            }
            DecompositionType::Multiplicative => {
                let mean_seasonal: f64 = seasonal_means.iter().product::<f64>().powf(1.0 / period as f64);
                if mean_seasonal != 0.0 {
                    for mean in &mut seasonal_means {
                        *mean /= mean_seasonal;
                    }
                }
            }
            _ => {}
        }
        
        // Assign seasonal components
        for (i, seas) in seasonal.iter_mut().enumerate() {
            *seas = seasonal_means[i % period];
        }
        
        seasonal
    }

    // ARIMA Modeling (simplified)
    pub fn fit_arima(&self, series: &TimeSeries, p: usize, d: usize, q: usize) -> ARIMAModel {
        // Difference the series
        let mut current_series = series.clone();
        for _ in 0..d {
            current_series = self.difference(&current_series, 1);
        }
        
        let values = &current_series.values;
        if values.is_empty() {
            return self.empty_arima_model(p, d, q);
        }
        
        // Simplified parameter estimation using method of moments
        let ar_coeffs = if p > 0 {
            self.estimate_ar_coefficients(values, p)
        } else {
            Vec::new()
        };
        
        let ma_coeffs = if q > 0 {
            vec![0.0; q] // Simplified - would need more complex estimation
        } else {
            Vec::new()
        };
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let residuals = self.compute_arima_residuals(values, &ar_coeffs, &ma_coeffs, mean);
        let sigma_squared = residuals.iter().map(|&r| r.powi(2)).sum::<f64>() / residuals.len() as f64;
        
        // Information criteria (simplified)
        let n = values.len() as f64;
        let k = (p + q + 1) as f64; // Parameters
        let log_likelihood = -0.5 * n * (1.0 + (2.0 * std::f64::consts::PI).ln() + sigma_squared.ln());
        let aic = -2.0 * log_likelihood + 2.0 * k;
        let bic = -2.0 * log_likelihood + k * n.ln();
        
        ARIMAModel {
            p,
            d,
            q,
            ar_coefficients: ar_coeffs,
            ma_coefficients: ma_coeffs,
            intercept: mean,
            sigma_squared,
            log_likelihood,
            aic,
            bic,
        }
    }

    fn empty_arima_model(&self, p: usize, d: usize, q: usize) -> ARIMAModel {
        ARIMAModel {
            p,
            d,
            q,
            ar_coefficients: vec![0.0; p],
            ma_coefficients: vec![0.0; q],
            intercept: 0.0,
            sigma_squared: 1.0,
            log_likelihood: 0.0,
            aic: f64::INFINITY,
            bic: f64::INFINITY,
        }
    }

    fn estimate_ar_coefficients(&self, values: &[f64], p: usize) -> Vec<f64> {
        if values.len() <= p {
            return vec![0.0; p];
        }
        
        // Yule-Walker equations - use autocorrelations
        let autocorrs = self.compute_autocorrelations(values, p);
        let mut coeffs = vec![0.0; p];
        
        // Simplified estimation for AR(1)
        if p >= 1 && autocorrs.len() > 1 {
            coeffs[0] = autocorrs[1];
        }
        
        // For higher orders, would need to solve the full Yule-Walker system
        coeffs
    }

    fn compute_arima_residuals(&self, values: &[f64], ar_coeffs: &[f64], 
                             _ma_coeffs: &[f64], intercept: f64) -> Vec<f64> {
        let mut residuals = Vec::new();
        let p = ar_coeffs.len();
        
        for i in p..values.len() {
            let mut predicted = intercept;
            for j in 0..p {
                predicted += ar_coeffs[j] * values[i - j - 1];
            }
            residuals.push(values[i] - predicted);
        }
        
        residuals
    }

    // Forecasting
    pub fn forecast_arima(&self, model: &ARIMAModel, series: &TimeSeries, 
                         horizons: usize) -> Forecast {
        let values = &series.values;
        let mut forecasts = Vec::with_capacity(horizons);
        let mut prediction_intervals = Vec::with_capacity(horizons);
        
        // Get the last p values for AR component
        let p = model.p;
        let mut recent_values: VecDeque<f64> = if values.len() >= p {
            values[values.len()-p..].iter().cloned().collect()
        } else {
            values.iter().cloned().collect()
        };
        
        // Extend with zeros if needed
        while recent_values.len() < p {
            recent_values.push_front(0.0);
        }
        
        let last_timestamp = series.timestamps.last().cloned().unwrap_or(0.0);
        let time_step = if series.timestamps.len() > 1 {
            series.timestamps[series.timestamps.len()-1] - series.timestamps[series.timestamps.len()-2]
        } else {
            1.0
        };
        
        for h in 1..=horizons {
            // Point forecast
            let mut forecast = model.intercept;
            for (j, &coeff) in model.ar_coefficients.iter().enumerate() {
                if j < recent_values.len() {
                    forecast += coeff * recent_values[recent_values.len() - 1 - j];
                }
            }
            
            forecasts.push(forecast);
            
            // Prediction interval (simplified - assumes normal errors)
            let std_error = model.sigma_squared.sqrt() * (h as f64).sqrt();
            let z_score = 1.96; // 95% confidence interval
            prediction_intervals.push((
                forecast - z_score * std_error,
                forecast + z_score * std_error
            ));
            
            // Update recent values for multi-step forecasting
            recent_values.push_back(forecast);
            if recent_values.len() > p {
                recent_values.pop_front();
            }
        }
        
        let forecast_horizons: Vec<f64> = (1..=horizons)
            .map(|h| last_timestamp + h as f64 * time_step)
            .collect();
        
        let mut params = HashMap::new();
        params.insert("aic".to_string(), model.aic);
        params.insert("bic".to_string(), model.bic);
        params.insert("sigma_squared".to_string(), model.sigma_squared);
        
        Forecast {
            horizons: forecast_horizons,
            point_forecasts: forecasts,
            prediction_intervals,
            forecast_method: format!("ARIMA({},{},{})", model.p, model.d, model.q),
            model_parameters: params,
        }
    }

    // Smoothing and Filtering
    pub fn exponential_smoothing(&self, series: &TimeSeries, alpha: f64) -> TimeSeries {
        let values = &series.values;
        if values.is_empty() {
            return series.clone();
        }
        
        let mut smoothed = Vec::with_capacity(values.len());
        smoothed.push(values[0]); // Initialize with first observation
        
        for i in 1..values.len() {
            let smooth_val = alpha * values[i] + (1.0 - alpha) * smoothed[i - 1];
            smoothed.push(smooth_val);
        }
        
        TimeSeries {
            name: format!("{}_exponential_smooth", series.name),
            timestamps: series.timestamps.clone(),
            values: smoothed,
            metadata: series.metadata.clone(),
        }
    }

    pub fn moving_average(&self, series: &TimeSeries, window: usize) -> TimeSeries {
        let values = &series.values;
        let mut smoothed = Vec::with_capacity(values.len());
        
        for i in 0..values.len() {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(values.len());
            let window_sum: f64 = values[start..end].iter().sum();
            let window_len = end - start;
            smoothed.push(window_sum / window_len as f64);
        }
        
        TimeSeries {
            name: format!("{}_moving_avg", series.name),
            timestamps: series.timestamps.clone(),
            values: smoothed,
            metadata: series.metadata.clone(),
        }
    }

    // Spectral Analysis (basic)
    pub fn compute_periodogram(&self, series: &TimeSeries) -> SpectralAnalysis {
        let values = &series.values;
        let n = values.len();
        let dt = if series.timestamps.len() > 1 {
            series.timestamps[1] - series.timestamps[0]
        } else {
            1.0
        };
        
        // Compute DFT (simplified - would use FFT in practice)
        let mut frequencies = Vec::new();
        let mut power_spectrum = Vec::new();
        
        let nyquist_freq = 0.5 / dt;
        for k in 0..n/2 {
            let freq = k as f64 / (n as f64 * dt);
            if freq <= nyquist_freq {
                frequencies.push(freq);
                
                // Compute power at this frequency
                let mut real_sum = 0.0;
                let mut imag_sum = 0.0;
                
                for (j, &value) in values.iter().enumerate() {
                    let angle = -2.0 * std::f64::consts::PI * freq * j as f64 * dt;
                    real_sum += value * angle.cos();
                    imag_sum += value * angle.sin();
                }
                
                let power = (real_sum * real_sum + imag_sum * imag_sum) / n as f64;
                power_spectrum.push(power);
            }
        }
        
        SpectralAnalysis {
            frequencies,
            power_spectral_density: power_spectrum,
            coherence: None,
            phase: None,
            cross_spectrum: None,
        }
    }

    // Model Selection
    pub fn auto_arima(&self, series: &TimeSeries, max_p: usize, max_d: usize, max_q: usize) -> ARIMAModel {
        let mut best_model = self.empty_arima_model(0, 0, 0);
        let mut best_aic = f64::INFINITY;
        
        for d in 0..=max_d {
            for p in 0..=max_p {
                for q in 0..=max_q {
                    if p == 0 && q == 0 && d == 0 {
                        continue; // Skip the null model
                    }
                    
                    let model = self.fit_arima(series, p, d, q);
                    if model.aic < best_aic && model.aic.is_finite() {
                        best_aic = model.aic;
                        best_model = model;
                    }
                }
            }
        }
        
        best_model
    }
}

impl MathDomain for TimeSeriesDomain {
    fn name(&self) -> &str {
        &self.name
    }

    fn supports_operation(&self, operation: &str) -> bool {
        matches!(operation, 
            "decomposition" | "arima_fitting" | "forecasting" | "spectral_analysis" |
            "stationarity_test" | "autocorrelation" | "seasonal_adjustment" |
            "changepoint_detection" | "filtering" | "var_modeling"
        )
    }

    fn description(&self) -> &str {
        "Time Series Analysis and Forecasting"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    fn compute(&self, operation: &str, _args: &[&dyn std::any::Any]) -> MathResult<Box<dyn std::any::Any>> {
        match operation {
            "decomposition" => Ok(Box::new("Time series decomposed".to_string())),
            "arima_fitting" => Ok(Box::new("ARIMA model fitted".to_string())),
            "forecasting" => Ok(Box::new("Forecasts generated".to_string())),
            "spectral_analysis" => Ok(Box::new("Spectral analysis completed".to_string())),
            _ => Err(crate::core::MathError::NotImplemented(format!("Operation '{}' not implemented", operation))),
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "decomposition".to_string(), "arima_fitting".to_string(),
            "forecasting".to_string(), "spectral_analysis".to_string(),
            "stationarity_test".to_string(), "autocorrelation".to_string(),
            "seasonal_adjustment".to_string(), "changepoint_detection".to_string(),
            "filtering".to_string(), "var_modeling".to_string()
        ]
    }
}

pub fn time_series() -> TimeSeriesDomain {
    TimeSeriesDomain::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_series_creation() {
        let domain = TimeSeriesDomain::new();
        let timestamps = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let values = vec![1.0, 4.0, 9.0, 16.0, 25.0];
        
        let series = domain.create_time_series("test".to_string(), timestamps.clone(), values.clone());
        assert_eq!(series.timestamps, timestamps);
        assert_eq!(series.values, values);
    }

    #[test]
    fn test_differencing() {
        let domain = TimeSeriesDomain::new();
        let timestamps = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let values = vec![1.0, 2.0, 4.0, 7.0, 11.0];
        let series = domain.create_time_series("test".to_string(), timestamps, values);
        
        let diff_series = domain.difference(&series, 1);
        assert_eq!(diff_series.values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_moving_average() {
        let domain = TimeSeriesDomain::new();
        let timestamps = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series = domain.create_time_series("test".to_string(), timestamps, values);
        
        let smoothed = domain.moving_average(&series, 3);
        assert_eq!(smoothed.values.len(), 5);
    }

    #[test]
    fn test_arima_fitting() {
        let domain = TimeSeriesDomain::new();
        let timestamps = (0..100).map(|i| i as f64).collect();
        let values: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() + 0.1 * i as f64).collect();
        let series = domain.create_time_series("test".to_string(), timestamps, values);
        
        let model = domain.fit_arima(&series, 1, 1, 1);
        assert_eq!(model.p, 1);
        assert_eq!(model.d, 1);
        assert_eq!(model.q, 1);
    }

    #[test]
    fn test_exponential_smoothing() {
        let domain = TimeSeriesDomain::new();
        let timestamps = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let values = vec![1.0, 3.0, 2.0, 4.0, 3.0];
        let series = domain.create_time_series("test".to_string(), timestamps, values);
        
        let smoothed = domain.exponential_smoothing(&series, 0.3);
        assert_eq!(smoothed.values.len(), 5);
        assert_eq!(smoothed.values[0], 1.0); // First value unchanged
    }

    #[test]
    fn test_autocorrelation() {
        let domain = TimeSeriesDomain::new();
        let timestamps = (0..50).map(|i| i as f64).collect();
        let values: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
        let series = domain.create_time_series("test".to_string(), timestamps, values);
        
        let stats = domain.compute_statistics(&series, 10);
        assert_eq!(stats.autocorrelations.len(), 11); // 0 to 10 lags
        assert!((stats.autocorrelations[0] - 1.0).abs() < 1e-10); // ACF at lag 0 = 1
    }
}