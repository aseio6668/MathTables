use crate::core::{MathDomain, MathResult, MathError};
use num_complex::Complex64;
use std::any::Any;

#[derive(Debug, Clone)]
pub struct Signal {
    pub samples: Vec<f64>,
    pub sample_rate: f64,
    pub duration: f64,
}

#[derive(Debug, Clone)]
pub struct FFTResult {
    pub frequencies: Vec<Complex64>,
    pub magnitude_spectrum: Vec<f64>,
    pub phase_spectrum: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FilterCoefficients {
    pub b_coeffs: Vec<f64>, // numerator coefficients
    pub a_coeffs: Vec<f64>, // denominator coefficients
}

#[derive(Debug, Clone)]
pub struct WindowFunction {
    pub window_type: WindowType,
    pub coefficients: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum WindowType {
    Rectangular,
    Hamming,
    Hanning,
    Blackman,
    Kaiser { beta: f64 },
}

pub struct SignalProcessingDomain;

impl SignalProcessingDomain {
    pub fn new() -> Self {
        Self
    }
    
    pub fn create_signal(samples: Vec<f64>, sample_rate: f64) -> MathResult<Signal> {
        if samples.is_empty() || sample_rate <= 0.0 {
            return Err(MathError::InvalidArgument("Invalid signal parameters".to_string()));
        }
        
        let duration = samples.len() as f64 / sample_rate;
        
        Ok(Signal {
            samples,
            sample_rate,
            duration,
        })
    }
    
    pub fn generate_sine_wave(frequency: f64, amplitude: f64, duration: f64, sample_rate: f64) -> MathResult<Signal> {
        if frequency < 0.0 || duration <= 0.0 || sample_rate <= 0.0 {
            return Err(MathError::InvalidArgument("Invalid sine wave parameters".to_string()));
        }
        
        let num_samples = (duration * sample_rate) as usize;
        let mut samples = Vec::with_capacity(num_samples);
        
        for n in 0..num_samples {
            let t = n as f64 / sample_rate;
            let sample = amplitude * (2.0 * std::f64::consts::PI * frequency * t).sin();
            samples.push(sample);
        }
        
        Ok(Signal {
            samples,
            sample_rate,
            duration,
        })
    }
    
    pub fn fft(signal: &Signal) -> MathResult<FFTResult> {
        let n = signal.samples.len();
        
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(MathError::InvalidArgument("FFT requires power-of-2 length signal".to_string()));
        }
        
        let mut x: Vec<Complex64> = signal.samples.iter()
            .map(|&sample| Complex64::new(sample, 0.0))
            .collect();
        
        Self::fft_recursive(&mut x);
        
        let magnitude_spectrum: Vec<f64> = x.iter()
            .map(|&c| c.norm())
            .collect();
            
        let phase_spectrum: Vec<f64> = x.iter()
            .map(|&c| c.arg())
            .collect();
        
        Ok(FFTResult {
            frequencies: x,
            magnitude_spectrum,
            phase_spectrum,
        })
    }
    
    fn fft_recursive(x: &mut [Complex64]) {
        let n = x.len();
        
        if n <= 1 {
            return;
        }
        
        let mut even: Vec<Complex64> = x.iter().step_by(2).copied().collect();
        let mut odd: Vec<Complex64> = x.iter().skip(1).step_by(2).copied().collect();
        
        Self::fft_recursive(&mut even);
        Self::fft_recursive(&mut odd);
        
        for k in 0..(n / 2) {
            let t = Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * k as f64 / n as f64) * odd[k];
            x[k] = even[k] + t;
            x[k + n / 2] = even[k] - t;
        }
    }
    
    pub fn ifft(fft_result: &FFTResult) -> MathResult<Signal> {
        let n = fft_result.frequencies.len();
        if n == 0 {
            return Err(MathError::InvalidArgument("Empty FFT result".to_string()));
        }
        
        let mut x: Vec<Complex64> = fft_result.frequencies.iter()
            .map(|&c| c.conj())
            .collect();
        
        Self::fft_recursive(&mut x);
        
        let samples: Vec<f64> = x.iter()
            .map(|c| c.re / n as f64)
            .collect();
        
        let duration = samples.len() as f64;
        Ok(Signal {
            samples,
            sample_rate: 1.0, // Default sample rate, should be provided by caller
            duration,
        })
    }
    
    pub fn apply_window(signal: &Signal, window: &WindowFunction) -> MathResult<Signal> {
        if signal.samples.len() != window.coefficients.len() {
            return Err(MathError::InvalidArgument("Signal and window lengths must match".to_string()));
        }
        
        let windowed_samples: Vec<f64> = signal.samples.iter()
            .zip(window.coefficients.iter())
            .map(|(&sample, &coeff)| sample * coeff)
            .collect();
        
        Ok(Signal {
            samples: windowed_samples,
            sample_rate: signal.sample_rate,
            duration: signal.duration,
        })
    }
    
    pub fn create_hamming_window(length: usize) -> WindowFunction {
        let coefficients: Vec<f64> = (0..length)
            .map(|n| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * n as f64 / (length - 1) as f64).cos())
            .collect();
        
        WindowFunction {
            window_type: WindowType::Hamming,
            coefficients,
        }
    }
    
    pub fn create_hanning_window(length: usize) -> WindowFunction {
        let coefficients: Vec<f64> = (0..length)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / (length - 1) as f64).cos()))
            .collect();
        
        WindowFunction {
            window_type: WindowType::Hanning,
            coefficients,
        }
    }
    
    pub fn create_blackman_window(length: usize) -> WindowFunction {
        let coefficients: Vec<f64> = (0..length)
            .map(|n| {
                let n_norm = n as f64 / (length - 1) as f64;
                0.42 - 0.5 * (2.0 * std::f64::consts::PI * n_norm).cos() 
                    + 0.08 * (4.0 * std::f64::consts::PI * n_norm).cos()
            })
            .collect();
        
        WindowFunction {
            window_type: WindowType::Blackman,
            coefficients,
        }
    }
    
    pub fn convolution(signal1: &Signal, signal2: &Signal) -> MathResult<Signal> {
        let n1 = signal1.samples.len();
        let n2 = signal2.samples.len();
        
        if n1 == 0 || n2 == 0 {
            return Err(MathError::InvalidArgument("Empty signals for convolution".to_string()));
        }
        
        let result_length = n1 + n2 - 1;
        let mut result_samples = vec![0.0; result_length];
        
        for i in 0..n1 {
            for j in 0..n2 {
                result_samples[i + j] += signal1.samples[i] * signal2.samples[j];
            }
        }
        
        Ok(Signal {
            samples: result_samples,
            sample_rate: signal1.sample_rate,
            duration: result_length as f64 / signal1.sample_rate,
        })
    }
    
    pub fn correlation(signal1: &Signal, signal2: &Signal) -> MathResult<Signal> {
        let n1 = signal1.samples.len();
        let n2 = signal2.samples.len();
        
        if n1 == 0 || n2 == 0 {
            return Err(MathError::InvalidArgument("Empty signals for correlation".to_string()));
        }
        
        let result_length = n1 + n2 - 1;
        let mut result_samples = vec![0.0; result_length];
        
        for i in 0..n1 {
            for j in 0..n2 {
                if i + n2 > j {
                    result_samples[i + n2 - 1 - j] += signal1.samples[i] * signal2.samples[j];
                }
            }
        }
        
        Ok(Signal {
            samples: result_samples,
            sample_rate: signal1.sample_rate,
            duration: result_length as f64 / signal1.sample_rate,
        })
    }
    
    pub fn butterworth_lowpass_filter(cutoff_freq: f64, sample_rate: f64, order: usize) -> MathResult<FilterCoefficients> {
        if cutoff_freq <= 0.0 || sample_rate <= 0.0 || order == 0 {
            return Err(MathError::InvalidArgument("Invalid filter parameters".to_string()));
        }
        
        let nyquist = sample_rate / 2.0;
        let normalized_cutoff = cutoff_freq / nyquist;
        
        if normalized_cutoff >= 1.0 {
            return Err(MathError::InvalidArgument("Cutoff frequency must be less than Nyquist frequency".to_string()));
        }
        
        let wc = (std::f64::consts::PI * normalized_cutoff).tan();
        
        let mut b_coeffs = vec![0.0; order + 1];
        let mut a_coeffs = vec![0.0; order + 1];
        
        for k in 0..=order {
            b_coeffs[k] = Self::binomial_coefficient(order, k) as f64 * wc.powi(k as i32);
        }
        
        for k in 0..=order {
            let mut sum = 0.0;
            for j in 0..=k {
                sum += Self::binomial_coefficient(k, j) as f64 * wc.powi(j as i32) * (-1.0f64).powi((k - j) as i32);
            }
            a_coeffs[k] = sum;
        }
        
        let norm_factor = a_coeffs[order];
        for coeff in &mut a_coeffs {
            *coeff /= norm_factor;
        }
        for coeff in &mut b_coeffs {
            *coeff /= norm_factor;
        }
        
        Ok(FilterCoefficients { b_coeffs, a_coeffs })
    }
    
    fn binomial_coefficient(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }
    
    pub fn apply_filter(signal: &Signal, filter: &FilterCoefficients) -> MathResult<Signal> {
        let n = signal.samples.len();
        let mut filtered_samples = vec![0.0; n];
        
        let b_order = filter.b_coeffs.len();
        let a_order = filter.a_coeffs.len();
        
        for i in 0..n {
            let mut output = 0.0;
            
            for j in 0..b_order {
                if i >= j {
                    output += filter.b_coeffs[j] * signal.samples[i - j];
                }
            }
            
            for j in 1..a_order {
                if i >= j {
                    output -= filter.a_coeffs[j] * filtered_samples[i - j];
                }
            }
            
            filtered_samples[i] = output;
        }
        
        Ok(Signal {
            samples: filtered_samples,
            sample_rate: signal.sample_rate,
            duration: signal.duration,
        })
    }
    
    pub fn signal_energy(signal: &Signal) -> f64 {
        signal.samples.iter().map(|&x| x * x).sum()
    }
    
    pub fn signal_power(signal: &Signal) -> f64 {
        Self::signal_energy(signal) / signal.samples.len() as f64
    }
    
    pub fn signal_rms(signal: &Signal) -> f64 {
        Self::signal_power(signal).sqrt()
    }
}

impl MathDomain for SignalProcessingDomain {
    fn name(&self) -> &str { "Signal Processing" }
    fn description(&self) -> &str { "Digital signal processing including FFT, filtering, windowing, and signal analysis" }
    fn version(&self) -> &str { "1.0.0" }
    
    fn compute(&self, operation: &str, args: &[&dyn Any]) -> MathResult<Box<dyn Any>> {
        match operation {
            "create_signal" => {
                if args.len() != 2 {
                    return Err(MathError::InvalidArgument("create_signal requires 2 arguments".to_string()));
                }
                let samples = args[0].downcast_ref::<Vec<f64>>().ok_or_else(|| MathError::InvalidArgument("First argument must be Vec<f64>".to_string()))?;
                let sample_rate = args[1].downcast_ref::<f64>().ok_or_else(|| MathError::InvalidArgument("Second argument must be f64".to_string()))?;
                Ok(Box::new(Self::create_signal(samples.clone(), *sample_rate)?))
            },
            "fft" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("fft requires 1 argument".to_string()));
                }
                let signal = args[0].downcast_ref::<Signal>().ok_or_else(|| MathError::InvalidArgument("Argument must be Signal".to_string()))?;
                Ok(Box::new(Self::fft(signal)?))
            },
            "signal_energy" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("signal_energy requires 1 argument".to_string()));
                }
                let signal = args[0].downcast_ref::<Signal>().ok_or_else(|| MathError::InvalidArgument("Argument must be Signal".to_string()))?;
                Ok(Box::new(Self::signal_energy(signal)))
            },
            "signal_power" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("signal_power requires 1 argument".to_string()));
                }
                let signal = args[0].downcast_ref::<Signal>().ok_or_else(|| MathError::InvalidArgument("Argument must be Signal".to_string()))?;
                Ok(Box::new(Self::signal_power(signal)))
            },
            "signal_rms" => {
                if args.len() != 1 {
                    return Err(MathError::InvalidArgument("signal_rms requires 1 argument".to_string()));
                }
                let signal = args[0].downcast_ref::<Signal>().ok_or_else(|| MathError::InvalidArgument("Argument must be Signal".to_string()))?;
                Ok(Box::new(Self::signal_rms(signal)))
            },
            _ => Err(MathError::InvalidOperation(format!("Unknown operation: {}", operation)))
        }
    }
    
    fn list_operations(&self) -> Vec<String> {
        vec![
            "create_signal".to_string(),
            "generate_sine_wave".to_string(),
            "fft".to_string(),
            "ifft".to_string(),
            "apply_window".to_string(),
            "create_hamming_window".to_string(),
            "create_hanning_window".to_string(),
            "create_blackman_window".to_string(),
            "convolution".to_string(),
            "correlation".to_string(),
            "butterworth_lowpass_filter".to_string(),
            "apply_filter".to_string(),
            "signal_energy".to_string(),
            "signal_power".to_string(),
            "signal_rms".to_string(),
        ]
    }
}