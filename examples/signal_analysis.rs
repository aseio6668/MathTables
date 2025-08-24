use mathtables::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MathTables Signal Processing Demo ===\n");

    // 1. Generate test signals
    println!("1. Generating Test Signals...");
    
    // Generate a composite signal: sine wave + higher frequency component + noise
    let sample_rate = 8000.0;
    let duration = 1.0;
    let num_samples = (sample_rate * duration) as usize;
    
    // Primary sine wave at 440 Hz (A4 note)
    let signal1 = SignalProcessingDomain::generate_sine_wave(440.0, 1.0, duration, sample_rate)?;
    println!("   Generated 440 Hz sine wave");
    
    // Higher frequency component at 1760 Hz (A6 note)
    let signal2 = SignalProcessingDomain::generate_sine_wave(1760.0, 0.5, duration, sample_rate)?;
    println!("   Generated 1760 Hz sine wave");
    
    // Create composite signal by adding them together
    let mut composite_samples = signal1.samples.clone();
    for i in 0..composite_samples.len().min(signal2.samples.len()) {
        composite_samples[i] += signal2.samples[i];
        // Add some noise
        composite_samples[i] += ((i as f64) * 0.01).sin() * 0.1;
    }
    
    let composite_signal = SignalProcessingDomain::create_signal(composite_samples, sample_rate)?;
    println!("   Created composite signal with {} samples", composite_signal.samples.len());
    
    // 2. Signal Analysis
    println!("\n2. Signal Analysis...");
    
    let energy = SignalProcessingDomain::signal_energy(&composite_signal);
    let power = SignalProcessingDomain::signal_power(&composite_signal);
    let rms = SignalProcessingDomain::signal_rms(&composite_signal);
    
    println!("   Signal Energy: {:.6}", energy);
    println!("   Signal Power: {:.6}", power);
    println!("   RMS Value: {:.6}", rms);
    
    // 3. Windowing Functions
    println!("\n3. Applying Window Functions...");
    
    // Create different window functions
    let hamming_window = SignalProcessingDomain::create_hamming_window(composite_signal.samples.len());
    let hanning_window = SignalProcessingDomain::create_hanning_window(composite_signal.samples.len());
    let blackman_window = SignalProcessingDomain::create_blackman_window(composite_signal.samples.len());
    
    println!("   Created Hamming window");
    println!("   Created Hanning window");
    println!("   Created Blackman window");
    
    // Apply Hamming window to reduce spectral leakage
    let windowed_signal = SignalProcessingDomain::apply_window(&composite_signal, &hamming_window)?;
    println!("   Applied Hamming window to composite signal");
    
    // Compare windowed vs original signal properties
    let windowed_rms = SignalProcessingDomain::signal_rms(&windowed_signal);
    println!("   Original RMS: {:.6}, Windowed RMS: {:.6}", rms, windowed_rms);
    
    // 4. Frequency Domain Analysis (FFT)
    println!("\n4. Frequency Domain Analysis...");
    
    // Pad signal to power of 2 for FFT efficiency
    let fft_size = 8192; // Next power of 2 >= num_samples
    let mut padded_samples = windowed_signal.samples.clone();
    padded_samples.resize(fft_size, 0.0);
    
    let padded_signal = Signal {
        samples: padded_samples,
        sample_rate: windowed_signal.sample_rate,
        duration: windowed_signal.duration,
    };
    
    println!("   Padded signal to {} samples for FFT", fft_size);
    
    // Compute FFT
    let fft_result = SignalProcessingDomain::fft(&padded_signal)?;
    println!("   FFT computed with {} frequency bins", fft_result.frequencies.len());
    
    // Analyze frequency content
    let bin_width = sample_rate / fft_size as f64;
    println!("   Frequency resolution: {:.2} Hz per bin", bin_width);
    
    // Find peaks in magnitude spectrum (first half due to symmetry)
    let half_bins = fft_size / 2;
    let mut peaks = Vec::new();
    
    for i in 1..(half_bins - 1) {
        let magnitude = fft_result.magnitude_spectrum[i];
        let prev_mag = fft_result.magnitude_spectrum[i - 1];
        let next_mag = fft_result.magnitude_spectrum[i + 1];
        
        // Simple peak detection: current bin is higher than neighbors and above threshold
        if magnitude > prev_mag && magnitude > next_mag && magnitude > 100.0 {
            let frequency = i as f64 * bin_width;
            peaks.push((frequency, magnitude));
        }
    }
    
    println!("   Detected {} spectral peaks:", peaks.len());
    for (freq, mag) in &peaks {
        println!("     {:.1} Hz: magnitude = {:.1}", freq, mag);
    }
    
    // 5. Digital Filtering
    println!("\n5. Digital Filtering...");
    
    // Design a low-pass filter to remove high-frequency components
    let cutoff_frequency = 1000.0; // Remove frequencies above 1 kHz
    let filter_order = 4;
    
    let lowpass_filter = SignalProcessingDomain::butterworth_lowpass_filter(
        cutoff_frequency, 
        sample_rate, 
        filter_order
    )?;
    
    println!("   Designed {}th order Butterworth lowpass filter", filter_order);
    println!("   Cutoff frequency: {} Hz", cutoff_frequency);
    
    // Apply filter to original composite signal
    let filtered_signal = SignalProcessingDomain::apply_filter(&composite_signal, &lowpass_filter)?;
    println!("   Applied filter to composite signal");
    
    // Compare filtered signal properties
    let filtered_rms = SignalProcessingDomain::signal_rms(&filtered_signal);
    println!("   Original RMS: {:.6}, Filtered RMS: {:.6}", rms, filtered_rms);
    
    // Analyze filtered signal spectrum
    let mut filtered_padded = filtered_signal.samples.clone();
    filtered_padded.resize(fft_size, 0.0);
    let filtered_padded_signal = Signal {
        samples: filtered_padded,
        sample_rate: filtered_signal.sample_rate,
        duration: filtered_signal.duration,
    };
    
    let filtered_fft = SignalProcessingDomain::fft(&filtered_padded_signal)?;
    println!("   Computed FFT of filtered signal");
    
    // 6. Convolution and Correlation
    println!("\n6. Convolution and Correlation...");
    
    // Create a simple impulse response for demonstration
    let impulse_samples = vec![0.5, 0.3, 0.1]; // Simple FIR filter
    let impulse_response = SignalProcessingDomain::create_signal(impulse_samples, sample_rate)?;
    
    // Convolve with a portion of our signal (to keep computational load reasonable)
    let short_signal_samples = composite_signal.samples[0..100].to_vec();
    let short_signal = SignalProcessingDomain::create_signal(short_signal_samples, sample_rate)?;
    
    let convolved = SignalProcessingDomain::convolution(&short_signal, &impulse_response)?;
    println!("   Performed convolution: {} samples -> {} samples", 
             short_signal.samples.len(), convolved.samples.len());
    
    // Auto-correlation of the short signal
    let autocorr = SignalProcessingDomain::correlation(&short_signal, &short_signal)?;
    println!("   Computed auto-correlation");
    
    // Find the peak in auto-correlation (should be at center)
    let max_idx = autocorr.samples.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    
    println!("   Auto-correlation peak at sample {}", max_idx);
    
    // 7. Inverse FFT Demonstration
    println!("\n7. Inverse FFT (Signal Reconstruction)...");
    
    // Take our FFT result and convert back to time domain
    let reconstructed = SignalProcessingDomain::ifft(&fft_result)?;
    println!("   Reconstructed signal from FFT");
    
    // Compare original vs reconstructed (should be very close)
    let original_slice = &padded_signal.samples[0..10];
    let reconstructed_slice = &reconstructed.samples[0..10];
    
    println!("   Comparison (first 10 samples):");
    println!("   Original:      {:?}", &original_slice[0..5]);
    println!("   Reconstructed: {:?}", &reconstructed_slice[0..5]);
    
    // Calculate reconstruction error
    let mut total_error = 0.0;
    let compare_length = original_slice.len().min(reconstructed_slice.len());
    
    for i in 0..compare_length {
        let error = (original_slice[i] - reconstructed_slice[i]).abs();
        total_error += error * error;
    }
    
    let rmse = (total_error / compare_length as f64).sqrt();
    println!("   Reconstruction RMSE: {:.2e}", rmse);
    
    // 8. Multi-Signal Processing Pipeline
    println!("\n8. Complete Signal Processing Pipeline...");
    
    // Demonstrate a complete audio processing pipeline
    println!("   Pipeline: Generation -> Windowing -> Filtering -> Analysis");
    
    // Generate another test signal
    let test_freq = 800.0;
    let test_signal = SignalProcessingDomain::generate_sine_wave(test_freq, 0.8, 0.5, sample_rate)?;
    
    // Apply window
    let window = SignalProcessingDomain::create_hanning_window(test_signal.samples.len());
    let windowed = SignalProcessingDomain::apply_window(&test_signal, &window)?;
    
    // Apply bandpass filtering (simulation with lowpass)
    let filter = SignalProcessingDomain::butterworth_lowpass_filter(1500.0, sample_rate, 3)?;
    let processed = SignalProcessingDomain::apply_filter(&windowed, &filter)?;
    
    // Final analysis
    let final_energy = SignalProcessingDomain::signal_energy(&processed);
    let final_rms = SignalProcessingDomain::signal_rms(&processed);
    
    println!("   Input frequency: {} Hz", test_freq);
    println!("   Final energy: {:.6}", final_energy);
    println!("   Final RMS: {:.6}", final_rms);
    
    // 9. Performance Summary
    println!("\n9. Processing Summary...");
    println!("   Total samples processed: {}", num_samples);
    println!("   Sample rate: {} Hz", sample_rate);
    println!("   Signal duration: {} seconds", duration);
    println!("   FFT size: {}", fft_size);
    println!("   Frequency resolution: {:.2} Hz", bin_width);
    
    println!("\n=== Signal Processing Demo Complete ===");
    println!("This demo showcased:");
    println!("• Signal generation and manipulation");
    println!("• Time-domain analysis (energy, power, RMS)");
    println!("• Window functions for spectral analysis");
    println!("• FFT for frequency domain analysis");
    println!("• Digital filtering with Butterworth filters");
    println!("• Convolution and correlation operations");
    println!("• Signal reconstruction with inverse FFT");
    println!("• Complete signal processing pipelines");
    
    Ok(())
}