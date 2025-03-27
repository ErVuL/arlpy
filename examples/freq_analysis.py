import arlpy.uwa as uwa
import arlpy.signal as usp
import numpy as _np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import scipy.signal as _sig

# Example usage
if __name__ == "__main__":
    
    # Generate chirp signal
    fs = 192000  # Sampling frequency
    duration = 1  # Duration in seconds
    t = _np.linspace(0, duration, int(fs * duration))  # Time vector
    
    # Define chirp parameters
    f0 = 1  # Start frequency of the chirp (Hz)
    f1 = fs/2  # End frequency of the chirp (Hz)
    t1 = duration  # Time at which f1 is reached (end of the chirp)
    method = 'linear'  # Frequency sweep method ('linear', 'quadratic', 'logarithmic', etc.)
    
    # Generate chirp signal
    signal_1 = 100*_sig.chirp(t, f0, t1, f1, method=method)
    
    # Add noise to the chirp signal
    signal_1 += _np.random.normal(0, 100, int(fs * duration))
    
    # Process signal_2 with more magnitude & phase variation
    def chebyshev_lowpass(signal, cutoff, fs, order=8, rp=0.5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = _sig.cheby1(order, rp, normal_cutoff, btype='low', analog=False)
        filtered_signal = _sig.lfilter(b, a, signal)
        return filtered_signal
    
    # Apply the modified filter
    signal_2 = chebyshev_lowpass(signal_1 * 10, fs/10, fs) + _np.random.normal(0, 50, int(fs * duration))
    
    # SEL
    sel = usp.SEL()
    sel.compute(signal_1, fs, chunk_size=fs)
    sel.plot(title="Example Signal")
    
    # PSD
    psd = usp.PSD()
    psd.compute(signal_1, fs)
    fig, ax = psd.plot(title="Example Signal", label='signal 1')
    
    psd.compute(signal_2, fs)
    psd.add2plot(ax, label="signal 2", linestyle='dashed')
    
    # FRF    
    frf = usp.FRF()
    frf.compute(signal_1, signal_2, fs, method='etfe')
    fig, ax = frf.plot(title="Example signal", label="Chebyshev LP")
    
    frf.compute(signal_1, signal_2, fs, method='welch', estimator='H1', nperseg=2048)
    frf.add2plot(ax, label="Chebyshev LP", linestyle='dashed')
    fig_coh, ax_coh = frf.plot_coh(label="Chebyshev LP")
    
    frf.compute(signal_1, signal_2, fs, method='p_etfe', nperseg=2048)
    frf.add2plot(ax, label="Chebyshev LP", linestyle='dashed')
        
    frf.compute(signal_1, signal_2, fs, method='welch', estimator='H2')
    frf.add2plot(ax, label="Chebyshev LP", linestyle='dashed')
    frf.add2plot_coh(ax_coh, label="Chebyshev LP", linestyle='dashed')
    
    frf.compute(signal_1, signal_2, fs, method='ls_fir', m='AIC', m_max=1024, stop_count=50)
    frf.add2plot(ax, label="Chebyshev LP", linestyle='dashed')
    
    frf.compute(signal_1, signal_2, fs, method='ls_fir', m='BIC', m_max=1024, stop_count=50)
    frf.add2plot(ax, label="Chebyshev LP", linestyle='dashed')
    
    frf.compute(signal_1, signal_2, fs, method='ls_fir', m='CP', m_max=1024, stop_count=50)
    frf.add2plot(ax, label="Chebyshev LP", linestyle='dashed')
    
    frf.compute(signal_1, signal_2, fs, method='ls_fir', m='FPE', m_max=1024, stop_count=50)
    frf.add2plot(ax, label="Chebyshev LP", linestyle='dashed')
    frf.plot_impulse_info(title="Example signal")
    
    # PSDPDF
    psdpdf = usp.PSDPDF(seg_duration=0.01, nperseg=1024, noverlap=1024/2, nbins=100)
    psdpdf.compute(signal_1, fs)
    psdpdf.plot(title="Example Signal")
    
    # Spectrogram
    spec = usp.Spectrogram()
    spec.compute(signal_1, fs)
    spec.plot(title="Example Signal", ymin=100, vmax=180)
    
    plt.show()
