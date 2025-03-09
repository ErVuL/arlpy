import arlpy.uwa as uwa
import arlpy.signal as usp
import numpy as _np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import scipy.signal as _sig

def lowpass(signal, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return lfilter(b, a, signal)  # Causal filtering, introducing phase delay

# Example usage
if __name__ == "__main__":
    
    # Generate chirp signal
    fs = 192000  # Sampling frequency
    duration = 5  # Duration in seconds
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
    
    # Process signal_2 (e.g., apply a lowpass filter and add noise)
    def lowpass(signal, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = _sig.butter(order, normal_cutoff, btype='low', analog=False)
        filtered_signal = _sig.lfilter(b, a, signal)
        return filtered_signal
    
    signal_2 = lowpass(signal_1 * 10, fs/10, fs) + _np.random.normal(0, 50, int(fs * duration))
    
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
    frf.compute(signal_1, signal_2, fs, method='welch', estimator='H1', nperseg=8192)
    fig, ax = frf.plot(title="Example signal", label="Butterworth LP")
    fig_coh, ax_coh = frf.plot_coh(label="Butterworth LP")
    
    frf.compute(signal_1, signal_2, fs, method='welch', estimator='H2', nperseg=8192)
    frf.add2plot(ax, label="Butterworth LP", linestyle='dashed')
    frf.add2plot_coh(ax_coh, label="Butterworth LP", linestyle='dashed')
    
    frf.compute(signal_1, signal_2, fs, method='ls-ir', m=64)
    frf.add2plot(ax, label="Butterworth LP", linestyle='dashed')
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
