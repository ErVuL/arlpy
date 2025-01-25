import arlpy.uwa as uwa
import numpy as _np
import matplotlib.pyplot as plt

# Example usage
if __name__ == "__main__":
    
    # Generate sample noise signal
    fs = 192000  # Sampling frequency
    duration = 10  # Duration in seconds
    t = _np.linspace(0, duration, int(fs*duration))

    # Create a signal with multiple frequency components
    signal = (_np.sin(2*_np.pi*1000*t) + 
              0.5*_np.sin(2*_np.pi*2000*t) + 
              0.25*_np.sin(2*_np.pi*4000*t))

    # Add some noise
    noise = _np.random.normal(0, 0.1, len(t))
    signal = signal + noise

    # Create and plot SEL
    sel = uwa.SEL()
    sel.compute(signal, fs)
    sel.plot(title="Example Signal")

    # Create and plot PSD
    psd = uwa.PSD()
    psd.compute(signal, fs)
    psd.plot(title="Example Signal")

    # Create and plot PSDPDF
    psdpdf = uwa.PSDPDF(seg_duration=0.1, nperseg=8192, noverlap=4096, nbins=50)
    psdpdf.compute(signal, fs)
    psdpdf.plot(title="Example Signal")

    # Create and plot Spectrogram
    spec = uwa.Spectrogram()
    spec.compute(signal, fs)
    spec.plot(title="Example Signal", ymin=100, vmax=180)
    
    plt.show()
