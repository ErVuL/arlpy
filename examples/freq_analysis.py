import arlpy.uwa as uwa
import arlpy.signal as usp
import numpy as _np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def lowpass(signal, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal)

# Example usage
if __name__ == "__main__":
    
    # Generate signal
    fs = 192000  # Sampling frequency
    duration = 10  # Duration in seconds
    signal = (_np.sin(2*_np.pi*1000*_np.linspace(0, duration, int(fs*duration))) + 
              0.5*_np.sin(2*_np.pi*2000*_np.linspace(0, duration, int(fs*duration))) + 
              0.25*_np.sin(2*_np.pi*4000*_np.linspace(0, duration, int(fs*duration))))
    signal = signal + _np.random.normal(0, 0.1, int(fs*duration))
    
    # SEL
    sel = usp.SEL()
    sel.compute(signal, fs, chunk_size=192000)
    sel.plot(title="Example Signal")
    
    # PSD
    psd = usp.PSD()
    psd.compute(signal, fs)
    fig, ax = psd.plot(title="Example Signal", label='signal 1')
    psd.compute(signal/2, fs)
    psd.add2plot(ax, label="signal 2", linestyle='dashed')
    
    # FRF    
    frf = usp.FRF()
    frf.compute(signal, lowpass(signal*10, 2000, fs), fs, nperseg=16384)
    fig, ax = frf.plot(title="Example Signal", label="sig 1")
    frf.compute(signal, lowpass(signal*10, 40000, fs), fs, method='stft', nperseg=16384)
    frf.add2plot(ax, label='sig 2', linestyle='dashed')
    
    # PSDPDF
    psdpdf = usp.PSDPDF(seg_duration=0.1, nperseg=4096, noverlap=4096/2, nbins=100)
    psdpdf.compute(signal, fs)
    psdpdf.plot(title="Example Signal")
    
    # Spectrogram
    spec = usp.Spectrogram()
    spec.compute(signal, fs)
    spec.plot(title="Example Signal", ymin=100, vmax=180)
    
    plt.show()
