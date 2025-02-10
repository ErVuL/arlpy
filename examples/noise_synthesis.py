import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy import signal
import arlpy.uwa as uwa

if __name__ == "__main__":

    # Example of required PSD in U**2/Hz
    iFxx = np.linspace(1,48000,48000)
    iPxx = np.array([170-110/48000*f+50*np.cos(2*np.pi*0.0002*f) for f in iFxx]) # ((U/ref)**2)/Hz
    ref=1e-6 # iPxx is expressed in ((U/ref)**2)/Hz

    # Generate signal in time domain with required PSD
    Gtime, Gsignal, fs = uwa.SSRP(iPxx, iFxx, 50, scale=1e-6)
    
    # Compute Welch periodogram
    Fxx, Pxx = signal.welch(Gsignal, fs=iFxx[-1]*2, window='hann', nperseg=16384, noverlap=8192, scaling='density')
    psd = uwa.PSD()

    # Plot the results
    fig0, ax0 =  plt.subplots(1,1)
    ax0.set_xlim([1, iFxx[-1]*2/2*0.8])
    Fxx, Pxx = psd.compute(Gsignal, fs)
    ax0.semilogx(Fxx,10.0*np.log10(Pxx/(ref**2)), 'k', label='Generated signal')
    ax0.semilogx(iFxx,10.0*np.log10(iPxx), 'r--', label='Goal')
    ax0.legend()
    ax0.set_xlabel('Frequency [Hz]')
    ax0.set_ylabel('Level [dB re 1µPa²/Hz]')
    ax0.set_title('Power Spectral Density', loc='left')
    ax0.grid(True, which="both")

    fig1, ax1 =  plt.subplots(1,1)
    ax1.plot(Gtime,Gsignal)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude [U]')
    ax1.set_title('Generated signal', loc='left')
    ax1.grid(True, which="both")
    

