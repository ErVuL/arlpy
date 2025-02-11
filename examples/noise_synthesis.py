import numpy as np
import matplotlib.pyplot as plt
import arlpy.uwa as uwa

if __name__ == "__main__":

    # PSD of required signal
    wenz = uwa.WenzModel(Fxx=np.linspace(1,192000,65536))
    duration = 10 # s

    # Signal synthesis
    Gtime, Gsignal, fs = uwa.SSRP(wenz.Pxx, wenz.Fxx, duration, scale=1)
    
    # PSD
    psd = uwa.PSD()
    Fxx, Pxx = psd.compute(Gsignal, fs)
    
    # Plot
    fig, ax = psd.plot(title='Example', label='Generated signal', color='black')
    psd.add2plot(ax=ax, Fxx=wenz.Fxx, Pxx=wenz.Pxx, label='Goal', linestyle='dashed', color='red')
    fig1, ax1 =  plt.subplots(1,1)
    ax1.plot(Gtime,Gsignal)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude [Pa]')
    ax1.set_title('Generated signal', loc='left')
    ax1.grid(True, which="both")
    wenz.plot()
        
    plt.show()

