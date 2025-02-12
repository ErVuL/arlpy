##############################################################################
#
# Copyright (c) 2016, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""Underwater acoustics toolbox."""

import numpy as _np
import math
import scipy.signal as _sp
import matplotlib.pyplot as plt

def soundspeed(temperature=27, salinity=35, depth=10):
    """Get the speed of sound in water.

    Uses Mackenzie (1981) to compute sound speed in water.

    :param temperature: temperature in deg C
    :param salinity: salinity in ppt
    :param depth: depth in m
    :returns: sound speed in m/s

    >>> import arlpy
    >>> arlpy.uwa.soundspeed()
    1539.1
    >>> arlpy.uwa.soundspeed(temperature=25, depth=20)
    1534.6
    """
    c = 1448.96 + 4.591*temperature - 5.304e-2*temperature**2 + 2.374e-4*temperature**3
    c += 1.340*(salinity-35) + 1.630e-2*depth + 1.675e-7*depth**2
    c += -1.025e-2*temperature*(salinity-35) - 7.139e-13*temperature*depth**3
    return c

def absorption(frequency, distance=1000, temperature=27, salinity=35, depth=10, pH=8.1):
    """Get the acoustic absorption in water.

    Computes acoustic absorption in water using Francois-Garrison model.

    :param frequency: frequency in Hz
    :param distance: distance in m
    :param temperature: temperature in deg C
    :param salinity: salinity in ppt
    :param depth: depth in m
    :param pH: pH of water
    :returns: absorption as a linear multiplier

    >>> import arlpy
    >>> arlpy.uwa.absorption(50000)
    0.2914
    >>> arlpy.utils.mag2db(arlpy.uwa.absorption(50000))
    -10.71
    >>> arlpy.utils.mag2db(arlpy.uwa.absorption(50000, distance=3000))
    -32.13
    """
    f = frequency/1000.0
    d = distance/1000.0
    c = 1412.0 + 3.21*temperature + 1.19*salinity + 0.0167*depth
    A1 = 8.86/c * 10**(0.78*pH-5)
    P1 = 1.0
    f1 = 2.8*_np.sqrt(salinity/35) * 10**(4-1245/(temperature+273))
    A2 = 21.44*salinity/c*(1+0.025*temperature)
    P2 = 1.0 - 1.37e-4*depth + 6.2e-9*depth*depth
    f2 = 8.17 * 10**(8-1990/(temperature+273)) / (1+0.0018*(salinity-35))
    P3 = 1.0 - 3.83e-5*depth + 4.9e-10*depth*depth
    if temperature < 20:
        A3 = 4.937e-4 - 2.59e-5*temperature + 9.11e-7*temperature*temperature - 1.5e-8*temperature*temperature*temperature
    else:
        A3 = 3.964e-4 - 1.146e-5*temperature + 1.45e-7*temperature*temperature - 6.5e-10*temperature*temperature*temperature
    a = A1*P1*f1*f*f/(f1*f1+f*f) + A2*P2*f2*f*f/(f2*f2+f*f) + A3*P3*f*f
    return 10**(-a*d/20.0)

def absorption_filter(fs, ntaps=31, nfreqs=64, distance=1000, temperature=27, salinity=35, depth=10):
    """Design a FIR filter with response based on acoustic absorption in water.

    :param fs: sampling frequency in Hz
    :param ntaps: number of FIR taps
    :param nfreqs: number of frequencies to use for modeling frequency response
    :param distance: distance in m
    :param temperature: temperature in deg C
    :param salinity: salinity in ppt
    :param depth: depth in m
    :returns: tap weights for a FIR filter that represents absorption at the given distance

    >>> import arlpy
    >>> import numpy as np
    >>> fs = 250000
    >>> b = arlpy.uwa.absorption_filter(fs, distance=500)
    >>> x = arlpy.signal.sweep(20000, 40000, 0.5, fs)
    >>> y = arlpy.signal.lfilter0(b, 1, x)
    >>> y /= 500**2      # apply spreading loss for 500m
    """
    nyquist = fs/2.0
    f = _np.linspace(0, nyquist, num=nfreqs)
    g = absorption(f, distance, temperature, salinity, depth)
    return _sp.firwin2(ntaps, f, g, nyq=nyquist)

def density(temperature=27, salinity=35):
    """Get the density of sea water near the surface.

    Computes sea water density using Fofonoff (1985 - IES 80).

    :param temperature: temperature in deg C
    :param salinity: salinity in ppt
    :returns: density in kg/m^3

    >>> import arlpy
    >>> arlpy.uwa.density()
    1022.7
    """
    t = temperature
    A = 1.001685e-04 + t * (-1.120083e-06 + t * 6.536332e-09)
    A = 999.842594 + t * (6.793952e-02 + t * (-9.095290e-03 + t * A))
    B = 7.6438e-05 + t * (-8.2467e-07 + t * 5.3875e-09)
    B = 0.824493 + t * (-4.0899e-03 + t * B)
    C = -5.72466e-03 + t * (1.0227e-04 - t * 1.6546e-06)
    D = 4.8314e-04
    return A + salinity * (B + C*_np.sqrt(salinity) + D*salinity)

def reflection_coeff(angle, rho1, c1, alpha=0, rho=density(), c=soundspeed()):
    """Get the Rayleigh reflection coefficient for a given angle.

    :param angle: angle of incidence in radians
    :param rho1: density of second medium in kg/m^3
    :param c1: sound speed in second medium in m/s
    :param alpha: attenuation
    :param rho: density of water in kg/m^3
    :param c: sound speed in water in m/s
    :returns: reflection coefficient as a linear multiplier

    >>> from numpy import pi
    >>> import arlpy
    >>> arlpy.uwa.reflection_coeff(pi/4, 1200, 1600)
    0.1198
    >>> arlpy.uwa.reflection_coeff(0, 1200, 1600)
    0.0990
    >>> arlpy.utils.mag2db(arlpy.uwa.reflection_coeff(0, 1200, 1600))
    -20.1
    """
    # Brekhovskikh & Lysanov
    n = float(c)/c1*(1+1j*alpha)
    m = float(rho1)/rho
    t1 = m*_np.cos(angle)
    t2 = _np.sqrt(n**2-_np.sin(angle)**2)
    V = (t1-t2)/(t1+t2)
    return V.real if V.imag == 0 else V

def doppler(speed, frequency, c=soundspeed()):
    """Get the Doppler-shifted frequency given relative speed between transmitter and receiver.

    The Doppler approximation used is only valid when `speed` << `c`. This is usually the case
    for underwater vehicles.

    :param speed: relative speed between transmitter and receiver in m/s
    :param frequency: transmission frequency in Hz
    :param c: sound speed in m/s
    :returns: the Doppler shifted frequency as perceived by the receiver

    >>> import arlpy
    >>> arlpy.uwa.doppler(2, 50000)
    50064.97
    >>> arlpy.uwa.doppler(-1, 50000)
    49967.51
    """
    return (1+speed/float(c))*frequency

def bubble_resonance(radius, depth=0.0, gamma = 1.4, p0 = 1.013e5, rho_water = 1022.476):
    """Compute resonance frequency of a freely oscillating has bubble in water,
    using implementation based on Medwin & Clay (1998). This ignores surface-tension,
    thermal, viscous and acoustic damping effects, and the pressure-volume relationship
    is taken to be adiabatic. Parameters:

    :radius: bubble `radius` in meters
    :depth: depth of bubble in water in meters
    :gamma: gas ratio of specific heats. Default 1.4 for air
    :p0: atmospheric pressure. Default 1.013e5 Pa
    :rho_water: Density of water. Default 1022.476 kg/m³

    >>> import arlpy
    >>> arlpy.uwa.bubble_resonance(100e-6)
    32465.56
    """
    g = 9.80665 #acceleration due to gravity
    p_air = p0 + rho_water*g*depth
    return 1/(2*_np.pi*radius)*_np.sqrt(3*gamma*p_air/rho_water)

def bubble_surface_loss(windspeed, frequency, angle):
    """Get the surface loss due to bubbles.

    The surface loss is computed based on APL model (1994).

    :param windspeed: windspeed in m/s (measured 10 m above the sea surface)
    :param frequency: frequency in Hz
    :param angle: incidence angle in radians
    :returns: absorption as a linear multiplier

    >>> import numpy
    >>> import arlpy
    >>> arlpy.utils.mag2db(uwa.bubble_surface_loss(3,10000,0))
    -1.44
    >>> arlpy.utils.mag2db(uwa.bubble_surface_loss(10,10000,0))
    -117.6
    """
    beta = _np.pi/2-angle
    f = frequency/1000.0
    if windspeed >= 6:
        a = 1.26e-3/_np.sin(beta) * windspeed**1.57 * f**0.85
    else:
        a = 1.26e-3/_np.sin(beta) * 6**1.57 * f**0.85 * _np.exp(1.2*(windspeed-6))
    return 10**(-a/20.0)

def bubble_soundspeed(void_fraction, c=soundspeed(), c_gas=340, relative_density=1000):
    """Get the speed of sound in a 2-phase bubbly water.

    The sound speed is computed based on Wood (1964) or Buckingham (1997).

    :param void_fraction: void fraction
    :param c: speed of sound in water in m/s
    :param c_gas: speed of sound in gas in m/s
    :param relative_density: ratio of density of water to gas
    :returns: sound speed in m/s

    >>> import arlpy
    >>> arlpy.uwa.bubble_soundspeed(1e-5)
    1402.133
    """
    m = _np.sqrt(relative_density)
    return 1/(1/c*_np.sqrt((void_fraction*(c/c_gas)**2*m+(1-void_fraction)/m)*(void_fraction/m+(1-void_fraction)*m)))

def pressure(x, sensitivity, gain, volt_params=None):
    """Convert the real signal x to an acoustic pressure signal in micropascal.

    :param x: real signal in voltage or bit depth (number of bits)
    :param sensitivity: receiving sensitivity in dB re 1V per micropascal
    :param gain: preamplifier gain in dB
    :param volt_params: (nbits, v_ref) is used to convert the number of bits
        to voltage where nbits is the number of bits of each sample and v_ref
        is the reference voltage, default to None
    :returns: acoustic pressure signal in micropascal

    If `volt_params` is provided, the sample unit of x is in number of bits,
    else is in voltage.

    >>> import arlpy
    >>> nbits = 16
    >>> V_ref = 1.0
    >>> x_volt = V_ref*signal.cw(64, 1, 512)
    >>> x_bit = x_volt*(2**(nbits-1))
    >>> sensitivity = 0
    >>> gain = 0
    >>> p1 = arlpy.uwa.pressure(x_volt, sensitivity, gain)
    >>> p2 = arlpy.uwa.pressure(x_bit, sensitivity, gain, volt_params=(nbits, V_ref))
    """
    nu = 10**(sensitivity/20)
    G = 10**(gain/20)
    if volt_params is not None:
        nbits, v_ref = volt_params
        x = x*v_ref/(2**(nbits-1))
    return x/(nu*G)

def spl(x, ref=1):
    """Get Sound Pressure Level (SPL) of the acoustic pressure signal x.

    :param x: acoustic pressure signal in micropascal
    :param ref: reference acoustic pressure in micropascal, default to 1
    :returns: average SPL in dB re micropascal

    In water, the common reference is 1 micropascal. In air, the common
    reference is 20 micropascal.

    >>> import arlpy
    >>> p = signal.cw(64, 1, 512)
    >>> arlpy.uwa.spl(p)
    -3.0103
    """
    rmsx = _np.sqrt(_np.mean(_np.power(_np.abs(x), 2)))
    return 20*_np.log10(rmsx/ref)

class WenzModel:
    """
    A class to calculate and plot underwater noise levels using the "Wenz" model.

    Based on :
    A simple yet practical ambient noise model
    Cristina D. S. Tollefsen, Sean Pecknold
    DRDC – Atlantic Research Centre
    May 2022

    The model calculates noise level (in dB re uPa) based on five components:
    (1) Shipping noise (Wenz, 1962)
    (2) Wind noise (Merklinger, 1979, and Piggott, 1964)
    (3) Rain noise (Torres and Costa, 2019)
    (4) Thermal noise (Mellen, 1952)
    (5) Turbulence noise (Nichols and Bradley, 2016)

    Table 1-1. Beaufort Wind Force and Sea State Numbers Vs Wind Speed
                ("AMBIENT NOISE IN THE SEA" R.J.URICK, 1984)
                                                Wind Speed
    Beaufort Number     Sea State       Knots       Meters/Sec
    0                   0               <1          0 - 0.2
    1                   1/2             1 - 3       0.3 - 1.5
    2                   1               4 - 6       1.6 - 3.3
    3                   2               7 - 10      3.4 - 5.4
    4                   3               11 - 16     5.5 - 7.9
    5                   4               17 - 21     8.0 - 10.7
    6                   5               22 - 27     10.8 - 13.8
    7                   6               28 - 33     13.9 - 17.1
    8                   6               34 - 40     17.2 - 20.7
    """


    def __init__(self, Fxx=_np.linspace(1,100000,100000), wind_speed=5, rain_rate='no', water_depth='deep', shipping_level='low'):

        """
        Initialize the Wenz model with parameters.

        Parameters:
            Fxx (array): Frequency vector in Hz
            wind_speed (float): Wind speed in knots
            rain_rate (str): 'no', 'light', 'moderate', 'heavy', or 'veryheavy'
            water_depth (str): 'shallow' or 'deep'
            shipping_level (str): 'no', 'low', 'medium', or 'high'
        """
        self.Fxx = _np.array(Fxx).flatten()
        self.wind_speed = wind_speed
        self.rain_rate = rain_rate
        self.water_depth = water_depth
        self.shipping_level = shipping_level
        self.compute()

    def _compute_wind_noise(self):
        """Calculate wind-based noise component."""
        if self.wind_speed == 0:
            return _np.zeros_like(self.Fxx)

        f_wind = 2000  # Cutoff for wind noise section
        s1w = 1.5     # Constant in wind calcs
        s2w = -5.0    # Constant in wind calc
        a = -25       # Curve melding exponent
        slope = s2w * (0.1 / _np.log10(2))  # Slope at high freq

        cst = 45 if self.water_depth == 'shallow' else 42

        i_wind = self.Fxx <= f_wind
        f_temp = self.Fxx[i_wind] if _np.any(i_wind) else _np.array([2000])

        f0w = 770 - 100 * _np.log10(self.wind_speed)
        L0w = cst + 20 * _np.log10(self.wind_speed) - 17 * _np.log10(f0w / 770)
        L1w = L0w + (s1w / _np.log10(2)) * _np.log10(f_temp / f0w)
        L2w = L0w + (s2w / _np.log10(2)) * _np.log10(f_temp / f0w)
        Lw = L1w * (1 + (L1w / L2w) ** (-a)) ** (1 / a)
        temp_noise_dist = 10 ** (Lw / 10)

        NL = _np.zeros_like(self.Fxx)
        if _np.any(i_wind):
            NL[i_wind] = temp_noise_dist
        if _np.any(~i_wind):
            prop_const = temp_noise_dist[-1] / f_temp[-1] ** slope
            NL[~i_wind] = prop_const * self.Fxx[~i_wind] ** slope

        return 10 * _np.log10(NL)

    def _compute_thermal_noise(self):
        """Calculate thermal noise component."""
        noise = -75.0 + 20.0 * _np.log10(self.Fxx)
        noise[noise <= 0] = 1
        return noise

    def _compute_shipping_noise(self):
        """Calculate shipping noise component."""
        c1 = 30 if self.water_depth == 'deep' else 65
        c2 = {'low': 1, 'medium': 4, 'high': 7, 'no': 0}.get(self.shipping_level, 4)

        if self.shipping_level != 'no':
            noise = 76 - 20 * (_np.log10(self.Fxx) - _np.log10(c1))**2 + 5 * (c2 - 4)
            noise[noise <= 0] = 1
            return noise
        return _np.zeros_like(self.Fxx)

    def _compute_turbulence_noise(self):
        """Calculate turbulence noise component."""

        noise = 108.5 - 32.5 * _np.log10(self.Fxx)
        noise[noise <= 0] = 1
        return noise

    def _compute_rain_noise(self):

        if self.rain_rate == "no":
            return _np.zeros(len(self.Fxx))

        """Calculate rain noise component."""
        r0 = [0, 51.0769, 61.5358, 65.1107, 74.3464]
        r1 = [0, 1.4687, 1.0147, 0.8226, 1.0131]
        r2 = [0, -0.5232, -0.4255, -0.3825, -0.4258]
        r3 = [0, 0.0335, 0.0277, 0.0251, 0.0277]

        i_rain = {'light': 1, 'moderate': 2, 'heavy': 3, 'veryheavy': 4}.get(self.rain_rate, 1)
        fk = self.Fxx / 1000
        noise = r0[i_rain] + r1[i_rain] * fk + r2[i_rain] * fk**2 + r3[i_rain] * fk**3

        slope = -5.0 * (0.1 / _np.log10(2))
        ind = _np.where(self.Fxx < 7000)[0][-1]
        temp_noise = 10**(noise[ind] / 10)
        prop_const = temp_noise / self.Fxx[ind]**slope
        noise[self.Fxx > 7000] = 10 * _np.log10(prop_const * self.Fxx[self.Fxx > 7000]**slope)

        return noise

    def _compute_total_noise(self):
        """Calculate total noise by combining all components."""
        return 10 * _np.log10(
            10**(self.thermal_noise/10) +
            10**(self.wind_noise/10) +
            10**(self.shipping_noise/10) +
            10**(self.turbulence_noise/10) +
            10**(self.rain_noise/10)
        )

    def compute(self):
        # Calculate all noise components
        self.thermal_noise = self._compute_thermal_noise()
        self.wind_noise = self._compute_wind_noise()
        self.shipping_noise = self._compute_shipping_noise()
        self.turbulence_noise = self._compute_turbulence_noise()
        self.rain_noise = self._compute_rain_noise()
        self.total_noise = self._compute_total_noise()
        self.Pxx = 10**(self.total_noise/10)*1e-12

    def plot(self, title='', **kwargs):
        """Plot all noise components and total noise."""
        fig, ax = plt.subplots()

        ax.semilogx(self.Fxx, self.total_noise,
                    label=f'Total noise ({self.water_depth} water)', color='black', **kwargs)
        ax.semilogx(self.Fxx, self.shipping_noise,
                    label=f'Shipping noise ({self.shipping_level} traffic)',
                    color='blue', linestyle='dashed', **kwargs)
        ax.semilogx(self.Fxx, self.wind_noise,
                    label=f'Wind noise ({self.wind_speed} kn)',
                    color='green', linestyle='dashed', **kwargs)
        ax.semilogx(self.Fxx, self.rain_noise,
                    label=f'Rain noise ({self.rain_rate} rain)',
                    color='orange', linestyle='dashed', **kwargs)
        ax.semilogx(self.Fxx, self.thermal_noise,
                    label='Thermal noise', color='red', linestyle='dashed', **kwargs)
        ax.semilogx(self.Fxx, self.turbulence_noise,
                    label='Turbulence noise', color='purple', linestyle='dashed', **kwargs)

        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Noise Level [dB re 1µPa²]')
        ax.set_title(f'[WENZ - Noise Level Estimate] {title}', loc='left')
        ax.set_xlim((self.Fxx[0], self.Fxx[-1]))
        ax.set_ylim((6, 146))
        ax.legend()
        ax.grid(True, 'both')
        plt.tight_layout()

        return fig, ax

class SEL:

    def __init__(self, fmin=8.9125, fmax=22387, band_type='third_octave', num_bands=30, ref=1e-6, integration_time=None):
        """
        Initialize SEL calculator.

        Args:
            fmin (float): Minimum frequency in Hz
            fmax (float): Maximum frequency in Hz
            band_type (str): Type of frequency bands ('octave', 'third_octave', or 'linear')
            num_bands (int): Number of bands for linear band_type
            ref (float): Reference pressure level in Pa
            integration_time (float): Integration time in seconds (if None, uses full signal length)
        """
        self.fmin = fmin
        self.fmax = fmax
        self.band_type = band_type
        self.num_bands = num_bands
        self.duration = None
        self.ref = ref  # Store the reference level as an attribute
        self.integration_time = integration_time

    def _adjust_fmin_fmax(self, fs):
        """
        Adjust minimum and maximum frequencies to match band boundaries.

        Args:
            fs (float): Sampling frequency in Hz
        """
        if self.band_type == 'octave':
            self.fmin = 2 ** _np.floor(math.log2(self.fmin))
            self.fmax = 2 ** _np.ceil(math.log2(self.fmax))
            if self.fmax > fs/2 :
                self.fmax = 2 ** _np.floor(math.log2(self.fmax))
        elif self.band_type == 'third_octave':
            base = math.pow(2, 1/6)
            self.fmin = base ** _np.floor(math.log(self.fmin, base))
            self.fmax = base ** _np.ceil(math.log(self.fmax, base))
            if self.fmax > fs/2 :
                self.fmax = base ** _np.floor(math.log(self.fmax, base))

    def _generate_frequency_bands(self, fs):
        """
        Generate frequency bands based on specified band_type.

        Args:
            fs (float): Sampling frequency in Hz

        Returns:
            list: List of tuples containing (low, center, high) frequencies for each band
        """
        if self.fmin <= 0 or self.fmax <= self.fmin:
            raise ValueError("fmin must be > 0 and fmax must be greater than fmin.")

        if self.band_type in ['octave', 'third_octave']:
            self._adjust_fmin_fmax(fs)

        bands = []

        if self.band_type == 'octave':
            base = math.sqrt(2)
            f_center = self.fmin
            while f_center < self.fmax:
                f_low = f_center / base
                f_high = f_center * base
                bands.append((f_low, f_center, f_high))
                f_center *= 2
            if bands and bands[-1][2] > self.fmax:
                bands[-1] = (bands[-1][0], bands[-1][1], self.fmax)

        elif self.band_type == 'third_octave':
            base = math.pow(2, 1/6)
            f_center = self.fmin
            while f_center < self.fmax:
                f_low = f_center / base
                f_high = f_center * base
                bands.append((f_low, f_center, f_high))
                f_center *= math.pow(2, 1/3)
            if bands and bands[-1][2] > self.fmax:
                bands[-1] = (bands[-1][0], bands[-1][1], self.fmax)

        elif self.band_type == 'linear':
            if self.num_bands <= 0:
                raise ValueError("num_bands must be a positive integer for linear bands.")
            band_width = (self.fmax - self.fmin) / self.num_bands
            f_low = self.fmin
            for _ in range(self.num_bands):
                f_high = f_low + band_width
                f_center = (f_low + f_high) / 2
                bands.append((f_low, f_center, f_high))
                f_low = f_high
            if bands and bands[-1][2] > self.fmax:
                bands[-1] = (bands[-1][0], bands[-1][1], self.fmax)

        else:
            raise ValueError("Invalid band_type. Choose 'octave', 'third_octave', or 'linear'.")

        return bands

    def compute(self, data, fs, chunk_size=262144, nfft=None):
        """
        Compute Sound Exposure Level for each frequency band.

        Args:
            data (array): Input time series data in Pa
            fs (float): Sampling frequency in Hz
            nfft (int, optional): Number of FFT points

        Returns:
            tuple: (sel, bands) where sel contains SEL values in Pa².s and bands contains frequency bands
        """
        # Determine how much data to process based on integration_time
        if self.integration_time is not None:
            samples_to_process = min(int(self.integration_time * fs), len(data))
            data = data[:samples_to_process]

        self.bands = self._generate_frequency_bands(fs)
        self.duration = len(data)/fs
        if chunk_size > len(data):
            self.chunk_size = len(data)
        else:
            self.chunk_size = chunk_size

        if nfft is None:
            nfft = fs

        window = _sp.windows.hann(nfft)

        # Initialize frequency axis to determine band indices
        f = _np.fft.rfftfreq(nfft, d=1/fs)

        # Initialize band indices
        band_indices = []
        for low, center, high in self.bands:
            idx = _np.logical_and(f >= low, f < high)
            band_indices.append(idx)

        # Initialize SEL accumulator
        self.sel = _np.zeros(len(self.bands))

        # Process data in chunks
        for i in range(0, len(data), chunk_size):
            chunk = data[i:min(i + chunk_size, len(data))]

            if len(chunk) < nfft:
                chunk = _np.pad(chunk, (0, nfft - len(chunk)))

            # Compute spectrogram for chunk
            f, t, Sxx = _sp.spectrogram(chunk, fs, window=window, noverlap=0,
                                      nfft=nfft, scaling='spectrum')
            Sxx_sum = _np.sum(Sxx, axis=1)

            # Accumulate SEL in each band
            for k, idx in enumerate(band_indices):
                self.sel[k] += _np.sum(Sxx_sum[idx])

        return self.sel, self.bands

    def plot(self, title='', ylim=(0, 200)):
        """
        Plot Sound Exposure Level spectrum.

        Args:
            title (str): Plot title
            ylim (tuple): Y-axis limits as (min, max)

        Returns:
            tuple: (figure, axis) matplotlib objects
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        Fedges = [low for low, _, _ in self.bands] + [self.bands[-1][2]]
        width = [Fedges[i + 1] - Fedges[i] for i in range(len(Fedges) - 1)]
        ax.bar(Fedges[:-1], 10 * _np.log10(self.sel / (self.ref ** 2)), width=width, align='edge', edgecolor='black')

        # If the duration is provided, include it in the title
        ax.set_title(f'[SEL {self.duration}s] {title}', loc='left')

        if self.ref == 1e-6:
            ref = "1µ"
        elif self.ref == 2e-5:
            ref ="20µ"
        else:
            ref = f"{self.ref:02e}"
        ax.set_ylabel(f'Level [dB re {ref}Pa²·s]')
        if self.band_type != 'linear':
            ax.set_xscale('log')
        ax.set_xlabel(f'Frequency ({self.band_type}) [Hz]')
        ax.set_ylim(ylim)
        ax.grid(which='both', alpha=0.75)
        ax.set_axisbelow(True)
        return fig, ax

class PSD:

    def __init__(self, ref=1e-6, **kwargs):
        """
        Power Spectral Density (PSD) computation and visualization class.

        Parameters:
        - ref: Reference level for dB scaling.
        - **kwargs: Additional arguments for scipy.signal.welch.
        """
        self.ref = ref

        # Default Welch parameters, overridden by kwargs if provided
        self.welch_params = {
            "nperseg": 8192,
            "noverlap": 4096,
            "window": "hann",
            "scaling": "density",
        }
        self.welch_params.update(kwargs)

    def compute(self, data, fs):
        """
        Compute the Power Spectral Density (PSD) using Welch's method.

        Parameters:
        - data: Input signal array (Pa).
        - fs: Sampling frequency of the signal (Hz).

        Returns:
        - freqs: Array of frequencies (Hz).
        - psd: Array of PSD values (linear scale).
        """
        # Compute Welch periodogram
        freqs, Pxx = _sp.welch(data, fs, **self.welch_params)

        # Store frequencies and PSD values
        self.freqs = freqs
        self.psd = Pxx
        return freqs, Pxx

    def plot(self, title="", label="", ymin=0, ymax=150, **kwargs):
        """
        Plot the computed PSD as a line plot.

        Parameters:
        - title: Plot title.
        - ylim: Y-axis limits (dB).
        """
        if not hasattr(self, "freqs") or not hasattr(self, "psd"):
            raise RuntimeError("You must compute the PSD before plotting it.")

        # Convert PSD to dB scale
        psd_db = 10 * _np.log10(self.psd / (self.ref ** 2))

        # Plot PSD
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogx(self.freqs, psd_db, label=label, **kwargs)

        # Customize plot appearance
        ax.set_title(f"[PSD] {title}", loc='left')
        ax.set_xlabel("Frequency [Hz]")
        if self.ref == 1e-6:
            ref = "1µ"
        elif self.ref == 2e-5:
            ref ="20µ"
        else:
            ref = f"{self.ref:02e}"
        ax.set_ylabel(f'Level [dB re {ref}Pa²/Hz]')
        ax.set_ylim((ymin,ymax))
        ax.set_xlim((_np.max((self.freqs[0],1)),self.freqs[-1]))
        ax.grid(which="both", alpha=0.75)
        plt.tight_layout()
        if label != "":
            ax.legend()

        return fig, ax

    def add2plot(self, ax, Fxx=None, Pxx=None, ref=None, label="", **kwargs):

        if Fxx is None and Pxx is None:
            Fxx = self.freqs
            Pxx = self.psd
        if ref is None:
            ref = self.ref
            
        psd_db = 10 * _np.log10(Pxx / (ref ** 2))
        ax.plot(Fxx, psd_db, label=label, **kwargs)
        if label != "":
            ax.legend()

        return ax


class FRF:
    def __init__(self, method='welch', **kwargs):
        """
        Transfer Function (Frequency Response Function, FRF) computation and visualization class.

        Parameters:
        - **kwargs: Additional arguments for scipy.signal.welch and scipy.signal.csd.

        Notes:
        The Transfer Function (FRF) is a complex function that relates the input and output of a linear time-invariant (LTI) system in the frequency domain.
        It is defined as:

            H(f) = Pxy(f) / Pxx(f)

        Where:
        - Pxx(f): Power Spectral Density (PSD) of the input signal (x).
        - Pxy(f): Cross-Power Spectral Density (CPSD) between input (x) and output (y).
        """
        # Default parameters, overridden by kwargs if provided
        self.params = {
            "nperseg": 8192,
            "noverlap": 0,
        }
        self.params.update(kwargs)
        self .method = method

    def compute(self, x, y, fs, method=None, nperseg=None, noverlap=None):

        if method != None:
            self.method = method

        if nperseg != None:
            self.params['nperseg'] = nperseg

        if noverlap != None:
            self.params['noverlap'] = noverlap

        if self.method == 'welch':
            freqs, mag, phase, coh = self.compute_welch(x, y, fs)
        elif self.method == 'stft':
            freqs, mag, phase, coh = self.compute_stft(x, y, fs)

        return freqs, mag, phase, coh

    def compute_welch(self, x, y, fs):
        """
        Compute the Frequency Response Function (FRF) using Welch's method.
        This method is more dedicated to stationary signals.
        Coherence indicates the degree of linear dependency between input (x) and output (y) at each frequency.

        Parameters:
        - x: Input signal array (reference).
        - y: Output signal array.
        - fs: Sampling frequency of the signals (Hz).

        Returns:
        - freqs: Array of frequencies (Hz).
        - tf: Array of transfer function values (complex).
        - coh: Array of coherence values.
        """
        # Compute cross-spectral densities
        freqs, Pxx = _sp.welch(x, fs, scaling='density', **self.params)
        _, Pyy = _sp.welch(y, fs, scaling='density', **self.params)
        _, Pxy = _sp.csd(x, y, fs, scaling='density', **self.params)

        # Compute transfer function and coherence
        tf = Pxy / Pxx
        coh = abs(Pxy)**2 / (Pxx * Pyy)

        # Store computed values
        self.freqs = freqs
        self.tf = tf
        self.coh = coh

        # Split transfer function into magnitude and phase
        mag = _np.abs(tf)
        phase = _np.angle(tf, deg=True)

        return freqs, mag, phase, coh

    def compute_stft(self, x, y, fs):
        """
        Compute the Frequency Response Function (FRF) using Short-Time Fourier Transform (STFT).
        This method is more dedicated to non-stationary signals.
        Coherence indicates the degree of linear dependency between input (x) and output (y) at each frequency.

        Parameters:
        - x: Input signal array (reference).
        - y: Output signal array.
        - fs: Sampling frequency of the signals (Hz).

        Returns:
        - freqs: Array of frequencies (Hz).
        - mag: Magnitude of the transfer function.
        - phase: Phase of the transfer function (degrees).
        - coh_avg: Average coherence values.
        """
        # Create ShortTimeFFT object
        stft = _sp.ShortTimeFFT(_sp.windows.hann(self.params["nperseg"]),
                                hop=self.params["nperseg"]-self.params["noverlap"],
                                fs=fs,
                                scale_to='psd')

        # Compute STFT for x and y
        Zxx = stft.spectrogram(x)
        Zyy = stft.spectrogram(y)
        freqs = _np.arange(stft.f_pts) * stft.delta_f

        # Compute cross-spectrogram
        Zxy = stft.spectrogram(x, y)

        # Compute transfer function
        tf = Zxy / Zxx
        tf_avg = _np.mean(tf, axis=1)  # Moyenne complexe de la fonction de transfert

        # Compute magnitude and phase after averaging
        mag = _np.abs(tf_avg)  # Module après moyennage
        phase = _np.angle(tf_avg, deg=True)  # Phase après moyennage

        # Compute coherence based on averaged TF
        Sxx_avg = _np.mean(_np.abs(Zxx), axis=1)
        Syy_avg = _np.mean(_np.abs(Zyy), axis=1)
        Sxy_avg = _np.mean(Zxy, axis=1)
        coh_avg = _np.abs(Sxy_avg)**2 / (Sxx_avg * Syy_avg)

        # Store computed values
        self.freqs = freqs
        self.tf = tf_avg
        self.coh = coh_avg

        return freqs, mag, phase, coh_avg


    def plot(self, title="", label="", ymin=-60, ymax=60, **kwargs):
        """
        Plot the computed Transfer Function as magnitude and phase plots.

        Parameters:
        - title: Plot title.
        - label: Legend label.
        - ymin, ymax: Y-axis limits for magnitude plot (dB).
        - **kwargs: Additional plotting arguments.

        Notes:
        The magnitude (in dB) is computed as:
            20 * log10(|H(f)|)

        Phase is given in degrees.
        Coherence is plotted to assess the reliability of the FRF.
        """
        if not hasattr(self, "freqs") or not hasattr(self, "tf"):
            raise RuntimeError("You must compute the Transfer Function before plotting it.")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        ax1.set_title(f"[FRF] {title}", loc="left")

        # Magnitude plot
        mag_db = 20 * _np.log10(_np.abs(self.tf))
        ax1.plot(self.freqs, mag_db, label=label, **kwargs)
        ax1.set_ylabel("Magnitude [dB]")
        ax1.set_xscale("log")
        ax1.set_ylim((ymin, ymax))
        ax1.set_xlim((_np.max((self.freqs[0], 1)), self.freqs[-1]))
        ax1.grid(which='major', alpha=0.75)
        ax1.grid(which='minor', alpha=0.25)
        ax1.set_xticklabels([])
        ax1.tick_params(axis='x', which='both', bottom=False)

        # Phase plot
        phase_deg = _np.angle(self.tf, deg=True)
        ax2.plot(self.freqs, phase_deg, label=label, **kwargs)
        ax2.set_ylabel("Phase [degrees]")
        ax2.set_xscale("log")
        ax2.set_ylim((-180, 180))
        ax2.set_xlim((_np.max((self.freqs[0], 1)), self.freqs[-1]))
        ax2.grid(which='major', alpha=0.75)
        ax2.grid(which='minor', alpha=0.25)
        ax2.set_xticklabels([])
        ax2.tick_params(axis='x', which='both', bottom=False)

        # Coherence plot
        ax3.plot(self.freqs, self.coh, label=label, **kwargs)
        ax3.set_xlabel("Frequency [Hz]")
        ax3.set_ylabel("Coherence")
        ax3.set_xscale("log")
        ax3.set_ylim((0.75, 1.01))
        ax3.set_xlim((_np.max((self.freqs[0], 1)), self.freqs[-1]))
        ax3.grid(which='major', alpha=0.75)
        ax3.grid(which='minor', alpha=0.25)
        ax3.tick_params(axis='x', which='both')

        if label != "":
            ax1.legend()
            ax2.legend()
            ax3.legend()

        plt.tight_layout()

        return fig, (ax1, ax2, ax3)

    def add2plot(self, axes, label="", **kwargs):
        """
        Add transfer function data to existing plots.

        Parameters:
        - axes: Tuple of (magnitude_axis, phase_axis, coherence_axis).
        - label: Legend label.
        - **kwargs: Additional plotting arguments.
        """
        ax1, ax2, ax3 = axes

        mag_db = 20 * _np.log10(_np.abs(self.tf))
        phase_deg = _np.angle(self.tf, deg=True)

        ax1.plot(self.freqs, mag_db, label=label, **kwargs)
        ax2.plot(self.freqs, phase_deg, label=label, **kwargs)
        ax3.plot(self.freqs, self.coh, label=label, **kwargs)

        if label != "":
            ax1.legend()
            ax2.legend()
            ax3.legend()

        return axes

class PSDPDF:

    def __init__(self, ref=1e-6, seg_duration=1, overlap_pct=0, nbins=100, lvlmin=40, lvlmax=150, **kwargs):
        """
        Class to compute and visualize the probability density function (PDF) of PSD over multiple segments.

        Parameters:
        - ref: Reference value for scaling in Pa (default 1e-6).
        - seg_duration: Duration of each segment for Welch computation (seconds).
        - overlap_pct: Percentage overlap between consecutive segments.
        - **kwargs: Additional arguments for scipy.signal.welch.
        """
        self.seg_duration = seg_duration
        self.overlap_pct = overlap_pct
        self.ref = ref
        self.nbins = nbins
        self.lvlmin = lvlmin
        self.lvlmax = lvlmax

        # Default Welch parameters, overridden by kwargs if provided
        self.welch_params = {
            "nperseg": 8192,
            "noverlap": 4096,
            "window": "hann",
            "scaling": "density"
        }
        self.welch_params.update(kwargs)

    def compute(self, data, fs):
        """
        Compute the PDF of PSD from signal segments.

        Parameters:
        - data: Input signal array.
        - fs: Sampling frequency of the signal (Hz).

        Returns:
        - freqs: Array of frequencies (Hz).
        - levels: Array of level bins (dB re ref²).
        - pdf: 2D array representing the probability density (normalized).
        """
        # Calculate chunk size and overlap in samples
        chunk_size = int(self.seg_duration * fs)
        overlap_samples = int(chunk_size * self.overlap_pct / 100)
        step = chunk_size - overlap_samples

        # Create level bins for histogram
        levels = _np.linspace(self.lvlmin, self.lvlmax, self.nbins)

        # Process data in chunks
        psd_list = []

        for i in range(0, len(data) - chunk_size + 1, step):
            chunk = data[i:min(i+chunk_size, len(data))]

            if len(chunk) < self.welch_params['nperseg']:
                chunk = _np.pad(chunk, (0, self.welch_params['nperseg'] - len(chunk)))

            freqs, psd = _sp.welch(chunk, fs, **self.welch_params)
            psd_list.append(psd)

        # Convert accumulated PSDs to dB scale
        psd_segments = 10 * _np.log10(_np.array(psd_list) / (self.ref ** 2))

        # Compute mean and standard deviation
        self.mean_psd = _np.mean(psd_segments, axis=0)
        self.std_psd = _np.std(psd_segments, axis=0)

        # Compute PDF using a histogram
        pdf = _np.zeros((len(levels) - 1, len(freqs)))
        for i in range(len(freqs)):
            hist, _ = _np.histogram(psd_segments[:, i], bins=levels, density=True)
            pdf[:, i] = hist

        # Replace zeros with NaNs
        pdf[pdf == 0] = _np.nan

        self.binwidth_dB = levels[1] - levels[0]
        self.freqs = freqs
        self.levels = 10**(levels/10)*(self.ref**2)
        self.pdf = pdf

        return freqs, levels, pdf

    def plot(self, title="", ymin=0, ymax=200, vmin=0, vmax=None):
        """
        Plot the computed PDF as a colormap.

        Parameters:
        - title: Plot title.
        - **kwargs: Additional arguments for plotting, including 'ylim' for y-axis limits and other settings.
        """

        if vmax == None:
            vmax = 1/self.binwidth_dB

        fig, ax = plt.subplots(figsize=(10, 6))
        align_ybins = self.binwidth_dB/2
        pcm = ax.pcolormesh(
            self.freqs,
            10*_np.log10(self.levels[:-1]/(self.ref**2))+align_ybins,
            self.pdf,
            cmap="jet",
            shading="auto",
            vmin=vmin,  # Setting the minimum value for color scaling
            vmax=vmax,  # Setting the maximum value for color scaling
            alpha=1,
        )
        fig.colorbar(pcm, ax=ax, label=f"Probability Density Estimate [{self.binwidth_dB:.1f} dB/bin]")

        # Plot mean and standard deviation
        ax.plot(self.freqs, self.mean_psd, 'k-', label='Mean', linewidth=1.5)
        ax.plot(self.freqs, self.mean_psd + self.std_psd, 'k--', label='Mean ± STD')
        ax.plot(self.freqs, self.mean_psd - self.std_psd, 'k--')

        ax.set_title(f"[PSD-PDF {self.seg_duration}s] {title}", loc='left')
        ax.set_xlabel("Frequency [Hz]")
        if self.ref == 1e-6:
            ref = "1µ"
        elif self.ref == 2e-5:
            ref = "20µ"
        else:
            ref = f"{self.ref:02e}"
        ax.set_ylabel(f'Level [dB re {ref}Pa²/Hz]')
        ax.set_xscale("log")
        ax.set_xlim((_np.max((self.freqs[0],1)), self.freqs[-1]))
        ax.set_ylim((ymin, ymax))
        ax.grid(which="both", alpha=0.5)
        ax.legend(loc='upper right')

        return fig, ax

class Spectrogram:

    def __init__(self, ref=1e-6, **kwargs):
        """
        Spectrogram computation and visualization class.

        Parameters:
        - ref: Reference level for dB scaling.
        - **kwargs: Additional arguments for scipy.signal.spectrogram.
        """
        self.ref = ref

        # Default spectrogram parameters, overridden by kwargs if provided
        self.spec_params = {
            "nperseg": 8192,
            "noverlap": 4096,
            "window": "hann",
        }
        self.spec_params.update(kwargs)

    def compute(self, data, fs):
        """
        Compute the spectrogram using scipy.signal.spectrogram.

        Parameters:
        - data: Input signal array (Pa).
        - fs: Sampling frequency of the signal (Hz).

        Returns:
        - freqs: Array of frequencies (Hz).
        - times: Array of time points (s).
        - Sxx: 2D array of spectrogram values.
        """
        freqs, times, Sxx = _sp.spectrogram(data, fs, scaling='density', mode='psd',  **self.spec_params)

        self.freqs = freqs
        self.times = times
        self.Sxx = Sxx

        return freqs, times, Sxx

    def plot(self, title="", ymin=1, ymax=None, vmin=0, vmax=200):
        """
        Plot the computed spectrogram as a colormap.

        Parameters:
        - title: Plot title.
        - ymin: Minimum frequency to display (Hz).
        - ymax: Maximum frequency to display (Hz).
        - vmin: Minimum value for color scaling (dB).
        - vmax: Maximum value for color scaling (dB).
        """
        if not hasattr(self, "freqs") or not hasattr(self, "times") or not hasattr(self, "Sxx"):
            raise RuntimeError("You must compute the spectrogram before plotting it.")

        # Convert to dB scale
        Sxx_db = 10 * _np.log10(self.Sxx / (self.ref ** 2))

        fig, ax = plt.subplots(figsize=(10, 6))
        pcm = ax.pcolormesh(
            self.times,
            self.freqs,
            Sxx_db,
            cmap="jet",
            shading="auto",
            vmin=vmin,
            vmax=vmax
        )
        cbar = fig.colorbar(pcm, ax=ax)

        if self.ref == 1e-6:
            ref = "1µ"
        elif self.ref == 2e-5:
            ref = "20µ"
        else:
            ref = f"{self.ref:02e}"
        cbar.set_label(f'Level [dB re {ref}Pa²/Hz]')
        ax.set_title(f"[Spectrogram] {title}", loc='left')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        #ax.set_yscale("log")

        if ymax is None:
            ymax = self.freqs[-1]

        ax.set_ylim((ymin, ymax))
        ax.grid(which="both", alpha=0.25, color='black')

        return fig, ax

def SSRP(Pxx, Fxx, duration=1, scale=1):
    """
    Spectral Synthesis of Random Processes.

    Create a noise in the time domain based on a given PSD.
    The PSD length should be 2**N for efficient computation.
    The signal will be sampled at Fxx[-1]*2.

    :param Pxx: Power spectral density in (U/scale)**2/Hz
    :param Fxx: Frequency array in Hz
    :param duration: Duration of the generated signal in seconds
    :param scale: Scale factor
    :returns: tuple (t, x, fs) where t is time array in s, x is signal array in U, fs is sampling frequency in Hz

    >>> import numpy as np
    >>> fs = 1000
    >>> f = np.linspace(0, fs/2, 512)
    >>> Pxx = np.ones_like(f)
    >>> t, x, fs = SSRP(Pxx, f, 1, 1)
    """
    dF = (Fxx[1]-Fxx[0])
    Pxx = Pxx*dF
    fmin = Fxx[0]
    fmax = Fxx[-1]
    fs = fmax*2
    v = Pxx * fmin / 4
    N = len(Pxx)

    # Calculate chunk parameters
    chunk_size = 2 * (N + 1)  # Size of each chunk
    overlap_size = chunk_size // 4  # 50% overlap
    samples_needed = int(duration * fs)
    num_chunks = int(_np.ceil(samples_needed / (chunk_size - overlap_size)))

    # Initialize output arrays
    x_total = _np.zeros(samples_needed)
    t_total = _np.arange(samples_needed) / fs

    # Generate chunks
    for i in range(num_chunks):
        # Generate chunk
        vi = _np.random.randn(N)
        vq = _np.random.randn(N)
        w = (vi + 1j * vq) * _np.sqrt(v)
        chunk = _np.fft.irfft(_np.concatenate(([0], w)), chunk_size)
        chunk = chunk * chunk_size

        # Create fade in/out windows
        fade = _np.ones(chunk_size)
        if i > 0:  # Fade in
            fade[:overlap_size] = _np.sin(_np.pi/2 * _np.linspace(0, 1, overlap_size))
        if i < num_chunks-1:  # Fade out
            fade[-overlap_size:] = _np.sin(_np.pi/2 * _np.linspace(1, 0, overlap_size))
        chunk = chunk * fade

        # Calculate chunk position
        start_idx = i * (chunk_size - overlap_size)
        end_idx = start_idx + chunk_size

        # Add chunk to output, accounting for final chunk potentially being too long
        if end_idx > samples_needed:
            chunk = chunk[:samples_needed-start_idx]
            end_idx = samples_needed

        x_total[start_idx:end_idx] += chunk[:end_idx-start_idx] * scale

    return t_total, x_total, int(fs)
