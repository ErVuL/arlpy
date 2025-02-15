##############################################################################
#
# Copyright (c) 2016, Mandar Chitre
#
# This file is part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
##############################################################################

"""Signal processing toolbox."""

import functools
import operator as _op
import numpy as _np
import scipy.signal as _sig
import arlpy.utils as _utils
import scipy.signal as _sp
import matplotlib.pyplot as plt
import math

def time(n, fs):
    """Generate a time vector for time series.

    :param n: time series, or number of samples
    :param fs: sampling rate in Hz
    :returns: time vector starting at time 0

    >>> import arlpy
    >>> t = arlpy.signal.time(100000, fs=250000)
    >>> t
    array([  0.00000000e+00,   4.00000000e-06,   8.00000000e-06, ...,
         3.99988000e-01,   3.99992000e-01,   3.99996000e-01])
    >>> x = arlpy.signal.cw(fc=27000, duration=0.5, fs=250000)
    >>> t = arlpy.signal.time(x, fs=250000)
    >>> t
    array([  0.00000000e+00,   4.00000000e-06,   8.00000000e-06, ...,
         4.99988000e-01,   4.99992000e-01,   4.99996000e-01])
    """
    if hasattr(n, "__len__"):
        n = _np.asarray(n).shape[0]
    return _np.arange(n, dtype=_np.float64)/fs

def cw(fc, duration, fs, window=None, complex_output=False):
    """Generate a sinusoidal pulse.

    :param fc: frequency of the pulse in Hz
    :param duration: duration of the pulse in s
    :param fs: sampling rate in Hz
    :param window: window function to use (``None`` means rectangular window)
    :param complex_output: True to return complex signal, False for a real signal

    For supported window functions, see documentation for :func:`scipy.signal.get_window`.

    >>> import arlpy
    >>> x1 = arlpy.signal.cw(fc=27000, duration=0.5, fs=250000)
    >>> x2 = arlpy.signal.cw(fc=27000, duration=0.5, fs=250000, window='hamming')
    >>> x3 = arlpy.signal.cw(fc=27000, duration=0.5, fs=250000, window=('kaiser', 4.0))
    """
    n = int(round(duration*fs))
    x = _np.exp(2j*_np.pi*fc*time(n, fs)) if complex_output else _np.sin(2*_np.pi*fc*time(n, fs))
    if window is not None:
        w = _sig.get_window(window, n, False)
        x *= w
    return x

def sweep(f1, f2, duration, fs, method='linear', window=None):
    """Generate frequency modulated sweep.

    :param f1: starting frequency in Hz
    :param f2: ending frequency in Hz
    :param duration: duration of the pulse in s
    :param fs: sampling rate in Hz
    :param method: type of sweep (``'linear'``, ``'quadratic'``, ``'logarithmic'``, ``'hyperbolic'``)
    :param window: window function to use (``None`` means rectangular window)

    For supported window functions, see documentation for :func:`scipy.signal.get_window`.

    >>> import arlpy
    >>> x1 = arlpy.signal.sweep(20000, 30000, duration=0.5, fs=250000)
    >>> x2 = arlpy.signal.sweep(20000, 30000, duration=0.5, fs=250000, window='hamming')
    >>> x2 = arlpy.signal.sweep(20000, 30000, duration=0.5, fs=250000, window=('kaiser', 4.0))
    """
    n = int(round(duration*fs))
    x = _sig.chirp(time(n, fs), f1, duration, f2, method)
    if window is not None:
        w = _sig.get_window(window, n, False)
        x *= w
    return x

def envelope(x, axis=-1):
    """Generate a Hilbert envelope of the real signal x.

    :param x: real passband signal
    :param axis: axis of the signal, if multiple signals specified
    """
    return _np.abs(_sig.hilbert(x, axis=axis))

def mseq(spec, n=None):
    """Generate m-sequence.

    m-sequences are sequences of :math:`\\pm 1` values with near-perfect discrete periodic
    auto-correlation properties. All non-zero lag periodic auto-correlations
    are -1. The zero-lag autocorrelation is :math:`2^m-1`, where m is the shift register
    length.

    This function currently supports shift register lengths between 2 and 30.

    :param spec: m-sequence specifier (shift register length or taps)
    :param n: length of sequence (``None`` means full length of :math:`2^m-1`)

    >>> import arlpy
    >>> x = arlpy.signal.mseq(7)
    >>> len(x)
    127
    """
    if isinstance(spec, int):
        if spec < 2 or spec > 30:
            raise ValueError('spec must be between 2 and 30')
        known_specs = {  # known m-sequences are specified as base 1 taps
             2: [1,2],          3: [1,3],          4: [1,4],          5: [2,5],
             6: [1,6],          7: [1,7],          8: [1,2,7,8],      9: [4,9],
            10: [3,10],        11: [9,11],        12: [6,8,11,12],   13: [9,10,12,13],
            14: [4,8,13,14],   15: [14,15],       16: [4,13,15,16],  17: [14,17],
            18: [11,18],       19: [14,17,18,19], 20: [17,20],       21: [19,21],
            22: [21,22],       23: [18,23],       24: [17,22,23,24], 25: [22,25],
            26: [20,24,25,26], 27: [22,25,26,27], 28: [25,28],       29: [27,29],
            30: [7,28,29,30]
        }
        spec = list(map(lambda x: x-1, known_specs[spec]))  # convert to base 0 taps
    spec.sort(reverse=True)
    m = spec[0]+1
    if n is None:
        n = 2**m-1
    reg = _np.ones(m, dtype=_np.uint8)
    out = _np.zeros(n)
    for j in range(n):
        b = functools.reduce(_op.xor, reg[spec], 0)
        reg = _np.roll(reg, 1)
        out[j] = float(2*reg[0]-1)
        reg[0] = b
    return out

def gmseq(spec, theta=None):
    """Generate generalized m-sequence.

    Generalized m-sequences are related to m-sequences but have an additional parameter
    :math:`\\theta`. When :math:`\\theta = \\pi/2`, generalized m-sequences become normal m-sequences. When
    :math:`\\theta < \\pi/2`, generalized m-sequences contain a DC-component that leads to an exalted
    carrier after modulation.

    When theta is :math:`\\arctan(\\sqrt{n})` where :math:`n` is the length of the m-sequence, the m-sequence
    is considered to be period matched. Period matched m-sequences are complex sequences
    with perfect discrete periodic auto-correlation properties, i.e., all non-zero lag
    periodic auto-correlations are zero. The zero-lag autocorrelation is :math:`n = 2^m-1`, where
    m is the shift register length.

    This function currently supports shift register lengths between 2 and 30.

    :param spec: m-sequence specifier (shift register length or taps)
    :param theta: transmission angle (``None`` to use period-matched angle)

    >>> import arlpy
    >>> x = arlpy.signal.gmseq(7)
    >>> len(x)
    127
    """
    x = mseq(spec)
    if theta is None:
        theta = _np.arctan(_np.sqrt(len(x)))
    return _np.cos(theta) + 1j*_np.sin(theta)*x

def bb2pb(x, fd, fc, fs=None, axis=-1):
    """Convert baseband signal to passband.

    For communication applications, one may wish to use :func:`arlpy.comms.upconvert` instead,
    as that function supports pulse shaping.

    The convention used in that exp(2j*pi*fc*t) is a positive frequency carrier.

    :param x: complex baseband signal
    :param fd: sampling rate of baseband signal in Hz
    :param fc: carrier frequency in passband in Hz
    :param fs: sampling rate of passband signal in Hz (``None`` => same as `fd`)
    :param axis: axis of the signal, if multiple signals specified
    :returns: real passband signal, sampled at `fs`
    """
    if fs is None or fs == fd:
        y = _np.array(x, dtype=_np.complex128)
        fs = fd
    else:
        y = _sig.resample_poly(_np.asarray(x, dtype=_np.complex128), fs, fd, axis=axis)
    osc = _np.sqrt(2)*_np.exp(2j*_np.pi*fc*time(y,fs))
    y *= _utils.broadcastable_to(osc, y.shape, axis)
    return y.real

def pb2bb(x, fs, fc, fd=None, flen=127, cutoff=None, axis=-1):
    """Convert passband signal to baseband.

    The baseband conversion uses a low-pass filter after downconversion, with a
    default cutoff frequency of `0.6*fd`, if `fd` is specified, or `1.1*fc` if `fd`
    is not specified. Alternatively, the user may specify the cutoff frequency
    explicitly.

    For communication applications, one may wish to use :func:`arlpy.comms.downconvert` instead,
    as that function supports matched filtering with a pulse shape rather than a generic
    low-pass filter.

    The convention used in that exp(2j*pi*fc*t) is a positive frequency carrier.

    :param x: passband signal
    :param fs: sampling rate of passband signal in Hz
    :param fc: carrier frequency in passband in Hz
    :param fd: sampling rate of baseband signal in Hz (``None`` => same as `fs`)
    :param flen: number of taps in the low-pass FIR filter
    :param cutoff: cutoff frequency in Hz (``None`` means auto-select)
    :param axis: axis of the signal, if multiple signals specified
    :returns: complex baseband signal, sampled at `fd`
    """
    if cutoff is None:
        cutoff = 0.6*fd if fd is not None else 1.1*_np.abs(fc)
    osc = _np.sqrt(2)*_np.exp(-2j*_np.pi*fc*time(x.shape[axis],fs))
    y = x * _utils.broadcastable_to(osc, x.shape, axis)
    hb = _sig.firwin(flen, cutoff=cutoff, nyq=fs/2.0)
    y = _sig.filtfilt(hb, 1, y, axis=axis)
    if fd is not None and fd != fs:
        y = _sig.resample_poly(y, 2*fd, fs, axis=axis)
        y = _np.apply_along_axis(lambda a: a[::2], axis, y)
    return y

def mfilter(s, x, complex_output=False, axis=-1):
    """Matched filter recevied signal using a reference signal.

    :param s: reference signal
    :param x: recevied signal
    :param complex_output: True to return complex signal, False for absolute value of complex signal
    :param axis: axis of the signal, if multiple recevied signals specified
    """
    hb = _np.conj(_np.flipud(s))
    if axis < 0:
        axis += len(x.shape)
    padding = []
    x = _np.apply_along_axis(lambda a: _np.pad(a, (0, len(s)-1), 'constant'), axis, x)
    y = _sig.lfilter(hb, 1, x, axis=axis)
    y = _np.apply_along_axis(lambda a: a[len(s)-1:], axis, y)
    if not complex_output:
        y = _np.abs(y)
    return y

def lfilter0(b, a, x, axis=-1):
    """Filter data with an IIR or FIR filter with zero DC group delay.

    :func:`scipy.signal.lfilter` provides a way to filter a signal `x` using a FIR/IIR
    filter defined by `b` and `a`. The resulting output is delayed, as compared to the
    input by the group delay. This function corrects for the group delay, resulting in
    an output that is synchronized with the input signal. If the filter as an acausal
    impulse response, some precursor signal from the output will be lost. To avoid this,
    pad input signal `x` with sufficient zeros at the beginning to capture the precursor.
    Since both, :func:`scipy.signal.lfilter` and this function return a signal with the
    same length as the input, some signal tail is lost at the end. To avoid this, pad
    input signal `x` with sufficient zeros at the end.

    See documentation for :func:`scipy.signal.lfilter` for more details.

    >>> import arlpy
    >>> import numpy as np
    >>> fs = 250000
    >>> b = arlpy.uwa.absorption_filter(fs, distance=500)
    >>> x = np.pad(arlpy.signal.sweep(20000, 40000, 0.5, fs), (127, 127), 'constant')
    >>> y = arlpy.signal.lfilter0(b, 1, x)
    """
    w, g = _sig.group_delay((b, a))
    ndx = _np.argmin(_np.abs(w))
    d = int(round(g[ndx]))
    x = _np.apply_along_axis(lambda a: _np.pad(a, (0, d), 'constant'), axis, x)
    y = _sig.lfilter(b, a, x, axis)[d:]
    return y

def _lfilter_gen(b, a):
    x = _np.zeros(len(b))
    y = _np.zeros(len(a))
    while True:
        x = _np.roll(x, 1)
        x[0] = yield y[0]
        y = _np.roll(y, 1)
        y[0] = _np.sum(b*x) - _np.sum(a[1:]*y[1:])

def lfilter_gen(b, a):
    """Generator form of an FIR/IIR filter.

    The filter is a direct form I implementation of the standard difference
    equation. Data samples can be passed to the filter using the :func:`send`
    method, and the output can be read a sample at a time.

    >>> import arlpy
    >>> import numpy as np
    >>> import scipy.signal as sp
    >>> b, a = sp.iirfilter(2, 0.1, btype='lowpass')  # generate a biquad lowpass
    >>> f = arlpy.signal.filter_gen(b, a)             # create the filter
    >>> x = np.random.normal(0, 1, 1000)              # get some random data
    >>> y = [f.send(v) for v in x]                    # filter data by stepping through it
    """
    b = _np.asarray(b, dtype=_np.float64)
    if not hasattr(a, "__len__") and a == 1:
        a = [1]
    a = _np.asarray(a, dtype=_np.float64)
    if a[0] != 1.0:
        raise ValueError('a[0] must be 1')
    f = _lfilter_gen(b, a)
    f.__next__()
    return f

def nco_gen(fc, fs=2.0, phase0=0, wrap=2*_np.pi, func=lambda x: _np.exp(1j*x)):
    """Generator form of a numerically controlled oscillator (NCO).

    Samples at the output of the oscillator can be generated using the
    :func:`next` method. The oscillator frequency can be modified during
    operation using the :func:`send` method, with `fc` as the argument.

    If fs is specified, fc is given in Hz, otherwise it is specified as
    normalized frequency (Nyquist = 1).

    The default oscillator function is ``exp(i*phase)`` to generate a complex
    sinusoid. Alternate oscillator functions that take in the phase angle
    and generate other outputs can be specifed. For example, a real sinusoid
    can be generated by specifying ``sin`` as the function. The phase angle
    can be generated by specifying ``None`` as the function.

    :param fc: oscillation frequency
    :param fs: sampling frequency in Hz
    :param phase0: initial phase in radians (default: 0)
    :param wrap: phase angle to wrap phase around to 0 (default: :math:`2\\pi`)
    :param func: oscillator function of phase angle (default: complex sinusoid)

    >>> import arlpy
    >>> import math
    >>> nco = arlpy.signal.nco_gen(27000, 108000, func=math.sin)
    >>> x = [nco.next() for i in range(12)]
    >>> x = np.append(x, nco.send(54000))      # change oscillation frequency
    >>> x = np.append(x, [nco.next() for i in range(4)])
    >>> x
    [0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 1, -1, 1, -1, 1]
    """
    p = phase0
    while True:
        fc1 = yield p if func is None else func(p)
        if fc1 is not None:
            fc = fc1
        p = _np.mod(p + 2*_np.pi*fc/fs, wrap)

def nco(fc, fs=2.0, phase0=0, wrap=2*_np.pi, func=lambda x: _np.exp(1j*x)):
    """Numerically controlled oscillator (NCO).

    If fs is specified, fc is given in Hz, otherwise it is specified as
    normalized frequency (Nyquist = 1).

    The default oscillator function is ``exp(i*phase)`` to generate a complex
    sinusoid. Alternate oscillator functions that take in the phase angle
    and generate other outputs can be specifed. For example, a real sinusoid
    can be generated by specifying ``sin`` as the function. The phase angle
    can be generated by specifying ``None`` as the function.

    :param fc: array of instantaneous oscillation frequency
    :param fs: sampling frequency in Hz
    :param phase0: initial phase in radians (default: 0)
    :param wrap: phase angle to wrap phase around to 0 (default: :math:`2\\pi`)
    :param func: oscillator function of phase angle (default: complex sinusoid)

    >>> import arlpy
    >>> import numpy as np
    >>> fc = np.append([27000]*12, [54000]*5)
    >>> x = arlpy.signal.nco(fc, 108000, func=np.sin)
    >>> x
    [0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 1, -1, 1, -1, 1]
    """
    p = 2*_np.pi*fc/fs
    p[0] = phase0
    p = _np.mod(_np.cumsum(p), wrap)
    return p if func is None else func(p)

def correlate_periodic(a, v=None):
    """Cross-correlation of two 1-dimensional periodic sequences.

    a and v must be sequences with the same length. If v is not specified, it is
    assumed to be the same as a (i.e. the function computes auto-correlation).

    :param a: input sequence #1
    :param v: input sequence #2
    :returns: discrete periodic cross-correlation of a and v
    """
    a_fft = _np.fft.fft(_np.asarray(a))
    if v is None:
        v_cfft = a_fft.conj()
    else:
        v_cfft = _np.fft.fft(_np.asarray(v)).conj()
    x = _np.fft.ifft(a_fft * v_cfft)
    if _np.isrealobj(a) and (v is None or _np.isrealobj(v)):
        x = x.real
    return x

def goertzel(f, x, fs=2.0, filter=False):
    """Goertzel algorithm for single tone detection.

    The output of the Goertzel algorithm is the same as a single bin DFT if
    ``f/(fs/N)`` is an integer, where ``N`` is the number of points in signal ``x``.

    The detection metric returned by this function is the magnitude of the output
    of the Goertzel algorithm at the end of the input block. If ``filter`` is set
    to ``true``, the complex time series at the output of the IIR filter is returned,
    rather than just the detection metric.

    :param f: frequency of tone of interest in Hz
    :param x: real or complex input sequence
    :param fs: sampling frequency of x in Hz
    :param filter: output complex time series if true, detection metric otherwise (default: false)
    :returns: detection metric or complex time series

    >>> import arlpy
    >>> x1 = arlpy.signal.cw(64, 1, 512)
    >>> g1 = arlpy.signal.goertzel(64, x1, 512)
    >>> g1
    256.0
    >>> g2 = arlpy.signal.goertzel(32, x1, 512)
    >>> g2
    0.0
    """
    n = x.size
    m = f/(fs/n)
    if filter:
        y = _np.empty(n, dtype=_np.complex128)
    w1 = 0
    w2 = 0
    for j in range(n):
        w0 = 2*_np.cos(2*_np.pi*m/n)*w1 - w2 + x[j]
        if filter:
            y[j] = w0 - _np.exp(-2j*_np.pi*m/n)*w1
        w2 = w1
        w1 = w0
    if filter:
        return y
    w0 = 2*_np.cos(2*_np.pi*m/n)*w1 - w2
    return _np.abs(w0 - _np.exp(-2j*_np.pi*m/n)*w1)

def detect_impulses(x, fs, k=10, tdist=1e-3):
    """Detect impulses in `x`

    The minimum height of impulses is defined by `a+k*b`
    where `a` is median of the envelope of `x` and `b` is median
    absolute deviation (MAD) of the envelope of `x`.

    :param x: real signal
    :param fs: sampling frequency in Hz
    :param k: multiple of MAD for the impulse minimum height (default: 10)
    :param tdist: minimum time difference between neighbouring impulses in sec (default: 1e-3)
    :returns: indices and heights of detected impulses

    >>> nsamp = 1000
    >>> ind_impulses = np.array([10, 115, 641, 888])
    >>> x = np.zeros((nsamp))
    >>> x[ind_impulses] = 1
    >>> x += np.random.normal(0, 0.1, nsamp)
    >>> ind_pks, h_pks = signal.detect_impulses(x, fs=100000, k=10, tdist=1e-3)
    """
    env = envelope(x)
    height = _np.median(env)+k*_np.median(_np.abs(env-_np.median(env)))
    distance = int(tdist*fs)
    ind_imp, properties = _sig.find_peaks(env, height=height, distance=distance)
    return ind_imp, properties["peak_heights"]


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
    def __init__(self, method='welch', estimate='H1', **kwargs):
        """
        Transfer Function (Frequency Response Function, FRF) computation and visualization class.
        
        Method:
        - Welch: use Welch periodogram for PSD estimate, dedicated to stationnary signals
        - STFT:  spectrogram based calculus, dedicated to non stationnary signals
        
        Estimation calculus:
        - H1: minimizes the effect of noise introduced at the system output
        - H2: minimizes the effect of noise introduced at the system input

        Parameters:
        - **kwargs: Additional arguments for scipy.signal.welch and scipy.signal.csd.

        Notes:
        The Transfer Function (FRF) is a complex function that relates the input and output of a linear time-invariant (LTI) system in the frequency domain.
        It is defined as:

            H1(f) = Pyx(f) / Pxx(f)
            H2(f) = Pyy(f) / Pxy(f)

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
        self.estimate = estimate

    def compute(self, x, y, fs, method=None, estimate=None, nperseg=None, noverlap=None):

        if method != None:
            self.method = method

        if nperseg != None:
            self.params['nperseg'] = nperseg

        if noverlap != None:
            self.params['noverlap'] = noverlap
        
        if estimate != None:
            self.estimate = estimate

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
        
        if self.estimate == 'H2':
            _, Pxy = _sp.csd(y, x, fs, scaling='density', **self.params)
            tf = Pyy / Pxy
            coh = abs(Pxy)**2 / (Pxx * Pyy)
        else:
            _, Pyx = _sp.csd(x, y, fs, scaling='density', **self.params)
            tf = Pyx / Pxx
            coh = abs(Pyx)**2 / (Pxx * Pyy)

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
        
        Sxx_avg = _np.mean(_np.abs(Zxx), axis=1)
        Syy_avg = _np.mean(_np.abs(Zyy), axis=1)

        if self.estimate == 'H2':
            Zxy = stft.spectrogram(x, y)
            tf = Zyy / Zxy
            Sxy_avg = _np.mean(Zxy, axis=1)
            coh_avg = _np.abs(Sxy_avg)**2 / (Sxx_avg * Syy_avg)
        else:
            Zyx = stft.spectrogram(y, x)
            tf = Zyx / Zxx
            Syx_avg = _np.mean(Zyx, axis=1)
            coh_avg = _np.abs(Syx_avg)**2 / (Sxx_avg * Syy_avg)        

        # Compute magnitude and phase after averaging
        tf_avg = _np.mean(tf, axis=1)
        mag = _np.abs(tf_avg)  # Module après moyennage
        phase = _np.angle(tf_avg, deg=True)  # Phase après moyennage

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
        
        if label != "":
            addstr = f"[{self.method}-{self.estimate}] "
            label = addstr.upper() + label

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

    def add2plot(self, axes, freqs=None, mag=None, phase=None, coh=None, method=None, estimate=None, label="", **kwargs):
        """
        Add transfer function data to existing plots.

        Parameters:
        - axes: Tuple of (magnitude_axis, phase_axis, coherence_axis).
        - label: Legend label.
        - **kwargs: Additional plotting arguments.
        """
        ax1, ax2, ax3 = axes
        
        if estimate is None:
            estimate = self.estimate
        if method is None:
            method = self.method
            
        if label != "":
            addstr = f"[{method}-{estimate}] "
            label = addstr.upper() + label
            
        if freqs is None or mag is None:
            ax1.plot(self.freqs, 20 * _np.log10(_np.abs(self.tf)), label=label, **kwargs)
        else:
            ax1.plot(freqs, 20 * _np.log10(mag), label=label, **kwargs)
        
        if freqs is None or phase is None:
            ax2.plot(self.freqs, _np.angle(self.tf, deg=True), label=label, **kwargs)
        else:
            ax2.plot(freqs, phase, label=label, **kwargs)
        
        if freqs is None or coh is None:
            ax3.plot(self.freqs, self.coh, label=label, **kwargs)
        else:
            ax3.plot(freqs, coh, label=label, **kwargs)

        if label != "":
            ax1.legend()
            ax2.legend()
            ax3.legend()

        return axes
    
def resample(data, UpSamplingFactor, DownSamplingFactor):
    return _sig.resample_poly(data.astype(float), up=UpSamplingFactor, down=DownSamplingFactor)