##############################################################################
#
# Copyright (c) 2018, Mandar Chitre
#
# This file is initially part of arlpy which is released under Simplified BSD License.
# See file LICENSE or go to http://www.opensource.org/licenses/BSD-3-Clause
# for full license details.
#
# Some code have been modified and added by Theo Bertet.
# 
##############################################################################

"""
Underwater acoustic propagation modeling toolbox.
"""

import os as _os
import re as _re
import subprocess as _proc
import numpy as _np
from scipy import interpolate as _interp
import pandas as _pd
from tempfile import mkstemp as _mkstemp
from struct import unpack as _unpack
import matplotlib.pyplot as plt
import pyram.PyRAM as ram
from struct import unpack
import os 
from matplotlib import rc

# Add acoustic toolbox path to Python path
os.environ['PATH'] = os.environ['PATH'].replace(':/opt/build/at/bin', '')+":/opt/build/at/bin"

# Configure LaTeX for matplotlib
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':16})
rc('text', usetex=True)

## Constants and models definition
###############################################################################

# @todo     Check if there is differences between linear/c-linear/n2-linear/spline, and remove useless definition !

# Interpolation methods
linear       = 'linear'
spline       = 'spline'
curvilinear  = 'curvilinear'
c_linear     = 'c-linear'
n2_linear    = 'n2-linear'
analytic     = 'analytic'

# Kraken modes and options
modes        = 'modes'
TL           = 'TL'

# Bellhop modes and options
incoherent   = 'incoherent'
semicoherent = 'semicoherent'
coherent     = 'coherent'
arrivals     = 'arrivals'
eigenrays    = 'eigenrays'
rays         = 'rays'

# Boundary condition
rigid           = 'rigid'
vacuum          = 'vacuum'
acousto_elastic = 'acousto-elastic'
file            = 'file'
soft_boss       = 'soft-boss'
hard_boss       = 'hard-boss'


# models (in order of preference)
_models = []

# Objects definition
###############################################################################
   
# @todo     create and manage env3d for 3 dimensionnal problem

def create_env2d(**kv):
    """Create a new 2D underwater environment.

    A basic environment is created with default values. To see all the parameters
    available and their default values. The environment parameters may be changed 
    by passing keyword arguments or modified later using a dictionary notation.
    """
    env = {
        'name': 'arlpy',
        'type': '2D',                   # 2D only
        'model': 'BELLHOP',             # Model: BELLHOP, KRAKEN, RAM
        'frequency': 25000,             # Source frequency in Hz
        'soundspeed': 1500,             # m/s
        'soundspeed_range': [0],        # m
        'soundspeed_depth': [0],        # m
        'soundspeed_interp': spline,    # spline/linear
        'bottom_soundspeed': 1600,      # m/s
        'bottom_density': 1600,         # kg/m^3
        'bottom_absorption': 0.1,       # dB/wavelength
        'bottom_roughness': 0,          # m (rms)
        'bottom_srange': [0],           # m (bottom settings range for density absorption and roughness)
        'bottom_sdepth': [0],           # m (bottom settings depth for density absorption and roughness)
        'surface': None,                # surface profile
        'surface_interp': linear,       # curvilinear/linear
        'tx_depth': 5,                  # m
        'tx_directionality': None,      # [(deg, dB)...]
        'rx_depth': 10,                 # m
        'rx_range': 1000,               # m
        'depth': 25,                    # m
        'depth_interp': linear,         # curvilinear/linear
        'min_angle': -80,               # deg
        'max_angle': 80,                # deg
        'nbeams': 0,                    # number of beams (0 = auto)
        'nmedia': 2,                    # number of medias
        
        # Kraken specific settings
        ##########################
        
        # Boundary conditions
        'top_Bdry': 'vacuum',            # (cf. docs => OPT(2:2))
        'bottom_Bdry': 'rigid',          # (cf. docs => (6) )
        
        # Top halfspace proprieties only for acousto-elastic boundary condition
        'top_PwaveSpeed': 2000,          # m/s
        'top_SwaveSpeed': 2000,          # m/s
        'top_density': 10,               # g/cm3
        'top_PwaveAttenuation': 100,     # dB/wavelength
        'top_SwaveAttenuation': 200,     # dB/wavelength
    }
    for k, v in kv.items():
        if k not in env.keys():
            raise KeyError('Unknown key: '+k)
        env[k] = _np.asarray(v, dtype=_np.float64) if not isinstance(v, _pd.DataFrame) and _np.size(v) > 1 else v
    env = check_env2d(env)
    return env

# @todo     Update assert for each model.
def check_env2d(env):
    """Check the validity of a 2D underwater environment definition.

    :param env: environment definition

    Exceptions are thrown with appropriate error messages if the environment is invalid.

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> check_env2d(env)
    """
    try:
        assert env['model'] is not None, 'model is not defined'
        assert env['type'] == '2D', 'Not a 2D environment'
        max_range = _np.max(env['rx_range'])
        if env['surface'] is not None:
            assert _np.size(env['surface']) > 1, 'surface must be an Nx2 array'
            assert env['surface'].ndim == 2, 'surface must be a scalar or an Nx2 array'
            assert env['surface'].shape[1] == 2, 'surface must be a scalar or an Nx2 array'
            assert env['surface'][0,0] <= 0, 'First range in surface array must be 0 m'
            assert env['surface'][-1,0] >= max_range, 'Last range in surface array must be beyond maximum range: '+str(max_range)+' m'
            assert _np.all(_np.diff(env['surface'][:,0]) > 0), 'surface array must be strictly monotonic in range'
            # assert env['surface_interp'] == curvilinear or env['surface_interp'] == linear, 'Invalid interpolation type: '+str(env['surface_interp'])
        if _np.size(env['depth']) > 1:
            assert env['depth'].ndim == 2, 'depth must be a scalar or an Nx2 array'
            assert env['depth'].shape[1] == 2, 'depth must be a scalar or an Nx2 array'
            assert env['depth'][0,0] <= 0, 'First range in depth array must be 0 m'
            assert env['depth'][-1,0] >= max_range, 'Last range in depth array must be beyond maximum range: '+str(max_range)+' m'
            assert _np.all(_np.diff(env['depth'][:,0]) > 0), 'Depth array must be strictly monotonic in range'
            assert env['depth_interp'] == curvilinear or env['depth_interp'] == linear, 'Invalid interpolation type: '+str(env['depth_interp'])
            max_depth = _np.max(env['depth'][:,1])
        else:
            max_depth = env['depth']
        if isinstance(env['soundspeed'], _pd.DataFrame):
            assert env['soundspeed'].shape[0] > 3, 'soundspeed profile must have at least 4 points'
            assert env['soundspeed_depth'][0] <= 0, 'First depth in soundspeed array must be 0 m'
            assert env['soundspeed_depth'][-1] >= max_depth, 'Last depth in soundspeed array must be beyond water depth: '+str(max_depth)+' m'
            assert _np.all(_np.diff(env['soundspeed'].index) > 0), 'Soundspeed array must be strictly monotonic in depth'
        elif _np.size(env['soundspeed']) > 1:
            #assert env['soundspeed'].ndim == 2, 'soundspeed must be a scalar or an Nx2 array'
            #assert env['soundspeed'].shape[1] == 2, 'soundspeed must be a scalar or an Nx2 array'
            assert env['soundspeed'].shape[0] > 3, 'soundspeed profile must have at least 4 points'
            assert env['soundspeed_depth'][0] <= 0, 'First depth in soundspeed array must be 0 m'
            assert env['soundspeed_depth'][-1] >= max_depth, 'Last depth in soundspeed array must be beyond water depth: '+str(max_depth)+' m'
            assert _np.all(_np.diff(env['soundspeed_depth']) > 0), 'Soundspeed array must be strictly monotonic in depth'
            #assert env['soundspeed_interp'] == spline or env['soundspeed_interp'] == linear, 'Invalid interpolation type: '+str(env['soundspeed_interp'])
            if not(max_depth in env['soundspeed_depth']):
                indlarger = _np.argwhere(env['soundspeed'][:,0]>max_depth)[0][0]
                if env['soundspeed_interp'] == spline:
                    tck = _interp.splrep(env['soundspeed'][:,0], env['soundspeed'][:,1], s=0)
                    insert_ss_val = _interp.splev(max_depth, tck, der=0)
                else:
                    insert_ss_val = _np.interp(max_depth, env['soundspeed'][:,0], env['soundspeed'][:,1])
                env['soundspeed'] = _np.insert(env['soundspeed'],indlarger,[max_depth,insert_ss_val],axis = 0)
                env['soundspeed'] = env['soundspeed'][:indlarger+1,:]
        assert _np.max(env['tx_depth']) <= max_depth, 'tx_depth cannot exceed water depth: '+str(max_depth)+' m'
        assert _np.max(env['rx_depth']) <= max_depth, 'rx_depth cannot exceed water depth: '+str(max_depth)+' m'
        assert env['min_angle'] > -90 and env['min_angle'] < 90, 'min_angle must be in range (-90, 90)'
        assert env['max_angle'] > -90 and env['max_angle'] < 90, 'max_angle must be in range (-90, 90)'
        if env['tx_directionality'] is not None:
            assert _np.size(env['tx_directionality']) > 1, 'tx_directionality must be an Nx2 array'
            assert env['tx_directionality'].ndim == 2, 'tx_directionality must be an Nx2 array'
            assert env['tx_directionality'].shape[1] == 2, 'tx_directionality must be an Nx2 array'
            assert _np.all(env['tx_directionality'][:,0] >= -180) and _np.all(env['tx_directionality'][:,0] <= 180), 'tx_directionality angles must be in [-90, 90]'
        return env
    except AssertionError as e:
        raise ValueError(e.args)

def print_env(env):
    """Display the environment in a human readable form.

    :param env: environment definition

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d(depth=40, soundspeed=1540)
    >>> pm.print_env(env)
    """
    env = check_env2d(env)
    keys = ['name'] + sorted(list(env.keys()-['name']))
    for k in keys:
        v = str(env[k])
        if '\n' in v:
            v = v.split('\n')
            print('%20s : '%(k) + v[0])
            for v1 in v[1:]:
                print('%20s   '%('') + v1)
        else:
            print('%20s : '%(k) + v)

def plot_env(env, surface_color='dodgerblue', bottom_color='peru', tx_color='orangered', rx_color='midnightblue', rx_plot=None, **kwargs):
    """Plots a visual representation of the environment.

    :param env: environment description
    :param surface_color: color of the surface (see `Bokeh colors <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`_)
    :param bottom_color: color of the bottom (see `Bokeh colors <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`_)
    :param tx_color: color of transmitters (see `Bokeh colors <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`_)
    :param rx_color: color of receviers (see `Bokeh colors <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`_)
    :param rx_plot: True to plot all receivers, False to not plot any receivers, None to automatically decide

    Other keyword arguments applicable for `arlpy.plot.plot()` are also supported.

    The surface, bottom, transmitters (marker: '*') and receivers (marker: 'o')
    are plotted in the environment. If `rx_plot` is set to None and there are
    more than 2000 receivers, they are not plotted.

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d(depth=[[0, 40], [100, 30], [500, 35], [700, 20], [1000,45]])
    >>> pm.plot_env(env)
    """
    env = check_env2d(env)
    min_x = 0
    max_x = _np.max(env['rx_range'])
    if max_x-min_x > 10000:
        divisor = 1000
        min_x /= divisor
        max_x /= divisor
        xlabel = 'Range (km)'
    else:
        divisor = 1
        xlabel = 'Range (m)'
    if env['surface'] is None:
        min_y = 0
    else:
        min_y = _np.min(env['surface'][:,1])
    if _np.size(env['depth']) > 1:
        max_y = _np.max(env['depth'][:,1])
    else:
        max_y = env['depth']
    mgn_x = 0.01*(max_x-min_x)
    mgn_y = 0.1*(max_y-min_y)
    """
    oh = _plt.hold()
    if env['surface'] is None:
        _plt.plot([min_x, max_x], [0, 0], xlabel=xlabel, ylabel='Depth (m)', xlim=(min_x-mgn_x, max_x+mgn_x), ylim=(-max_y-mgn_y, -min_y+mgn_y), color=surface_color, **kwargs)
    else:
        # linear and curvilinear options use the same altimetry, just with different normals
        s = env['surface']
        _plt.plot(s[:,0]/divisor, -s[:,1], xlabel=xlabel, ylabel='Depth (m)', xlim=(min_x-mgn_x, max_x+mgn_x), ylim=(-max_y-mgn_y, -min_y+mgn_y), color=surface_color, **kwargs)
    if _np.size(env['depth']) == 1:
        _plt.plot([min_x, max_x], [-env['depth'], -env['depth']], color=bottom_color)
    else:
        # linear and curvilinear options use the same bathymetry, just with different normals
        s = env['depth']
        _plt.plot(s[:,0]/divisor, -s[:,1], color=bottom_color)
    txd = env['tx_depth']
    _plt.plot([0]*_np.size(txd), -txd, marker='*', style=None, color=tx_color)
    if rx_plot is None:
        rx_plot = _np.size(env['rx_depth'])*_np.size(env['rx_range']) < 2000
    if rx_plot:
        rxr = env['rx_range']
        if _np.size(rxr) == 1:
            rxr = [rxr]
        for r in _np.array(rxr):
            rxd = env['rx_depth']
            _plt.plot([r/divisor]*_np.size(rxd), -rxd, marker='o', style=None, color=rx_color)
    _plt.hold(oh)
    """
    
def plot_ssp(env, **kwargs):
    """Plots the sound speed profile.

    :param env: environment description

    Other keyword arguments applicable for `arlpy.plot.plot()` are also supported.

    If the sound speed profile is range-dependent, this function only plots the first profile.

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d(soundspeed=[[ 0, 1540], [10, 1530], [20, 1532], [25, 1533], [30, 1535]])
    >>> pm.plot_ssp(env)
    """
    env = check_env2d(env)
    svp = env['soundspeed']
    if isinstance(svp, _pd.DataFrame):
        svp = _np.hstack((_np.array([svp.index]).T, _np.asarray(svp)))
    if _np.size(svp) == 1:
        if _np.size(env['depth']) > 1:
            max_y = _np.max(env['depth'][:,1])
        else:
            max_y = env['depth']
        """
        _plt.plot([svp, svp], [0, -max_y], xlabel='Soundspeed (m/s)', ylabel='Depth (m)', **kwargs)
        """
    elif env['soundspeed_interp'] == spline:
        s = svp
        ynew = _np.linspace(_np.min(svp[:,0]), _np.max(svp[:,0]), 100)
        tck = _interp.splrep(svp[:,0], svp[:,1], s=0)
        xnew = _interp.splev(ynew, tck, der=0)
        """
        _plt.plot(xnew, -ynew, xlabel='Soundspeed (m/s)', ylabel='Depth (m)', hold=True, **kwargs)
        _plt.plot(svp[:,1], -svp[:,0], marker='.', style=None, **kwargs)
        """
    else:
        """
        _plt.plot(svp[:,1], -svp[:,0], xlabel='Soundspeed (m/s)', ylabel='Depth (m)', **kwargs)
        """
        pass

def compute_arrivals(env, model=None, debug=False):
    """Compute arrivals between each transmitter and receiver.

    :param env: environment definition
    :param model: propagation model to use (None to auto-select)
    :param debug: generate debug information for propagation model
    :returns: arrival times and coefficients for all transmitter-receiver combinations

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> arrivals = pm.compute_arrivals(env)
    >>> pm.plot_arrivals(arrivals)
    """
    env = check_env2d(env)
    (model_name, model) = _select_model(env, arrivals, model)
    if debug:
        print('[DEBUG] Model: '+model_name)
    return model.run(env, arrivals, debug)

def compute_eigenrays(env, tx_depth_ndx=0, rx_depth_ndx=0, rx_range_ndx=0, model=None, debug=False):
    """Compute eigenrays between a given transmitter and receiver.

    :param env: environment definition
    :param tx_depth_ndx: transmitter depth index
    :param rx_depth_ndx: receiver depth index
    :param rx_range_ndx: receiver range index
    :param model: propagation model to use (None to auto-select)
    :param debug: generate debug information for propagation model
    :returns: eigenrays paths

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> rays = pm.compute_eigenrays(env)
    >>> pm.plot_rays(rays, width=1000)
    """
    env = check_env2d(env)
    env = env.copy()
    if _np.size(env['tx_depth']) > 1:
        env['tx_depth'] = env['tx_depth'][tx_depth_ndx]
    if _np.size(env['rx_depth']) > 1:
        env['rx_depth'] = env['rx_depth'][rx_depth_ndx]
    if _np.size(env['rx_range']) > 1:
        env['rx_range'] = env['rx_range'][rx_range_ndx]
    (model_name, model) = _select_model(env, eigenrays, model)
    if debug:
        print('[DEBUG] Model: '+model_name)
    return model.run(env, eigenrays, debug)

def compute_rays(env, tx_depth_ndx=0, model=None, debug=False):
    """Compute rays from a given transmitter.

    :param env: environment definition
    :param tx_depth_ndx: transmitter depth index
    :param model: propagation model to use (None to auto-select)
    :param debug: generate debug information for propagation model
    :returns: ray paths

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> rays = pm.compute_rays(env)
    >>> pm.plot_rays(rays, width=1000)
    """
    env = check_env2d(env)
    if _np.size(env['tx_depth']) > 1:
        env = env.copy()
        env['tx_depth'] = env['tx_depth'][tx_depth_ndx]
    (model_name, model) = _select_model(env, rays, model)
    if debug:
        print('[DEBUG] Model: '+model_name)
    return model.run(env, rays, debug)

def compute_transmission_loss(env, tx_depth_ndx=0, mode=coherent, model=None, debug=False):
    """Compute transmission loss from a given transmitter to all receviers.

    :param env: environment definition
    :param tx_depth_ndx: transmitter depth index
    :param mode: coherent, incoherent or semicoherent
    :param model: propagation model to use (None to auto-select)
    :param debug: generate debug information for propagation model
    :returns: complex transmission loss at each receiver depth and range

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> tloss = pm.compute_transmission_loss(env, mode=pm.incoherent)
    >>> pm.plot_transmission_loss(tloss, width=1000)
    """    
    
    env = check_env2d(env)
    
    if env['model'] == 'BELLHOP':
        if mode not in [coherent, incoherent, semicoherent]:
            print('[WARN] BELLHOP: Unknown transmission loss mode: '+mode+', using coherent as default.')
            mode = coherent
    if env['model'] == 'RAM':
        mode = TL
    if env['model'] == 'KRAKEN':
        mode = TL
    
    if _np.size(env['tx_depth']) > 1:
        env = env.copy()
        env['tx_depth'] = env['tx_depth'][tx_depth_ndx]

    (model_name, model_process) = _select_model(env, mode, env['model'])
    if debug:
        print('[DEBUG] Model: '+model_name)
        
    return model_process.run(env, mode, debug)
        
def compute_modes(env, debug=False):
    """
    Compute modes.

    :param env: environment definition
    :param debug: generate debug information for propagation model
    :returns: modes

    """
    env = check_env2d(env)

    if env['model'] == 'KRAKEN':
        mode = modes
        
    (model_name, model_process) = _select_model(env, mode, env['model'])
    if debug:
        print('[DEBUG] Model: '+model_name)
 
    return model_process.run(env, mode, debug)

def arrivals_to_impulse_response(arrivals, fs, abs_time=False):
    """Convert arrival times and coefficients to an impulse response.

    :param arrivals: arrivals times (s) and coefficients
    :param fs: sampling rate (Hz)
    :param abs_time: absolute time (True) or relative time (False)
    :returns: impulse response

    If `abs_time` is set to True, the impulse response is placed such that
    the zero time corresponds to the time of transmission of signal.

    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d()
    >>> arrivals = pm.compute_arrivals(env)
    >>> ir = pm.arrivals_to_impulse_response(arrivals, fs=192000)
    """
    t0 = 0 if abs_time else min(arrivals.time_of_arrival)
    irlen = int(_np.ceil((max(arrivals.time_of_arrival)-t0)*fs))+1
    ir = _np.zeros(irlen, dtype=_np.complex128)
    for _, row in arrivals.iterrows():
        ndx = int(_np.round((row.time_of_arrival.real-t0)*fs))
        ir[ndx] = row.arrival_amplitude
    return ir

def plot_ir(ir, env, Title='', fs=96000, dB=False, color='steelblue', **kwargs):
    """
    Plots the impulse response.

    Parameters:
        ir (array): Impulse response.
        env (dict): Environment definition.
        Title (str): Title for the plot.
        fs (int): Sampling frequency in samples per second (default is 96000).
        dB (bool): True to plot in dB, False for linear scale.
        color (str): Line color (see `Matplotlib colors <https://matplotlib.org/stable/gallery/color/named_colors.html>`_).
        **kwargs: Other keyword arguments applicable for `matplotlib.pyplot.stem()`.

    Returns:
        fig, ax: Figure and axis objects for the plot.
    """

    # Create the x-axis values based on the length of the impulse response
    x = _np.arange(len(ir))
    
    # If dB is True, convert the impulse response to dB
    if dB:
        ir = 20 * _np.log10(_np.abs(ir) + _np.finfo(float).eps)

    # Plot the impulse response using stem plot
    fig, ax = plt.subplots()
    ax.stem(x, ir, linefmt=color, markerfmt=color, basefmt='k', **kwargs)
    ax.set_xlabel('Sample [S]')
    ax.set_ylabel('Amplitude')
    ax.set_title(f"[ {env['model']} - Impulse response @ {fs} S/s ] {Title}")
    ax.grid('all')
    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_arrivals(arrivals, env, Title='', dB=False, color='steelblue', **kwargs):
    """
    Plots the arrival times and amplitudes.

    Parameters:
        arrivals (DataFrame): Arrival times (s) and coefficients.
        env (dict): Environment definition.
        Title (str): Title for the plot.
        dB (bool): True to plot in dB, False for linear scale.
        color (str): Line color (see `Matplotlib colors <https://matplotlib.org/stable/gallery/color/named_colors.html>`_).
        **kwargs: Other keyword arguments applicable for `matplotlib.pyplot.plot()`.

    Returns:
        fig, ax: Figure and axis objects for the plot.
    """

    t0 = min(arrivals.time_of_arrival)
    t1 = max(arrivals.time_of_arrival)

    fig, ax = plt.subplots()

    if dB:
        min_y = 20 * _np.log10(_np.max(_np.abs(arrivals.arrival_amplitude))) - 60
        ylabel = 'Amplitude [dB]'
    else:
        ylabel = 'Amplitude'
        ax.plot([t0, t1], [0, 0], 'r')
        min_y = 0

    for _, row in arrivals.iterrows():
        t = row.time_of_arrival.real
        y = _np.abs(row.arrival_amplitude)

        if dB:
            y = max(20 * _np.log10(_np.finfo(float).eps + y), min_y)

        ax.stem(t, y, linefmt=color, markerfmt=color, basefmt='k')

    ax.set_ylabel(ylabel)
    ax.set_title(f"[ {env['model']} - Arrivals ] {Title}")
    ax.set_xlabel('Arrival time [s]')
    ax.grid('all')
    plt.tight_layout()
    plt.show()

    return fig, ax

    
def plot_rays(rays, env, Title='', invert_colors=False, **kwargs):
    """
    Plots ray paths.

    Parameters:
        rays (DataFrame): Ray paths.
        env (dict): Environment definition.
        Title (str): Title for the plot.
        invert_colors (bool): False to use black for high-intensity rays, True to use white.

    Returns:
        fig, ax: Figure and axis objects for the plot.
    """

    # Sorting rays by bottom bounces in descending order
    rays = rays.sort_values('bottom_bounces', ascending=False)

    # Determine the maximum amplitude of bottom bounces
    max_amp = _np.max(_np.abs(rays.bottom_bounces)) if len(rays.bottom_bounces) > 0 else 0

    if max_amp <= 0:
        max_amp = 1

    divisor = 1
    xlabel = 'Range [m]'
    r = []

    # Flatten ray coordinates for determining the range
    for _, row in rays.iterrows():
        r += list(row.ray[:, 0])

    if max(r) - min(r) > 10000:
        divisor = 1000
        xlabel = 'Range [km]'

    fig, ax = plt.subplots()

    for _, row in rays.iterrows():
        num_bnc = row.bottom_bounces + row.surface_bounces
        if row.bottom_bounces == 0 and row.surface_bounces == 0:
            ax.plot(row.ray[:, 0] / divisor, row.ray[:, 1], color='k', alpha=.5)
        elif num_bnc > 1:
            ax.plot(row.ray[:, 0] / divisor, row.ray[:, 1], color='r', alpha=.5)
        elif row.surface_bounces == 1:
            ax.plot(row.ray[:, 0] / divisor, row.ray[:, 1], color='b', alpha=.5)
        elif row.bottom_bounces == 1:
            ax.plot(row.ray[:, 0] / divisor, row.ray[:, 1], color='saddlebrown', alpha=.5)

    ax.plot(env['depth'][:, 0] / divisor, env['depth'][:, 1], 'saddlebrown', linewidth=3)
    ax.plot(env['surface'][:, 0] / divisor, env['surface'][:, 1], 'b', linewidth=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Depth [m]")
    ax.set_ylim((_np.min(env['surface']), _np.max(env['depth'][:, 1])))
    ax.set_xlim((0, env['rx_range'] / divisor))
    ax.set_title(f"[ {env['model']} - Rays ] {Title}")
    ax.scatter(0, env['tx_depth'], label="Source", color="k", s=250, marker="*")
    ax.scatter(env['rx_range'], env['rx_depth'], label="Receiver", color="k", s=250, marker="o")
    ax.invert_yaxis()
    ax.grid('all')
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def plot_transmission_loss(tloss, env, Title='', vmin=-180, vmax=0, **kwargs):
    """
    Plots transmission loss.

    Parameters:
        tloss (array): Complex transmission loss.
        env (dict): Environment definition.

    Returns:
        fig, ax: Figure and axis objects for the plot.
    """

    fig, ax = plt.subplots()
    X = env['rx_range']
    Y = env['rx_depth']

    tlossplt = 20 * _np.log10(_np.finfo(float).eps + _np.abs(_np.array(tloss)))

    # Remove TL in sediment/surface
    for ii, x in enumerate(X):
        ylim = _np.interp(x, env['depth'][:, 0], env['depth'][:, 1])
        for jj, y in enumerate(Y):
            if y > ylim:
                tlossplt[jj, ii] = vmax

    if env['model'] == 'BELLHOP' and env['surface'] is not None:
        for ii, x in enumerate(X):
            ylim = _np.interp(x, env['surface'][:, 0], env['surface'][:, 1])
            for jj, y in enumerate(Y):
                if y < ylim:
                    tlossplt[jj, ii] = _np.NaN
        ax.plot(env['surface'][:, 0] / 1000, env['surface'][:, 1], 'b', linewidth=3)

    elif env['model'] == 'RAM':
        ax.plot([0, X[-1] / 1000], [0, 0], 'b', linewidth=3)

    X, Y = _np.meshgrid(X, Y)
    im1 = ax.pcolormesh(X / 1000, Y, tlossplt, cmap='jet', shading='gouraud', vmin=vmin, vmax=vmax)
    ax.plot(env['depth'][:, 0] / 1000, env['depth'][:, 1], 'k', linewidth=3)
    ax.set_xlim((X[0, 0] / 1000, X[-1, -1] / 1000))
    ax.set_ylim((Y[0, 0], Y[-1, -1]))
    ax.set_xlabel('Range [km]')
    ax.set_ylabel('Depth [m]')
    ax.set_title(f"[ {env['model']} - Propagation Loss @ {env['frequency']} Hz ] {Title}")
    cbar1 = fig.colorbar(im1, ax=ax)
    cbar1.ax.set_ylabel('Loss [dB]')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return fig, ax
    
def plot_absorption(env, Title='', vmin=0, vmax=20, Nxy=500, **kwargs):
    """
    Plots the absorption profile in the environment.

    Parameters:
        env (dict): Environmental parameters.
        Title (str): Title for the plot.
        vmin (float): Minimum value for the absorption color map.
        vmax (float): Maximum value for the absorption color map.
        Nxy (int): Number of points in the x and y directions.
        **kwargs: Additional keyword arguments for customization.

    Returns:
        fig, ax: Figure and axis objects for the plot.
    """

    fig, ax = plt.subplots()

    Xb = _np.array(env['bottom_srange'])
    Yb = _np.array(env['bottom_sdepth'] - _np.max(env['depth'][:, 1]), ndmin=1)
    Zb = _np.array(_np.array(env['bottom_absorption'], ndmin=2))

    if env['model'] == 'BELLHOP':
        Xb = [0, env['rx_range'][-1]]
        Zb = _np.full((len(Yb), 2), _np.mean(Zb))

    Xg = _np.linspace(0, env['rx_range'][-1], Nxy)
    Yg = _np.linspace(0, env['rx_depth'][-1], Nxy)
    Zg = _np.zeros([len(Yg), len(Xg)])

    # Bathy
    rb = _np.array(env['depth'][:, 0])
    zb = _np.array(env['depth'][:, 1])

    # Re-compute map over grid
    for ii, x in enumerate(Xg):  # For all map pixels
        for jj, y in enumerate(Yg):
            if y > _np.interp(x, rb, zb):  # If in sediment (interpolation of bathymetry line between samples)
                x_idx = _np.argmin(_np.abs(Xb - x))
                y_idx = _np.argmin(_np.abs(Yb - y))
                Zg[jj, ii] = Zb[y_idx, x_idx]
            else:  # Else it is in water column
                # Set minimum value
                Zg[jj, ii] = vmin

    # Plot surface if Bellhop
    if env['model'] == 'BELLHOP' and env['surface'] is not None:
        for ii, x in enumerate(Xg):  # For all map pixels
            ylim = _np.interp(x, env['surface'][:, 0], env['surface'][:, 1])
            Zg[Yg < ylim, ii] = _np.NaN
        ax.plot(env['surface'][:, 0]/1000, env['surface'][:, 1], 'b', linewidth=3)
    elif env['model'] == 'RAM':
        ax.plot([0, Xg[-1]/1000], [0, 0], 'b', linewidth=3)

    # Plot
    Xg, Yg = _np.meshgrid(Xg/1000, Yg)
    im = ax.pcolormesh(Xg, Yg, Zg, cmap='jet', shading='gouraud', vmin=vmin, vmax=vmax)
    ax.plot(rb/1000, zb, 'k', linewidth=3)
    ax.scatter(0, env['tx_depth'], label="Stars", color="r", s=500, marker="*")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Attenuation [dB/$\lambda$]')
    ax.set_xlabel('Range [km]')
    ax.set_ylabel('Depth [m]')
    ax.set_title(f"[ {env['model']} - Attenuation in sediment ] {Title}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    return fig, ax

def plot_beam(env, Title='', vmin=-60, vmax=20, **kwargs):
    """
    Plots the beam pattern of the source in the environment.

    Parameters:
        env (dict): Environmental parameters.
        Title (str): Title for the plot.
        vmin (float): Minimum value for the beam pattern color map.
        vmax (float): Maximum value for the beam pattern color map.
        **kwargs: Additional keyword arguments for customization.

    Returns:
        fig, ax: Figure and axis objects for the plot.
    """
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    if env['model'] == 'RAM' or env['tx_directionality'] is None:
        ax.plot(_np.linspace(0, 2*_np.pi, 1000), _np.zeros(1000))
    elif env['model'] == 'BELLHOP':
        ax.plot((env['tx_directionality'][:, 0]+180)/360*2*_np.pi-_np.pi, env['tx_directionality'][:, 1])
        
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    ax.set_ylim(vmin, vmax)
    ax.set_title(f"[ {env['model']} - Source directivity [dB] ] {Title}", va='bottom')
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def plot_density(env, Title, vmin=1000, vmax=1750, Nxy=500, **kwargs):
    """
    Plots the density profile of the sediment in the environment.

    Parameters:
        env (dict): Environmental parameters.
        Title (str): Title for the plot.
        vmin (float): Minimum value for the density color map.
        vmax (float): Maximum value for the density color map.
        Nxy (int): Number of points in the x and y directions.
        **kwargs: Additional keyword arguments for customization.

    Returns:
        fig, ax: Figure and axis objects for the plot.
    """
    
    fig, ax = plt.subplots()
    
    Xb = _np.array(env['bottom_srange'])
    Yb = _np.array(env['bottom_sdepth'] - _np.max(env['depth'][:, 1]), ndmin=1)
    Zb = _np.array(env['bottom_density'], ndmin=2)
    
    if env['model'] == 'BELLHOP':
        Xb = [0, env['rx_range'][-1]]
        Zb = _np.full((len(Yb), 2), _np.mean(Zb))
    
    Xg = _np.linspace(0, env['rx_range'][-1], Nxy)
    Yg = _np.linspace(0, env['rx_depth'][-1], Nxy)
    Zg = _np.zeros([len(Yg), len(Xg)])
    
    # Bathy
    rb = _np.array(env['depth'][:, 0])
    zb = _np.array(env['depth'][:, 1])
    
    # Re-compute map over grid
    for ii, x in enumerate(Xg):  # For all map pixels
        for jj, y in enumerate(Yg):
            if y > _np.interp(x, rb, zb):  # If in sediment (interpolation of bathymetry line between samples)
                x_idx = _np.argmin(_np.abs(Xb - x))
                y_idx = _np.argmin(_np.abs(Yb - y))
                Zg[jj, ii] = Zb[y_idx, x_idx]
            else:  # Else it is in water column
                # Set minimum value
                Zg[jj, ii] = vmin

    # Plot surface if Bellhop
    if env['model'] == 'BELLHOP' and env['surface'] is not None:
        for ii, x in enumerate(Xg):  # For all map pixels
            ylim = _np.interp(x, env['surface'][:, 0], env['surface'][:, 1])
            Zg[Yg < ylim, ii] = _np.nan
        ax.plot(env['surface'][:, 0]/1000, env['surface'][:, 1], 'b', linewidth=3)
    elif env['model'] == 'RAM':
        ax.plot([0, Xg[-1]/1000], [0, 0], 'b', linewidth=3)

    # Plot
    Xg, Yg = _np.meshgrid(Xg/1000, Yg)
    im = ax.pcolormesh(Xg, Yg, Zg, cmap='jet', shading='gouraud', vmin=vmin, vmax=vmax)
    ax.plot(rb/1000, zb, 'k', linewidth=3)
    ax.scatter(0, env['tx_depth'], label="Stars", color="r", s=500, marker="*")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Density [$g.m^{3}$]')
    ax.set_xlabel('Range [km]')
    ax.set_ylabel('Depth [m]')
    ax.set_title(f"[ {env['model']} - Density in sediment ] {Title}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    return fig, ax



def plot_soundspeed_map(env, Title='', Nxy=500, **kwargs):
    """
    Plots the sound speed profile of the environment.
    
    Parameters:
    env (dict): Environmental parameters.
    Title (str): Title for the plot.
    Nxy (int): Number of points in the x and y directions.
    **kwargs: Additional keyword arguments for customization.
    
    Returns:
    fig, ax: Figure and axis objects for the plot.
    """
    
    fig, ax = plt.subplots()
            
    X, Y, Z = _np.array(env['soundspeed_range']), _np.array(env['soundspeed_depth']), _np.array(env['soundspeed'])
    
    Xb, Yb, Zb = _np.array(env['bottom_srange']), _np.array(env['bottom_sdepth']-_np.max(env['depth'][:,1]),ndmin=1), _np.array(env['bottom_soundspeed'],ndmin=2)
    
    if env['model'] == 'BELLHOP':
        Xb, Zb = [0, env['rx_range'][-1]], _np.full((len(Yb), 2), _np.mean(Zb))
        X, Z = [0, env['rx_range'][-1]], _np.column_stack((_np.mean(Z, axis=1), _np.mean(Z, axis=1)))
    
    Xg = _np.linspace(0, env['rx_range'][-1], Nxy)
    Yg = _np.linspace(0, Y[-1], Nxy)
    Zg = _np.zeros([len(Yg), len(Xg)])
    
    # Bathy
    rb, zb = _np.array(env['depth'][:,0]), _np.array(env['depth'][:,1])
    
    # Re-compute map over grid
    for ii, x in enumerate(Xg):
        for jj, y in enumerate(Yg):
            if y > _np.interp(x, rb, zb):  # If in sediment (interpolation of bathymetry line between samples)
                y_idx = _np.argmin(_np.abs(Yb - y))
                x_idx = _np.argmin(_np.abs(Xb - x))
                Zg[jj, ii] = Zb[y_idx, x_idx]
            else:  # Else it is in water column
                y_idx = _np.argmin(_np.abs(Y - y))
                x_idx = _np.argmin(_np.abs(X - x))
                Zg[jj, ii] = Z[y_idx, x_idx]
     
    # Plot surface if Bellhop
    if env['model'] == 'BELLHOP' and env['surface'] is not None:
        for ii, x in enumerate(Xg):
            ylim = _np.interp(x, env['surface'][:,0], env['surface'][:,1])
            Zg[Yg < ylim, ii] = _np.nan
        ax.plot(env['surface'][:,0]/1000, env['surface'][:,1], 'b', linewidth=3)    
    elif env['model'] == 'RAM':
        ax.plot([0, Xg[-1]/1000], [0, 0], 'b', linewidth=3) 
        
    # Plot
    Xg, Yg = _np.meshgrid(Xg/1000, Yg)
    im = ax.pcolormesh(Xg, Yg, Zg, cmap='jet', shading='gouraud')
    ax.plot(rb/1000, zb, 'k', linewidth=3)
    ax.scatter(0, env['tx_depth'], label="Stars", color="r", s=500, marker="*") 
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Sound speed [m/s]')
    ax.set_xlabel('Range [km]')
    ax.set_ylabel('Depth [m]')
    ax.set_title(f"[ {env['model']} - Sound speed profile ] {Title}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return fig, ax
    
def models(env=None, task=None):
    """List available models.

    :param env: environment to model
    :param task: arrivals/eigenrays/rays/coherent/incoherent/semicoherent
    :returns: list of models that can be used

    >>> import arlpy.uwapm as pm
    >>> pm.models()
    ['bellhop']
    >>> env = pm.create_env2d()
    >>> pm.models(env, task=coherent)
    ['bellhop']
    """
    if env is not None:
        env = check_env2d(env)
    if (env is None and task is not None) or (env is not None and task is None):
        raise ValueError('env and task should be both specified together')
    rv = []
    for m in _models:
        if m[1]().supports(env, task):
            rv.append(m[0])
    return rv

def _select_model(env, task, model):
    if model is not None:
        for m in _models:
            if m[0] == model:
                return (m[0], m[1]())
        raise ValueError('Unknown model: '+model)
    for m in _models:
        mm = m[1]()
        if mm.supports(env, task):
            return (m[0], mm)
    raise ValueError('No suitable propagation model available')

### Bellhop propagation model ###

class _Bellhop:

    def __init__(self):
        pass

    def supports(self, env=None, task=None):
        if env is not None and env['type'] != '2D':
            return False
        fh, fname = _mkstemp(suffix='.env')
        _os.close(fh)
        fname_base = fname[:-4]
        self._unlink(fname_base+'.env')
        rv = self._bellhop(fname_base)
        self._unlink(fname_base+'.prt')
        self._unlink(fname_base+'.log')
        return rv

    def run(self, env, task, debug=False):
        taskmap = {
            arrivals:     ['A', self._load_arrivals],
            eigenrays:    ['E', self._load_rays],
            rays:         ['R', self._load_rays],
            coherent:     ['C', self._load_shd],
            incoherent:   ['I', self._load_shd],
            semicoherent: ['S', self._load_shd]
        }
        fname_base = self._create_env_file(env, taskmap[task][0])
        results = None
        if self._bellhop(fname_base):
            err = self._check_error(fname_base)
            if err is not None:
                print(err)
            else:
                try:
                    results = taskmap[task][1](fname_base)
                except FileNotFoundError:
                    print('[WARN] Bellhop did not generate expected output file')
        if debug:
            print('[DEBUG] Bellhop working files: '+fname_base+'.*.')
        else:
            self._unlink(fname_base+'.env')
            self._unlink(fname_base+'.bty')
            self._unlink(fname_base+'.ssp')
            self._unlink(fname_base+'.ati')
            self._unlink(fname_base+'.sbp')
            self._unlink(fname_base+'.prt')
            self._unlink(fname_base+'.log')
            self._unlink(fname_base+'.arr')
            self._unlink(fname_base+'.ray')
            self._unlink(fname_base+'.shd')
        return results

    def _bellhop(self, *args):
        try:
            _proc.run(f'bellhop.exe {" ".join(list(args))}', 
                      stderr=_proc.STDOUT, stdout=_proc.PIPE,
                      shell=True)
        except OSError:
            return False
        return True

    def _unlink(self, f):
        try:
            _os.unlink(f)
        except:
            pass

    def _print(self, fh, s, newline=True):
        _os.write(fh, (s+'\n' if newline else s).encode())

    def _print_array(self, fh, a):
        if _np.size(a) == 1:
            self._print(fh, "1")
            self._print(fh, "%0.6f /" % (a))
        else:
            self._print(fh, str(_np.size(a)))
            for j in a:
                self._print(fh, "%0.6f " % (j), newline=False)
            self._print(fh, "/")

    def _create_env_file(self, env, taskcode):
        fh, fname = _mkstemp(suffix='.env')
        fname_base = fname[:-4]
        self._print(fh, "'"+env['name']+"'")
        self._print(fh, "%0.6f" % (env['frequency']))
        self._print(fh, "1")
        if _np.size(env['soundspeed'],axis=1) > 1:
            print(f"[INFO] {env['model']}: Multiple sound profiles not supported, using average value.")
            mn = _np.mean(env['soundspeed'], axis=1)
            svp = _np.column_stack((env['soundspeed_depth'], mn))
        else:
            svp = _np.column_stack((env['soundspeed_depth'], env['soundspeed']))
        svp_depth = 0.0
        svp_interp = 'S' if env['soundspeed_interp'] == spline else 'C'
        if isinstance(svp, _pd.DataFrame):
            svp_depth = svp.index[-1]
            if len(svp.columns) > 1:
                svp_interp = 'Q'
            else:
                svp = _np.hstack((_np.array([svp.index]).T, _np.asarray(svp)))
        if env['surface'] is None:
            self._print(fh, "'%cVWT'" % svp_interp)
        else:
            self._print(fh, "'%cVWT*'" % svp_interp)
            self._create_bty_ati_file(fname_base+'.ati', env['surface'], env['surface_interp'])
        #max depth should be the depth of the acoustic domain, which can be deeper than the max depth bathymetry
        max_depth = env['depth'] if _np.size(env['depth']) == 1 else max(_np.max(env['depth'][:,1]), svp_depth)
        self._print(fh, "1 0.0 %0.6f" % (max_depth))
        if _np.size(svp) == 1:
            self._print(fh, "0.0 %0.6f /" % (svp))
            self._print(fh, "%0.6f %0.6f /" % (max_depth, svp))
        elif svp_interp == 'Q':
            for j in range(svp.shape[0]):
                self._print(fh, "%0.6f %0.6f /" % (svp.index[j], svp.iloc[j,0]))
            self._create_ssp_file(fname_base+'.ssp', svp)
        else:
            for j in range(svp.shape[0]):
                self._print(fh, "%0.6f %0.6f /" % (svp[j,0], svp[j,1]))
        depth = env['depth']
        if _np.size(depth) == 1:
            self._print(fh, "'A' %0.6f" % (env['bottom_roughness']))
        else:
            self._print(fh, "'A*' %0.6f" % (env['bottom_roughness']))
            self._create_bty_ati_file(fname_base+'.bty', depth, env['depth_interp'])  
        if env['bottom_soundspeed'].ndim > 0:
            print(f"[INFO] {env['model']}: Multiple bottom soundspeed profiles not supported, using average value.")
        if env['bottom_density'].ndim > 0:
            print(f"[INFO] {env['model']}: Multiple bottom density profiles not supported, using average value.")
        if env['bottom_absorption'].ndim > 0:
            print(f"[INFO] {env['model']}: Multiple bottom absorption profiles not supported, using average value.")
        self.bts = _np.mean(env['bottom_soundspeed'])
        self.btd = _np.mean(env['bottom_density'])
        self.bta = _np.mean(env['bottom_absorption'])
        self._print(fh, "%0.6f %0.6f 0.0 %0.6f %0.6f /" % (max_depth, self.bts, self.btd/1000, self.bta))
        self._print_array(fh, env['tx_depth'])
        self._print_array(fh, env['rx_depth'])
        self._print_array(fh, env['rx_range']/1000)
        if env['tx_directionality'] is None:
            self._print(fh, "'"+taskcode+"'")
        else:
            self._print(fh, "'"+taskcode+" *'")
            self._create_sbp_file(fname_base+'.sbp', env['tx_directionality'])
        self._print(fh, "%d" % (env['nbeams']))
        self._print(fh, "%0.6f %0.6f /" % (env['min_angle'], env['max_angle']))
        self._print(fh, "0.0 %0.6f %0.6f" % (1.01*max_depth, 1.01*_np.max(env['rx_range'])/1000))
        _os.close(fh)
        
        return fname_base

    def _create_bty_ati_file(self, filename, depth, interp):
        with open(filename, 'wt') as f:
            f.write("'%c'\n" % ('C' if interp == curvilinear else 'L'))
            f.write(str(depth.shape[0])+"\n")
            for j in range(depth.shape[0]):
                f.write("%0.6f %0.6f\n" % (depth[j,0]/1000, depth[j,1]))

    def _create_sbp_file(self, filename, dir):
        with open(filename, 'wt') as f:
            f.write(str(dir.shape[0])+"\n")
            for j in range(dir.shape[0]):
                f.write("%0.6f %0.6f\n" % (dir[j,0], dir[j,1]))

    def _create_ssp_file(self, filename, svp):
        with open(filename, 'wt') as f:
            f.write(str(svp.shape[1])+"\n")
            for j in range(svp.shape[1]):
                f.write("%0.6f%c" % (svp.columns[j]/1000, '\n' if j == svp.shape[1]-1 else ' '))
            for k in range(svp.shape[0]):
                for j in range(svp.shape[1]):
                    f.write("%0.6f%c" % (svp.iloc[k,j], '\n' if j == svp.shape[1]-1 else ' '))

    def _readf(self, f, types, dtype=str):
        if type(f) is str:
            p = _re.split(r' +', f.strip())
        else:
            p = _re.split(r' +', f.readline().strip())
        for j in range(len(p)):
            if len(types) > j:
                p[j] = types[j](p[j])
            else:
                p[j] = dtype(p[j])
        return tuple(p)

    def _check_error(self, fname_base):
        err = None
        try:
            with open(fname_base+'.prt', 'rt') as f:
                for lno, s in enumerate(f):
                    if err is not None and s != '\n':
                        err += '[ERROR] BELLHOP:' + s[:-1] + '.\n'
                    elif '*** FATAL ERROR ***' in s:
                        err = '[ERROR] BELLHOP:' + s
        except:
            pass
        return err

    def _load_arrivals(self, fname_base):
        with open(fname_base+'.arr', 'rt') as f:
            hdr = f.readline()
            if hdr.find('2D') >= 0:
                freq = self._readf(f, (float,))
                tx_depth_info = self._readf(f, (int,), float)
                tx_depth_count = tx_depth_info[0]
                tx_depth = tx_depth_info[1:]
                assert tx_depth_count == len(tx_depth)
                rx_depth_info = self._readf(f, (int,), float)
                rx_depth_count = rx_depth_info[0]
                rx_depth = rx_depth_info[1:]
                assert rx_depth_count == len(rx_depth)
                rx_range_info = self._readf(f, (int,), float)
                rx_range_count = rx_range_info[0]
                rx_range = rx_range_info[1:]
                assert rx_range_count == len(rx_range)
            else:
                freq, tx_depth_count, rx_depth_count, rx_range_count = self._readf(hdr, (float, int, int, int))
                tx_depth = self._readf(f, (float,)*tx_depth_count)
                rx_depth = self._readf(f, (float,)*rx_depth_count)
                rx_range = self._readf(f, (float,)*rx_range_count)
            arrivals = []
            for j in range(tx_depth_count):
                f.readline()
                for k in range(rx_depth_count):
                    for m in range(rx_range_count):
                        count = int(f.readline())
                        for n in range(count):
                            data = self._readf(f, (float, float, float, float, float, float, int, int))
                            arrivals.append(_pd.DataFrame({
                                'tx_depth_ndx': [j],
                                'rx_depth_ndx': [k],
                                'rx_range_ndx': [m],
                                'tx_depth': [tx_depth[j]],
                                'rx_depth': [rx_depth[k]],
                                'rx_range': [rx_range[m]],
                                'arrival_number': [n],
                                # 'arrival_amplitude': [data[0]*_np.exp(1j * data[1]* _np.pi/180)],
                                'arrival_amplitude': [data[0] * _np.exp( -1j * (_np.deg2rad(data[1]) + freq[0] * 2 * _np.pi * (data[3] * 1j +  data[2])))],
                                'time_of_arrival': [data[2]],
                                'complex_time_of_arrival': [data[2] + 1j*data[3]],
                                'angle_of_departure': [data[4]],
                                'angle_of_arrival': [data[5]],
                                'surface_bounces': [data[6]],
                                'bottom_bounces': [data[7]]
                            }, index=[len(arrivals)+1]))
        return _pd.concat(arrivals)

    def _load_rays(self, fname_base):
        with open(fname_base+'.ray', 'rt') as f:
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            rays = []
            while True:
                s = f.readline()
                if s is None or len(s.strip()) == 0:
                    break
                a = float(s)
                pts, sb, bb = self._readf(f, (int, int, int))
                ray = _np.empty((pts, 2))
                for k in range(pts):
                    ray[k,:] = self._readf(f, (float, float))
                rays.append(_pd.DataFrame({
                    'angle_of_departure': [a],
                    'surface_bounces': [sb],
                    'bottom_bounces': [bb],
                    'ray': [ray]
                }))
        return _pd.concat(rays)

    def _load_shd(self, fname_base):
        with open(fname_base+'.shd', 'rb') as f:
            recl, = _unpack('i', f.read(4))
            title = str(f.read(80))
            f.seek(4*recl, 0)
            ptype = f.read(10).decode('utf8').strip()
            assert ptype == 'rectilin', 'Invalid file format (expecting ptype == "rectilin")'
            f.seek(8*recl, 0)
            nfreq, ntheta, nsx, nsy, nsd, nrd, nrr, atten = _unpack('iiiiiiif', f.read(32))
            assert nfreq == 1, 'Invalid file format (expecting nfreq == 1)'
            assert ntheta == 1, 'Invalid file format (expecting ntheta == 1)'
            assert nsd == 1, 'Invalid file format (expecting nsd == 1)'
            f.seek(32*recl, 0)
            pos_r_depth = _unpack('f'*nrd, f.read(4*nrd))
            f.seek(36*recl, 0)
            pos_r_range = _unpack('f'*nrr, f.read(4*nrr))
            pressure = _np.zeros((nrd, nrr), dtype=_np.complex128)
            for ird in range(nrd):
                recnum = 10 + ird
                f.seek(recnum*4*recl, 0)
                temp = _np.array(_unpack('f'*2*nrr, f.read(2*nrr*4)))
                pressure[ird,:] = temp[::2] + 1j*temp[1::2]
        return _pd.DataFrame(pressure, index=pos_r_depth, columns=pos_r_range)
    
_models.append(('BELLHOP', _Bellhop))


class HS:
    def __init__(self, alphaR=_np.array([]), betaR=_np.array([]), rho=_np.array([]), alphaI=_np.array([]), betaI=_np.array([])):
        self.alphaR = _np.array(alphaR)
        self.betaR = _np.array(betaR)
        self.rho = _np.array(rho)
        self.alphaI = _np.array(alphaI)
        self.betaI = _np.array(betaI)
        
class BotBndry:
    def __init__(self, Opt, Hs, depth=[], betaI=[0]):
        self.Opt = Opt # 'A' for analytic or 'CVW' for interpolated ssp
        self.hs = Hs

class TopBndry:
    def __init__(self, Opt, depth=[]):
        self.Opt = Opt
        self.cp = None
        self.cs = None
        self.rho = None 

class Modes:
    def __init__(self, **kwargs):
        self.M = kwargs['M']
        self.k = kwargs['modes_k']
        self.z = _np.array(kwargs['z'])
        self.phi = kwargs['modes_phi']
        self.top = kwargs['top']
        self.bot = kwargs['bot']
        self.N = kwargs['N'] 
        self.Nfreq = kwargs['Nfreq']
        self.Nmedia = kwargs['Nmedia']
        self.depth = kwargs['depth']
        self.rho = kwargs['rho']
        self.freqvec = kwargs['freqVec']
        self.init_dict = kwargs
        self.num_modes = self.M # easier to remember

    def get_excited_modes(self, sd, threshold):
        '''
        return an array of modes that are excited by a source at sd meters depth
        threshold 
        
        also populate some structures in moded caled excited_phi and excited_k

        '''
        if sd not in self.z:
            raise ValueError("sd not in the depth array, are you sure it's the right depth you're passing?")
        depth_ind = [i for i in range(len(self.z)) if self.z[i] == sd][0]
        vals = self.phi[depth_ind,:]
        const = _np.max(abs(vals)) 
        filtered_inds = [i for i in range(len(self.k)) if abs(self.phi[depth_ind, i]) / const > threshold]
        self.excited_phi = self.phi[:, filtered_inds]
        self.excited_k = self.k[filtered_inds]
        return self.excited_phi, self.excited_k

    def get_source_depth_ind(self, sd):
        """
        sd is an int 
        """
        tol = 1e-2
        sind = [i for i in range(len(self.z)) if abs(self.z[i]-sd) < tol]
        if len(sind) == 0 :
            raise ValueError("sd not in the depth array, are you sure it's the right depth you're passing?")
        else:
            self.sind = sind[0]
        return  self.sind

    def remove_source_pos(self, sd):
        """
        Take the source at sd from the mode matrix
        Initiate a new attribute to hold the source
        modal value
        """
        sind = self.get_source_depth_ind(sd)
        new_pos_len = len(self.z) - 1
        new_phi = _np.zeros((new_pos_len, self.num_modes), dtype=self.phi.dtype)
        new_phi[:sind, :] = self.phi[:sind, :]
        new_phi[sind:,:] = self.phi[sind+1:,:]
        self.phi = new_phi
        new_z = _np.zeros((new_pos_len), dtype=self.z.dtype)
        new_z[:sind] = self.z[:sind]
        new_z[sind:] = self.z[sind+1:]
        self.z = new_z
        self.source_strength = self.phi[sind,:]
        return 
            
    def get_source_strength(self, sd):
        """
        Get the value of each mode at the source depth sd (meters)
        Initialize new attribute for the source strength
        """
        sind = self.get_source_depth_ind(sd)
        vals = self.phi[sind,:]
        self.source_strength = vals
        return  vals

    def get_receiver_modes(self, zr):
        """
        zr is array like 
        """
        tol = 1e-3
        r_inds = [i for i in range(len(self.z)) if _np.min(abs(self.z[i]-zr)) < tol]
        receiver_modes = self.phi[r_inds, :]
        self.receiver_modes = receiver_modes
        return receiver_modes

    def get_source_excitation(self, zs):
        """
        For case where there is a track, there may be multiple repeats in zs
        """
        tol = 1e-3
        r_inds = [_np.argmin(abs(zs[i] - self.z)) for i in range(len(zs))]
        strength_modes = self.phi[r_inds, :]
        self.strength_modes = strength_modes
        return strength_modes
        

    def plot(self):
        figs = []
        if self.M > 5:
            for i in range(self.M):
                fig = plt.figure(i)
                plt.plot(self.phi[:,i], -self.z)
                figs.append(fig)
        return figs

    def __repr__(self):
        return 'Modes object with ' + str(self.M) + ' distinct modes'

class _Kraken:

    def __init__(self):
        pass

    def supports(self, env=None, task=None):
        if env is not None and env['type'] != '2D':
            return False
        fh, fname = _mkstemp(suffix='.env')
        _os.close(fh)
        fname_base = fname[:-4]
        self._unlink(fname_base+'.env')
        rv = self._kraken(fname_base)
        self._unlink(fname_base+'.prt')
        self._unlink(fname_base+'.log')
        return rv

    def run(self, env, task, debug=False):
        
        taskmap = {
            TL:           ['C', self._load_shd],
            modes:        ['', self._load_modes]
        }
        fname_base = self._create_env_file(env, taskmap[task][0], debug=debug)
        results = None
        if self._kraken(fname_base):
            err = self._check_error(fname_base)
            if err is not None:
                print(err)
            else:
                if self._field(fname_base):
                    err = self._check_error(fname_base)
                    if err is not None:
                        print(err)
                try:
                    results = taskmap[task][1](fname_base)
                except FileNotFoundError:
                    print('[ERROR] KRAKEN: Fortran execution did not generate expected output file !')
                    self._unlink(fname_base+'.env')
                    self._unlink(fname_base+'.mod')
                    self._unlink(fname_base+'.shd')
        if debug:
            print('[DEBUG] KRAKEN: Working files: '+fname_base+'.*.')
        else:
            self._unlink(fname_base+'.env')
            self._unlink(fname_base+'.mod')
            self._unlink(fname_base+'.shd')

        return results

    def _kraken(self, *args):
        try:
            _proc.run(f'kraken.exe {" ".join(list(args))}', 
                      stderr=_proc.STDOUT, stdout=_proc.PIPE,
                      shell=True)
        except OSError:
            return False
        return True
    
    def _field(self, fname):
        try:
            _proc.run(f'field.exe {fname}', 
                      stderr=_proc.STDOUT, stdout=_proc.PIPE,
                      shell=True)
        except OSError:
            return False
        return True        

    def _unlink(self, f):
        try:
            _os.unlink(f)
        except:
            pass

    def _print(self, fh, s, newline=True):
        _os.write(fh, (s+'\n' if newline else s).encode())

    def _print_array(self, fh, a):
        if _np.size(a) == 1:
            self._print(fh, "1")
            self._print(fh, "%0.6f /" % (a))
        else:
            self._print(fh, str(_np.size(a)))
            for j in a:
                self._print(fh, "%0.6f " % (j), newline=False)
            self._print(fh, "/")

    def _create_env_file(self, env, taskcode, debug = 0, **kwargs):
        fh, fname = _mkstemp(suffix='.env')
        fname_base = fname[:-4]
        self._print(fh, "'"+env['name']+"'")
        self._print(fh, "%0.6f" % (env['frequency']))
        self._print(fh, "%d" % (env['nmedia']))
                
        if env['soundspeed_interp'] == spline:
            svp_interp = 'S'
        elif env['soundspeed_interp'] == c_linear:
            svp_interp = 'C' 
        elif env['soundspeed_interp'] == n2_linear:
            svp_interp = 'N'
        elif env['soundspeed_interp'] == analytic:
            svp_interp = 'A'
          
        # @todo     Add all boundary conditions.
        if env['topBdry'] == rigid:
            topBdry = 'R'
        elif env['topBdry'] == vacuum:
            topBdry = 'V' 
        else:
            print('[WARN] KRAKEN: Unknown top boundary condition, using vacuum instead !')
            topBdry = 'V'
            
        # @todo     Manage thorp attenuation option.  
        self._print(fh, "'%c%cWT'" % (svp_interp, topBdry))
        
        if _np.size(env['soundspeed'],axis=1) > 1:
            print(f"[INFO] {env['model']}: Multiple sound profiles not supported, using average value.")
            mn = _np.mean(env['soundspeed'], axis=1)
            svp = _np.column_stack((env['soundspeed_depth'], mn))
        else:
            svp = _np.column_stack((env['soundspeed_depth'], env['soundspeed']))
        svp_depth = 0.0
        
        #max depth should be the depth of the acoustic domain, which can be deeper than the max depth bathymetry
        if _np.size(env['depth']) != 1:
            print("[INFO] KRAKEN: Range dependent bathymetry not supported, using average value.")
            max_depth = max(_np.mean(env['depth'][:,1]), svp_depth)
        else:
            max_depth = env['depth']
            
        self._print(fh, "%d 0.0 %0.6f" % (len(env['rx_range']),max_depth))
        if _np.size(svp) == 1:
            self._print(fh, "0.0 %0.6f /" % (svp))
            self._print(fh, "%0.6f %0.6f /" % (max_depth, svp))
        else:
            for j in range(svp.shape[0]):
                self._print(fh, "%0.6f %0.6f /" % (svp[j,0], svp[j,1]))
        
        # @todo     Add all boundary conditions.
        if env['botBdry'] == rigid:
            botBdry = 'R'
        elif env['botBdry'] == vacuum:
            botBdry = 'V' 
        elif env['botBdry'] == analytic:
            botBdry = 'A' 
        else:
            print('[WARN] KRAKEN: Unknown bottom boundary condition, using perfectly rigid instead !')
            botBdry = 'R'

        self._print(fh, "'%c' %0.6f" % (botBdry, env['bottom_roughness']))    
  
            
        if env['bottom_soundspeed'].ndim > 0:
            print(f"[INFO] {env['model']}: Multiple bottom soundspeed profiles not supported, using average value.")
            self.bts = _np.mean(env['bottom_soundspeed'])
            
        if env['bottom_density'].ndim > 0:
            print(f"[INFO] {env['model']}: Multiple bottom density profiles not supported, using average value.")
            self.btd = _np.mean(env['bottom_density'])
            
        if env['bottom_absorption'].ndim > 0:
            print(f"[INFO] {env['model']}: Multiple bottom absorption profiles not supported, using average value.")
            self.bta = _np.mean(env['bottom_absorption'])
    
        # @todo    Not sure if this condition is required !
        if env['botBdry'] == analytic:
            self._print(fh, "%0.6f %0.6f 0.0 %0.6f %0.6f /" % (max_depth, self.bts, self.btd/1000, self.bta))
            
        self._print(fh, "%0.6f %0.6f" %(0, _np.max(env['soundspeed'])))                    # C0 min and max
        self._print(fh, "%0.6f" %(env['rx_range'][-1]/1000))                               # Max range in km 
        self._print(fh, "%d" %(1))                                                         # Number of sources depth
        self._print(fh, "%0.6f" %(env['tx_depth']))                                        # Source depths
        self._print(fh, "%d" %(len(env['rx_depth'])))                                      # Number of receiver depths
        self._print(fh, "%0.6f %0.6f /" %(env['rx_depth'][0], env['rx_depth'][-1]))        # Receiver depths
                
        _os.close(fh)
        if debug:
            print_file_content(fname_base+'.env')
            
        return fname_base

    # @todo     Remove useless file creation.

    def _create_bty_ati_file(self, filename, depth, interp):
        with open(filename, 'wt') as f:
            f.write("'%c'\n" % ('C' if interp == curvilinear else 'L'))
            f.write(str(depth.shape[0])+"\n")
            for j in range(depth.shape[0]):
                f.write("%0.6f %0.6f\n" % (depth[j,0]/1000, depth[j,1]))

    def _create_sbp_file(self, filename, dir):
        with open(filename, 'wt') as f:
            f.write(str(dir.shape[0])+"\n")
            for j in range(dir.shape[0]):
                f.write("%0.6f %0.6f\n" % (dir[j,0], dir[j,1]))

    def _create_ssp_file(self, filename, svp):
        with open(filename, 'wt') as f:
            f.write(str(svp.shape[1])+"\n")
            for j in range(svp.shape[1]):
                f.write("%0.6f%c" % (svp.columns[j]/1000, '\n' if j == svp.shape[1]-1 else ' '))
            for k in range(svp.shape[0]):
                for j in range(svp.shape[1]):
                    f.write("%0.6f%c" % (svp.iloc[k,j], '\n' if j == svp.shape[1]-1 else ' '))

    def _readf(self, f, types, dtype=str):
        if type(f) is str:
            p = _re.split(r' +', f.strip())
        else:
            p = _re.split(r' +', f.readline().strip())
        for j in range(len(p)):
            if len(types) > j:
                p[j] = types[j](p[j])
            else:
                p[j] = dtype(p[j])
        return tuple(p)

    def _check_error(self, fname_base):
        err = None
        try:
            with open(fname_base+'.prt', 'rt') as f:
                for lno, s in enumerate(f):
                    if err is not None and s != '\n':
                        err += '[ERROR] KRAKEN:' + s[:-1] + '.\n'
                    elif '*** FATAL ERROR ***' in s:
                        err = '[ERROR] KRAKEN:' + s
        except:
            pass
        return err

    def _load_modes(self, fname_base, debug = 0, freq=0, modes=[0,1,2,3,4,5], **kwargs):
        
        '''
         Read the modes produced by KRAKEN
         usage:
         keys are 'fname', 'freq', 'modes'
        'fname' and 'freq' are mandatory, 'modes' is if you only want a subset of modes
            [ Modes ] = read_modes_bin( filename, modes )
         filename is without the extension, which is assumed to be '.moA'
         freq is the frequency (involved for broadband runs)
           (you can use freq=0 if there is only one frequency)
         modes is an optional vector of mode indices

         derived from readKRAKEN.m    Feb 12, 1996 Aaron Thode

         Translated to python by Hunter Akins 2019

         Modes.M          number of modes
         Modes.k          wavenumbers
         Modes.z          sample depths for modes
         Modes.phi        modes

         Modes.Top.bc
         Modes.Top.cp
         Modes.Top.cs
         Modes.Top.rho
         Modes.Top.depth

         Modes.Bot.bc
         Modes.Bot.cp
         Modes.Bot.cs
         Modes.Bot.rho
         Modes.Bot.depth

         Modes.N          Number of depth points in each medium
         Modes.Mater      Material type of each medium (acoustic or elastic)
         Modes.Nfreq      Number of frequencies
         Modes.Nmedia     Number of media
         Modes.depth      depths of interfaces
         Modes.rho        densities in each medium
         Modes.freqVec    vector of frequencies for which the modes were calculated
        '''

        filename = fname_base+'.mod'

        if debug:
            if _os.path.exists(filename):
                print(f"[DEBUG] KRAKEN: The .mod file exists at {filename}.")
                
        if not _os.path.exists(filename):
            print(f"[ERROR] KRAKEN: The .mod file does not exist at {filename} !")
    
        if 'freq' in kwargs.keys():
            freq = kwargs['freq']
            
        if 'modes' in kwargs.keys():
            modes = kwargs['modes']
            
        with open(filename, 'rb') as f:
                        
            iRecProfile = 1;   # (first time only)
            
            lrecl     = 4*unpack('<I', f.read(4))[0];     #record length in bytes

            rec = iRecProfile - 1;

            f.seek(rec * lrecl + 4) # do I need to do this ?

            title    = unpack('80s', f.read(80))
            Nfreq  = unpack('<I', f.read(4))[0]
            Nmedia = unpack('<I', f.read(4))[0]
            Ntot = unpack('<l', f.read(4))[0]
            Nmat = unpack('<l', f.read(4))[0]
            N = []
            Mater = []


            if Ntot < 0:
                return

            # N and Mater
            rec   = iRecProfile;
            f.seek(rec * lrecl); # reposition to next level
            for Medium in range(Nmedia):
               N.append(unpack('<I', f.read(4))[0])
               Mater.append(unpack('8s', f.read(8))[0])


            # depth and density
            rec = iRecProfile + 1
            f.seek(rec * lrecl)
            bulk        = unpack('f'*2*Nmedia, f.read(4*2*Nmedia))
            depth = [bulk[i] for i in range(0,2*Nmedia,2)]
            rho = [bulk[i+1] for i in range(0,2*Nmedia,2)]

            # frequencies
            rec = iRecProfile + 2;
            f.seek(rec * lrecl);
            freqVec = unpack('d', f.read(8))[0]
            freqVec = _np.array(freqVec)

            # z
            rec = iRecProfile + 3
            f.seek(rec * lrecl)
            z = unpack('f'*Ntot, f.read(Ntot*4))

            # read in the modes

            # identify the index of the frequency closest to the user-specified value
            freqdiff = abs(freqVec - freq );
            freq_index = _np.argmin( freqdiff );

            # number of modes, m
            iRecProfile = iRecProfile + 4;
            rec = iRecProfile;

            # skip through the mode file to get to the chosen frequency
            for ifreq in range(freq_index+1):
                f.seek(rec * lrecl);
                M = unpack('l', f.read(8))[0]
           
               # advance to the next frequency
                if ( ifreq < freq_index ):
                    iRecProfile = iRecProfile + 2 + M + 1 + _np.floor( ( 2 * M - 1 ) / lrecl );   # advance to next profile
                    rec = iRecProfile;
                    f.seek(rec * lrecl)

            if 'modes' not in kwargs.keys():
                modes = _np.linspace(0, M-1, M, dtype=int);    # read all modes if the user didn't specify

            # Top and bottom halfspace info

            # Top
            rec = iRecProfile + 1
            f.seek(rec * lrecl)
            top_bc    = unpack('c', f.read(1))[0]
            cp        = unpack('ff', f.read(8))
            top_cp    = complex( cp[ 0 ], cp[ 1 ] )
            cs        = unpack('ff', f.read(8))
            top_cs    = complex( cs[ 1 ], cs[ 1 ] )
            top_rho   = unpack('f', f.read(4))[0]
            top_depth = unpack('f', f.read(4))[0]
        
            top_hs = HS(alphaR=top_cp.real, alphaI=top_cp.imag, betaR=top_cs.real, betaI=top_cs.imag)
            top = TopBndry(top_bc, depth=top_depth)  

            # Bottom
            bot_bc    = unpack('c', f.read(1))[0]
            cp        = unpack('ff', f.read(8))
            bot_cp    = complex( cp[ 0 ], cp[ 1 ] )
            cs        = unpack('ff', f.read(8))
            bot_cs    = complex( cs[ 1 ], cs[ 1 ] )
            bot_rho   = unpack('f', f.read(4))[0]
            bot_depth = unpack('f', f.read(4))[0]
      
            bot_hs = HS(alphaR=bot_cp.real, alphaI=bot_cp.imag, betaR=bot_cs.real, betaI=bot_cs.imag)
            bot = BotBndry(bot_bc, bot_hs, depth=bot_depth)  

            rec = iRecProfile
            f.seek(rec * lrecl)
            # if there are modes, read them
            if ( M == 0 ):
               modes_phi = []
               modes_k   = []
            else:
                modes_phi = _np.zeros((Nmat, len( modes )),dtype=_np.complex128)   # number of modes
               
                for ii in range(len(modes)):
                    rec = iRecProfile + 2 + int(modes[ ii ])
                    f.seek(rec * lrecl)
                    phi = unpack('f'*2*Nmat, f.read(2*Nmat*4)) # Data is read columwise
                    #t0 = time.time()
                    phi = _np.array(phi)
                    phir = phi[::2]
                    phii = phi[1::2]
                    #print('aray cast time', time.time()-t0)
                    #t0 = time.time()
                    #phir = np.array([phi[i] for i in range(0,2*Nmat,2)])
                    #phii = np.array([phi[i+1] for i in range(0,2*Nmat,2)])
                    #print('aray list time', time.time()-t0)
                    modes_phi[ :, ii ] = phir + complex(0, 1)*phii;
               
                rec = iRecProfile + 2 + M;
                f.seek(rec * lrecl)
                k    = unpack('f'*2*M, f.read(4*2*M))
                kr = _np.array([k[i] for i in range(0,2*M,2)])
                ki = _np.array([k[i+1] for i in range(0,2*M,2)])
                modes_k = kr+ complex(0,1) * ki
                modes_k = _np.array([modes_k[i] for i in modes], dtype=_np.complex128)  # take the subset that the user specified

        input_dict = {'M':M, 'modes_k': modes_k, 'z':z, 'modes_phi':modes_phi, 'top':top, 'bot':bot, 'N':N, 'Mater':Mater, 'Nfreq': Nfreq, 'Nmedia': Nmedia, 'depth':depth, 'rho':rho, 'freqVec':freqVec}
        modes = Modes(**input_dict)
        return modes

    def _load_shd(self, fname_base, debug = 0, **kwargs):
        
        if debug:
            if _os.path.exists(fname_base):
                print(f"[DEBUG] KRAKEN: The .shd file exists at {fname_base}.")
                
        if not _os.path.exists(fname_base):
            print(f"[ERROR] KRAKEN: The .shd file does not exist at {fname_base} !")
            
        with open(fname_base+'.shd', 'rb') as f:
            print("SHD file open !")
            recl, = _unpack('i', f.read(4))
            title = str(f.read(80))
            f.seek(4*recl, 0)
            ptype = f.read(10).decode('utf8').strip()
            assert ptype == 'rectilin', 'Invalid file format (expecting ptype == "rectilin")'
            f.seek(8*recl, 0)
            nfreq, ntheta, nsx, nsy, nsd, nrd, nrr, atten = _unpack('iiiiiiif', f.read(32))
            assert nfreq == 1, 'Invalid file format (expecting nfreq == 1)'
            assert ntheta == 1, 'Invalid file format (expecting ntheta == 1)'
            assert nsd == 1, 'Invalid file format (expecting nsd == 1)'
            f.seek(32*recl, 0)
            pos_r_depth = _unpack('f'*nrd, f.read(4*nrd))
            f.seek(36*recl, 0)
            pos_r_range = _unpack('f'*nrr, f.read(4*nrr))
            pressure = _np.zeros((nrd, nrr), dtype=_np.complex128)
            for ird in range(nrd):
                recnum = 10 + ird
                f.seek(recnum*4*recl, 0)
                temp = _np.array(_unpack('f'*2*nrr, f.read(2*nrr*4)))
                pressure[ird,:] = temp[::2] + 1j*temp[1::2]
        return _pd.DataFrame(pressure, index=pos_r_depth, columns=pos_r_range)
    
_models.append(('KRAKEN', _Kraken))


        
class _RAM:
    """
    A class representing the RAM model for underwater acoustic propagation.

    Attributes:
        pyram: RAM model instance.
        tl_tol (float): Tolerable mean difference in TL (dB) with reference result.
        lines (numpy.ndarray): Array representing depths for computation.
        columns (numpy.ndarray): Array representing ranges for computation.

    Methods:
        __init__(): Initializes the RAM model instance.
        supports(env=None, task='TL'): Checks if the RAM model supports a specific task for the given environment.
        run(env, task='TL', debug=False): Runs the RAM computation for a specified task and environment.
    """
    # @todo     Understand and manage CP task
    
    def __init__(self):
        pass

    def supports(self, env=None, task=TL):
        """
        Checks if the RAM model supports the specified task for the given environment.
        
        Parameters:
        env (dict): Environmental parameters.
        task (str): Task identifier ('TL' for Transmission Loss, 'CP' for Cepstral Peak).
        
        Returns:
        bool: True if the task is supported, False otherwise.
        """
        if task == TL or task == 'CP':
            return True    
        else:
            print("[ERROR] KRAKEN: Unknown mode: {task} !")
            
        return False
    
    def run(self, env, task=TL, debug=False):
        """
        Runs the RAM computation for the specified task and environment.
        
        Parameters:
            env (dict): Environmental parameters.
            task (str): Task identifier ('TL' for Transmission Loss, 'CP' for Cepstral Peak).
            debug (bool): Debugging flag.
        
        Returns:
            pd.DataFrame or bool: Resulting data as a DataFrame if successful, False otherwise.
        """
        if env['surface'] is not None:
            print(f"[INFO] {env['model']}: Surface not supported, considering flat air/water interface.")
        
        if env['tx_directionality'] is not None:
            print(f"[INFO] {env['model']}: Beam pattern not supported, using omnidirectionnal instead.")


        # Initialize RAM environment
        # ram.PyRAM(freq, zs, zr, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb, **kwargs)       
        self.pyram = ram.PyRAM(env['frequency'],
                           env['tx_depth'],
                           env['rx_depth'][-1],
                           _np.array(env['soundspeed_depth']),
                           _np.array(env['soundspeed_range']),
                           _np.array(env['soundspeed']),
                           _np.array(env['bottom_sdepth']-_np.max(env['depth'][:,1]),ndmin=1),
                           _np.array(env['bottom_srange']),
                           _np.array(env['bottom_soundspeed'],ndmin=2),
                           _np.array(env['bottom_density'],ndmin=2)/1000,
                           _np.array(env['bottom_absorption'],ndmin=2),
                           _np.array((env['depth'][:,0],env['depth'][:,1])).transpose(),
                           rmax = env['rx_range'][-1],
                           dr=env['rx_range'][2]-env['rx_range'][1],
                           dz=env['rx_depth'][2]-env['rx_depth'][1],
                           zmplt=env['rx_depth'][-1],
                           c0=_np.mean(env['soundspeed'][:,1])
                           )

        self.tl_tol  = 1e-2  # Tolerable mean difference in TL (dB) with reference result
        self.lines   = env['rx_depth']
        self.columns = env['rx_range']
        
        # Run computation
        results = self.pyram.run()
        
        if task == TL:
            # If necessary resize the results grid by adding NaNs
            if _np.size(results['TL Grid'], axis=0) < len(self.lines):
                results['TL Grid'] = _np.insert(results['TL Grid'], 0, _np.zeros(_np.size(results['TL Grid'],axis=1)), axis=0)
                results['TL Grid'][0,:] = _np.nan
            if _np.size(results['TL Grid'], axis=1) < len(self.columns):
                results['TL Grid'] = _np.insert(results['TL Grid'], 0, _np.zeros(_np.size(results['TL Grid'],axis=0)), axis=1) 
                results['TL Grid'][:,0] = _np.nan
            if _np.size(results['TL Grid'], axis=1) < len(self.columns):
                results['TL Grid'] = _np.insert(results['TL Grid'], _np.size(results['TL Grid'],axis=1), _np.zeros(_np.size(results['TL Grid'],axis=0)), axis=1)
                results['TL Grid'][:,-1] = _np.nan
            
            # Return data as DataFrame with range and depth as index
            return _pd.DataFrame(10**(-results['TL Grid']/20), index=self.lines, columns=self.columns)
        
        if task == 'CP':
            # If necessary resize the results grid by adding NaNs
            if _np.size(results['CP Grid'], axis=0) < len(self.lines):
                results['CP Grid'] = _np.insert(results['CP Grid'], 0, _np.zeros(_np.size(results['CP Grid'],axis=1)), axis=0)
                results['CP Grid'][0,:] = _np.nan
            if _np.size(results['CP Grid'], axis=1) < len(self.columns):
                results['CP Grid'] = _np.insert(results['CP Grid'], 0, _np.zeros(_np.size(results['CP Grid'],axis=0)), axis=1) 
                results['CP Grid'][:,0] = _np.nan
            if _np.size(results['CP Grid'], axis=1) < len(self.columns):
                results['CP Grid'] = _np.insert(results['CP Grid'], _np.size(results['CP Grid'],axis=1), _np.zeros(_np.size(results['CP Grid'],axis=0)), axis=1)
                results['CP Grid'][:,-1] = _np.nan
            
            # Return data as DataFrame with range and depth as index
            return _pd.DataFrame(results['CP Grid'], index=self.lines, columns=self.columns)    
        
        return False

# Add model to available models
_models.append(('RAM', _RAM))


def plot_recwPSD(Fxx, Pxx, maxval=2**24-1, vpk=3, sh=-205, gain=0, Title='', **kwargs):
    """
    Plot Welch Power Spectral Density (PSD) of a recorded signal.

    Parameters:
    - Fxx: Frequency vector of the Welch periodogram.
    - Pxx: Welch periodogram computed with the recorded signal.
    - maxval: Recorded signal maximum limit value (e.g., 2**24-1 for a 24-bit signal). Default is 2**24-1.
    - vpk: Electrical voltage corresponding to maxval. Default is 3.
    - sh: Hydrophone sensitivity in dB re V/uPa. Default is -205.
    - gain: Preamplification gain in dB. Default is 0.
    - Title: Title for the plot. Default is an empty string.
    - **kwargs: Additional keyword arguments to pass to the plot.

    Returns:
    - fig: Matplotlib figure.
    - ax: Matplotlib axis.

    """
    
    fig, ax = plt.subplots()
    ax.semilogx(Fxx, 10*_np.log10(Pxx)+20*_np.log10(vpk/maxval)-sh-gain, **kwargs)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Level [dB re 1$\mu Pa / \sqrt{Hz}$]')
    ax.set_title(f"[ WELCH - Power Spectral Density ] {Title}")
    ax.grid(True)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def plot_PSD(Fxx, Lvl_dB, Title='', **kwargs):
    """
    Plot Power Spectral Density (PSD).

    Parameters:
    - Fxx: Frequency vector of the PSD.
    - Lvl_dB: PSD expressed in dB re 1uPa/vHz.
    - Title: Title for the plot. Default is an empty string.
    - **kwargs: Additional keyword arguments to pass to the plot.

    Returns:
    - fig: Matplotlib figure.
    - ax: Matplotlib axis.

    Code adapted from the original version by [Original Authors].
    """
    
    fig, ax = plt.subplots()
    ax.semilogx(Fxx, Lvl_dB, **kwargs)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Level [dB re 1$\mu Pa / \sqrt{Hz}$]')
    ax.set_title(f"[ Power Spectral Density ] {Title}")
    ax.grid(True)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return fig, ax
    


def compute_windnoise(f, u, water_depth='deep', sumOnly=False):
    """
    Calculates wind-based noise (in dB re uPa) with adjustment for shallow water based on Piggot (1964).
    Adapted from IDL code originally by Dan Hutt, rewritten by Vic Young,
    and obtained through Sean Pecknold.
    Any mistakes propagated could have been theirs.

    Parameters:
        f (_np.ndarray or float): Single frequency or vector of frequencies (Hz). Assumes a 1-Hz band for a single frequency.
        u (float): Wind speed (knots).
        water_depth (str): 'shallow' or 'deep' (default: 'deep').

    Optional Parameters:
        sumOnly (bool): If True, the noise is summed across frequency bands. Default is False.
                       In this case, the calculation is still valid for non-constant bandwidth -
                       band limits are assumed to be halfway between input frequencies.

    Returns:
        NL (_np.ndarray): Vector containing the noise (dB per muPa^2/Hz) at frequencies f.
                         If sumOnly is True, the output is the wind noise summed across the band.

    Code translated from "A simple yet practical ambient noise model"
    by Cristina D. S. Tollefsen and Sean Pecknold.
    """
    
    # This all breaks down if u == 0 so account for that
    if u == 0:
        NL = _np.zeros_like(f)
    else:
        n2 = f.size
        f = _np.array(f).flatten()  # Make sure f is a 1D array
        if sumOnly:
            f2 = _np.concatenate(([0], f, 2 * f[-1] - f[-2]))
            df = (f2[2:] - f2[:-2]) / 2
        else:
            df = _np.ones_like(f)

        # Bookkeeping:
        # Some constants
        f_wind = 2000  # Cutoff for wind noise section
        s1w    = 1.5   # Constant in wind calcs
        s2w    = -5.0  # Constant in wind calc
        a      = -25   # Curve melding exponent
        slope  = s2w * (0.1 / _np.log10(2))  # Slope at high freq
        NL     = _np.zeros_like(f)

        # Do the wind part for f <= 2000 Hz
        if water_depth == 'shallow':
            cst = 45
        elif water_depth == 'deep':
            cst = 42
        else:
            cst = 42 # default
            
        i_wind = f <= f_wind
        f_temp = f[i_wind] if _np.any(i_wind) else _np.array([2000])  # so that it doesn’t crash if only f > 2000 are entered, admittedly this is a total arbitrary hack

        # These confusing letters were taken directly from the old code
        f0w             = 770 - 100 * _np.log10(u)
        L0w             = cst + 20 * _np.log10(u) - 17 * _np.log10(f0w / 770)
        L1w             = L0w + (s1w / _np.log10(2)) * _np.log10(f_temp / f0w)
        L2w             = L0w + (s2w / _np.log10(2)) * _np.log10(f_temp / f0w)
        Lw              = L1w * (1 + (L1w / L2w) ** (-a)) ** (1 / a)
        temp_noise_dist = 10 ** (Lw / 10)

        if _np.any(i_wind):
            NL[i_wind] = temp_noise_dist * df[i_wind]

        # Meld with a sensible line at freqs greater than 2000 Hz
        if _np.any(~i_wind):
            prop_const  = temp_noise_dist[-1] / f_temp[-1] ** slope
            NL[~i_wind] = prop_const * f[~i_wind] ** slope * df[~i_wind]

        NL = 10 * _np.log10(NL)

        if n2 != 1:
            NL = NL.reshape((n2,))

    return NL



def compute_wenz(f, u, rain_rate='none', water_depth='deep', shipping_level='medium', totalOnly=False):
    """
    Calculates the noise level (in dB re uPa) based on five components:
    (1) Shipping noise (Wenz, 1962)
    (2) Wind noise (Merklinger, 1979, and Piggott, 1964)
    (3) Rain noise (Torres and Costa, 2019)
    (4) Thermal noise (Mellen, 1952)
    (5) Turbulence noise (Nichols and Bradley, 2016)

    Parameters:
        f (_np.ndarray or float): Single frequency or vector of frequencies (Hz). Assumes a 1-Hz band for a single frequency.
        u (float): Wind speed (knots).
        shipping_level (str): 'no', 'low', 'medium', or 'high' (default: 'medium').
        water_depth (str): 'shallow' or 'deep' (default: 'deep').
        rain_rate (str): 'no', 'light', 'moderate', 'heavy', or 'veryheavy' (default: 'none').
        totalOnly (bool): False to get all noises separately, including the total; True to get only the total.

    Returns:
        NL (_np.ndarray): Column vector containing the noise (dB per muPa^2/Hz) at frequencies f. 
        If totalOnly is True, NL includes [total, noise_ship, noise_wind, noise_rain, noise_therm, noise_turb].

    Code translated from "A simple yet practical ambient noise model"
    by Cristina D. S. Tollefsen and Sean Pecknold.
    
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
    
    f = _np.array(f).flatten()

    # Thermal noise
    noise_therm                   = -75.0 + 20.0 * _np.log10(f)
    noise_therm[noise_therm <= 0] = 1

    # Wind noise
    noise_wind = compute_windnoise(f, u, water_depth)

    # Shipping noise
    c1 = 30 if water_depth == 'deep'   else 65 if water_depth == 'shallow'   else 30
    c2 = 1  if shipping_level == 'low' else 4  if shipping_level == 'medium' else 7  if shipping_level == 'high' else 4
    
    if shipping_level != 'no':
        noise_ship                  = 76 - 20 * (_np.log10(f) - _np.log10(c1))**2 + 5 * (c2 - 4)
        noise_ship[noise_ship <= 0] = 1
    else:
        noise_ship = _np.zeros_like(f)
     
    # Turbulence noise
    noise_turb                  = 108.5 - 32.5 * _np.log10(f)
    noise_turb[noise_turb <= 0] = 1

    # Rain rate noise
    r0 = [0, 51.0769, 61.5358, 65.1107, 74.3464]
    r1 = [0, 1.4687, 1.0147, 0.8226, 1.0131]
    r2 = [0, -0.5232, -0.4255, -0.3825, -0.4258]
    r3 = [0, 0.0335, 0.0277, 0.0251, 0.0277]

    i_rain     = {'no': 0, 'light': 1, 'moderate': 2, 'heavy': 3, 'veryheavy': 4}.get(rain_rate, 1)
    fk         = f / 1000  # convert to kHz for this equation
    noise_rain = r0[i_rain] + r1[i_rain] * fk + r2[i_rain] * fk**2 + r3[i_rain] * fk**3

    # Only good up to about 7 kHz, so meld with a sensible line above that
    # Technique borrowed from wind-driven noise
    slope                = -5.0 * (0.1 / _np.log10(2))  # slope at high freq
    ind                  = _np.where(f < 7000)[0][-1]
    temp_noise           = 10**(noise_rain[ind] / 10)
    prop_const           = temp_noise / f[ind]**slope
    noise_rain[f > 7000] = 10 * _np.log10(prop_const * f[f > 7000]**slope)

    # Sum
    if totalOnly:
        NL = 10 * _np.log10(10**(noise_therm / 10) + 10**(noise_wind / 10) + 10**(noise_ship / 10) + 10**(noise_turb / 10) + 10**(noise_rain / 10))
    else:
        total = 10 * _np.log10(10**(noise_therm / 10) + 10**(noise_wind / 10) + 10**(noise_ship / 10) + 10**(noise_turb / 10) + 10**(noise_rain / 10))
        NL    = _np.column_stack((total, noise_ship, noise_wind, noise_rain, noise_therm, noise_turb))

    return NL



def plot_wenz(Fxx, NL, wind_speed, rain_rate, water_depth, shipping_level, Title=''):
    """
    Plot noise levels estimated with the WENZ model.

    Parameters:
        Fxx (array): Frequency vector.
        NL (array): Noise levels in dB re 1uPa calculated for different components.

    Returns:
        fig, ax: Matplotlib figure and axis objects.
    """
    fig, ax = plt.subplots()

    if NL.shape[1] == 1:
        # If only total noise is provided, plot it
        ax.semilogx(Fxx, NL, label=f'Total noise ({water_depth} water, {shipping_level} traffic, {wind_speed} kn, {rain_rate} rain)', color='black')
    else:
        # Plot noise levels for different components
        ax.semilogx(Fxx, NL[:, 0], label=f'Total noise ({water_depth} water)', color='black')
        ax.semilogx(Fxx, NL[:, 1], label=f'Shipping noise ({shipping_level} traffic)', color='blue', linestyle='dashed')
        ax.semilogx(Fxx, NL[:, 2], label=f'Wind noise ({wind_speed} kn)', color='green', linestyle='dashed')
        ax.semilogx(Fxx, NL[:, 3], label=f'Rain noise ({rain_rate} rain)', color='orange', linestyle='dashed')
        ax.semilogx(Fxx, NL[:, 4], label='Thermal noise', color='red', linestyle='dashed')
        ax.semilogx(Fxx, NL[:, 5], label='Turbulence noise', color='purple', linestyle='dashed')

    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Noise Level [dB re 1$\mu$Pa]')
    ax.set_title(f'[ WENZ - Noise Level Estimate ] {Title}')
    ax.set_xlim((Fxx[0], Fxx[-1]))
    ax.set_ylim((6, 146))  # Adjusted y-axis limits for better visibility
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def print_file_content(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            print("================================================================================================")
            print(f"Content of the file: '{file_path}'.")
            print("================================================================================================")
            print(f"{content}")
            print("================================================================================================")

    except FileNotFoundError:
        print(f"[ERROR] Unable to print file, '{file_path}' does not exist.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during print file: {e}")