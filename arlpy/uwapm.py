##############################################################################
#
# cpright (c) 2018, Mandar Chitre
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
import copy

# Add acoustic toolbox path to Python path
os.environ['PATH'] = os.environ['PATH'].replace(':/opt/build/at/bin', '')+":/opt/build/at/bin"

# Configure LaTeX for matplotlib
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':12})
rc('text', usetex=True)

## Constants and models definition
###############################################################################

# @todo     Check if there is differences between linear/c-linear/n2-linear/spline, and remove useless definition !

# Surface/bottom interpolation methods
linear            = 'linear'
curvilinear       = 'curvilinear'
piecewise_linear  = 'piecewise linear'

# SSP interpolation methods
spline            = 'spline'         # Cubic Spline
c_linear          = 'c-linear'
n2_linear         = 'n2-linear'
analytic          = 'analytic'
quadrilatteral    = 'quadrilatteral' # Bellhop only, with SSP file
hexahedral        = 'hexahedral'     # Bellhop 3D only with SSP file   
hermite           = 'hermite'        # Piecewise Cubic Hermite Interpolating Polynomial for which model ?


# Attenuation units
nepers_meter      = 'Nepers/m'
dB_wavelength     = 'dB/wavelength'
dB_meter          = 'dB/m'
dB_meter_fScaled  = 'dB/m freq scaled'
dB_kmHz           = 'dB/(kmHz)'
quality_factor    = 'quality-factor' 
loss_parameter    = 'loss-parameter'

# Volume attenuation
Thorp             = 'Thorp'
Francois_Garrison = 'Francois Garrison'
biological        = 'biological'

# @todo remove it !
# Kraken modes and options
modes             = 'modes'
TL                = 'TL'

# Bellhop modes and options
incoherent        = 'incoherent'
semicoherent      = 'semicoherent'
coherent          = 'coherent'
arrivals          = 'arrivals'
eigenrays         = 'eigenrays'
rays              = 'rays'

# Boundary condition
rigid             = 'rigid'
vacuum            = 'vacuum'
acousto_elastic   = 'acousto-elastic'
file              = 'file'
grain_size        = 'grain size'
precalculated     = 'precalculated'
soft_boss         = 'soft-boss'     # Kraken only
soft_boss_amp     = 'soft-boss-amp' # Kraken only
hard_boss         = 'hard-boss'     # Kraken only
hard_boss_amp     = 'hard-boss-amp' # Kraken only

# Kraken mode
coupled           = 'coupled' 
adiabatic         = 'adiabatic' 

# models (in order of preference)
_models = []

# Objects definition
###############################################################################

# @todo     create and manage env3d for 3 dimensionnal problem
        
def make_env2d(**kv):
    """
    Create a new 2D underwater environment.

    This function creates a 2D underwater environment with default parameters. It allows customization of the environment
    by specifying keyword arguments or modifying the parameters later using dictionary notation.

    Parameters:
        auto_xSettings (bool, optional): Whether to automatically set numerical box boundaries. Defaults to True.
        auto_xBox (bool, optional): Whether to adjust the environment border by padding input data. Defaults to True.
        cp (bool, optional): Whether to return a shallow cp of the environment. Defaults to False.
        **kv: Additional keyword arguments to customize the environment parameters.

    Returns:
        dict: A dictionary containing the environment parameters.

    Raises:
        KeyError: If an unknown key is provided in the keyword arguments.

    Note:
        Surface/bottom interpolation methods:
            - linear
            - curvilinear
            - piecewise linear
        
        SSP interpolation methods:
            - spline: Cubic Spline
            - c_linear
            - n2_linear
            - analytic
            - quadrilatteral: Bellhop only, with SSP file
            - hexahedral: Bellhop 3D only with SSP file
            - hermite: Piecewise Cubic Hermite Interpolating Polynomial

        Attenuation units:
            - nepers_meter
            - dB_wavelength
            - dB_meter
            - dB_meter_fScaled
            - dB_kmHz
            - quality_factor
            - loss_parameter

        Volume attenuation:
            - Thorp
            - Francois_Garrison
            - biological

        Boundary condition:
            - rigid
            - vacuum
            - acousto_elastic
            - file
            - grain_size
            - precalculated
            - soft_boss: Kraken only
            - soft_boss_amp: Kraken only
            - hard_boss: Kraken only
            - hard_boss_amp: Kraken only
    """
    env = {
            'name'            : '',                     
            'dimension'       : '2D',                        # 2D only
            'pad_inputData'   : True,                        # pad input data to the border of the calculation range

            'mode'            : coherent,                    # Propagation loss mode
            'volume_attn'     : None,                        # Added volume attenuation
            
            'rx_depth'        : None,                        # m
            'rx_range'        : None,                        # m
            
            'top_boundary'    : vacuum,
            'top_interface'   : None,                        # surface profile
            'top_roughness'   : 0,                           # m (rms)
            
            'ssp_range'       : 0,                           # m
            'ssp_depth'       : 0,                           # m
            'ssp'             : 1500,                        # m/s
            'ssp_interp'      : c_linear,                    # spline/linear/quadrilatteral/..
            'water_density'   : 1.03,                        # g/cm3, KRAKEN only    
            
            'tx_freq'         : None,                        # Source frequency in Hz
            'tx_depth'        : None,                        # m
            'tx_beam'         : None,                        # [[deg, dB]...]
            'tx_nbeam'        : 0,                           # number of beams (0 = auto)
            'tx_minAngle'     : -180,                        # deg
            'tx_maxAngle'     : 180,                         # deg
            
            'bot_boundary'    : rigid,
            'bot_interface'   : None,                        # m
            'bot_roughness'   : 0,                           # m (rms) 
            'bot_range'       : 0,                           # m (bottom settings range for ssp, density and absorption)
            'bot_depth'       : 0,                           # m (bottom settings depth for ssp, density and absorption)            

            # OALIB ONLY
            ## Acouso-elastic boundary condition, bottom typical values for rock
            'attn_unit'       : dB_wavelength,               # Attenuation units 
            'top_density'     : None,                        # g/cm3 (RHOT)
            'top_PwaveSpeed'  : None,                        # m/s (CPT)
            'top_SwaveSpeed'  : None,                        # m/s (CST)
            'top_PwaveAttn'   : None,                        # attn_unit (APT)
            'top_SwaveAttn'   : None,                        # attn_unit (AST)
            'bot_density'     : None,                        # g/cm3 (RHOB)   
            'bot_PwaveSpeed'  : None,                        # m/s (CPB)
            'bot_SwaveSpeed'  : None,                        # m/s (CSB)
            'bot_PwaveAttn'   : None,                        # attn_unit (APB)
            'bot_SwaveAttn'   : None,                        # attn_unit (ASB)
        
            # KRAKEN ONLY
            # Twersky scatter parameters for soft/hard-boss Twersky top boundary condition only (4c)
            'twy_bumpDensity' : None,                       # ridges/km (BUMDEN)
            'twy_radius1'     : None,                       # m (ETA)
            'twy_radius2'     : None,                       # m (XI)
            'nmedia'          : 1,                          # Number of media except infinite top and bottom 
            'theory'          : adiabatic,                 
            'nmode'           : 999999999,
                    
            }
    
    for k, v in kv.items():
        if k not in env.keys():
            raise KeyError('Unknown key: '+k)
        env[k] = _np.array(v, dtype=_np.float64) if _np.size(v) > 1 else v
    
    # Ensure consistency of dimensions for SSP (Sound Speed Profile) and bottom settings    
    if env['bot_interface'][0,0] is None or not _np.isfinite(env['bot_interface'][0,0]):
        env['bot_interface'][0,0] = 0
    if _np.size(env['ssp_range']) == _np.size(env['ssp']):
        if _np.size(env['ssp']) > 1:
            env['ssp'] = _np.hstack(env['ssp'])
    if _np.size(env['ssp_depth']) == _np.size(env['ssp']):
        if _np.size(env['ssp']) > 1:
            env['ssp'] = _np.vstack(env['ssp'])
    if env['bot_PwaveAttn'] is not None and _np.size(env['bot_PwaveAttn']) > 1:
        if _np.size(env['bot_range']) == _np.size(env['bot_PwaveAttn']):
            env['bot_PwaveAttn'] = _np.hstack(env['bot_PwaveAttn'])
        if _np.size(env['bot_depth']) == _np.size(env['bot_PwaveAttn']):
            env['bot_PwaveAttn'] = _np.vstack(env['bot_PwaveAttn'])
    if env['bot_PwaveSpeed'] is not None and _np.size(env['bot_PwaveSpeed']) > 1:
        if _np.size(env['bot_range']) == _np.size(env['bot_PwaveSpeed']):
            env['bot_PwaveSpeed'] = _np.hstack(env['bot_PwaveSpeed'])
        if _np.size(env['bot_depth']) == _np.size(env['bot_PwaveSpeed']):
            env['bot_PwaveSpeed'] = _np.vstack(env['bot_PwaveSpeed'])
    if env['bot_density'] is not None and _np.size(env['bot_density']) > 1:
        if _np.size(env['bot_range']) == _np.size(env['bot_density']):
            env['bot_density'] = _np.hstack(env['bot_density'])
        if _np.size(env['bot_depth']) == _np.size(env['bot_density']):
            env['bot_density'] = _np.vstack(env['bot_density'])

    # Adjust environment border by padding input data if required for OALIB and RAM
    if env['pad_inputData'] == True:
        
        # Define OALIB numerical box
        rBox = 1.01*_np.max(_np.abs(env['rx_range']))
        zBox = 1.01*_np.max((_np.max(env['bot_interface'][:,-1]), _np.max(env['rx_depth'])))
        
        # Ensure top interface is initalize and sized
        if env['top_interface'] is not None:
            env['top_interface'] = _adjust_2D(env['top_interface'], -1.001*rBox, 1.001*rBox)
        else:
            env['top_interface'] = _np.array((0,0), ndmin=2)
        zSSPmin = _np.min(env['top_interface'][:,1])-zBox/100
        
        # Get at least 2 ssp values at depth 0 and 1.001*zBox
        if (_np.size(env['ssp_range']) == 1 and _np.size(env['ssp_depth']) == 1 and _np.size(env['ssp']) == 1) or (_np.size(env['ssp_depth']) == 1 and _np.size(env['ssp_range']) == _np.size(env['ssp'])) or _np.size(env['ssp']) == 1:
            env['ssp'] = _np.vstack([env['ssp'], env['ssp']])
            env['ssp_depth'] = _np.array([zSSPmin, 1.001*zBox])

        # Ensure settings overlap and size 
        env['bot_interface'] = _adjust_2D(env['bot_interface'], -1.001*rBox, 1.001*rBox)
        env['ssp'], env['ssp_range'], env['ssp_depth']  = _adjust_3D(env['ssp'], env['ssp_range'], env['ssp_depth'], -1.001*rBox, 1.001*rBox, zSSPmin,1.001*zBox)                                                               
        env['bot_PwaveAttn'], _, _  = _adjust_3D(env['bot_PwaveAttn'], env['bot_range'], env['bot_depth'], -1.001*rBox, 1.001*rBox, -1.001*_np.min(_np.abs(env['bot_interface'][:,1])), 1.001*zBox)                                                                 
        env['bot_density'], _, _  = _adjust_3D(env['bot_density'], env['bot_range'], env['bot_depth'], -1.001*rBox, 1.001*rBox, -1.001*_np.min(_np.abs(env['bot_interface'][:,1])), 1.001*zBox)                                                                
        env['bot_PwaveSpeed'], env['bot_range'], env['bot_depth']  = _adjust_3D(env['bot_PwaveSpeed'], env['bot_range'], env['bot_depth'], -1.001*rBox, 1.001*rBox, -1.001*_np.min(_np.abs(env['bot_interface'][:,1])), 1.001*zBox)

    # @todo 
    # env = check_env2d(env)
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
        if env['top_interface'] is not None:
            assert _np.size(env['top_interface']) > 1, 'surface must be an Nx2 array'
            assert env['top_interface'].ndim == 2, 'surface must be a scalar or an Nx2 array'
            assert env['top_interface'].shape[1] == 2, 'surface must be a scalar or an Nx2 array'
            assert env['top_interface'][0,0] <= 0, 'First range in surface array must be 0 m'
            assert env['top_interface'][-1,0] >= max_range, 'Last range in surface array must be beyond maximum range: '+str(max_range)+' m'
            assert _np.all(_np.diff(env['top_interface'][:,0]) > 0), 'surface array must be strictly monotonic in range'
            # assert env['surface_interp'] == curvilinear or env['surface_interp'] == linear, 'Invalid interpolation type: '+str(env['surface_interp'])
        if _np.size(env['bot_interface']) > 1:
            assert env['bot_interface'].ndim == 2, 'depth must be a scalar or an Nx2 array'
            assert env['bot_interface'].shape[1] == 2, 'depth must be a scalar or an Nx2 array'
            assert env['bot_interface'][0,0] <= 0, 'First range in depth array must be 0 m'
            assert env['bot_interface'][-1,0] >= max_range, 'Last range in depth array must be beyond maximum range: '+str(max_range)+' m'
            assert _np.all(_np.diff(env['bot_interface'][:,0]) > 0), 'Depth array must be strictly monotonic in range'
            assert env['depth_interp'] == curvilinear or env['depth_interp'] == linear, 'Invalid interpolation type: '+str(env['depth_interp'])
            max_depth = _np.max(env['bot_interface'][:,1])
        else:
            max_depth = env['bot_interface']
        if isinstance(env['ssp'], _pd.DataFrame):
            assert env['ssp'].shape[0] > 3, 'soundspeed profile must have at least 4 points'
            assert env['ssp_depth'][0] <= 0, 'First depth in soundspeed array must be 0 m'
            assert env['ssp_depth'][-1] >= max_depth, 'Last depth in soundspeed array must be beyond water depth: '+str(max_depth)+' m'
            assert _np.all(_np.diff(env['ssp'].index) > 0), 'Soundspeed array must be strictly monotonic in depth'
        elif _np.size(env['ssp']) > 1:
            #assert env['ssp'].ndim == 2, 'soundspeed must be a scalar or an Nx2 array'
            #assert env['ssp'].shape[1] == 2, 'soundspeed must be a scalar or an Nx2 array'
            assert env['ssp'].shape[0] > 3, 'soundspeed profile must have at least 4 points'
            assert env['ssp_depth'][0] <= 0, 'First depth in soundspeed array must be 0 m'
            assert env['ssp_depth'][-1] >= max_depth, 'Last depth in soundspeed array must be beyond water depth: '+str(max_depth)+' m'
            assert _np.all(_np.diff(env['ssp_depth']) > 0), 'Soundspeed array must be strictly monotonic in depth'
            #assert env['ssp_interp'] == spline or env['ssp_interp'] == linear, 'Invalid interpolation type: '+str(env['ssp_interp'])
            if not(max_depth in env['ssp_depth']):
                indlarger = _np.argwhere(env['ssp'][:,0]>max_depth)[0][0]
                if env['ssp_interp'] == spline:
                    tck = _interp.splrep(env['ssp'][:,0], env['ssp'][:,1], s=0)
                    insert_ss_val = _interp.splev(max_depth, tck, der=0)
                else:
                    insert_ss_val = _np.interp(max_depth, env['ssp'][:,0], env['ssp'][:,1])
                env['ssp'] = _np.insert(env['ssp'],indlarger,[max_depth,insert_ss_val],axis = 0)
                env['ssp'] = env['ssp'][:indlarger+1,:]
        assert _np.max(env['tx_depth']) <= max_depth, 'tx_depth cannot exceed water depth: '+str(max_depth)+' m'
        assert _np.max(env['rx_depth']) <= max_depth, 'rx_depth cannot exceed water depth: '+str(max_depth)+' m'
        assert env['tx_minAngle'] > -90 and env['tx_minAngle'] < 90, 'min_angle must be in range (-90, 90)'
        assert env['tx_maxAngle'] > -90 and env['tx_maxAngle'] < 90, 'max_angle must be in range (-90, 90)'
        if env['tx_beam'] is not None:
            assert _np.size(env['tx_beam']) > 1, 'tx_directionality must be an Nx2 array'
            assert env['tx_beam'].ndim == 2, 'tx_directionality must be an Nx2 array'
            assert env['tx_beam'].shape[1] == 2, 'tx_directionality must be an Nx2 array'
            assert _np.all(env['tx_beam'][:,0] >= -180) and _np.all(env['tx_beam'][:,0] <= 180), 'tx_directionality angles must be in [-90, 90]'
        return env
    except AssertionError as e:
        raise ValueError(e.args)

def print_env(env):
    """
    Display the environment in a human-readable form.

    Parameters:
        env (dict): Environment definition.

    Example:
    >>> import arlpy.uwapm as pm
    >>> env = pm.create_env2d(depth=40, soundspeed=1540)
    >>> pm.print_env(env)
    """
    # Sort keys and include 'name' first
    keys = ['name'] + sorted(list(env.keys() - {'name'}))
    
    # Iterate through each key-value pair in the environment
    for k in keys:
        v = str(env[k])
        
        # If the value is too long, truncate it
        if len(v) > 40:
            if '\n' in v:
                v = v.split('\n')
                print('%20s : ' % (k) + v[0] + '...')
                print('%26s %s' % ('...', v[-1]))
            else:
                print('%20s : ' % (k) + v[:20] + '...')
                print('%26s %s' % ('...', v[-20:]))
                
        # If the value contains newline characters, format it accordingly
        elif '\n' in v:
            v = v.split('\n')
            if len(v) > 3:
                print('%20s : ' % (k) + v[0])
                print('%26s\n%30s' % ('...', v[-1]))
            else:
                print('%20s : ' % (k) + v[0])
                for v1 in v[1:]:
                    print('%20s   ' % ('') + v1)
        else:
            # Print the key-value pair
            print('%20s : ' % (k) + v)

def arrivals_to_impulse_response(arrivals, fs, abs_time=False):
    """
    Convert arrival times and coefficients to an impulse response.

    Parameters:
        arrivals (DataFrame): Arrivals times (s) and coefficients.
        fs (int): Sampling rate (Hz).
        abs_time (bool): Absolute time (True) or relative time (False).

    Returns:
        ndarray: Impulse response.

    If `abs_time` is set to True, the impulse response is placed such that
    the zero time corresponds to the time of transmission of signal.
    """
    # Determine the reference time for the impulse response
    t0 = 0 if abs_time else min(arrivals.time_of_arrival)
    
    # Calculate the length of the impulse response
    irlen = int(_np.ceil((max(arrivals.time_of_arrival) - t0) * fs)) + 1
    
    # Initialize the impulse response array
    ir = _np.zeros(irlen, dtype=_np.complex128)
    
    # Fill the impulse response array with arrival coefficients
    for _, row in arrivals.iterrows():
        ndx = int(_np.round((row.time_of_arrival.real - t0) * fs))
        ir[ndx] = row.arrival_amplitude
        
    return ir
    
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
class BELLHOP:

    def __init__(self, env=None, cp=False):
        
        if env is None:
            self.env           = None
        else:
            self.set_env(env, cp=cp)
        self.transmission_loss = None 
        self.eigen_rays        = None
        self.arrivals          = None
        self.impulse_response  = None
        self.step              = 0

    def set_step(self, step):
        self.step = step

    def set_env(self, env, cp=False, **kwargs):
        
        """Set a new 2D underwater environment.

        A basic environment is created with default values. To see all the parameters
        available and their default values. The environment parameters may be changed 
        by passing keyword arguments or modified later using a dictionary notation.
        """
        if cp:
            self.env =  dict(env)
        else:
            self.env = env
        
        # Numerical box definition
        self.rbox = 1.01*_np.max(_np.abs(env['rx_range']))
        self.zbox = 1.01*_np.max((_np.max(env['bot_interface'][:,-1]), _np.max(env['rx_depth'])))
                
        self.check_env()

        return self.env
        
    def check_env(self):
        # @todo
       
        return True

    def compute_transmission_loss(self, debug=False, **kwargs):
        
        # Assert environment
        self.check_env
        
        # Define mode taskcode
        if self.env['mode'] == coherent:
            taskcode = 'C'
        elif self.env['mode'] == incoherent:
            taskcode = 'I'
        elif self.env['mode'] == semicoherent:
            taskcode = 'S'
        else:
            raise Exception("[ERROR] BELLHOP: Mode not known !")
           
        # Generate temporary env file and get base name used for all temporary files
        fname_base = self._create_env_file(taskcode, debug=debug)
        
        # Compute TL
        if self._bellhop(fname_base):
            err = self._check_error(fname_base)
            if err is not None:
                raise RuntimeError(err) 
            else:
                try:
                    self.transmission_loss = self._load_shd(fname_base)
                except FileNotFoundError:
                    raise FileNotFoundError('BELLHOP: Fortran execution did not generate expected output file !')
        
        # Delete temporary generated files
        self._unlink_all(fname_base)     
        
        return self.transmission_loss
     
    def flip_env(self):
        
        self.env['rx_range']           = -_np.flip( self.env['rx_range'])
        self.env['bot_interface'][:,0] = -_np.flip(self.env['bot_interface'][:,0])
        self.env['bot_interface'][:,1] =  _np.flip(self.env['bot_interface'][:,1])
        self.env['top_interface'][:,0] = -_np.flip(self.env['top_interface'][:,0])
        self.env['top_interface'][:,1] =  _np.flip(self.env['top_interface'][:,1])
        if _np.size(self.env['ssp_range']) > 1:
            self.env['ssp']                =  _np.fliplr(self.env['ssp'])
            self.env['ssp_range']          = -_np.flip(self.env['ssp_range'])
        
    def compute_arrivals(self, debug=False):
        
        # Assert environment
        self.check_env()
        
        # Define mode taskcode
        taskcode = 'A'
        flip = False
        if self.env['rx_range'] < 0:
            self.flip_env()
            flip = True
           
        # Generate temporary env file and get base name used for all temporary files
        fname_base = self._create_env_file(taskcode, debug=debug)
        
        # Compute Arrivals
        if self._bellhop(fname_base):
            err = self._check_error(fname_base)
            if err is not None:
                raise RuntimeError(err) 
            else:
                try:
                    self.arrivals = self._load_arrivals(fname_base)
                except FileNotFoundError:
                    raise FileNotFoundError('BELLHOP: Fortran execution did not generate expected output file !')
        
        # Delete temporary generated files
        self._unlink_all(fname_base)    
        
        if flip:
            self.flip_env()
            flip = False
        
        return self.arrivals
        

    def compute_rays(self, debug=False):
        
        # Assert environment
        self.check_env()
        
        # Define mode taskcode
        taskcode = 'R'
        
        flip = False
        if self.env['rx_range'] < 0:
            self.flip_env()
            flip = True
              
        # Generate temporary env file and get base name used for all temporary files
        fname_base = self._create_env_file(taskcode, debug=debug)
        
        # Compute Arrivals
        if self._bellhop(fname_base):
            err = self._check_error(fname_base)
            if err is not None:
                raise RuntimeError(err) 
            else:
                try:
                    self.rays = self._load_rays(fname_base)
                except FileNotFoundError:
                    raise FileNotFoundError('BELLHOP: Fortran execution did not generate expected output file !')
        
        # Delete temporary generated files
        self._unlink_all(fname_base)     
        
        if flip:
            self.flip_env()
            flip = False
            
        return self.rays
    
    def compute_eigen_rays(self, debug=False):
        
        # Assert environment
        self.check_env()
        
        # Define mode taskcode
        taskcode = 'E'
           
        flip = False
        
        if self.env['rx_range'] < 0:
            self.flip_env()
            flip = True
           
            
        # Generate temporary env file and get base name used for all temporary files
        fname_base = self._create_env_file(taskcode, debug=debug)
        
        # Compute Arrivals
        if self._bellhop(fname_base):
            err = self._check_error(fname_base)
            if err is not None:
                raise RuntimeError(err) 
            else:
                try:
                    self.eigen_rays = self._load_rays(fname_base)
                except FileNotFoundError:
                    raise FileNotFoundError('BELLHOP: Fortran execution did not generate expected output file !')
        
        # Delete temporary generated files
        self._unlink_all(fname_base)     
        
        if flip:
            self.flip_env()
            flip = False
            
        return self.eigen_rays
    
    def compute_impulse_respsonse(self, fs=24000, nArrival=10, debug=False):
        
        self.compute_arrivals(debug=debug)
        self.impulse_response = arrivals_to_impulse_response(self.arrivals[0:min(nArrival, len(self.arrivals)-1)], fs, abs_time=False)
        self.impulse_response_fs = fs
        return self.impulse_response
        
        
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
        
    def _unlink_all(self, fname_base):
        
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

    def _print(self, fh, s, newline=True):
        _os.write(fh, (s+'\n' if newline else s).encode())

    def _print_array(self, fh, a):
        if _np.size(a) == 1:
            self._print(fh, "1")
            self._print(fh, "%0.6f /" % (a))
        else:
            self._print(fh, str(_np.size(a)))
            self._print(fh, "%0.6f %0.6f " % (a.min(), a.max()), newline=False)
            #for j in a:
            #   self._print(fh, "%0.6f " % (j), newline=False)
            self._print(fh, "/")

    def _create_env_file(self, taskcode, debug=False, **kwargs):
        
        fh, fname = _mkstemp(suffix='.env')
        fname_base = fname[:-4]
                
        # Title
        self._print(fh, "'"+self.env['name']+"'")
        
        # Freq
        self._print(fh, "%0.6f" % (self.env['tx_freq']))
        
        # Nmedia
        self._print(fh, "%d" % (1)) # Bellhop is limited to one media and ignore this parameter
        
        # Option (1:1) SSP interp      
        if _np.size(self.env['ssp_range']) > 1 and self.env['ssp_interp'] != quadrilatteral:
            print('[INFO] BELLHOP: SSP interp replaced by quadrilatteral for range dependant SSP.')
            ssp_interp = 'Q'
        elif _np.ndim(self.env['ssp']) == 1 and self.env['ssp_interp'] == quadrilatteral:
            print('[INFO] BELLHOP: Quadrilatteral interpolation is for range dependant SSP only, using C-linear instead.')
            ssp_interp = 'C'
        elif _np.ndim(self.env['ssp']) == 2 and self.env['ssp_interp'] == quadrilatteral:
            ssp_interp = 'Q'
        elif self.env['ssp_interp'] == spline:
            ssp_interp = 'S'
        elif self.env['ssp_interp'] == c_linear:
            ssp_interp = 'C' 
        elif self.env['ssp_interp'] == n2_linear:
            ssp_interp = 'N'
        elif self.env['ssp_interp'] == analytic:
            #ssp_interp = 'A'   
            # @todo
            print('[WARNING] BELLHOP: Analytic SSP interpolation not yet coded, using C-linear instead !')
            ssp_interp = 'C' 
        elif self.env['ssp_interp'] == hermite:
            ssp_interp = 'P'
        
        # Option (2:2) Top boundary condition
        # @todo     Manage all boundary conditions.
        if self.env['top_boundary'] == rigid:
            topBdry = 'R'
        elif self.env['top_boundary'] == vacuum:
            topBdry = 'V' 
        elif self.env['top_boundary'] == acousto_elastic:
            topBdry = 'A' 
        elif self.env['top_boundary'] == file:
            #topBdry = 'F'
            # @todo
            print('[WARNING] BELLHOP: Top boundary condition from file not yet coded, using vacuum instead !')
            topBdry = 'V' 
        
        # Option (3:3) Attenuation units
        if self.env['attn_unit'] == nepers_meter:
            attnUnit = 'N'
        elif self.env['attn_unit'] == dB_wavelength:
            attnUnit = 'W'
        elif self.env['attn_unit'] == dB_meter:
            attnUnit = 'M'        
        elif self.env['attn_unit'] == dB_meter_fScaled:
            #attnUnit = 'm' 
            # @todo
            print('[WARNING] BELLHOP: Attenuation unit in dB/m scaled with frequency not yet coded, using dB/m instead !')
            attnUnit = 'M'   
        elif self.env['attn_unit'] == dB_kmHz:
            attnUnit = 'F'
        elif self.env['attn_unit'] == quality_factor:
            attnUnit = 'Q'            
        elif self.env['attn_unit'] == loss_parameter:
            attnUnit = 'L'              

        # Option (4:4) Added volume attenuation
        # @todo     Manage Francois Garrison and biological attenuation
        if self.env['volume_attn'] == None:
            vAttn = ' '
        elif self.env['volume_attn'] == Thorp:
            vAttn = 'T'
        elif self.env['volume_attn'] == Francois_Garrison:
            #vAttn = 'F'
            # @todo
            print('[WARNING] BELLHOP: Francois Garrison attenuation formula not yet coded, using Thorp formula instead !')
            vAttn = 'T'
        elif self.env['volume_attn'] == biological:
            #vAttn = 'B'
            # @todo
            print('[WARNING] BELLHOP: Biological attenuation formula not yet coded, using Thorp formula instead !')
            vAttn = 'T'
        
        # Option (5:5) Altimetry option
        if self.env['top_interface'] is None:
            alt = '_'
        else:
            alt = '*'
            # Automatic interpolation selection
            if ((self.env['top_interface'][-1,0]-self.env['top_interface'][0,0])/self.env['top_interface'][:,0].size) < (_np.mean(self.env['ssp'])/self.env['tx_freq']*10) :
                self._create_bty_ati_file(fname_base+'.ati', self.env['top_interface'], curvilinear, debug=debug)
            else:
                self._create_bty_ati_file(fname_base+'.ati', self.env['top_interface'], piecewise_linear, debug=debug)
        
        # All options string
        self._print(fh, "'%c%c%c%c%c'" % (ssp_interp, topBdry, attnUnit, vAttn, alt))

        # Extra line (4a) or (4b)
        # @todo     Manage biological attenuation
        if topBdry == 'A':
            self._print(fh, "%0.6f %0.6f %0.6f %0.6f %0.6f %0.6f" % (_np.max(self.env['bot_interface']), self.env['top_PwaveSpeed'], self.env['top_SwaveSpeed'], self.env['top_density'], self.env['top_PwaveAttn'], self.env['top_SwaveAttn']))
        
        # Medium info / Sound Speed Profile (5)
        nmesh = 0 # Unused by Bellhop
        self._print(fh, "%d %0.1f %0.6f" % (nmesh, self.env['top_roughness'], _np.max(self.env['ssp_depth'])))
        if ssp_interp == 'Q': 
            for j in range(len(self.env['ssp_depth'])):
                self._print(fh, "%0.6f %0.6f /" % (self.env['ssp_depth'][j], self.env['ssp'][j,0]))
            self._create_ssp_file(fname_base+'.ssp',self.env['ssp_range'], self.env['ssp'], debug=debug)
        elif _np.size(self.env['ssp_depth']) > 1:
            for j in range(self.env['ssp_depth'].size):
                self._print(fh, "%0.6f %0.6f /" % (self.env['ssp_depth'][j], self.env['ssp'][j]))
        else:
            self._print(fh, "%0.6f %0.6f /" % (_np.min(self.env['rx_depth']), self.env['ssp']))
            self._print(fh, "%0.6f %0.6f /" % (_np.max(self.env['rx_depth']), self.env['ssp']))
        
        # Bottom option (6)
        if self.env['bot_boundary'] == rigid:
            botBdry = 'R'
        elif self.env['bot_boundary'] == vacuum:
            botBdry = 'V' 
        elif self.env['bot_boundary'] == acousto_elastic:
            botBdry = 'A'
        elif self.env['bot_boundary'] == file:
            #botBdry = 'F'
            # @todo 
            print('[WARNING] BELLHOP: File bottom boundary condition not yet coded, using perfectly rigid instead !')
            botBdry = 'R'
        elif self.env['bot_boundary'] == grain_size:
            #botBdry = 'G'   
            # @todo 
            print('[WARNING] BELLHOP: Grain size bottom boundary condition not yet coded, using perfectly rigid instead !')
            botBdry = 'R'
        elif self.env['bot_boundary'] == precalculated:
            #botBdry = 'P'   
            # @todo
            print('[WARNING] BELLHOP: Precalculated bottom boundary condition not yet coded, using perfectly rigid instead !')
            botBdry = 'R'
        
        # Bottom options string
        if _np.ndim(self.env['bot_interface']) == 2:
            self._print(fh, "'%c*' %0.6f" % (botBdry, self.env['bot_roughness']))
            if ((self.env['bot_interface'][-1,0]-self.env['bot_interface'][0,0])/self.env['bot_interface'][:,0].size) < (_np.mean(self.env['ssp'])/self.env['tx_freq']*10) :
                self._create_bty_ati_file(fname_base+'.bty', self.env['bot_interface'], curvilinear, debug=debug)
            else:
                self._create_bty_ati_file(fname_base+'.bty', self.env['bot_interface'], piecewise_linear, debug=debug)
        else:
            self._print(fh, "'%c_' %0.6f" % (botBdry, self.env['bot_roughness']))

        
        # Bottom halfspace extra lines (6a) (6b)
        # @todo     Add Grain size 
        if  _np.size(self.env['bot_PwaveSpeed']) > 1:
            print("[INFO] BELLHOP: Do not support multiple Pwave speed definition, using median value instead.")
            bot_PwaveSpeed = _np.median(self.env['bot_PwaveSpeed'])
        else:
            bot_PwaveSpeed = self.env['bot_PwaveSpeed']
        
        if  _np.size(self.env['bot_SwaveSpeed']) > 1:
            print("[INFO] BELLHOP: Do not support multiple Swave speed definition, using median value instead.")
            bot_SwaveSpeed = _np.median(self.env['bot_SwaveSpeed'])
        else:
            bot_SwaveSpeed = self.env['bot_SwaveSpeed']
            
        if  _np.size(self.env['bot_density']) > 1:
            print("[INFO] BELLHOP: Do not support multiple bottom density definition, using median value instead.")
            bot_density = _np.median(self.env['bot_density'])
        else:
            bot_density = self.env['bot_density']    
            
        if  _np.size(self.env['bot_PwaveAttn']) > 1:
            print("[INFO] BELLHOP: Do not support multiple Pwave attn definition, using median value instead.")
            bot_PwaveAttn = _np.median(self.env['bot_PwaveAttn'])
        else:
            bot_PwaveAttn = self.env['bot_PwaveAttn']   
            
        if  _np.size(self.env['bot_SwaveAttn']) > 1:
            print("[INFO] BELLHOP: Do not support multiple Swave speed definition, using median value instead.")
            bot_SwaveAttn = _np.median(self.env['bot_SwaveAttn'])
        else:
            bot_SwaveAttn = self.env['bot_SwaveAttn']     
     
        if botBdry == 'A':
            self._print(fh, "%0.6f %0.6f %0.6f %0.6f %0.6f %0.6f" % (_np.max(self.env['bot_interface']), bot_PwaveSpeed, bot_SwaveSpeed, bot_density, bot_PwaveAttn, bot_SwaveAttn))
        
        # Number of source depth  and source depths (m) (7)
        self._print_array(fh, self.env['tx_depth'])
        
        # Number of receiver depth  and receiver depths (m) (7)
        self._print_array(fh, self.env['rx_depth'])
        
        # Number of receiver range and receiver range (km) (7)
        self._print_array(fh, self.env['rx_range']/1000)
        
        # Run type (8)
        # @todo     Manage all options  (line source in priority ?) 
        if self.env['tx_beam'] is None:
            self._print(fh, "'"+taskcode+"'")
        else:
            self._print(fh, "'"+taskcode+" *'")
            self._create_sbp_file(fname_base+'.sbp', self.env['tx_beam'], debug=debug)
            
        # Beam fan (9)
        self._print(fh, "%d" % (self.env['tx_nbeam']))
        self._print(fh, "%0.6f %0.6f /" % (self.env['tx_minAngle'], self.env['tx_maxAngle']))
        
        # Numerical integrator info (10)
        self._print(fh, "%0.6f %0.6f %0.6f" % (self.step, self.zbox, self.rbox/1000))
        
        _os.close(fh)
        
        if debug:
            print_file_content(fname_base+'.env')
        
        return fname_base


    def plot_interpSSP(self):
        # @todo Plot SSP in 1D or 2D taking inoto account interpolation process for validation !!
        pass
    
    def plot_beam(self, vmin=-60, vmax=20, **kwargs):
        """
        Plots the beam pattern.

        Parameters:
            vmin (float): Minimum value in dB (default: -60).
            vmax (float): Maximum value in dB (default: 20).
            **kwargs: Additional keyword arguments for customization.

        Returns:
            fig, ax: Figure and axis objects for the plot.
        """
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        if self.env['tx_beam'] is not None:
            # Plot the beam pattern if available
            ax.plot((self.env['tx_beam'][:, 0] + 180) / 360 * 2 * _np.pi - _np.pi, self.env['tx_beam'][:, 1])
        else:
            # Plot a flat line if beam pattern data is not available
            ax.plot(_np.linspace(0, 2 * _np.pi, 1000), _np.zeros(1000))

        # Move radial labels away from the plotted line
        ax.set_rlabel_position(-22.5)
        ax.grid(True)
        ax.set_ylim(vmin, vmax)
        ax.set_title(f"[BELLHOP - Source directivity [dB]] {self.env['name']}", va='bottom')
        plt.tight_layout()
        plt.show()

        return fig, ax
    
    def plot_transmission_loss(self, vmin=-120, vmax=0, debug=False):
        """
        Plots transmission loss.

        Parameters:
            vmin (float): Minimum value for color scale (default: -120 dB).
            vmax (float): Maximum value for color scale (default: 0 dB).
            debug (bool): Whether to enable debug mode (default: False).

        Returns:
            fig, ax: Figure and axis objects for the plot.
        """

        fig, ax = plt.subplots()
        X = self.env['rx_range']
        Y = self.env['rx_depth']

        tlossplt = 20 * _np.log10(_np.finfo(float).eps + _np.abs(_np.array(self.transmission_loss)))

        if _np.size(self.env['bot_interface'][:, 0]) > 1:
            for i, range_val in enumerate(X):
                ylim = _np.interp(range_val, self.env['top_interface'][:, 0], self.env['top_interface'][:, 1])
                for j, depth_val in enumerate(Y):
                    if depth_val < ylim:
                        tlossplt[j, i] = _np.NaN
            ax.plot(self.env['top_interface'][:, 0] / 1000, self.env['top_interface'][:, 1], 'b', linewidth=3)
        else:
            ax.plot([X[0] / 1000, X[-1] / 1000], [self.env['top_interface'][0, 1], self.env['top_interface'][0, 1]], 'b', linewidth=3)
            
        # Plot the transmission loss map using imshow
        im1 = ax.imshow(tlossplt, extent=[X[0] / 1000, X[-1] / 1000, Y[-1], Y[0]], cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
        
        # Plot the bottom interface
        if _np.size(self.env['bot_interface'][:, 0]) > 1:
            ax.plot(self.env['bot_interface'][:, 0] / 1000, self.env['bot_interface'][:, 1], 'k', linewidth=3)
        else:
            ax.plot([X[0] / 1000, X[-1] / 1000], [self.env['bot_interface'][0, 1], self.env['bot_interface'][0, 1]], 'k', linewidth=3)
        
        # Remove transmission loss in sediment/surface
        interp_y = _np.interp(X, self.env['bot_interface'][:, 0], self.env['bot_interface'][:, 1])
        ax.fill_between(X / 1000, interp_y, Y[-1], color='brown')
        
        # Set plot properties
        ax.set_xlim((X[0] / 1000, X[-1] / 1000))
        ax.set_ylim((Y[0], Y[-1]))
        ax.set_xlabel('Range [km]')
        ax.set_ylabel('Depth [m]')
        ax.set_title(f"[BELLHOP - Transmission loss @ {self.env['tx_freq']} Hz] {self.env['name']}")
        
        # Add color bar
        cbar1 = fig.colorbar(im1, ax=ax)
        cbar1.ax.set_ylabel('Loss [dB]')
        
        # Invert y-axis for depth
        ax.invert_yaxis()
        
        # Adjust layout and display plot
        plt.tight_layout()
        plt.show()

        return fig, ax
    
    def plot_ssp(self, Nxy=500, **kwargs):
        """
        Plots the sound speed profile of the environment.
        
        Parameters:
            Nxy (int): Number of points in the x and y directions.
            **kwargs: Additional keyword arguments for customization.
            
        Returns:
            fig, ax: Figure and axis objects for the plot.
        """
        fig, ax = plt.subplots()
        
        vmax = kwargs.get('vmax', _np.max(self.env['ssp'])+_np.abs(_np.mean(self.env['ssp']))/100)
        vmin = kwargs.get('vmin', _np.min(self.env['ssp'])-_np.abs(_np.mean(self.env['ssp']))/100)
        
        if _np.size(self.env['ssp_range']) > 1: # 2D SSP data
        
            X, Y, Z = _np.array(self.env['ssp_range']), _np.array(self.env['ssp_depth']), _np.array(self.env['ssp'])
            
            Zb = kwargs.get('vmax', _np.max(self.env['ssp']) * 4) 
            
            Xg = _np.linspace(self.env['rx_range'][0], self.env['rx_range'][-1], Nxy)
            Yg = _np.linspace(0, Y[-1], Nxy)
            Zg = _np.zeros([len(Yg), len(Xg)])
            
            # Bathy
            rb, zb = _np.array(self.env['bot_interface'][:, 0]), _np.array(self.env['bot_interface'][:, 1])
            
            # Re-compute map over grid
            for ii, x in enumerate(Xg):
                interpolation = _np.interp(x, rb, zb)
                for jj, y in enumerate(Yg):
                    y_idx = _np.argmin(_np.abs(Y - y))
                    x_idx = _np.argmin(_np.abs(X - x))
                    Zg[jj, ii] = Z[y_idx, x_idx]
             
            # Plot surface
            for ii, x in enumerate(Xg):
                ylim = _np.interp(x, self.env['top_interface'][:, 0], self.env['top_interface'][:, 1])
                Zg[Yg < ylim, ii] = _np.nan
            ax.plot(self.env['top_interface'][:, 0] / 1000, self.env['top_interface'][:, 1], 'b', linewidth=3)    
    
            # Remove transmission loss in sediment/surface
            interp_y = _np.interp(Xg, self.env['bot_interface'][:, 0], self.env['bot_interface'][:, 1])
            ax.fill_between(Xg / 1000, interp_y, Y[-1], color='brown')
            
            # Plot
            im = ax.imshow(Zg, cmap='jet', aspect='auto', extent=[Xg[0]/1000, Xg[-1]/1000, Yg[-1], Yg[0]], **kwargs)
            ax.plot(rb / 1000, zb, 'k', linewidth=3)
            ax.scatter(0, self.env['tx_depth'], label="Stars", color="r", s=500, marker="*") 
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Sound speed [m/s]')
            ax.set_xlim((self.env['rx_range'][0] / 1000, self.env['rx_range'][-1] / 1000))
            ax.set_ylim((self.env['rx_depth'][0], self.env['rx_depth'][-1]))
            ax.set_xlabel('Range [km]')
            ax.set_ylabel('Depth [m]')
            ax.set_title(f"[BELLHOP - Sound speed profile] {self.env['name']}")
            ax.invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        else: # 1D SSP data
        
            if _np.size(self.env['ssp_depth']) > 1:
                Y, Z = _np.array(self.env['ssp_depth']), _np.array(self.env['ssp'])
            else:
                Y, Z = _np.array([0, _np.max(self.env['bot_interface'][:,1])]), _np.array([self.env['ssp'], self.env['ssp']])
            ax.set_title(f"[BELLHOP - Sound speed profile] {self.env['name']}")
            ax.set_xlim((vmin, vmax))
            ax.invert_yaxis()
            ax.grid(True)
            ax.set_ylabel('Depth [m]')
            ax.set_xlabel('Sound speed [m/s]')
            ax.plot(Z, Y, 'k', linewidth=3)
            plt.tight_layout()
            plt.show()
        
        return fig, ax
        
    def plot_arrivals(self, dB=False, color='steelblue', **kwargs):
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

        t0 = min(self.arrivals.time_of_arrival)
        t1 = max(self.arrivals.time_of_arrival)

        fig, ax = plt.subplots()

        if dB:
            min_y = 20 * _np.log10(_np.max(_np.abs(self.arrivals.arrival_amplitude))) - 60
            ylabel = 'Amplitude [dB]'
        else:
            ylabel = 'Amplitude'
            ax.plot([t0, t1], [0, 0], 'r')
            min_y = 0

        for _, row in self.arrivals.iterrows():
            t = row.time_of_arrival.real
            y = _np.abs(row.arrival_amplitude)

            if dB:
                y = max(20 * _np.log10(_np.finfo(float).eps + y), min_y)

            ax.stem(t, y, linefmt=color, markerfmt=color, basefmt='k')

        ax.set_ylabel(ylabel)
        ax.set_title(f"[BELLHOP - Arrivals] {self.env['name']}")
        ax.set_xlabel('Arrival time [s]')
        ax.grid('all')
        plt.tight_layout()
        plt.show()

        return fig, ax
    
    def plot_rays(self, nRay=100, invert_colors=False, **kwargs):
        """
        Plot rays.

        Parameters:
        - number: Number of rays to plot (default: np.Inf).
        - invert_colors: If True, invert colors of the plot (default: False).

        Returns:
        - fig: The figure object.
        - ax: The axes object.
        """
        
        nInit = nRay
        
        # Sorting rays by bottom bounces in ascending order
        self.rays = self.rays.sort_values('surface_bounces', ascending=True)

        # Determine the maximum amplitude of bottom bounces
        max_amp = _np.max(_np.abs(self.rays.bottom_bounces)) if len(self.rays.bottom_bounces) > 0 else 0
        if max_amp <= 0:
            max_amp = 1

        divisor = 1
        xlabel = 'Range [m]'
        r = []

        # Flatten ray coordinates for determining the range
        for _, row in self.rays.iterrows():
            r += list(row.ray[:, 0])

        # Check if range exceeds 10,000 meters, if so, change divisor and x-label
        if max(r) - min(r) > 10000:
            divisor = 1000
            xlabel = 'Range [km]'

        # Create figure and axes
        fig, ax = plt.subplots()

        # Plot rays
        for _, row in self.rays.iterrows():
            num_bnc = row.bottom_bounces + row.surface_bounces
            if nRay > 0:
                if row.bottom_bounces == 0 and row.surface_bounces == 0:
                    ax.plot(row.ray[:, 0] / divisor, row.ray[:, 1], color='r', alpha=.5)  # Plot direct path
                    nRay -= 1
                elif num_bnc > 1:
                    ax.plot(row.ray[:, 0] / divisor, row.ray[:, 1], color='k', alpha=.5)  # Plot multi-bounce path
                    nRay -= 1
                elif row.surface_bounces == 1:
                    ax.plot(row.ray[:, 0] / divisor, row.ray[:, 1], color='b', alpha=.5)  # Plot surface bounce path
                    nRay -= 1
                elif row.bottom_bounces == 1:
                    ax.plot(row.ray[:, 0] / divisor, row.ray[:, 1], color='g', alpha=.5)  # Plot bottom bounce path
                    nRay -= 1
            else:
                break
            
        # Check if receiver range is negative, if so, flip x-axis
        if self.env['rx_range'] < 0:
            ax.plot(-_np.flip(self.env['bot_interface'][:, 0]) / divisor, _np.flip(self.env['bot_interface'][:, 1]), 'k', linewidth=3)
            ax.plot(-_np.flip(self.env['top_interface'][:, 0]) / divisor, _np.flip(self.env['top_interface'][:, 1]), 'b', linewidth=3)
            ax.scatter(-self.env['rx_range'] / divisor, self.env['rx_depth'], label="Receiver", color="k", s=250, marker="o")
            ax.set_xlim((-self.env['rx_range'] / divisor, 0))
        else:
            ax.plot(self.env['bot_interface'][:, 0] / divisor, self.env['bot_interface'][:, 1], 'k', linewidth=3)
            ax.plot(self.env['top_interface'][:, 0] / divisor, self.env['top_interface'][:, 1], 'b', linewidth=3)
            ax.scatter(self.env['rx_range'] / divisor, self.env['rx_depth'], label="Receiver", color="k", s=250, marker="o")
            ax.set_xlim((0, self.env['rx_range'] / divisor))

        # Set labels, title, and legend
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Depth [m]")
        ax.set_ylim((_np.min(self.env['top_interface'][:,1]), _np.max(self.env['bot_interface'][:, 1])))
        ax.set_title(f"[BELLHOP - Rays ({nInit-nRay})] {self.env['name']}")
        ax.scatter(0, self.env['tx_depth'], label="Source", color="k", s=250, marker="*")

        # Invert y-axis, add grid, and display plot
        ax.invert_yaxis()
        ax.grid('all')
        plt.tight_layout()
        plt.show()

        return fig, ax
    
    def plot_eigen_rays(self, nRay=10, invert_colors=False, **kwargs):
        """
        Plot eigen rays.
        
        Parameters:
        - number: Number of eigen rays to plot (default: 10).
        - invert_colors: If True, invert colors of the plot (default: False).
        
        Returns:
        - fig: The figure object.
        - ax: The axes object.
        """
        
        nInit = nRay
        
        # Sorting rays by bottom bounces in descending order
        self.eigen_rays = self.eigen_rays.sort_values('bottom_bounces', ascending=True)

        # Determine the maximum amplitude of bottom bounces
        max_amp = _np.max(_np.abs(self.eigen_rays.bottom_bounces)) if len(self.eigen_rays.bottom_bounces) > 0 else 0
        max_amp = max_amp if max_amp > 0 else 1

        divisor = 1
        xlabel = 'Range [m]'
        r = []

        # Flatten ray coordinates for determining the range
        for _, row in self.eigen_rays.iterrows():
            r.extend(list(row.ray[:, 0]))

        # Check if range exceeds 10,000 meters, if so, change divisor and x-label
        if max(r) - min(r) > 10000:
            divisor = 1000
            xlabel = 'Range [km]'

        # Create figure and axes
        fig, ax = plt.subplots()

        # Plot each eigen ray
        for _, row in self.eigen_rays.iterrows():
            num_bnc = row.bottom_bounces + row.surface_bounces
            if nRay > 0:
                if row.bottom_bounces == 0 and row.surface_bounces == 0:
                    ax.plot(row.ray[:, 0] / divisor, row.ray[:, 1], color='r', alpha=.5)  # Plot direct path
                    nRay -= 1
                elif num_bnc > 1:
                    ax.plot(row.ray[:, 0] / divisor, row.ray[:, 1], color='k', alpha=.5)  # Plot multi-bounce path
                    nRay -= 1
                elif row.surface_bounces == 1:
                    ax.plot(row.ray[:, 0] / divisor, row.ray[:, 1], color='b', alpha=.5)  # Plot surface bounce path
                    nRay -= 1
                elif row.bottom_bounces == 1:
                    ax.plot(row.ray[:, 0] / divisor, row.ray[:, 1], color='g', alpha=.5)  # Plot bottom bounce path
                    nRay -= 1
            else:
                break

        # Check if receiver range is negative, if so, flip x-axis
        if self.env['rx_range'] < 0:
            ax.plot(-_np.flip(self.env['bot_interface'][:, 0]) / divisor, _np.flip(self.env['bot_interface'][:, 1]), 'k', linewidth=3)
            ax.plot(-_np.flip(self.env['top_interface'][:, 0]) / divisor, _np.flip(self.env['top_interface'][:, 1]), 'b', linewidth=3)
            ax.scatter(-self.env['rx_range'] / divisor, self.env['rx_depth'], label="Receiver", color="k", s=250, marker="o")
            ax.set_xlim((-self.env['rx_range'] / divisor, 0))
        else:
            ax.plot(self.env['bot_interface'][:, 0] / divisor, self.env['bot_interface'][:, 1], 'k', linewidth=3)
            ax.plot(self.env['top_interface'][:, 0] / divisor, self.env['top_interface'][:, 1], 'b', linewidth=3)
            ax.scatter(self.env['rx_range'] / divisor, self.env['rx_depth'], label="Receiver", color="k", s=250, marker="o")
            ax.set_xlim((0, self.env['rx_range'] / divisor))

        # Set labels, title, and legend
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Depth [m]")
        ax.set_ylim((_np.min(self.env['top_interface'][:, 1]), _np.max(self.env['bot_interface'][:, 1])))
        ax.set_title(f"[BELLHOP - Eigen rays ({nInit-nRay})] {self.env['name']}")
        ax.scatter(0, self.env['tx_depth'], label="Source", color="k", s=250, marker="*")

        # Invert y-axis, add grid, and display plot
        ax.invert_yaxis()
        ax.grid('all')
        plt.tight_layout()
        plt.show()

        return fig, ax
    
    def plot_impulse_response(self, dB=False, nArrival=10, color='steelblue', **kwargs):
        """
        Plot impulse response.

        Parameters:
        - dB: If True, convert the impulse response to dB (default: False).
        - nArrival: Number of arrivals to plot (default: 10).
        - color: Color of the plot (default: 'steelblue').

        Returns:
        - fig: The figure object.
        - ax: The axes object.
        """
        # If dB is True, convert the impulse response to dB
        if dB:
            ir = 20 * _np.log10(_np.abs(self.impulse_response) + _np.finfo(float).eps)  # Convert to dB
        else:
            ir = self.impulse_response.real  # Use real part of impulse response

        irlen = 0
        nn    = 0
        for ii,val in enumerate(ir):
            if val != 0:
                nn += 1
                irlen = ii
            if nn == nArrival:
                break
        
        # Plot the impulse response using stem plot
        fig, ax = plt.subplots()
        ax.plot(ir, color=color)  # Plot impulse response
        ax.set_xlim([0, irlen])  # Set x-axis limit
        ax.set_xlabel('Sample [S]')  # Set x-axis label
        ax.set_ylabel('Amplitude')  # Set y-axis label
        ax.set_title(f"[BELLHOP - Impulse response ({nn} @ {self.impulse_response_fs} S/s)] {self.env['name']}")  # Set title
        ax.grid('all')  # Add grid
        plt.tight_layout()  # Adjust layout
        plt.show()  # Show plot

        return fig, ax
    
    def _create_bty_ati_file(self, filename, depth, interp, debug=False):
        with open(filename, 'wt') as f:
            # @todo     Manage 'LL' option for bty
            f.write("'%c'\n" % ('C' if interp == curvilinear else 'L'))
            f.write(str(depth.shape[0])+"\n")
            for j in range(depth.shape[0]):
                f.write("%0.6f %0.6f\n" % (depth[j,0]/1000, depth[j,1]))
        if debug:
            print_file_content(filename)

    def _create_sbp_file(self, filename, dir, debug=False):
        with open(filename, 'wt') as f:
            f.write(str(dir.shape[0])+"\n")
            for j in range(dir.shape[0]):
                f.write("%0.6f %0.6f\n" % (dir[j,0], dir[j,1]))
        if debug:
            print_file_content(filename)

    def _create_ssp_file(self, filename, ssp_range, ssp, debug=False):
        with open(filename, 'wt') as f:
            f.write(str(ssp_range.size)+"\n")
            for j in range(ssp_range.size):
                f.write("%0.6f%c" % (ssp_range[j] /1000, '\n' if j == ssp_range.size-1 else ' '))
            for k in range(ssp.shape[0]):
                for j in range(ssp_range.size):
                    f.write("%0.6f%c" % (ssp[k,j], '\n' if j == ssp_range.size-1 else ' '))
        if debug:
            print_file_content(filename)
            
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

    def _load_shd(self, fname_base, debug=0, **kwargs):
        
        if not _os.path.exists(fname_base+'.shd'):
            print(f"[ERROR] BELLHOP: {fname_base}.shd not found !")
            
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
    
_models.append(('BELLHOP', BELLHOP))


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
        self.depth = kwargs['bot_interface']
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

class KRAKEN:

    def __init__(self, env=None, cp=False):
        
        if env is None:
            self.env           = None
        else:
            self.set_env(env, cp=cp)
        self.transmission_loss = None 
        self.modes             = None
        self.step              = 0

    def set_step(self, step):
        self.step = step

    def set_env(self, env, cp=False, **kwargs):
        
        """Set a new 2D underwater environment.

        A basic environment is created with default values. To see all the parameters
        available and their default values. The environment parameters may be changed 
        by passing keyword arguments or modified later using a dictionary notation.
        """
        if cp:
            self.env =  dict(env)
        else:
            self.env = env
        
        # Numerical box definition
        self.rbox = 1.01*_np.max(_np.abs(env['rx_range']))
        self.zbox = 1.01*_np.max((_np.max(env['bot_interface'][:,-1]), _np.max(env['rx_depth'])))
                
        self.check_env()

        return self.env
    
    def check_env(self):
        # @todo
        return True
    
    def compute_transmission_loss(self, debug=False, **kwargs):
        
        # Assert environment
        self.check_env()
        
        # Define mode taskcode
        taskcode = ''
        TL_L = None
        TL_R = None
        
        # Manage left propagation
        flip = False
        if _np.min(self.env['rx_range']) < 0:
            self.env['rx_range'] = -_np.flip(self.env['rx_range'])
            flip = True
        
            # Generate temporary env file and get base name used for all temporary files
            fname_base = self._create_env_file(taskcode, debug=debug)
            
            if self._kraken(fname_base):
                err = self._check_error(fname_base)
                if err is not None:
                    raise RuntimeError(err) 
                else:
                    try:
                        self._create_flp_file(fname_base, debug=debug)
                        if self._field(fname_base):
                            try:
                                self.modes = self._load_modes(fname_base)
                                TL_L = self._load_shd(fname_base)
                            except FileNotFoundError:
                                raise FileNotFoundError('KRAKEN: Fortran execution did not generate expected output file !')
                    except Exception as e:
                        raise Exception(e)   
                        
                # Delete temporary files
                self._unlink_all(fname_base)  
         

        # Manage right propagation      
         
        # Unflip env
        if flip:
            self.env['rx_range'] = -_np.flip(self.env['rx_range'])
            flip = False
        
        if _np.max(self.env['rx_range']) >= 0:
    
             # Generate temporary env file and get base name used for all temporary files
             fname_base = self._create_env_file(taskcode, debug=debug)
    
             if self._kraken(fname_base):
                 err = self._check_error(fname_base)
                 if err is not None:
                     raise RuntimeError(err) 
                 else:
                     try:
                         self._create_flp_file(fname_base, debug=debug)
                         if self._field(fname_base):
                             try:
                                 self.modes = self._load_modes(fname_base)
                                 TL_R = self._load_shd(fname_base)
                             except FileNotFoundError:
                                 raise FileNotFoundError('KRAKEN: Fortran execution did not generate expected output file !')
    
                     except Exception as e:
                         raise Exception(e)
    
                 # Delete temporary files
                 self._unlink_all(fname_base) 

        if TL_L is not None and TL_R is not None:
            self.transmission_loss = _np.maximum(_np.fliplr(TL_L), TL_R)
        elif TL_L is not None:
            self.transmission_loss = _np.fliplr(TL_L)
        elif TL_R is not None:
            self.transmission_loss = TL_R
        else:
            print("[ERROR] KRAKEN: Transmission loss results are empty !")
            
        return self.transmission_loss

    def compute_modes(self, debug=False, **kwargs):
        
        # Assert environment
        self.check_env()
        
        # Define mode taskcode
        taskcode = ''

        # Manage left propagation limit
        flip = False
        if (_np.abs(_np.min(self.env['rx_range'])) > _np.abs(_np.max(self.env['rx_range']))):
            self.env['rx_range'] = -_np.flip(self.env['rx_range'])
            flip = True
            
        # Generate temporary env file and get base name used for all temporary files
        fname_base = self._create_env_file(taskcode, debug=debug)
        
        # Compute modes
        if self._kraken(fname_base):
            err = self._check_error(fname_base)
            if err is not None:
                raise RuntimeError(err) 
            else:
                try:
                    self.modes = self._load_modes(fname_base)
                except FileNotFoundError:
                    raise FileNotFoundError('KRAKEN: Fortran execution did not generate expected output file !')
        if flip:
            self.env['rx_range'] = -_np.flip(self.env['rx_range'])
            flip = False
            
        # Delete temporary generated files
        self._unlink_all(fname_base)     
        
        return self.modes
    

    def _kraken(self, *args):
        try:
            _proc.run(f'kraken.exe {" ".join(list(args))}', 
                      stderr=_proc.STDOUT, stdout=_proc.PIPE,
                      shell=True)
        except OSError:
            return False
        return True
    
    def _create_flp_file(self, fname_base, debug=False, **kwargs):
                
        with open(fname_base+'.flp', 'wt') as fh:
            
            fh = fh.fileno()
            
            # Title read from .mod file
            self._print(fh, "/")
                    
            # OPT (1:1)
            rx_type = 'R'
            
            # OPT (2:2)
            if self.env['theory'] == coupled:
                th = 'C'
            elif self.env['theory'] == adiabatic:
                th = 'A'
            else:
                print("[WARNING] KRAKEN: Unknown theory, using adiabatic instead !")
                th = 'A'
            
            # OPT (3:3)
            if self.env['tx_beam'] is not None and _np.size(self.env['tx_beam'][:,0]) > 1:
                bp = '*'
                self._create_sbp_file(fname_base+'.sbp', self.env['tx_beam'], debug=debug)
            else:
                bp = 'O'
                
            # OPT (4:4)
            if self.env['mode'] == coherent:
                mode = 'C'
            elif self.env['mode'] == incoherent:
                mode = 'I'
            else:
                print('[WARNING] KRAKEN: Unknown mode, using incoherent instead !')
                mode = 'C'
            
            # Options string
            self._print(fh, "'%c%c%c%c'" % (rx_type, th, bp, mode))
            
            # NUMBER OF MODES (3)
            nmode = self.env['nmode']
            self._print(fh, "%d" % (nmode))
            
            # PROFILE RANGES (4)
            # Nprof
            self._print(fh, "%d" % (1))
            self._print(fh, "%0.6f" % (0.0))
            
            # SOURCE/RECEIVER LOCATIONS (6)
            # NRr R
            self._print(fh,"%d" % (_np.size(self.env['rx_range'])))
            self._print(fh,"%0.6f %0.6f /" % (_np.min((_np.min(self.env['rx_range']/1000), 0)), _np.max(self.env['rx_range']/1000)))
            
            # NSz Sz
            self._print_array(fh, self.env['tx_depth'])
            
            # NRz Rz
            self._print_array(fh, self.env['rx_depth']) 
            
            # NRro
            self._print(fh, "%d" % (_np.size(self.env['rx_depth'])))
            
            # RR
            #self._print(fh, "%0.6f %0.6f" % (_np.min(self.env['rx_range']), _np.max(self.env['rx_range'])))
            self._print(fh, "%0.6f /" % (0.0))
            
        if debug:
            print_file_content(fname_base+'.flp')
    
    def _create_sbp_file(self, fname_base, dir, debug=False):
        
        with open(fname_base+'.sbp', 'wt') as f:
            f.write(str(dir.shape[0])+"\n")
            for j in range(dir.shape[0]):
                f.write("%0.6f %0.6f\n" % (dir[j,0], dir[j,1]))
        if debug:
            print_file_content(fname_base+'.sbp')
            
    def _field(self, fname_base, *args):
        
        try:
            _proc.run(f'field.exe {fname_base}', 
                      stderr=_proc.STDOUT, stdout=_proc.PIPE,
                      shell=True)
        except OSError:
            return False
        return True        

    def _unlink(self, f):
        try:
            _os.unlink(f)
        except:
            try:
                _os.remove(f)
            except:
                pass
        
    def _unlink_all(self, fname_base):
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
            self._unlink(fname_base+'.mod')
            self._unlink(fname_base+'.flp')
            
    def _print(self, fh, s, newline=True):
        _os.write(fh, (s+'\n' if newline else s).encode())

    def _print_array(self, fh, a):
        if _np.size(a) == 1:
            self._print(fh, "1")
            self._print(fh, "%0.6f /" % (a))
        else:
            self._print(fh, str(_np.size(a)))
            self._print(fh, "%0.6f %0.6f " % (a.min(), a.max()), newline=False)
            #for j in a:
            #   self._print(fh, "%0.6f " % (j), newline=False)
            self._print(fh, "/")

    def _create_env_file(self, taskcode, debug=0, **kwargs):
        
        fh, fname = _mkstemp(suffix='.env')
        fname_base = fname[:-4]
                
        # Title
        self._print(fh, "'"+self.env['name']+"'")
        
        # Freq
        self._print(fh, "%0.6f" % (self.env['tx_freq']))
        
        # Nmedia
        if(self.env['nmedia'] > 1):
            print("[WARNING] KRAKEN: Multiple media not yet coded, using only the first one.")
        self._print(fh, "%d" % (1))
        
        # Option (1:1) SSP interp      
        if self.env['ssp_interp'] == spline:
            ssp_interp = 'S'
        elif self.env['ssp_interp'] == c_linear:
            ssp_interp = 'C' 
        elif self.env['ssp_interp'] == n2_linear:
            ssp_interp = 'N'
        elif self.env['ssp_interp'] == analytic:
            ssp_interp = 'A'
        else:
            print("[WARNING] KRAKEN: Unknown ssp interpolation method, using c-linear instead !")
            ssp_interp = 'C'
            
          
        # Option (2:2) Top boundary condition
        # @todo     Manage all boundary conditions.
        if self.env['top_boundary'] == rigid:
            topBdry = 'R'
        elif self.env['top_boundary'] == vacuum:
            topBdry = 'V' 
        elif self.env['top_boundary'] == acousto_elastic:
            topBdry = 'A' 
        elif self.env['top_boundary'] == file:
            # @todo
            raise Exception('KRAKEN: Rreflection coefficient from file not available, it need to be coded !') 
        elif self.env['top_boundary'] == soft_boss:
            topBdry = 'S' 
        elif self.env['top_boundary'] == hard_boss:
            topBdry = 'H' 
        elif self.env['top_boundary'] == soft_boss_amp:
            topBdry = 'T' 
        elif self.env['top_boundary'] == hard_boss_amp:
            topBdry = 'I' 
        else:
            print('[WARNING] KRAKEN: Unknown top boundary condition, using vacuum instead !')
            topBdry = 'V'
            
        # Option (3:3) Attenuation units
        # Kraken ignores material attenuation in elastic media, Krakenc treats it properly.
        attnUnit = 'W' # Unused but necessary option
        
        # Option (4:4)
        if self.env['volume_attn'] is None:
            vAttn = ''
        elif self.env['volume_attn'] == Francois_Garrison:
            print('[WARNING] KRAKEN: Francois Garrison attenuation formula not supported, using Thorp formula instead !')
            vAttn = 'T'
        elif self.env['volume_attn'] == biological:
            print('[WARNING] KRAKEN: Biological attenuation formula not supported, using Thorp formula instead !')
            vAttn = 'T'
        elif self.env['volume_attn'] == Thorp:
            vAttn = 'T'
        else:
            print('[WARNING] KRAKEN: Unknown volume attenuation, using Thorp formula instead !')
            vAttn = 'T'
        
        # Option (5:5)
        # Available in KRAKENC only
        
        # All options string
        if vAttn != '':
            self._print(fh, "'%c%c%c%c'" % (ssp_interp, topBdry, attnUnit, vAttn))
        else:
            self._print(fh, "'%c%c%c'" % (ssp_interp, topBdry, attnUnit))
        
        # Top boundary extra line (4a) or (4c)            
        if topBdry == 'A':
            self._print(fh, "%0.6f %0.6f %0.6f %0.6f %0.6f %0.6f" % (_np.max(self.env['bot_interface']), self.env['top_PwaveSpeed'], self.env['top_SwaveSpeed'], self.env['top_density'], self.env['top_PwaveAttn'], self.env['top_SwaveAttn']))
        elif topBdry == 'F' or topBdry == 'I':
            self._print(fh, "%0.6f %0.6f %0.6f" % (self.env['twy_bumpDensity'], self.env['twy_radius1'], self.env['twy_radius2']))
        
        # Medium info (5)
        # @todo     Manage manual NMESH
        # @todo     Add bottom_idepth for interface depth to use with roughness ?
        
        nmesh  = 0 # Automatic mesh points calculation
        '''
        for media in range(env['nmedia']): # len(bottom_idepth ?)

            # Water as first media
            if media == 0: 
                
                nmesh  = 0 # Automatic mesh points calculation
                sigma  = 0 # ??
                z_nssp = 0 # max_depth ??
                
                self._print(fh, "%d %0.6f %0.6f" % (nmesh, sigma, z_nssp))
                
                # Sound speed profile (5a)
                if _np.size(env['ssp'],axis=1) > 1:
                    print(f"[INFO] {env['model']}: Multiple sound profiles not supported, using median value.")
                    mn = _np.median(env['ssp'], axis=1)
                    ssp = _np.column_stack((env['ssp_depth'], mn))
                else:
                    ssp = _np.column_stack((env['ssp_depth'], env['ssp']))
                
                if ssp_interp != 'A':
                    pass
                
            # Other bottom medium
            else:
                pass
            ssp_depth = self.env['ssp_depth']
        '''
        
        
        if _np.size(self.env['ssp_range']) > 1:
            print("[WARNING] KRAKEN: Range dependant ssp not supported, using median values instead !")
            try:
                ssp = _np.median(self.env['ssp'], axis=1)
            except:
                try:
                    ssp = _np.median(self.env['ssp'], axis=0)
                except:
                    ssp = self.env['ssp']
        else:
            ssp = self.env['ssp']
        
        if _np.size(self.env['ssp_depth']) > 1 and _np.size(self.env['ssp']) > 1:      
            self._print(fh, "%d %0.1f %0.6f" % (nmesh, self.env['top_roughness'], _np.max(self.env['ssp_depth'])))
            firstLine = True
            for j in range(self.env['ssp_depth'].size):
                if firstLine:
                    firstLine = False
                    self._print(fh, "%0.6f %0.6f 0.0 %0.6f 0.0 0.0 /" % (_np.abs(self.env['ssp_depth'][j]), ssp[j], self.env['water_density']))
                else:
                    self._print(fh, "%0.6f %0.6f /" % (_np.abs(self.env['ssp_depth'][j]), ssp[j]))
                    
        else:
            self._print(fh, "%0.6f %0.6f /" % (_np.min(self.env['rx_depth']), ssp))
            self._print(fh, "%0.6f %0.6f /" % (_np.max(self.env['rx_depth']), ssp))
        
        # Bottom option (6)
        if self.env['bot_boundary'] == rigid:
            botBdry = 'R'
        elif self.env['bot_boundary'] == vacuum:
            botBdry = 'V' 
        elif self.env['bot_boundary'] == acousto_elastic:
            botBdry = 'A'
        elif self.env['bot_boundary'] == file:
            #botBdry = 'F'
            # @todo 
            print('[WARNING] KRAKEN: File bottom boundary condition not yet coded, using perfectly rigid instead !')
            botBdry = 'R'
        elif self.env['bot_boundary'] == grain_size:
            #botBdry = 'G'   
            # @todo 
            print('[WARNING] KRAKEN: Grain size bottom boundary condition not yet coded, using perfectly rigid instead !')
            botBdry = 'R'
        elif self.env['bot_boundary'] == precalculated:
            #botBdry = 'P'   
            # @todo
            print('[WARNING] KRAKEN: Precalculated bottom boundary condition not yet coded, using perfectly rigid instead !')
            botBdry = 'R'
       
        # Bottom options string
        # if self.env['bot_interface'].ndim == 2:
        #     self._print(fh, "'%c*' %0.6f" % (botBdry, self.env['bot_roughness']))
        # else:
        self._print(fh, "'%c' %0.6f" % (botBdry, self.env['bot_roughness']))
        
        # Bottom halfspace extra lines (6a) (6b)
        # @todo     Add Grain size 
        if  _np.size(self.env['bot_PwaveSpeed']) > 1:
            print("[INFO] KRAKEN: Do not support multiple Pwave speed definition, using median value instead.")
            bot_PwaveSpeed = _np.median(self.env['bot_PwaveSpeed'])
        else:
            bot_PwaveSpeed = self.env['bot_PwaveSpeed']
        
        if  _np.size(self.env['bot_SwaveSpeed']) > 1:
            print("[INFO] KRAKEN: Do not support multiple Swave speed definition, using median value instead.")
            bot_SwaveSpeed = _np.median(self.env['bot_SwaveSpeed'])
        else:
            bot_SwaveSpeed = self.env['bot_SwaveSpeed']
            
        if  _np.size(self.env['bot_density']) > 1:
            print("[INFO] KRAKEN: Do not support multiple bottom density definition, using median value instead.")
            bot_density = _np.median(self.env['bot_density'])
        else:
            bot_density = self.env['bot_density']    
            
        if  _np.size(self.env['bot_PwaveAttn']) > 1:
            print("[INFO] KRAKEN: Do not support multiple Pwave attn definition, using median value instead.")
            bot_PwaveAttn = _np.median(self.env['bot_PwaveAttn'])
        else:
            bot_PwaveAttn = self.env['bot_PwaveAttn']   
            
        if  _np.size(self.env['bot_SwaveAttn']) > 1:
            print("[INFO] KRAKEN: Do not support multiple Swave speed definition, using median value instead.")
            bot_SwaveAttn = _np.median(self.env['bot_SwaveAttn'])
        else:
            bot_SwaveAttn = self.env['bot_SwaveAttn']     
     
        if botBdry == 'A':
            self._print(fh, "%0.6f %0.6f %0.6f %0.6f %0.6f %0.6f" % (_np.max(self.env['bot_interface']), bot_PwaveSpeed, bot_SwaveSpeed, bot_density, bot_PwaveAttn, bot_SwaveAttn))

        # C0 min and max (c0min = 0 => automatic calculation mode)
        self._print(fh, "%0.6f %0.6f" %(0, 999999999))                           
        
        # Max range in km self._print_array(fh, self.env['rx_depth'])
        self._print(fh, "%0.6f" %(self.rbox/1000))                                         
        
        # Number of source depth  and source depths (m) (7)
        self._print_array(fh, self.env['tx_depth'])         
         
        # Number of receiver depth  and receiver depths (m) (7)
        self._print_array(fh, self.env['rx_depth'])      

        # Number of receiver range and receiver range (km) (7)
        self._print_array(fh, self.env['rx_range']/1000)
               
        _os.close(fh)
        
        if debug:
            print_file_content(fname_base+'.env')
            
        return fname_base


    def plot_ssp(self, **kwargs):
        
        fig, ax = plt.subplots()
        vmax = kwargs.get('vmax', _np.max(self.env['ssp'])+_np.abs(_np.mean(self.env['ssp']))/100)
        vmin = kwargs.get('vmin', _np.min(self.env['ssp'])-_np.abs(_np.mean(self.env['ssp']))/100)
        
        if _np.size(_np.array(self.env['ssp_range'])) > 1:
            print("[INFO] KRAKEN: Do not support range dependant sound speed, using median values instead !")
            try:
                ssp = _np.median(self.env['ssp'], axis=1)
            except:
                ssp = _np.median(self.env['ssp'], axis=0)
        else:
            ssp = self.env['ssp']
            
        Y, Z = _np.array(self.env['ssp_depth']), _np.array(ssp)
        ax.set_title(f"[KRAKEN - Sound speed profile] {self.env['name']}")
        ax.set_xlim((vmin, vmax))
        ax.invert_yaxis()
        ax.grid(True)
        ax.set_ylabel('Depth [m]')
        ax.set_xlabel('Sound speed [m/s]')
        ax.plot(Z, Y, 'k', linewidth=3)
        plt.tight_layout()
        plt.show()
            
        return fig, ax
    
    def plot_transmission_loss(self, vmin=-120, vmax=0, **kwargs):
        """
        Plots the transmission loss of the environment.

        Parameters:
            vmin (float): Minimum value for the transmission loss.
            vmax (float): Maximum value for the transmission loss.
            **kwargs: Additional keyword arguments for customization.

        Returns:
            fig, ax: Figure and axis objects for the plot.
        """
        fig, ax = plt.subplots()
        X = self.env['rx_range']
        Y = self.env['rx_depth']

        tlossplt = 20 * _np.log10(_np.finfo(float).eps + _np.abs(_np.array(self.transmission_loss)))

        # Remove transmission loss in sediment/surface
        for i, x in enumerate(X):
            ylim = _np.interp(x, self.env['bot_interface'][:, 0], self.env['bot_interface'][:, 1])
            for j, y in enumerate(Y):
                if y > ylim:
                    tlossplt[j, i] = vmax
                    
        # Remove transmission loss in sediment/surface
        interp_y = _np.interp(X, self.env['bot_interface'][:, 0], self.env['bot_interface'][:, 1])
        ax.fill_between(X, _np.mean(self.env['bot_interface'][:,1]), Y[-1], color='brown')
        
        # Plot the surface
        ax.plot([X[0]/1000, X[-1]/1000], [0, 0], 'b', linewidth=3)

        # Plot the transmission loss map using imshow
        im1 = ax.imshow(tlossplt, extent=[X[0] / 1000, X[-1] / 1000, Y[-1], Y[0]], cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')

        # Plot the bottom interface
        ax.plot([X[0]/1000, X[-1]/1000], [_np.mean(self.env['bot_interface'][:,1]), _np.mean(self.env['bot_interface'][:,1])], 'k', linewidth=3)

        # Set plot properties
        ax.set_xlim((X[0] / 1000, X[-1] / 1000))
        ax.set_ylim((Y[0], Y[-1]))
        ax.set_xlabel('Range [km]')
        ax.set_ylabel('Depth [m]')
        ax.set_title(f"[KRAKEN - Transmission loss @ {self.env['tx_freq']} Hz] {self.env['name']}")

        # Add color bar
        cbar1 = fig.colorbar(im1, ax=ax)
        cbar1.ax.set_ylabel('Loss [dB]')

        # Invert y-axis for depth
        ax.invert_yaxis()

        # Adjust layout and display plot
        plt.tight_layout()
        plt.show()

        return fig, ax
    # @todo     Remove useless file creation.


    def plot_modes(self, n=10, vmin=-0.2, vmax=0.2):
        
        fig, ax = plt.subplots()
        
        ax.plot(_np.real(self.modes.phi[:,0:n]), self.modes.z)
        ax.set_ylabel('Depth [m]')
        ax.grid()
        ax.set_ylim([0,_np.max(self.env['bot_interface'][:,1])])
        ax.set_xlim([vmin, vmax])
        ax.set_title(f"[KRAKEN - Modes ({n})] {self.env['name']}")
        ax.invert_yaxis()
        
        return fig, ax
            
    def _create_bty_ati_file(self, filename, depth, interp):
        with open(filename, 'wt') as f:
            f.write("'%c'\n" % ('C' if interp == curvilinear else 'L'))
            f.write(str(depth.shape[0])+"\n")
            for j in range(depth.shape[0]):
                f.write("%0.6f %0.6f\n" % (depth[j,0]/1000, depth[j,1]))
                

    def _create_sbp_file(self, filename, dir, debug=False):
        with open(filename, 'wt') as f:
            f.write(str(dir.shape[0])+"\n")
            for j in range(dir.shape[0]):
                f.write("%0.6f %0.6f\n" % (dir[j,0], dir[j,1]))
        if debug:
            print_file_content(filename)

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
                
        if not _os.path.exists(filename):
            print(f"[ERROR] KRAKEN: {filename} not found! !")
    
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

        input_dict = {'M':M, 'modes_k': modes_k, 'z':z, 'modes_phi':modes_phi, 'top':top, 'bot':bot, 'N':N, 'Mater':Mater, 'Nfreq': Nfreq, 'Nmedia': Nmedia, 'bot_interface':depth, 'rho':rho, 'freqVec':freqVec}
        modes = Modes(**input_dict)
        return modes

    def _load_shd(self, fname_base, debug=0, *args, **kwargs):
        '''
        Code imported and modified from pyat.
        '''
        
        filename = fname_base+'.shd'
        if not _os.path.exists(filename):
            print(f"[ERROR] KRAKEN: {filename} not found !")

        # optional frequency
        freq = _np.NaN
    
        # optional source (x,y) coordinate
        xs = _np.NaN
        ys = _np.NaN
            
        with open( filename, 'rb' ) as f:
    
            recl     = unpack('<I', f.read(4))[0];     #record length in bytes will be 4*recl
            title    = unpack('80s', f.read(80))
        
            f.seek(4 * recl); #reposition to end of first record
            PlotType = unpack('10s', f.read(10))
        
            f.seek(2 * 4 * recl); #reposition to end of second record
            Nfreq, Ntheta, Nsx, Nsy, Nsd, Nrd, Nrr, atten = _unpack('iiiiiiif', f.read(32))
            
            f.seek(3 * 4 * recl); #reposition to end of record 3
            freqVec = unpack(str(Nfreq) +'d', f.read(Nfreq*8))
        
            f.seek(4 * 4 * recl) ; #reposition to end of record 4
            theta   = unpack(str(Ntheta) +'f', f.read(4*Ntheta))[0]
        
            if ( PlotType[ 1 : 2 ] != 'TL' ):
                f.seek(5 * 4 * recl); #reposition to end of record 5
                x     = unpack(str(Nsx)+'f',  f.read(Nsx*4))
                f.seek( 6 * 4 * recl); #reposition to end of record 6
                y     = unpack(str(Nsy) + 'f', f.read(Nsy*4))
            else:   # compressed format for TL from FIELD3D
                f.seek(5 * 4 * recl, -1 ); #reposition to end of record 5
                x     = f.read(2,    'float32' )
                x     = _np.linspace( x[0], x[-1], Nsx )
                f.seek(6 * 4 * recl, -1 ); #reposition to end of record 6
                y     = f.read(2,    'float32' )
                y     = _np.linspace( y[0], y[-1], Nsy )
        
            # Source depth
            f.seek(7 * 4 * recl); #reposition to end of record 7
            sdepth = unpack(str(Nsd)+'f', f.read(Nsd*4))
            sdepth = _np.array(sdepth)
        
            # Receiver depth
            f.seek(8 * 4 * recl); #reposition to end of record 8
            rdepth = _np.fromfile(f, dtype=_np.float32, count=Nrd)
            rdepth = rdepth.reshape((1,-1))
        
            # Receiver range
            f.seek(9 * 4 * recl); #reposition to end of record 9
            rrange = _np.fromfile(f, dtype=_np.float64, count=Nrr)
            rrange = rrange.reshape((1,-1))

            ##
            # Each record holds data from one source depth/receiver depth pair
        
            if PlotType == 'rectilin  ':
                pressure = _np.zeros(( Ntheta, Nsd, Nrd, Nrr ), dtype=_np.complex128)
                Nrcvrs_per_range = Nrd
            if PlotType == 'irregular ':
                pressure = _np.zeros(( Ntheta, Nsd,   1, Nrr ), dtype=_np.complex128)
                Nrcvrs_per_range = 1
            else:
                pressure = _np.zeros(( Ntheta, Nsd, Nrd, Nrr ), dtype=_np.complex128)
                Nrcvrs_per_range = Nrd
        
            ##
            if _np.isnan( xs ):    # Just read the first xs, ys, but all theta, sd, and rd
                # get the index of the frequency if one was selected
                ifreq = 0
                if not _np.isnan(freq):
                   freqdiff = [abs( x - freq ) for x in freqVec]
                   ifreq = min( freqdiff )
        
                for itheta in range (Ntheta):
                    for isd in range(Nsd):
                        # disp( [ 'Reading data for source at depth ' num2str( isd ) ' of ' num2str( Nsd ) ] )
                        for ird in range( Nrcvrs_per_range):
                            recnum = 10 + ( ifreq   ) * Ntheta * Nsd * Nrcvrs_per_range + \
                                          ( itheta  )          * Nsd * Nrcvrs_per_range + \
                                          ( isd     )                * Nrcvrs_per_range + \
                                            ird    
                            status = f.seek(int(recnum) * 4 * recl); #Move to end of previous record
                            if ( status == -1 ):
                                raise ValueError( 'Seek to specified record failed in read_shd_bin' )
                            temp = unpack(str(2*Nrr)+'f', f.read(2 * Nrr*4));    #Read complex data
                            pressure[ itheta, isd, ird, : ] = temp[ 0 : 2 * Nrr -1 : 2 ] + complex(0,1) *_np.array((temp[ 1 : 2 * Nrr :2]))
                            # Transmission loss matrix indexed by  theta x sd x rd x rr
                            
            else:              # read for a source at the desired x, y, z.
                
                xdiff = abs( x[0] - xs * 1000.0 )
                [ holder, idxX ] = min( xdiff )
                ydiff = abs( y - ys * 1000.0 )
                [ holder, idxY ] = min( ydiff )
                
                # show the source x, y that was found to be closest
                # [ x( idxX ) y( idxY ) ]
                for itheta in range(Ntheta):
                    for isd in range(Nsd):
                        # disp( [ 'Reading data for source at depth ' num2str( isd ) ' of ' num2str( Nsd ) ] )
                        for ird in range(Nrcvrs_per_range):
                            recnum = 10 + ( idxX   - 1 ) * Nsy * Ntheta * Nsd * Nrcvrs_per_range +   \
                                          ( idxY   - 1 )       * Ntheta * Nsd * Nrcvrs_per_range +  \
                                          ( itheta - 1 )                * Nsd * Nrcvrs_per_range +  \
                                          ( isd    - 1 )                      * Nrcvrs_per_range + ird - 1
                            status = f.seek(recnum * 4 * recl); # Move to end of previous record
                            if ( status == -1 ):
                                raise ValueError( 'Seek to specified record failed in read_shd_bin' )
                            
                            temp = f.read(2 * Nrr, 'float32' );    #Read complex data
                            pressure[ itheta, isd, ird, : ] = temp[ 1 : 2 : 2 * Nrr ] + complex(0,1) * _np.array(temp[ 2 : 2 : 2 * Nrr ])
                            # Transmission loss matrix indexed by  theta x sd x rd x rr
                            
        return _np.nan_to_num(pressure[0,0,:,:], nan=-_np.inf)
        

_models.append(('KRAKEN', KRAKEN))


        
class RAM:
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
    
    def __init__(self, env=None, cp=False):
              
        self.transmission_loss = None 
        self.step              = 0  # 0:Automatic => lambda/8
        self.PadeTerm          = 8
        self.attnLayerWidth    = 20 # wavelength
        self.nStabConst        = 1
        self.maxStabRange      = 0  # Automatic => rbox
        
        if env is None:
            self.env           = None
        else:
            self.set_env(env, cp=cp)

        

    def set_env(self, env, cp=False, **kwargs):
        
        if cp:
            self.env =  dict(env)
        else:
            self.env = env
        
        if hasattr(self, 'step') and self.step is not None and self.step > 0:
            step  = self.step
        else:
            step = _np.mean(self.env['ssp'])/self.env['tx_freq']/8 # lambda/8
            self.step = 0
        
        if _np.min(_np.abs(self.env['rx_range'])) != 0:          
            neg_min = _np.max(self.env['rx_range'][self.env['rx_range'] < 0])
            pos_min = _np.min(self.env['rx_range'][self.env['rx_range'] > 0])
            if _np.abs(neg_min) > pos_min:
                err = neg_min
            else:
                err = pos_min
            print(f"[WARNING] RAM: Output grid range is not 0 centered, maximum relative error = {err:.3f} m !")   
         
        if _np.min(_np.abs(self.env['rx_depth'])) != 0:          
            err = _np.max(self.env['rx_depth'][self.env['rx_depth'] < 0])
            print(f"[WARNING] RAM: Output grid depth is not 0 centered, maximum relative error = {err:.3f} m !")   
            
        # Pad bottom settings to minimum required size of 2x2
        bot_PwaveSpeed     = self.env['bot_PwaveSpeed']
        bot_attn    = self.env['bot_PwaveAttn']
        bot_density = self.env['bot_density']
        bot_range   = self.env['bot_range']
        bot_depth   = self.env['bot_depth']
        if _np.size(self.env['bot_range']) == 1 and _np.size(self.env['bot_depth']) == 1:
            if _np.size(self.env['bot_PwaveSpeed']) == 1:
                bot_PwaveSpeed = _np.array([[self.env['bot_PwaveSpeed'], self.env['bot_PwaveSpeed']],
                                     [self.env['bot_PwaveSpeed'], self.env['bot_PwaveSpeed']]])
            if _np.size(self.env['bot_PwaveAttn']) == 1:
                bot_attn = _np.array([[self.env['bot_PwaveAttn'], self.env['bot_PwaveAttn']],
                                      [self.env['bot_PwaveAttn'], self.env['bot_PwaveAttn']]])
            if _np.size(self.env['bot_density']) == 1:
                bot_density = _np.array([[self.env['bot_density'], self.env['bot_density']],
                                         [self.env['bot_density'], self.env['bot_density']]])
            bot_range = _np.array([_np.min(self.env['rx_range']), _np.max(self.env['rx_range'])])
            bot_depth = _np.array([_np.min(self.env['bot_interface'][:,1]), _np.max((_np.max(self.env['rx_depth']), _np.max(self.env['bot_interface'][:,1])))])
          
        # Positive depth only
        if _np.size(_np.where(self.env['rx_depth'] > 0)):
            dMin = _np.where(self.env['rx_depth'] > 0)[0][0]
        else: 
            dMin = _np.size(self.env['rx_depth'])-1
        ratio_d = 1-dMin/_np.size(self.env['rx_depth'])
            
        # Right propagation
        if _np.size(_np.where(self.env['rx_range'] >= 0)):
            iiMin = _np.where(self.env['rx_range'] >= 0)[0][0]
        else: 
            iiMin = _np.size(self.env['rx_range'])-1
        ratio = 1-iiMin/_np.size(self.env['rx_range'])
  
        if _np.size(self.env['rx_range']) > 1:
            ndr = _np.max((_np.ceil((self.env['rx_range'][1]-self.env['rx_range'][0])/step*ratio), 1))
            dr = (self.env['rx_range'][1]-self.env['rx_range'][0])/ndr
        else:
            ndr = _np.max((_np.ceil((self.env['rx_range'][iiMin:])/step*ratio), 1))
            dr = self.env['rx_range']/ndr
            
        if _np.size(self.env['rx_depth']) > 1:
            ndz = _np.max((_np.ceil((self.env['rx_depth'][1]-self.env['rx_depth'][0])/step*ratio_d), 1))
            dz = (self.env['rx_depth'][1]-self.env['rx_depth'][0])/ndz
        else:
            ndz = _np.max((_np.ceil((self.env['rx_depth'])/step*ratio_d), 1))
            dz = self.env['rx_depth']/ndz
    
        self.rbox  = ndr*dr*_np.size(self.env['rx_range'])*ratio+1e-6
        self.zbox  = ndz*dz*_np.size(self.env['rx_depth'])*ratio_d+1e-6

        if self.maxStabRange == 0:
            self.maxStabRange = self.rbox
            rs_auto = True
        else:
            rs_auto = False
            
        self.pyramR = ram.PyRAM(self.env['tx_freq'],
                           self.env['tx_depth'],
                           _np.max(self.env['rx_depth']),
                           _np.array(self.env['ssp_depth']),
                           _np.array(self.env['ssp_range']),
                           _np.array(self.env['ssp'], ndmin=2),
                           _np.array(bot_depth),
                           _np.array(bot_range),
                           _np.array(bot_PwaveSpeed,ndmin=2),
                           _np.array(bot_density,ndmin=2),
                           _np.array(bot_attn,ndmin=2),
                           _np.array(self.env['bot_interface'],ndmin=2),
                           rmax  = self.rbox,
                           dr    = dr,
                           dz    = dz,
                           ndr   = ndr,
                           ndz   = ndz,
                           zmplt = self.zbox,
                           np    = self.PadeTerm,
                           lyrw  = self.attnLayerWidth,
                           ns    = self.nStabConst,
                           rs    = self.maxStabRange
                           )
        
        # Left propagation  
        ratio = 1-ratio
                
        if _np.size(self.env['rx_range']) > 1:
            ndr = _np.max((_np.floor((self.env['rx_range'][1]-self.env['rx_range'][0])/step*ratio), 1))
            dr = (self.env['rx_range'][1]-self.env['rx_range'][0])/ndr
        else:
            ndr = _np.max(_np.floor(((self.env['rx_range'])/step*ratio), 1))
            dr = self.env['rx_range']/ndr

        self.rbox  = ndr*dr*_np.size(self.env['rx_range'])*ratio+1e-6
        self.zbox  = ndz*dz*_np.size(self.env['rx_depth'])*ratio_d+1e-6
        
        if rs_auto:
            self.maxStabRange = self.rbox
            
        self.pyramL = ram.PyRAM(self.env['tx_freq'],
                           self.env['tx_depth'],
                           _np.max(self.env['rx_depth']),
                           _np.array(self.env['ssp_depth']),
                           _np.flip(-_np.array(self.env['ssp_range'])),
                           _np.fliplr(_np.array(self.env['ssp'],ndmin=2)),
                           _np.array(bot_depth),
                           _np.flip(-_np.array(bot_range)),
                           _np.fliplr(_np.array(bot_PwaveSpeed,ndmin=2)),
                           _np.fliplr(_np.array(bot_density,ndmin=2)),
                           _np.fliplr(_np.array(bot_attn,ndmin=2)),
                           _np.column_stack((_np.flip(-self.env['bot_interface'][:,0]), _np.flip(self.env['bot_interface'][:,1]))),
                           rmax  = self.rbox,
                           dr    = dr,
                           dz    = dz,
                           ndr   = ndr,
                           ndz   = ndz,
                           zmplt = self.zbox,
                           np    = self.PadeTerm,
                           lyrw  = self.attnLayerWidth,
                           ns    = self.nStabConst,
                           rs    = self.maxStabRange
                           )
        
    def set_step(self, step):
        self.step = step
        self.set_env(self.env)
        
    def check_env(self):
        # Right propagation
        iiMin = _np.where(self.env['rx_range'] >= 0)[0][0]
        ratio = 1-iiMin/_np.size(self.env['rx_range'])
        if self.env['mesh_computeBox'] == 'manual':
            step  = self.step
        else:
            step = _np.mean(self.env['ssp'])/self.env['tx_freq']/8 # lambda/8
        if _np.size(self.env['rx_range']) > 1:
            ndr = _np.max((_np.round((self.env['rx_range'][1]-self.env['rx_range'][0])/(step)*ratio), 1))
            dr = (self.env['rx_range'][1]-self.env['rx_range'][0])/ndr
        else:
            ndr = _np.max((_np.round((self.env['rx_range'])/(step)*ratio), 1))
            dr = self.env['rx_range']/ndr
        if _np.size(self.env['rx_depth']) > 1:
            ndz = _np.max((_np.round((self.env['rx_depth'][1]-self.env['rx_depth'][0])/(step)), 1))
            dz = (self.env['rx_depth'][1]-self.env['rx_depth'][0])/ndz
        else:
            ndz = _np.max((_np.round((self.env['rx_depth'])/(step)), 1))
            dz = self.env['rx_depth']/ndz

        
        return self.pyramR.check_inputs(self.env['tx_freq'],
                           self.env['tx_depth'],
                           self.env['rx_depth'][-1],
                           _np.array(self.env['ssp_depth']),
                           _np.array(self.env['ssp_range']),
                           _np.array(self.env['ssp']),
                           _np.array(self.env['bot_depth']),
                           _np.array(self.env['bot_range']),
                           _np.array(self.env['bot_PwaveSpeed'],ndmin=2),
                           _np.array(self.env['bot_density'],ndmin=2),
                           _np.array(self.env['bot_PwaveAttn'],ndmin=2),
                           _np.array(self.env['bot_interface'],ndmin=2),
                           rmax  = ndr*dr*(_np.size(self.env['rx_range'])*ratio),
                           dr    = dr,
                           dz    = dz,
                           ndr   = ndr,
                           ndz   = ndz,
                           zmplt = ndz*dz*_np.size(self.env['rx_depth'])
                           )
    
    
    def compute_transmission_loss(self, debug=False):
        """
        Compute Transmission Loss.
        """
        
        # Check if there are multiple receiver ranges or depths
        if _np.size(self.env['rx_range']) > 1 or _np.size(self.env['rx_depth']) > 1:
            
            tlgL       = None
            tlgR       = None
            replicated = False

            # Right propagation
            if _np.max(self.env['rx_range']) > 0:    
                self.pyramR.run()
                if _np.any(self.env['rx_range'] == 0):
                    replicated_column = self.pyramR.tlg[:, [0]]
                    tlgR = _np.hstack((replicated_column, self.pyramR.tlg))
                    replicated_column = self.pyramR.cpg[:, [0]]
                    cpgR = _np.hstack((replicated_column, self.pyramR.cpg))
                    replicated = True
                else:
                    tlgR = self.pyramR.tlg
                    cpgR = self.pyramR.cpg
             
            # Left propagation
            if _np.min(self.env['rx_range']) < 0:
                self.pyramL.run()
                if _np.any(self.env['rx_range'] == 0) and not replicated:
                    replicated_column = self.pyramL.tlg[:, [0]]
                    tlgL = _np.hstack((replicated_column, self.pyramL.tlg))
                    replicated_column = self.pyramL.cpg[:, [0]]
                    cpgL = _np.hstack((replicated_column, self.pyramL.cpg))
                else:
                    tlgL = self.pyramL.tlg
                    cpgL = self.pyramL.cpg
        
        # Combine the transmission loss matrices
        if tlgL is not None and tlgR is not None:
            tlg = _np.hstack((_np.fliplr(tlgL), tlgR))
            cpg = _np.hstack((_np.fliplr(cpgL), cpgR))
        elif tlgL is not None:
            tlg = _np.fliplr(tlgL)
            cpg = _np.fliplr(cpgL)
        elif tlgR is not None:
            tlg = tlgR
            cpg = cpgR
        else:
            print("[ERROR] RAM: No results found !")
         
        # Handle cases where the result matrix depth is smaller than expected   
        if _np.size(tlg[:,0]) < _np.size(self.env['rx_depth']):
            tlg = _np.vstack((tlg[0,:],tlg))
            cpg = _np.vstack((cpg[0,:],cpg))
        num_columns = tlg.shape[1]
        nan_array = _np.empty((1, num_columns))
        nan_array[:] = _np.nan        
        while _np.size(tlg[:,0]) < _np.size(self.env['rx_depth']):
            tlg = _np.vstack((nan_array,tlg))
            cpg = _np.vstack((nan_array,cpg))
                
        # Compute transmission loss and complex pressure
        self.transmission_loss = 10**(-tlg/20)
        self.complex_pressure  = cpg
                
        return self.transmission_loss

    def compute_cp(self, debug=False):
        """
        Compute complex pressure. Does not include cylindrical spreading term 1/sqrt(r) or phase term exp(-j*k0*r).
        
        Parameters:
            debug (bool): If True, print debug information.
            
        Returns:
            array-like: Complex pressure.
        """
        
        # Check if there are multiple receiver ranges or depths
        if _np.size(self.env['rx_range']) > 1 or _np.size(self.env['rx_depth']) > 1:
            
            tlgL       = None
            tlgR       = None
            replicated = False

            # Right propagation
            if _np.max(self.env['rx_range']) > 0:    
                self.pyramR.run()
                if _np.any(self.env['rx_range'] == 0):
                    replicated_column = self.pyramR.tlg[:, [0]]
                    tlgR = _np.hstack((replicated_column, self.pyramR.tlg))
                    replicated_column = self.pyramR.cpg[:, [0]]
                    cpgR = _np.hstack((replicated_column, self.pyramR.cpg))
                    replicated = True
                else:
                    tlgR = self.pyramR.tlg
                    cpgR = self.pyramR.cpg
             
            # Left propagation
            if _np.min(self.env['rx_range']) < 0:
                self.pyramL.run()
                if _np.any(self.env['rx_range'] == 0) and not replicated:
                    replicated_column = self.pyramL.tlg[:, [0]]
                    tlgL = _np.hstack((replicated_column, self.pyramL.tlg))
                    replicated_column = self.pyramL.cpg[:, [0]]
                    cpgL = _np.hstack((replicated_column, self.pyramL.cpg))
                else:
                    tlgL = self.pyramL.tlg
                    cpgL = self.pyramL.cpg
        
        # Combine the transmission loss matrices
        if tlgL is not None and tlgR is not None:
            tlg = _np.hstack((_np.fliplr(tlgL), tlgR))
            cpg = _np.hstack((_np.fliplr(cpgL), cpgR))
        elif tlgL is not None:
            tlg = _np.fliplr(tlgL)
            cpg = _np.fliplr(cpgL)
        elif tlgR is not None:
            tlg = tlgR
            cpg = cpgR
        else:
            print("[ERROR] RAM: No results found !")
         
        # Handle cases where the result matrix depth is smaller than x grid (values over 0)   
        if _np.size(tlg[:,0]) < _np.size(self.env['rx_depth']):
            tlg = _np.vstack((tlg[0,:],tlg))
            cpg = _np.vstack((cpg[0,:],cpg))
        num_columns = tlg.shape[1]
        nan_array = _np.empty((1, num_columns))
        nan_array[:] = _np.nan        
        while _np.size(tlg[:,0]) < _np.size(self.env['rx_depth']):
            tlg = _np.vstack((nan_array,tlg))
            cpg = _np.vstack((nan_array,cpg))
                
        # Compute transmission loss and complex pressure
        self.transmission_loss = 10**(-tlg/20)
        self.complex_pressure  = cpg
        
        return self.complex_pressure
    
    def plot_transmission_loss(self, vmin=-120, vmax=0, **kwargs):
        """
        Plots the transmission loss of the environment.

        Parameters:
            vmin (float): Minimum value for the transmission loss.
            vmax (float): Maximum value for the transmission loss.
            **kwargs: Additional keyword arguments for customization.

        Returns:
            fig, ax: Figure and axis objects for the plot.
        """
        fig, ax = plt.subplots()
        X = self.env['rx_range']
        Y = self.env['rx_depth']

        tlossplt = 20 * _np.log10(_np.finfo(float).eps + _np.abs(_np.array(self.transmission_loss)))

        # Remove transmission loss in sediment
        interp_y = _np.interp(X, self.env['bot_interface'][:, 0], self.env['bot_interface'][:, 1])
        ax.fill_between(X / 1000, interp_y, Y[-1], color='brown')

        # Plot the surface
        ax.plot([X[0]/1000, X[-1]/1000], [0, 0], 'b', linewidth=3)

        # Plot the transmission loss map using imshow
        im1 = ax.imshow(tlossplt, extent=[X[0] / 1000, X[-1] / 1000, Y[-1], Y[0]], cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')

        # Plot the bottom interface
        if _np.size(self.env['bot_interface'][:, 0]) > 1:
            ax.plot(self.env['bot_interface'][:, 0] / 1000, self.env['bot_interface'][:, 1], 'k', linewidth=3)
        else:
            ax.plot([X[0] / 1000, X[-1] / 1000], [self.env['bot_interface'][0, 1], self.env['bot_interface'][0, 1]], 'k', linewidth=3)

        # Set plot properties
        ax.set_xlim((X[0] / 1000, X[-1] / 1000))
        ax.set_ylim((Y[0], Y[-1]))
        ax.set_xlabel('Range [km]')
        ax.set_ylabel('Depth [m]')
        ax.set_title(f"[RAM - Transmission loss @ {self.env['tx_freq']} Hz] {self.env['name']}")

        # Add color bar
        cbar1 = fig.colorbar(im1, ax=ax)
        cbar1.ax.set_ylabel('Loss [dB]')

        # Invert y-axis for depth
        ax.invert_yaxis()

        # Adjust layout and display plot
        plt.tight_layout()
        plt.show()

        return fig, ax

    def plot_bot_density(self, vmin=0, vmax=4, Nxy=500, **kwargs):
        """
        Plots the density profile of the sediment in the environment.
    
        Parameters:
            Title (str): Title for the plot.
            vmin (float): Minimum value for the density color map.
            vmax (float): Maximum value for the density color map.
            Nxy (int): Number of points in the x and y directions.
            **kwargs: Additional keyword arguments for customization.
    
        Returns:
            fig, ax: Figure and axis objects for the plot.
        """
        
        fig, ax = plt.subplots()
        
        # Extract density data
        Xb = _np.array(self.env['bot_range'])
        Yb = _np.array(self.env['bot_depth'])
        Zb = _np.array(self.env['bot_density'], ndmin=2)
        
        # Generate grid
        Xg = _np.linspace(self.env['rx_range'][0], self.env['rx_range'][-1], Nxy)
        Yg = _np.linspace(self.env['rx_depth'][0], self.env['rx_depth'][-1], Nxy)
        Zg = _np.zeros([len(Yg), len(Xg)])
        
        # Bathy
        rb, zb = _np.array(self.env['bot_interface'][:, 0]), _np.array(self.env['bot_interface'][:, 1])
        
        # Re-compute map over grid
        for ii, x in enumerate(Xg):  # For all map pixels
            intrepolation = _np.interp(x, rb, zb)
            for jj, y in enumerate(Yg):
                if y > intrepolation:  # If in sediment (interpolation of bathymetry line between samples)
                    x_idx = _np.argmin(_np.abs(Xb - x))
                    y_idx = _np.argmin(_np.abs(Yb - y))
                    Zg[jj, ii] = Zb[y_idx, x_idx]
                else:  # Else it is in water column
                    # Set minimum value
                    Zg[jj, ii] = vmin
    
        # Plot surface
        ax.plot([Xg[0]/1000, Xg[-1]/1000], [0, 0], 'b', linewidth=3)
    
        # Plot
        im = ax.imshow(Zg, cmap='jet', extent=[Xg[0] / 1000, Xg[-1] / 1000, Yg[-1], Yg[0]], aspect='auto', vmin=vmin, vmax=vmax)
        ax.plot(rb/1000, zb, 'k', linewidth=3)
        ax.scatter(0, self.env['tx_depth'], label="Stars", color="r", s=500, marker="*")
        ax.set_xlim((Xg[0] / 1000, Xg[-1] / 1000))
        ax.set_ylim((Yg[0], Yg[-1]))
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Density [g/cm$^{3}$]')
        ax.set_xlabel('Range [km]')
        ax.set_ylabel('Depth [m]')
        ax.set_title(f"[RAM - Bottom density] {self.env['name']}")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()
    
        return fig, ax

    def plot_ssp(self, Nxy=500, **kwargs):
        """
        Plots the sound speed profile of the environment.
        
        Parameters:
            Nxy (int): Number of points in the x and y directions.
            **kwargs: Additional keyword arguments for customization.
            
        Returns:
            fig, ax: Figure and axis objects for the plot.
        """
        fig, ax = plt.subplots()

        if _np.size(self.env['ssp_range']) > 1:  # 2D SSP data
            
            vmin = kwargs.get('vmin', _np.min(self.env['ssp'])-_np.abs(_np.mean(self.env['bot_PwaveSpeed']))/20)
            vmax = kwargs.get('vmax', _np.max(self.env['bot_PwaveSpeed'])+_np.abs(_np.mean(self.env['bot_PwaveSpeed']))/20)
        
            # Extract data
            X, Y, Z = _np.array(self.env['ssp_range']), _np.array(self.env['ssp_depth']), _np.array(self.env['ssp'])
            Xb, Yb, Zb = _np.array(self.env['bot_range']), _np.array(self.env['bot_depth']), _np.array(self.env['bot_PwaveSpeed'], ndmin=2)
            
            # Generate grid
            Xg = _np.linspace(self.env['rx_range'][0], self.env['rx_range'][-1], Nxy)
            Yg = _np.linspace(self.env['rx_depth'][0], self.env['rx_depth'][-1], Nxy)
            Zg = _np.zeros([len(Yg), len(Xg)])
            
            # Bathy
            rb, zb = _np.array(self.env['bot_interface'][:, 0]), _np.array(self.env['bot_interface'][:, 1])
            
            # Re-compute map over grid
            for ii, x in enumerate(Xg):
                intrepolation = _np.interp(x, rb, zb)
                for jj, y in enumerate(Yg):
                    if y > intrepolation:  # If in sediment
                        y_idx = _np.argmin(_np.abs(Yb - y))
                        x_idx = _np.argmin(_np.abs(Xb - x))
                        Zg[jj, ii] = Zb[y_idx, x_idx]
                    else:  # Else in water column
                        y_idx = _np.argmin(_np.abs(Y - y))
                        x_idx = _np.argmin(_np.abs(X - x))
                        Zg[jj, ii] = Z[y_idx, x_idx]
            
            # Plot            
            ax.plot([Xg[0] / 1000, Xg[-1] / 1000], [0, 0], 'b', linewidth=3)  # Flat surface
            im = ax.imshow(Zg, cmap='jet', extent=[Xg[0] / 1000, Xg[-1] / 1000, Yg[-1], Yg[0]], aspect='auto', vmin=vmin, vmax=vmax)
            ax.plot(rb / 1000, zb, 'k', linewidth=3)  # Bathy
            ax.scatter(0, self.env['tx_depth'], label="Stars", color="r", s=500, marker="*")  # Stars
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Sound speed [m/s]')
            ax.set_xlim((self.env['rx_range'][0] / 1000, self.env['rx_range'][-1] / 1000))
            ax.set_ylim((self.env['rx_depth'][0], self.env['rx_depth'][-1]))
            ax.set_xlabel('Range [km]')
            ax.set_ylabel('Depth [m]')
            ax.set_title(f"[RAM - Sound speed profile] {self.env['name']}")
            ax.invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        else:  # 1D SSP data
  
            vmin = kwargs.get('vmin', _np.min(self.env['ssp'])-_np.abs(_np.mean(self.env['ssp']))/100)
            vmax = kwargs.get('vmax', _np.max(self.env['ssp'])+_np.abs(_np.mean(self.env['ssp']))/100)
            
            if _np.size(self.env['ssp_depth']) > 1:
                Y, Z = _np.array(self.env['ssp_depth']), _np.array(self.env['ssp'])
            else:
                Y, Z = _np.array([0, _np.max(self.env['bot_interface'][:,1])]), _np.array([self.env['ssp'], self.env['ssp']])
            ax.set_title(f"[RAM - Sound speed profile] {self.env['name']}")
            ax.set_xlim((vmin, vmax))
            ax.invert_yaxis()
            ax.grid(True)
            ax.set_ylabel('Depth [m]')
            ax.set_xlabel('Sound speed [m/s]')
            ax.plot(Z, Y, 'k', linewidth=3)
            plt.tight_layout()
            plt.show()
            
        return fig, ax

    def plot_bot_attn(self, vmin=0, vmax=0.04, Nxy=500, **kwargs):
        """
        Plots the absorption profile in the environment.
    
        Parameters:
            vmin (float): Minimum value for the absorption color map.
            vmax (float): Maximum value for the absorption color map.
            Nxy (int): Number of points in the x and y directions.
            **kwargs: Additional keyword arguments for customization.
    
        Returns:
            fig, ax: Figure and axis objects for the plot.
        """
    
        fig, ax = plt.subplots()
    
        # Extract absorption data
        Xb = _np.array(self.env['bot_range'])
        Yb = _np.array(self.env['bot_depth'])
        Zb = _np.array(_np.array(self.env['bot_PwaveAttn'], ndmin=2))
    
        # Generate grid
        Xg = _np.linspace(self.env['rx_range'][0], self.env['rx_range'][-1], Nxy)
        Yg = _np.linspace(self.env['rx_depth'][0], self.env['rx_depth'][-1], Nxy)
        Zg = _np.zeros([len(Yg), len(Xg)])
    
        # Bathy
        rb, zb = _np.array(self.env['bot_interface'][:, 0]), _np.array(self.env['bot_interface'][:, 1])
    
        # Re-compute map over grid
        for ii, x in enumerate(Xg):  # For all map pixels
            intrepolation = _np.interp(x, rb, zb)
            for jj, y in enumerate(Yg):
                if y > intrepolation:  # If in sediment (interpolation of bathymetry line between samples)
                    x_idx = _np.argmin(_np.abs(Xb - x))
                    y_idx = _np.argmin(_np.abs(Yb - y))
                    Zg[jj, ii] = Zb[y_idx, x_idx]
                else:  # Else it is in water column
                    # Set minimum value
                    Zg[jj, ii] = vmin
    
        # Plot surface
        ax.plot([Xg[0]/1000, Xg[-1]/1000], [0, 0], 'b', linewidth=3)
    
        # Plot
        im = ax.imshow(Zg, cmap='jet', aspect='auto', extent=[Xg[0]/1000, Xg[-1]/1000, Yg[-1], Yg[0]], vmin=vmin, vmax=vmax)
        ax.plot(rb/1000, zb, 'k', linewidth=3)
        ax.scatter(0, self.env['tx_depth'], label="Stars", color="r", s=500, marker="*")
        ax.set_xlim((self.env['rx_range'][0] / 1000, self.env['rx_range'][-1] / 1000))
        ax.set_ylim((self.env['rx_depth'][0], self.env['rx_depth'][-1]))
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Attenuation [dB/$\lambda$]')
        ax.set_xlabel('Range [km]')
        ax.set_ylabel('Depth [m]')
        ax.set_title(f"[RAM - Bottom attenuation] {self.env['name']}")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()
    
        return fig, ax

    
# Add model to available models
_models.append(('RAM', RAM))

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
    ax.semilogx(Fxx, 10*_np.log10(_np.finfo(float).eps+Pxx)+20*_np.log10(_np.finfo(float).eps+vpk/maxval)-sh-gain, **kwargs)
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
    '''
    Prints the content of a file.
    
    Parameters:
        file_path (str): Path to the file.
    '''
    try:
        # Attempt to open the file for reading
        with open(file_path, 'r') as file:
            # Read the content of the file
            content = file.read()
            # Print file content with a separator
            print("================================================================================================")
            print(f"Content of the file: '{file_path}'.")
            print("================================================================================================")
            print(f"{content}")
            print("================================================================================================")

    except FileNotFoundError:
        # Handle case where the file does not exist
        print(f"[ERROR] Unable to print file, '{file_path}' does not exist.")
    except Exception as e:
        # Handle unexpected errors during file reading
        print(f"[ERROR] An unexpected error occurred during print file: {e}")

def _adjust_2D(vect2D, vmin, vmax):
    '''
    Adjusts 2D vector to ensure consistency in dimensions and boundary conditions along one axis.
    
    Parameters:
        vect2D (array-like): 2D vector to adjust.
        vmin (float): Minimum value for the axis.
        vmax (float): Maximum value for the axis.
        
    Returns:
        array-like: Adjusted 2D vector.
    '''
    
    # Check if the input vector has more than two elements
    if _np.size(vect2D) > 2:
        
        # Add extra point if needed for vmin
        if vect2D[0, 0] > vmin:
            lineF = _np.array([vmin, vect2D[0, 1]])
            vect2D = _np.vstack((lineF, vect2D))
        
        # Add extra point if needed for vmax
        if vect2D[-1, 0] < vmax:
            lineF = _np.array([vmax, vect2D[-1, 1]])
            vect2D = _np.vstack((vect2D, lineF))
            
    return vect2D
    

def _adjust_3D(vect2D, x, y, xmin, xmax, ymin, ymax):
    '''
    Adjusts 2D vector to ensure consistency in dimensions and boundary conditions for both horizontal and vertical axes.
    
    Parameters:
        vect2D (array-like): 2D vector to adjust.
        x (array-like): Horizontal axis values.
        y (array-like): Vertical axis values.
        xmin (float): Minimum value for horizontal axis.
        xmax (float): Maximum value for horizontal axis.
        ymin (float): Minimum value for vertical axis.
        ymax (float): Maximum value for vertical axis.
        
    Returns:
        tuple: Adjusted 2D vector, adjusted horizontal axis values, and adjusted vertical axis values.
    '''
    # Adjust vertically first
    vect2D, x = _adjust_3D_vertical(vect2D, x, y, xmin, xmax)
    
    # Adjust horizontally after vertical adjustments
    vect2D, y = _adjust_3D_horizontal(vect2D, x, y, ymin, ymax)
    
    return vect2D, x, y

def _adjust_3D_vertical(vect2D, x, y, xmin, xmax):
    '''
    Adjusts 2D vector to ensure consistency in dimensions and boundary conditions.
    
    Parameters:
        vect2D (array-like): 2D vector to adjust.
        x (array-like): Horizontal axis values.
        y (array-like): Vertical axis values.
        xmin (float): Minimum value for horizontal axis.
        xmax (float): Maximum value for horizontal axis.
        
    Returns:
        tuple: Adjusted 2D vector and adjusted horizontal axis values.
    '''
    
    # Check if the input vector has more than one element
    if _np.size(vect2D) > 1:
        
        # Check if horizontal axis has more than one element
        if _np.size(x) > 1:
            
            # If the input vector is 1D, adjust it
            if _np.ndim(vect2D) == 1:
                
                # Check if vertical axis has more than one element
                if _np.size(y) > 1:
                    raise Exception("[ERROR] BELLHOP: Size of input vector is inconsistent !")  
                vect2D = _np.vstack(vect2D)
                
                # Add extra row if needed for xmin
                if x[0] > xmin:
                    vect2D = _np.vstack((vect2D[0],vect2D))
                    x = _np.hstack((xmin, x))
                    
                # Add extra row if needed for xmax
                if x[-1] < xmax:
                    vect2D = _np.vstack((vect2D,vect2D[-1]))
                    x = _np.hstack((x, xmax))
                    
            else:
                
                # Add extra column if needed for xmin
                if x[0] > xmin:
                    first_column = vect2D[:, 0]
                    if first_column.ndim == 1:
                        first_column = _np.expand_dims(first_column, axis=1)
                    vect2D = _np.concatenate((first_column, vect2D), axis=1)
                    x = _np.hstack((xmin, x))
                    
                # Add extra column if needed for xmax
                if x[-1] < xmax:
                    last_column = vect2D[:, -1]
                    if last_column.ndim == 1:
                        last_column = _np.expand_dims(last_column, axis=1)
                    vect2D = _np.concatenate((vect2D, last_column), axis=1)
                    x = _np.hstack((x, xmax))
                
    return vect2D, x

def _adjust_3D_horizontal(vect2D, x, y, ymin, ymax):
    '''
    Adjusts 2D vector to ensure consistency in dimensions and boundary conditions, emphasizing vertical adjustments before horizontal.
    
    Parameters:
        vect2D (array-like): 2D vector to adjust.
        x (array-like): Horizontal axis values.
        y (array-like): Vertical axis values.
        ymin (float): Minimum value for vertical axis.
        ymax (float): Maximum value for vertical axis.
        
    Returns:
        tuple: Adjusted 2D vector and adjusted vertical axis values.
    '''
    
    # Check if the input vector has more than one element
    if _np.size(vect2D) > 1:
        
        # Check if either vertical or horizontal axis has more than one element
        if _np.size(y) > 1 or _np.size(x) > 1:
            
            # If the input vector is 1D, adjust it
            if _np.ndim(vect2D) == 1:
                
                # Reshape vector to be 2D
                vect2D = _np.hstack(vect2D)
                
                # Add extra column if needed for ymin
                if y[0] > ymin:
                    vect2D = _np.hstack((vect2D[0], vect2D))
                    y = _np.hstack((ymin, y))
                    
                # Add extra column if needed for ymax
                if y[-1] < ymax:
                    vect2D = _np.hstack((vect2D, vect2D[-1]))
                    y = _np.hstack((y, ymax))
            else:
                
                # Add extra row if needed for ymin
                if y[0] > ymin:
                    first_row = vect2D[0]
                    if first_row.ndim == 1:
                        first_row = _np.expand_dims(first_row, axis=0)
                    vect2D = _np.concatenate((first_row, vect2D), axis=0)
                    y = _np.hstack((ymin, y))
                    
                # Add extra row if needed for ymax
                if y[-1] < ymax:
                    last_row = vect2D[-1]
                    if last_row.ndim == 1:
                        last_row = _np.expand_dims(last_row, axis=0)
                    vect2D = _np.concatenate((vect2D, last_row), axis=0)
                    y = _np.hstack((y, ymax))
                
    return vect2D, y

