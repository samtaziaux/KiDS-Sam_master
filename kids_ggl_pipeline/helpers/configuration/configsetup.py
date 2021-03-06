from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from glob import glob
import numpy as np
import distutils
import distutils.util


_default_entries = {
    'lnk_bins': 'number of (log-spaced) bins in wavenumber k',
    'lnk_max': 'maximum value for lnk in the power spectrum calculations',
    'lnk_min': 'minimum value for lnk in the power spectrum calculations',
    'logM_bins': 'number of (log-spaced) bins in halo mass',
    'logM_max': 'maximum value of logM for halo mass function calculations',
    'logM_min': 'minimum value of logM for halo mass function calculations',
    'transfer': 'name of the transfer function used by hmf',
    # these three used only for mock production
    'logR_bins': 'number of (log-spaced) bins in lens-source separation',
    'logR_max': 'maximum value of logR for mock observations',
    'logR_min': 'minimum value of logR for mock observations',
    'pi_max': 'projection length for wp',
    'kaiser_correction': 'Kaiser effect correction for wp',
    # these are used for k-space filtering
    'bin_edges': 'radial bin edges, used for k-space filtering',
    'bin_samples': 'number of bins to sample for k-space filtering',
    'bin_sampling_max': 'max radius to sample for k-space filtering',
    'kfilter': 'file name(s) containing the k-space filter. Requires pixell',
    #
    'return_extra': 'any additional quantities the model should return for' \
        ' later access.',
    }

_default_values = {
    'backend': 'ccl',
    'bin_edges': [],
    'bin_samples': 201,
    'bin_sampling_max': 50,
    'kfilter': '',
    'lnk_bins': 10000,
    'lnk_min': -15.,
    'lnk_max': 10.,
    'logM_bins': 200,
    'logM_min': 10.,
    'logM_max': 16.,
    'transfer':'EH',
    'logR_bins': 20,
    'logR_min': -1.3,
    'logR_max': 1.,
    # these are used in the unit conversion when loading the data
    'R_unit': 'Mpc',
    'esd_unit': 'Msun/pc^2',
    'cov_unit': 'Msun^2/pc^4',
    'pi_max': 100.0,
    'kaiser_correction': 'False',
    # return extras
    'return_extra': [],
    }

_necessary_entries = {
    'delta': 'overdensity for mass definition (typically 200 or 500)',
    'delta_ref': 'background density reference: {"FOF", "SOCritical", "SOMean", "SOVirial"}',
    'distances': 'whether to use "proper" or "comoving" distances',
    'return': 'which quantity should the halo model return',
    'R_unit': 'units of the radial bins',
    'esd_unit': 'units of the lensing observable (e.g., ESD).' \
        ' Ignored if the lensing observable is kappa.',
    'cov_unit': 'units of the lensing covariance.' \
        ' Ignored if the lensing observable is kappa.',
    }

_valid_entries = {
    'backend': ('ccl', 'hmf'),
    'delta': float,
    'delta_ref': ('FOF', 'SOCritical', 'SOMean', 'SOVirial', 'critical', 'matter'),
    'distances': ('comoving', 'proper', 'angular'),
    'bin_edges': str,
    'bin_samples': int,
    'bin_sampling_max': float,
    'kfilter': str,
    'lnk_bins': int,
    'lnk_min': float,
    'lnk_max': float,
    'logM_bins': int,
    'logM_min': float,
    'logM_max': float,
    'logR_bins': int,
    'logR_min': float,
    'logR_max': float,
    'transfer': ('CAMB', 'EH', 'EH_NoBAO', 'BBKS', 'BondEfs'),
    'return': ('esd', 'kappa', 'power', 'sigma', 'xi', 'wp', 'all', 'esd_wp'),
    # will implement others in the future, require handling different
    # x-values
    #'return': list,
    'return_extra': list,
    #'return': ('esd', 'wp', 'esd_wp'),
    'R_unit': str,
    'esd_unit': str,
    'cov_unit': str,
    'pi_max': str,
    'kaiser_correction': ('True', 'False'),
    }


def add_defaults(setup):
    for key, value in _default_values.items():
        if key not in setup:
            setup[key] = value
    return setup


def add_kfilter(setup):
    if not setup['kfilter']:
        return setup
    if len(setup['bin_edges']) == 0:
        msg = 'must provide bin_edges if applying a k-space filter'
        raise ValueError(msg)
    try:
        import pixell
    except ImportError:
        err = 'you must install the pixell library in order to use' \
              ' a k-space filter with KiDS-GGL. Please install with\n' \
              '    pip install pixell'
        raise ImportError(err)
    else:
        try:
            from profiley.filtering import Filter
        except ImportError:
            err = 'you must install profiley in order to use a k-space' \
                  ' filter with KiDS-GGL. Please install with\n' \
                  '    pip install pixell'
        else:
            setup['kfilter'] = Filter(setup['kfilter'])
    # radial bin edges
    setup['bin_edges'] = setup['bin_edges'].split()
    setup['bin_edges'][0] = sorted(glob(setup['bin_edges'][0]))
    return setup


def add_mass_range(setup):
    # endpoint must be False for mass_range to be equal to hmf.m
    try:
        setup['mass_range'] = 10**np.linspace(
            setup['logM_min'], setup['logM_max'], setup['logM_bins'],
            endpoint=False)
        setup['mstep'] = (setup['logM_max']-setup['logM_min']) \
            / setup['logM_bins']
    except KeyError:
        pass
    return setup


def add_rvir_range(setup):
    setup['rvir_range_3d'] = np.logspace(-3.2, 4, 250, endpoint=True)
    #setup['rvir_range_3d_interp'] = np.logspace(-2.5, 1.2, 25, endpoint=True)
    setup['rvir_range_3d_interp'] = np.logspace(-2.5, 2.5, 30, endpoint=True)
    return setup


def add_wavenumber(setup):
    try:
        setup['k_step'] = (setup['lnk_max']-setup['lnk_min']) / setup['lnk_bins']
    except KeyError:
        pass
    else:
        #setup['k_range'] = np.arange(
            #setup['lnk_min'], setup['lnk_max'], setup['k_step'])
        setup['k_range'] = np.linspace(
            setup['lnk_min'], setup['lnk_max'], setup['lnk_bins'])
        setup['k_range_lin'] = np.exp(setup['k_range'])
    return setup


def append_entry(line, setup):
    """
    `line` should be a `ConfigLine` object
    """
    """
    for dtype in (int, float, str):
        try:
            setup[line.words[0]] = dtype(line.words[1])
            break
        except ValueError:
            pass
    """
    key, value = line.words[:2]
    # this only allows for list of strings but that's all we need for now
    if _valid_entries[key] == list:
        setup[key] = value.split(',')
    elif np.iterable(_valid_entries[key]):
        if value not in _valid_entries[key]:
            err = f'value {value} not allowed for setup entry {key}.' \
                  f' Allowed values are {_valid_entries[key]}.'
            raise ValueError(err)
        setup[key] = value
    else:
        try:
            setup[key] = _valid_entries[key](value)
        except ValueError:
            err = f'cannot convert value {value} in entry {key} to required' \
                  f'type {_valid_entries[key]}'
            raise ValueError(err)
    return setup


def check_entry_types(setup):
    for key, value in setup.items():
        valid = _valid_entries[key]
        if isinstance(valid, type):
            try:
                setup[key] = valid(value)
            except ValueError:
                msg = 'setup entry "{0}" must be {1}'.format(
                   key, valid.__name__)
                raise ValueError(msg)
        else:
            if setup[key] not in valid:
                msg = 'setup entry "{0}" can only be one of {1}'.format(
                    key, valid)
                raise ValueError(msg)
    return


def check_necessary_entries(setup):
    """Check that all necessary entries

    """
    for key in _necessary_entries.keys():
        if key not in setup:
            msg = 'setup entry "{0}" missing from config file'.format(key)
            raise ValueError(msg)
    return
    
    
def convert_to_bool(setup):
    for key, value in setup.items():
        if key in ['kaiser_correction']:
        # if value in ['True', 'False']: ????
            setup[key] = bool(distutils.util.strtobool(value))
        if key in ['pi_max']:
            setup[key] = np.array(value.split(','), dtype=np.float)
    
    return setup


def check_return(setup):
    # should not have esd_wp here - each element should be a combination
    # of observable type and measurement, e.g., "gm.esd"
    valid = ('esd', 'kappa', 'power', 'sigma', 'xi', 'wp', 'all', 'esd_wp')
    # ...


def check_setup(setup):
    """Run all checks to ensure that the setup section complies"""
    check_necessary_entries(setup)
    setup = add_defaults(setup)
    check_entry_types(setup)
    setup = add_mass_range(setup)
    setup = add_rvir_range(setup)
    setup = add_wavenumber(setup)
    setup = convert_to_bool(setup)
    setup = add_kfilter(setup)
    return setup


def test_definitions():
    """TO BE WRITTEN"""
    good = np.array_equal(sorted(_default_entries), sorted(_default_values))
    #if not good:
        #raise 
    return


