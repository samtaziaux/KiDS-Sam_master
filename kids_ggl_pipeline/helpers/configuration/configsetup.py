from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


_default_entries = {
    'lnk_bins': 'number of (log-spaced) bins in wavenumber k',
    'lnk_max': 'maximum value for lnk in the power spectrum calculations',
    'lnk_min': 'minimum value for lnk in the power spectrum calculations',
    'logM_bins': 'number of (log-spaced) bins in halo mass',
    'logM_max': 'maximum value of logM for halo mass function calculations',
    'logM_min': 'minimum value of logM for halo mass function calculations',
    'transfer': 'name of the transfer function used by hmf'
    }

_default_values = {
    'lnk_bins': 10000,
    'lnk_max': 17.,
    'lnk_min': -13.,
    'logM_bins': 200,
    'logM_max': 16.,
    'logM_min': 5.,
    'transfer':'EH',
    }

_necessary_entries = {
    'delta': 'overdensity for mass definition (typically 200 or 500)',
    'delta_ref': 'background density reference: {"mean","crit"}',
    'distances': 'whether to use "proper" or "comoving" distances',
    'return': 'which quantity should the halo model return'
    }

_valid_entries = {
    'delta': float,
    'delta_ref': ('mean', 'crit', 'critical'),
    'distances': ('comoving', 'proper'),
    'lnk_bins': int,
    'lnk_min': float,
    'lnk_max': float,
    'logM_bins': int,
    'logM_min': float,
    'logM_max': float,
    'transfer': ('CAMB', 'EH'),
    'return': ('esd', 'power', 'sigma', 'xi')
    }


def add_defaults(setup):
    for key, value in _default_values.items():
        if key not in setup:
            setup[key] = value
    return setup


def append_entry(line, setup):
    """
    `line` should be a `ConfigLine` object
    """
    for dtype in (int, float, str):
        try:
            setup[line.words[0]] = dtype(line.words[1])
            break
        except ValueError:
            pass
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


def check_setup(setup):
    """Run all checks to ensure that the setup section complies"""
    check_necessary_entries(setup)
    setup = add_defaults(setup)
    check_entry_types(setup)
    return setup


def test_definitions():
    """TO BE WRITTEN"""
    good = np.array_equal(sorted(_default_entries), sorted(_default_values))
    #if not good:
        #raise 
    return


