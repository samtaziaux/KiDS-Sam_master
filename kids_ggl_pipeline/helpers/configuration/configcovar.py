from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


_default_entries = {
    'gauss': 'calculate the gaussian part of the covariance? [bool]',
    'non_gauss': 'calculate the non-gaussian part of the covariance? [bool]',
    'ssc': 'calculate the super-sample part of the covariance? [bool]',
    'cross': 'calculate the cross terms between observables? [bool]',
    'subtract_randoms': 'is the signal from random points subtracted from the data? [bool]',
    'threads': 'number of threads to use to calculate the covariance matrix',
    'pi_max': 'integration length of the 3D clustering correlation function used to obtain w_p(r_p) [Mpc/h]',
    'area': 'area of the survey [deg^2]',
    'eff_density': 'effective galaxy density as defined in KiDS papers [gal/arcmin^2]',
    'variance_squared': 'variance of the galaxy ellipticity (as defined in KiDS papers)',
    'mean_survey_redshift': 'mean survey redshift (about 0.6 for KiDS as an example)',
    'output': 'output file name [ASCII document]',
    }

_default_values = {
    'pi_max': 100,
    'area': 180,
    'eff_density': 8.53,
    'variance_squared': 0.082,
    'mean_survey_redshift': 0.6,
    'gauss': 'True',
    'non_gauss': 'False',
    'ssc': 'False',
    'cross': 'True',
    'subtract_randoms': 'False',
    'threads': 1,
    'output': 'analytical_covariance.txt',
    }

_necessary_entries = {
    }

_valid_entries = {
    'pi_max': float,
    'area': float,
    'eff_density': float,
    'variance_squared': float,
    'mean_survey_redshift': float,
    'gauss': str,
    'non_gauss': str,
    'ssc': str,
    'cross': str,
    'subtract_randoms': str,
    'threads': int,
    'output': str,
    #'spec_cat': str, # Not yet implemented
    }


def add_defaults(covar):
    for key, value in _default_values.items():
        if key not in covar:
            covar[key] = value
    return covar


def append_entry(line, covar):
    """
    `line` should be a `ConfigLine` object
    """
    for dtype in (int, float, str):
        try:
            covar[line.words[0]] = dtype(line.words[1])
            break
        except ValueError:
            pass
    return covar


def check_entry_types(covar):
    for key, value in covar.items():
        valid = _valid_entries[key]
        if isinstance(valid, type):
            try:
                covar[key] = valid(value)
            except ValueError:
                msg = 'covariance entry "{0}" must be {1}'.format(
                   key, valid.__name__)
                raise ValueError(msg)
        else:
            if covar[key] not in valid:
                msg = 'covariance entry "{0}" can only be one of {1}'.format(
                    key, valid)
                raise ValueError(msg)
    return


def check_necessary_entries(covar):
    """Check that all necessary entries

    """
    for key in _necessary_entries.keys():
        if key not in covar:
            msg = 'setup entry "{0}" missing from config file'.format(key)
            raise ValueError(msg)
    return


def check_covar(covar):
    """Run all checks to ensure that the setup section complies"""
    check_necessary_entries(covar)
    setup = add_defaults(covar)
    check_entry_types(covar)
    return covar


def test_definitions():
    """TO BE WRITTEN"""
    good = np.array_equal(sorted(_default_entries), sorted(_default_values))
    #if not good:
        #raise 
    return

