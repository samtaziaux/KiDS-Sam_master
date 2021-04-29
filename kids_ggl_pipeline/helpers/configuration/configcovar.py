from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import distutils
import distutils.util


_default_entries = {
    'gauss': 'calculate the gaussian part of the covariance? [bool]',
    'non_gauss': 'calculate the non-gaussian part of the covariance? [bool]',
    'ssc': 'calculate the super-sample part of the covariance? [bool]',
    'mlf_gauss': 'calculate the gaussian part of the SMF/LF covariance? [bool]',
    'mlf_ssc': 'calculate the super-sample part of the SMF/LF covariance? [bool]',
    'cross': 'calculate the cross terms between observables? [bool]',
    'subtract_randoms': 'is the signal from random points subtracted from the data? [bool]',
    'threads': 'number of threads to use to calculate the covariance matrix',
    'pi_max': 'integration length of the 3D clustering correlation function used to obtain w_p(r_p) [Mpc/h]',
    'area': 'area of the survey [deg^2]',
    'healpy': 'if healpix mask map is to be used to determine survey variance and area',
    'healpy_data': 'healpix map',
    'eff_density': 'effective galaxy density as defined in KiDS papers [gal/arcmin^2]',
    'variance_squared': 'variance of the galaxy ellipticity (as defined in KiDS papers)',
    'mean_survey_redshift': 'mean survey redshift (about 0.6 for KiDS as an example)',
    'output': 'output file name [ASCII document]',
    'kids_sigma_crit': 'if KiDS specific sigma_crit is to be used',
    'z_epsilon': 'offset redshift between lenses and sources',
    'z_max': 'max redshift used in photo-z calibration',
    'lens_photoz': 'if we have photometric lenses',
    'lens_photoz_sigma': 'photo-z lens uncertainty',
    'lens_photoz_zdep': 'if photometric lenses uncertainty depends on redshift',
    'specz_file': 'file containing weights for photo-z calibration',
    'vmax_file': 'file containing Vmax values',
    }

_default_values = {
    'pi_max': 100,
    'area': 180,
    'healpy': 'False',
    'eff_density': 8.53,
    'variance_squared': 0.082,
    'mean_survey_redshift': 0.6,
    'gauss': 'True',
    'non_gauss': 'False',
    'ssc': 'False',
    'mlf_gauss': 'False',
    'mlf_ssc': 'False',
    'cross': 'True',
    'subtract_randoms': 'False',
    'kids_sigma_crit': 'False',
    'z_epsilon': 0.0,
    'z_max': 1.2,
    'lens_photoz': 'False',
    'lens_photoz_sigma': '0.0',
    'lens_photoz_zdep': 'False',
    'specz_file': 'None',
    'threads': 1,
    'output': 'analytical_covariance.txt',
    'vmax_file': 'None',
    }

_necessary_entries = {
    }

_valid_entries = {
    'pi_max': float,
    'area': float,
    'healpy': ('True', 'False'),
    'healpy_data': str,
    'eff_density': float,
    'variance_squared': float,
    'mean_survey_redshift': float,
    'gauss': ('True', 'False'),
    'non_gauss': ('True', 'False'),
    'ssc': ('True', 'False'),
    'mlf_gauss': ('True', 'False'),
    'mlf_ssc': ('True', 'False'),
    'cross': ('True', 'False'),
    'subtract_randoms': ('True', 'False'),
    'kids_sigma_crit': ('True', 'False'),
    'z_epsilon': float,
    'z_max': float,
    'lens_photoz': ('True', 'False'),
    'lens_photoz_sigma': float,
    'lens_photoz_zdep': ('True', 'False'),
    'specz_file': str,
    'threads': int,
    'output': str,
    'vmax_file': str,
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


def convert_to_bool(covar):
    for key, value in covar.items():
        if key in ['healpy', 'gauss', 'non_gauss', 'ssc',
                   'mlf_gauss', 'mlf_ssc', 'cross', 'subtract_randoms',
                   'kids_sigma_cirt', 'lens_photoz', 'lens_photoz_zdep']:
            covar[key] = bool(distutils.util.strtobool(covar[key]))
    
    return covar


def check_covar(covar):
    """Run all checks to ensure that the setup section complies"""
    check_necessary_entries(covar)
    setup = add_defaults(covar)
    check_entry_types(covar)
    setup = convert_to_bool(covar)
    return covar


def test_definitions():
    """TO BE WRITTEN"""
    good = np.array_equal(sorted(_default_entries), sorted(_default_values))
    #if not good:
        #raise 
    return


