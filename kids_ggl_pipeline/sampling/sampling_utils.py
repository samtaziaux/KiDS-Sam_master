from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os
import six
import sys
from glob import glob

if sys.version_info[0] == 2:
    input = raw_input
    range = xrange


def load_datapoints(datafile, datacols, exclude_bins=None):
    if isinstance(datafile, six.string_types):
        R, esd = np.loadtxt(datafile, usecols=datacols[:2]).T
        # better in Mpc
        if R[-1] > 500:
            R /= 1000
        if len(datacols) == 3:
            oneplusk = np.loadtxt(datafile, usecols=[datacols[2]]).T
            esd /= oneplusk
        R = np.array([R])
        esd = np.array([esd])
    else:
        R, esd = np.transpose([np.loadtxt(df, usecols=datacols[:2])
                                  for df in datafile], axes=(2,0,1))
        if len(datacols) == 3:
            oneplusk = np.array([np.loadtxt(df, usecols=[datacols[2]])
                              for df in datafile])
            esd /= oneplusk
        for i in range(len(R)):
            if R[i][-1] > 500:
                R[i] /= 1000
    if exclude_bins is not None:
        R = np.array([[Ri[j] for j in range(len(Ri))
                          if j not in exclude_bins] for Ri in R])
        esd = np.array([[esdi[j] for j in range(len(esdi))
                            if j not in exclude_bins] for esdi in esd])
    return R, esd


def load_covariance(covfile, covcols, Nobsbins, Nrbins, exclude_bins=None):
    cov = np.loadtxt(covfile, usecols=[covcols[0]])
    if len(covcols) == 2:
        cov /= np.loadtxt(covfile, usecols=[covcols[1]])
    # 4-d matrix
    if exclude_bins is None:
        nexcl = 0
    else:
        nexcl = len(exclude_bins)
    cov = cov.reshape((Nobsbins,Nobsbins,Nrbins+nexcl,Nrbins+nexcl))
    cov2d = cov.transpose(0,2,1,3)
    cov2d = cov2d.reshape((Nobsbins*(Nrbins+nexcl),
                           Nobsbins*(Nrbins+nexcl)))
    icov = np.linalg.inv(cov2d)
    # are there any bins excluded?
    if exclude_bins is not None:
        for b in exclude_bins[::-1]:
            cov = np.delete(cov, b, axis=3)
            cov = np.delete(cov, b, axis=2)
    # product of the determinants
    detC = np.array([np.linalg.det(cov[m][n])
                        for m in range(Nobsbins)
                        for n in range(Nobsbins)])
    prod_detC = detC[detC > 0].prod()
    # likelihood normalization
    likenorm = -(Nobsbins**2*np.log(2*np.pi) + np.log(prod_detC)) / 2
    # switch axes to have the diagonals aligned consistently to make it
    # a 2d array
    cov2d = cov.transpose(0,2,1,3)
    cov2d = cov2d.reshape((Nobsbins*Nrbins,Nobsbins*Nrbins))
    # errors are sqrt of the diagonal of the covariance matrix
    esd_err = np.sqrt(np.diag(cov2d)).reshape((Nobsbins,Nrbins))
    # invert
    icov = np.linalg.inv(cov2d)
    # reshape back into the desired shape (with the right axes order)
    icov = icov.reshape((Nobsbins,Nrbins,Nobsbins,Nrbins))
    icov = icov.transpose(2,0,3,1)

    # Hartlap correction
    #icov = (45.0 - Nrbins - 2.0)/(45.0 - 1.0)*icov

    return cov, icov, likenorm, esd_err, cov2d


def read_config(config_file, version='0.5.7', path_data='',
                path_covariance=''):
    valid_types = ('normal', 'lognormal', 'uniform', 'exp',
                   'fixed', 'read', 'function')
    exclude_bins = None
    path = ''
    path_data = ''
    path_covariance = ''
    config = open(config_file)
    for line in config:
        if line.replace(' ', '').replace('\t', '')[0] == '#':
            continue
        line = line.split()
        if len(line) == 0:
            continue
        if line[0] == 'path':
            path = line[1]
        if line[0] == 'path_data':
            path_data = line[1]
        if line[0] == 'data':
            datafiles = os.path.join(path_data, line[1]).replace(
                '<path>', path)
            datacols = [int(i) for i in line[2].split(',')]
            if len(datacols) not in (2,3):
                msg = 'datacols must have either two or three elements'
                raise ValueError(msg)
        elif line[0] == 'exclude_bins':
            exclude_bins = np.array([int(i) for i in line[1].split(',')])
        if line[0] == 'path_covariance':
            path_covariance = line[1]
        elif line[0] == 'covariance':
            covfile = os.path.join(path_covariance, line[1]).replace(
                '<path>', path)
            covcols = [int(i) for i in line[2].split(',')]
            if len(covcols) not in (1,2):
                msg = 'covcols must have either one or two elements'
                raise ValueError(msg)
        elif line[0] == 'sampler_output':
            output = line[1]
            if output[-5:].lower() != '.fits' and \
                output[-4:].lower() != '.fit':
                output += '.fits'
            # create folder if it doesn't exist, asking the user first
            output_folder = os.path.split(output)[0]
            if not os.path.isdir(output_folder):
                create_folder = input(
                    '\nOutput folder {0} does not exist. Would you like' \
                    ' to create it? [Y/n] '.format(output_folder))
                if create_folder.lower().startswith('n'):
                    msg = 'You chose not to create the non-existent' \
                          ' output folder, so the MCMC samples cannot' \
                          ' be stored. Exiting.'
                    raise SystemExit(msg)
                else:
                    print('Creating folder {0}'.format(output_folder))
                    os.makedirs(output_folder)
        # all of this will have to be made more flexible to allow for
        # non-emcee options
        elif line[0] == 'sampler':
            sampler = line[1]
        elif line[0] == 'nwalkers':
            nwalkers = int(line[1])
        elif line[0] == 'nsteps':
            nsteps = int(line[1])
        elif line[0] == 'nburn':
            nburn = int(line[1])
        elif line[0] == 'thin':
            thin = int(line[1])
        # this k is only needed for mis-centred groups in my implementation
        # so maybe it should go in the hm_utils?
        elif line[0] == 'k':
            k = int(line[1])
        elif line[0] == 'threads':
            threads = int(line[1])
        elif line[0] == 'sampler_type':
            sampler_type = line[1]
        elif line[0] == 'update_freq':
            update_freq = int(line[1])
    datafiles = sorted(glob(datafiles))
    covfile = glob(covfile)
    if len(covfile) > 1:
        msg = 'ambiguous covariance filename'
        raise ValueError(msg)
    covfile = covfile[0]

    out = (datafiles, datacols, covfile, covcols,
           exclude_bins, output,
           sampler, nwalkers, nsteps, nburn, thin, k, threads,
           sampler_type, update_freq)
    return out


def read_function(function):
    print('Reading function', function)
    function_path = function.split('.')
    if len(function_path) < 2:
        print('ERROR: the parent module(s) must be given with a function')
        exit()
    else:
        module = __import__(function)
        for attr in function_path:
            func = getattr(func, attr)
    return func


def setup_integrand(R, k=7):
    """
    These are needed for integration and interpolation and should always
    be used. k=7 gives a precision better than 1% at all radii

    """
    if R.shape[0] == 1:
        Rrange = np.logspace(np.log10(0.99*R.min()),
                                np.log10(1.01*R.max()), 2**k)
        # this assumes that a value at R=0 will never be provided, which is
        # obviously true in real observations
        R = np.array([np.append(0, R)])
        Rrange = np.append(0, Rrange)
    else:
        Rrange = [np.logspace(np.log10(0.99*Ri.min()),
                                 np.log10(1.01*Ri.max()), 2**k)
                  for Ri in R]
        R = [np.append(0, Ri) for Ri in R]
        Rrange = [np.append(0, Ri) for Ri in Rrange]
        R = np.array(R)
        Rrange = np.array(Rrange)
    return R, Rrange
