"""Input/output utilities"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.io import ascii, fits
from astropy.io.fits import BinTableHDU, Column
from astropy.table import Table
from glob import glob
from itertools import count
import numpy as np
import os
import six
from time import ctime

import sys
if sys.version_info[0] == 2:
    from itertools import izip as zip
    range = xrange

# local
# this shouldn't happen!
from ..sampling import sampling_utils


def finalize_hdr(sampler, hdrfile):
    with open(hdrfile, 'a') as hdr:

        try:
            print('acceptance_fraction =', sampler.acceptance_fraction)
            print('acceptance_fraction =', file=hdr, end=' ')
            for af in sampler.acceptance_fraction:
                print(af, file=hdr, end=' ')
        except ImportError:
            pass
        #try:
            #print('acor =', sampler.acor)
            #print('\nacor =', file=hdr, end=' ')
            #for ac in sampler.acor:
                #print(ac, file=hdr, end=' ')
        #except ImportError:
            #pass
        # acor and get_autocorr_time() are the same
        try:
            print('acor =', sampler.get_autocorr_time())
            print('\nacor_time =', file=hdr, end=' ')
            for act in sampler.get_autocorr_time(c=5):
                print(act, file=hdr, end=' ')
        #except AttributeError:
            #pass
        #except emcee.autocorr.AutocorrError:
            #pass
        except:
            pass
        # acor and get_autocorr_time() are the same
        #try:
            #print('acor_time =', sampler.get_autocorr_time())
            #print('\nacor_time =', file=hdr, end=' ')
            #for act in sampler.get_autocorr_time():
                #print(act, file=hdr, end=' ')
        #except AttributeError:
            #pass
        print('\nFinished', ctime(), file=hdr)
    print('Saved to', hdrfile)
    return


def load_covariance(covfile, covcols, Nobsbins, Nrbins, exclude=None):
    """Load covariance from a kids_ggl-compatible covariance file

    Parameters
    ----------
    covfile : str
        filename of covariance file
    covcols : list-like or int
        the index of the column(s) containing the covariance matrix
        and, if applicable, the multiplicative bias correction, (1+K)
        in the notation of Viola et al. (2015, MNRAS, 452, 3529)
    Nrbins : int
        number of data points per observable bin
    Nobsbins : int, optional
        number of observable bins (default 1). Must be provided if >1
    exclude : list-like, optional
        bins to be excluded from the covariance. Only supports one set
        of bins, which is excluded from the entire dataset

    Returns
    -------
    cov : array, shape (Nobsbins,Nobsbins,Nrbins,Nrbins)
        four-dimensional covariance matrix
    icov : array, shape (Nobsbins,Nobsbins,Nrbins,Nrbins)
        four-dimensional inverse covariance matrix
    likenorm : float
        constant additive term for the likelihood
    esd_err : array, shape (Nobsbins,Nrbins)
        square root of the diagonal elements of the covariance matrix
    cov2d : array, shape (Nobsbins*Nrbins,Nobsbins*Nrbins)
        re-shaped covariance matrix, for plotting or other uses
    """
    if not hasattr(covcols, '__iter__'):
        covcols = [covcols]
    cov = np.loadtxt(covfile, usecols=[covcols[0]])
    if len(covcols) == 2:
        cov /= np.loadtxt(covfile, usecols=[covcols[1]])
    if exclude is None:
        nexcl = 0
    else:
        if not hasattr(exclude, '__iter__'):
            exclude = [exclude]
        nexcl = len(exclude)
    # 4-d matrix
    cov = cov.reshape((Nobsbins,Nobsbins,Nrbins+nexcl,Nrbins+nexcl))
    cov2d = cov.transpose(0,2,1,3)
    cov2d = cov2d.reshape(
        (Nobsbins*(Nrbins+nexcl), Nobsbins*(Nrbins+nexcl)))
    icov = np.linalg.inv(cov2d)
    # are there any bins excluded?
    if exclude is not None:
        for b in exclude[::-1]:
            cov = np.delete(cov, b, axis=3)
            cov = np.delete(cov, b, axis=2)
    # product of the determinants
    detC = np.array(
        [np.linalg.det(cov[m][n])
         for m in range(Nobsbins) for n in range(Nobsbins)])
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


def load_data(options):
    # need to run this without exclude to throw out invalid values in it
    R, esd = load_datapoints(*options['data'])
    if options['exclude'] is not None:
        options['exclude'] = \
            options['exclude'][options['exclude'] < esd.shape[1]]
    R, esd = load_datapoints(
        options['data'][0], options['data'][1], options['exclude'])

    Nobsbins, Nrbins = esd.shape
    # load covariance
    cov = load_covariance(
          options['covariance'][0], options['covariance'][1],
          Nobsbins, Nrbins, options['exclude'])
    # needed for offset central profile
    # only used in nfw_stack, not in the halo model proper
    R, Rrange = sampling_utils.setup_integrand(
        R, options['precision'])
    angles = np.linspace(0, 2*np.pi, 540)
    return R, esd, cov, Rrange, angles


def load_datapoints(datafiles, datacols, exclude=None):
    if isinstance(datafiles, six.string_types):
        datafiles = sorted(glob(datafiles))
    R, esd = np.transpose([np.loadtxt(df, usecols=datacols[:2])
                           for df in datafiles], axes=(2,0,1))
    if len(datacols) == 3:
        oneplusk = np.array([np.loadtxt(df, usecols=[datacols[2]])
                             for df in datafiles])
        esd /= oneplusk
    if exclude is not None:
        R = np.array([[Ri[j] for j in range(len(Ri))
                       if j not in exclude] for Ri in R])
        esd = np.array([[esdi[j] for j in range(len(esdi))
                         if j not in exclude] for esdi in esd])
    return R, esd


def read_ascii(file, columns=None):
    """Read a table file into a np array

    np.loadtxt is doing weird things to read priors such that the array
    cannot be converted to a float numpy array
    """
    if columns is None:
        data = []
    else:
        data = [[] for i in columns]
    with open(file) as f:
        for line in f:
            if line.lstrip()[0] == '#':
                continue
            if '#' in line:
                line = line[:line.index('#')]
            line = line.split()
            if columns is None:
                for col in line:
                    data.append(col)
            else:
                for i, col in enumerate(columns):
                    data[i].append(line[col])
    return np.array(data, dtype=float)


def write_chain(sampler, options, chi2, names, jfree, output, metadata,
                iternum, nwritten, Nobsbins, fail_value):
    nchi2 = len(chi2)
    # the two following lines should remain consistent if modified
    chi2_loc = -2
    lnprior, chi2, lnlike = chi2
    if os.path.isfile(options['output']):
        os.remove(options['output'])
    chain = np.transpose(sampler.chain, axes=(2,1,0))
    columns = [Column(name=param, format='E', array=data[:iternum].flatten())
               for param, data in zip(names[jfree], chain)]
    columns.append(Column(name='lnprob', format='E',
                          array=sampler.lnprobability.T[:iternum].flatten()))

    meta_names, fits_format = output
    if len(meta_names) > 0:
        # save only the last chunk (starting from t),
        # all others are already in metadata.
        # j iterates over MCMC samples. Then each blob is a list of length
        # nwalkers containing the metadata
        nparams = len(metadata)
        for j, blob in zip(range(nwritten, iternum),
                           sampler.blobs[nwritten:]):
            data = [np.zeros((options['nwalkers'],m.shape[1]))
                    if len(m.shape) == 2 else np.zeros(options['nwalkers'])
                    for m in metadata]
            # this loops over the walkers. In each iteration we then access
            # a list corresponding to the number of hm_output's
            for i, walker in enumerate(blob):
                # apparently the format is different when the sample failed;
                # we are skipping those
                if walker[chi2_loc] == fail_value[-1]:
                    continue
                n = 0
                for param in walker[:-nchi2]:
                    # if it's a scalar
                    if not hasattr(param, '__iter__'):
                        data[n] = param
                        n += 1
                        continue
                    if len(data[n].shape) == 2 and len(param.shape) == 1:
                        data[n][i] = param
                        n += 1
                        continue
                    for entry in param:
                        data[n][i] = entry
                        n += 1
            # store data
            n = 0
            write_start = j * options['nwalkers']
            write_end = (j+1) * options['nwalkers']
            for k in range(len(data)):
                # if using a single bin, there can be scalars
                if not hasattr(data[k], '__iter__'):
                    metadata[n][write_start:write_end] = data[k]
                    n += 1
                    continue
                shape = data[k].shape
                if len(shape) == 3:
                    for datum in data[k]:
                        for m, val in enumerate(datum):
                            metadata[n][write_start:write_end] = val
                            n += 1
                elif len(shape) == 2 and len(fits_format[n]) == 1:
                    datum = data[k].T
                    for i in range(len(datum)):
                        metadata[n][write_start:write_end] = datum[i]
                        n += 1
                else:
                    metadata[n][write_start:write_end] = data[k]
                    n += 1
            lnprior[write_start:write_end] = np.array([b[-3] for b in blob])
            chi2[write_start:write_end] = np.array([b[-2] for b in blob])
            lnlike[write_start:write_end] = np.array([b[-1] for b in blob])
        # this handles options['exclude'] properly
        for name, val, fmt in zip(meta_names, metadata, fits_format):
            val = np.squeeze(val)
            if len(val.shape) == 1:
                fmt = fmt[-1]
            else:
                fmt = '{0}{1}'.format(val.shape[1], fmt[-1])
            columns.append(Column(name=name, array=val, format=fmt))
    columns.append(Column(name='lnprior', format='E', array=lnprior))
    columns.append(Column(name='chi2', format='E', array=chi2))
    columns.append(Column(name='lnlike', format='E', array=lnlike))
    fitstbl = BinTableHDU.from_columns(columns)
    fitstbl.writeto(options['output'])
    nwritten = iternum
    print('{2}: Saved to {0} with {1} samples'.format(
            options['output'], nwritten*options['nwalkers'], ctime()))
    if options['thin'] > 1:
        print('(printing every {0}th sample)'.format(options['thin']))
    #print('acceptance fraction =', sampler.acceptance_fraction)
    # these two are the same
    #print('autocorrelation length =', sampler.acor)
    #print('autocorrelation time =', sampler.get_autocorr_time())
    return metadata, nwritten


def write_hdr(options, function, parameters, names, prior_types):
    hdrfile = '.'.join(options['output'].split('.')[:-1]) + '.hdr'
    iparams = parameters[0].index('parameters')
    parameters[1][iparams] = np.transpose(parameters[1][iparams])
    with open(hdrfile, 'w') as hdr:
        print('Started', ctime(), file=hdr)
        print('datafile', ','.join(options['data'][0]), file=hdr)
        print('cols', ','.join([str(c) for c in options['data'][1]]),
              file=hdr)
        print('covfile', options['covariance'][0], file=hdr)
        print('covcols',
              ','.join([str(c) for c in options['covariance'][1]]),
              file=hdr)
        if options['exclude'] is not None:
            print('exclude',
                  ','.join([str(c) for c in options['exclude']]),
                  file=hdr)
        print('model {0}'.format(function), file=hdr)
        # being lazy for now
        print('observables  {0}'.format(
            parameters[1][parameters[0].index('observables')]), file=hdr)
        ingredients = parameters[1][parameters[0].index('ingredients')]
        print('ingredients {0}'.format(
            ','.join([key for key, item in ingredients.items() if item])),
            file=hdr)
        params = parameters[1][iparams]
        for p, pt, v1, v2, v3, v4 in zip(names, prior_types, *params.T):
            try:
                line = '%s  %s  ' %(p, pt)
                line += ','.join(np.array(v1, dtype=str))
            except TypeError:
                line = '%s  %s  %s' %(p, pt, str(v1))
                if pt not in ('fixed', 'function', 'read'):
                    # I don't know why v2, v3 and v4 are single-element
                    # arrays instead of floats but who cares if all the rest
                    # works... I'm leaving the try statement just in case they
                    # might be floats at some point
                    try:
                        line += '  %s  %s  %s' \
                                %tuple([str(v[0]) for v in (v2,v3,v4)])
                    except TypeError:
                        line += '  %s  %s  %s' \
                                %tuple([str(v) for v in (v2,v3,v4)])
            print(line, file=hdr)

        print('nwalkers  {0:5d}'.format(options['nwalkers']), file=hdr)
        print('nsteps    {0:5d}'.format(options['nsteps']), file=hdr)
        print('nburn     {0:5d}'.format(options['nburn']), file=hdr)
        print('thin      {0:5d}'.format(options['thin']), file=hdr)

    # back to its original shape (do we need this??)
    parameters[1][iparams] = np.transpose(parameters[1][iparams])
    return hdrfile


def write_signal(R, y, options, setup, err=None, output_path='mock'):
    """
    Write the signal to an ascii file
    """
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    outputs = []
    if len(y.shape) == 1:
        y = np.array([y])
    yname = setup['return']
    for i, yi in zip(count(), y):
        output = os.path.join(
            output_path,
            'kids_ggl_mock_{0}_{1}.txt'.format(yname, i))
        ascii.write(
            [R, yi], output, names=('R',yname), format='commented_header',
            formats={'R':'.2e',yname:'.2e'})
        outputs.append(output)
    return outputs

