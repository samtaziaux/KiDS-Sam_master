"""Input/output utilities"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.io import fits
from astropy.io.fits import BinTableHDU, Column
import numpy as np
import os
from time import ctime


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


