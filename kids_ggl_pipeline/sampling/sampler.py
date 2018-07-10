#!/usr/bin/env python
"""
Galaxy-galaxy lensing EMCEE wrapper

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import emcee
import numpy as np
import os
import sys
from astropy.io.fits import BinTableHDU, Column, Header, PrimaryHDU
from itertools import count
from matplotlib import cm, pylab
from numpy import all as npall, array, append, concatenate, dot, inf, isnan, \
                  isfinite, log, log10, outer, pi, sqrt, squeeze, transpose, \
                  where, zeros
from os import remove
from os.path import isfile
from scipy import stats
from time import ctime, time
#import pickle

# Python 2/3 compatibility
if sys.version_info[0] == 2:
    from itertools import izip as zip
    range = xrange
    input = raw_input

# local
from . import priors, sampling_utils


def run_emcee(hm_options, sampling, args):

    function, parameters, names, prior_types, nparams, starting, output = \
        hm_options

    val1, val2, val3, val4 = parameters[2]
    params_join = []

    print('Running KiDS-GGL pipeline - sampler\n')
    if args.demo:
        print(' ** Running demo only **')
    elif isfile(sampling['output']) and not args.force_overwrite:
        msg = 'Warning: output file {0} exists. Overwrite? [y/N] '.format(
            sampling['output'])
        answer = input(msg)
        if len(answer) == 0:
            exit()
        if answer.lower() not in ('y', 'yes'):
            exit()
    if not args.demo:
        print('\n{0}: Started\n'.format(ctime()))

    # load data files
    Ndatafiles = len(sampling['data'][0])
    # need to run this without exclude_bins to throw out invalid values in it
    R, esd = sampling_utils.load_datapoints(*sampling['data'])
    #if 'exclude' in sampling:
    if sampling['exclude'] is not None:
        sampling['exclude'] = \
            sampling['exclude'][sampling['exclude'] < esd.shape[1]]
    #else:
        #exclude_bins = None
    R, esd = sampling_utils.load_datapoints(
        sampling['data'][0], sampling['data'][1],
        sampling['exclude'])

    Nobsbins, Nrbins = esd.shape
    rng_obsbins = range(Nobsbins)
    rng_rbins = range(Nrbins)
    # load covariance
    cov = sampling_utils.load_covariance(
        sampling['covariance'][0], sampling['covariance'][1],
        Nobsbins, Nrbins, sampling['exclude'])
    cov, icov, likenorm, esd_err, cov2d = cov

    # needed for offset central profile
    # only used in nfw_stack, not in the halo model proper
    # this should *not* be part of the sampling dictionary
    # but doing it this way so it is an optional parameter
    if 'precision' not in sampling:
        sampling['precision'] = 7
    R, Rrange = sampling_utils.setup_integrand(
        R, sampling['precision'])
    angles = np.linspace(0, 2*pi, 540)
    val1 = np.append(val1, [Rrange, angles])

    # identify fixed and free parameters

    jfree = np.array([(p in priors.free_priors) for p in prior_types])
    ndim = len(val1[where(jfree)])
    assert len(starting) == ndim, \
        'ERROR: Not all starting points defined for free parameters.'
    print('Starting values =', starting)


    meta_names, fits_format = output
    mshape = meta_names.shape
    # this assumes that all parameters are floats -- can't imagine a
    # different scenario
    metadata = [[] for m in meta_names]
    for j, fmt in enumerate(fits_format):
        n = 1 if len(fmt) == 1 else int(fmt[0])
        # is this value a scalar?
        if len(fmt) == 1:
            size = sampling['nwalkers'] * sampling['nsteps'] \
                // sampling['thin']
        else:
            size = [sampling['nwalkers']*sampling['nsteps']//sampling['thin'],
                    int(fmt[:-1])]
            # only for ESDs. Note that there will be trouble if outputs
            # other than the ESD have the same length, so avoid them at
            # all cost.
            if sampling['exclude'] is not None \
                    and size[1] == esd.shape[-1]+len(sampling['exclude']):
                size[1] -= len(sampling['exclude'])
        metadata[j].append(zeros(size))
    metadata = [array(m) for m in metadata]
    metadata = [m[0] if m.shape[0] == 1 else m for m in metadata]
    fail_value = []
    for m in metadata:
        shape = list(m.shape)
        shape.remove(max(shape))
        fail_value.append(zeros(shape))
    # the last numbers are data chi2, lnLdata
    for i in range(3):
        fail_value.append(9999)
    # initialize
    lnprior = zeros(ndim)

    # are we just running a demo?
    if args.demo:
        run_demo(function, R, esd, esd_err, cov, icov, prior_types,
                 parameters, params_join, starting, jfree, nparams, names,
                 lnprior, rng_obsbins, fail_value, Ndatafiles, array, dot,
                 inf, outer, pi, zip)
        sys.exit()

    # write header file
    hdrfile = write_hdr(sampling, function, parameters, names, prior_types)

    # initialize sampler
    sampler = emcee.EnsembleSampler(
        sampling['nwalkers'], ndim, lnprob, threads=sampling['threads'],
        args=(R,esd,icov,function,names,prior_types[jfree],
              parameters,nparams,params_join,jfree,lnprior,likenorm,
              rng_obsbins,fail_value,array,dot,inf,zip,outer,pi))

    # set up starting point for all walkers
    po = sampler_ball(
        names, prior_types, jfree, starting, parameters,
        sampling['nwalkers'], ndim)

    # burn-in
    if sampling['nburn'] > 0:
        pos, prob, state, blobs = sampler.run_mcmc(po, sampling['nburn'])
        sampler.reset()
        print('{1}: {0} Burn-in steps finished'.format(
            sampling['nburn'], ctime()))
    else:
        pos = po

    # incrementally save output
    # this array contains lnprior, chi2, lnlike
    chi2 = [zeros(sampling['nwalkers']*sampling['nsteps']//sampling['thin'])
            for i in range(3)]
    nwritten = 0
    for i, result in enumerate(
            sampler.sample(pos, iterations=sampling['nsteps'],
                           thin=sampling['thin'])):
        if i > 0 and ((i+1)*sampling['nwalkers'] \
                      > sampling['nwalkers']*nwritten + sampling['update']):
            out = write_to_fits(
                sampler, sampling, chi2, names, jfree, output, metadata, i,
                nwritten, Nobsbins, fail_value, array, BinTableHDU,
                Column, ctime, enumerate, isfile, zip, transpose, range)
            metadata, nwritten = out

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

    tmp = sampling['output'].replace('.fits', '.temp.fits')
    cmd = 'mv {0} {1}'.format(sampling['output'], tmp)
    print(cmd)
    os.system(cmd)
    print('Saving everything to {0}...'.format(sampling['output']))
    write_to_fits(
        sampler, sampling, chi2, names, jfree, output, metadata, i+1,
        nwritten, Nobsbins, fail_value, array, BinTableHDU, Column, ctime,
        enumerate, isfile, zip, transpose, range)
    if os.path.isfile(tmp):
        os.remove(tmp)
    print('Everything saved to {0}!'.format(sampling['output']))
    return


def lnprob(theta, R, esd, icov, function, names, prior_types,
           parameters, nparams, params_join, jfree, lnprior, likenorm,
           rng_obsbins, fail_value, array, dot, inf, zip, outer, pi):
    """
    Probability of a model given the data, i.e., log-likelihood of the data
    given a model, times the prior on the model parameters.

    Parameters
    ----------
        theta
            whatever *free* parameters are received by the model selected.
        R
            lens-source separations
        esd
            excess surface density at distances R
        icov
            inverse covariance
        function
            the model used to calculate the likelihood
        prior_types
            one value per parameter in *theta*, {'normal', 'uniform', 'fixed'}
        parameters
            list of length 4, containing all parameters of the models. In
            terms of the configuration file, these are [ingredients,
            observable, parameters, setup]; where `parameters` is another
            length-4 list
        jfree
            indices of the free values
        lnprior
            just a placeholder, should be an array with a length
            equal to theta, so that it doesn't have to be defined every
            time

    """
    lnprior_total = priors.calculate_lnprior(
        lnprior, theta, prior_types, parameters, jfree)
    if not isfinite(lnprior_total):
        return -inf, fail_value

    # all other types ('fixed', 'read') do not contribute to the prior
    # run the given model
    v1 = parameters[2][0].copy()
    v1[where(jfree)] = theta
    if params_join is not None:
        v1j = list(v1)
        for p in params_join:
            # without this list comprehension numpy can't keep track of the
            # data type. I believe this is because there are elements of
            # different types in val1 and therefore its type is not
            # well defined (so it gets "object")
            v1j[p[0]] = array([v1[pi] for pi in p])
        # need to delete elements backwards to preserve indices
        aux = [[v1j.pop(pi) for pi in p[1:][::-1]]
                for p in params_join[::-1]]
        v1 = v1j #array(v1j) ??
    # join into sections and subsections as per the config file
    # by doing this I'm losing Rrange and angles, but I suppose
    # those should be defined in the config file as well.
    v1 = np.array([v1[sum(nparams[:i]):sum(nparams[:i+1])]
                   for i in range(len(nparams))])
    # note that by now we discard the other v's!
    # we don't want to overwrite the old list now that we've
    # changed one of its components
    newp = [p for p in parameters]
    newp[2] = v1

    model = function(newp, R)
    # no covariance
    #chi2 = (((esd-model[0]) / esd_err) ** 2).sum()
    # full covariance included
    #residuals = esd - model[0]
    #chi2 = array([dot(residuals[m], dot(icov[m][n], residuals[n]))
    #              for m in rng_obsbins for n in rng_obsbins]).sum()

    if 'model' in str(function):

        # model assumes comoving separations, changing data to accomodate for that
        redshift = read_redshift(v1, names, nparams)
        esd = esd / (1+redshift)**2
        residuals = (esd - model[0]) / (1+redshift)**2
    else:
        residuals = esd - model[0]

    chi2 = array([dot(residuals[m], dot(icov[m][n], residuals[n]))
                  for m in rng_obsbins for n in rng_obsbins]).sum()

    if not isfinite(chi2):
        return -inf, fail_value
    # remember that the last value returned by the models must be a lnprior
    # from the derived parameters
    lnlike = -chi2/2. + likenorm
    model.append(lnprior_total)
    model.append(chi2)
    model.append(lnlike)
    # return to original content for future calls
    #parameters[2] = [val1, val2, val3, val4]
    return lnlike + lnprior_total, model


def run_demo(function, R, esd, esd_err, cov, icov, prior_types, parameters,
             params_join, starting, jfree, nparams, names, lnprior,
             rng_obsbins, fail_value, Ndatafiles, array, dot, inf, outer,
             pi, zip):
    def plot_demo(ax, Ri, gt, gt_err, f):
        Ri = Ri[1:]
        ax.errorbar(Ri, gt, yerr=gt_err, fmt='ko', ms=10)
        ax.plot(Ri, f, 'r-', lw=3)
        ax.set_xscale('log')
        #for x, fi, gti, gei in zip(Ri, f, gt, gt_err):
            #ax.annotate('{0:.2f}'.format((fi-gti)/gei),
                        #xy=(x,gti+20), ha='center', va='bottom',
                        #color='r')
        return

    parameters[2][0][where(jfree)] = starting
    lnlike, model = lnprob(
        starting, R, esd, icov, function, names, prior_types[jfree],
        parameters, nparams, params_join, jfree, lnprior, 0, rng_obsbins,
        fail_value, array, dot, inf, zip, outer, pi)
    chi2 = model[-2]
    dof = esd.size - starting.size - 1
    print(' ** chi2 = {0:.2f}/{1:d} **'.format(chi2, dof))

    # the rest are corrected in lnprob. We should work to remove
    # this anyway
    if 'model' in str(function):
        redshift = read_redshift(parameters[2][0], names)
        esd_err = esd_err / (1+redshift)**2

    fig, axes = pylab.subplots(figsize=(4*Ndatafiles,4), ncols=Ndatafiles)
    if Ndatafiles == 1:
        plot_demo(axes, R[0], esd[0], esd_err[0], model[0][0])
    else:
        for i in zip(axes, R, esd, esd_err, model[0]):
            plot_demo(*i)
    if npall(esd - esd_err > 0) or 'model' in str(function):
        if Ndatafiles == 1:
            axes.set_yscale('log')
        else:
            for ax in axes:
                ax.set_yscale('log')
    #fig.tight_layout(w_pad=0.01)
    pylab.show()
    fig, axes = pylab.subplots(
        figsize=(10,8), nrows=cov.shape[0], ncols=cov.shape[0])
    vmin, vmax = np.percentile(cov, [1,99])
    if Ndatafiles == 1:
        axes = [[axes]]
        #for m in range(1):
            #for n in range(1):
                #axes.imshow(cov[m][-n-1][::-1], interpolation='nearest',
                            #cmap=cm.CMRmap_r, vmin=vmin, vmax=vmax)
    #else:
    for m, axm in enumerate(axes):
        for n, axmn in enumerate(axm):
            axmn.imshow(cov[m][-n-1][::-1], interpolation='nearest',
                        vmin=vmin, vmax=vmax)
    #fig.tight_layout()
    pylab.show()
    return


def write_to_fits(sampler, sampling, chi2, names, jfree, output, metadata,
                  iternum, nwritten, Nobsbins, fail_value, array, BinTableHDU,
                  Column, ctime, enumerate, isfile, zip, transpose, range):
    nchi2 = len(chi2)
    # the two following lines should remain consistent if modified
    chi2_loc = -2
    lnprior, chi2, lnlike = chi2
    if isfile(sampling['output']):
        remove(sampling['output'])
    chain = transpose(sampler.chain, axes=(2,1,0))
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
            data = [zeros((sampling['nwalkers'],m.shape[1]))
                    if len(m.shape) == 2 else zeros(sampling['nwalkers'])
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
            write_start = j * sampling['nwalkers']
            write_end = (j+1) * sampling['nwalkers']
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
            lnprior[write_start:write_end] = array([b[-3] for b in blob])
            chi2[write_start:write_end] = array([b[-2] for b in blob])
            lnlike[write_start:write_end] = array([b[-1] for b in blob])
        # this handles sampling['exclude'] properly
        for name, val, fmt in zip(meta_names, metadata, fits_format):
            val = squeeze(val)
            if len(val.shape) == 1:
                fmt = fmt[-1]
            else:
                fmt = '{0}{1}'.format(val.shape[1], fmt[-1])
            columns.append(Column(name=name, array=val, format=fmt))
    columns.append(Column(name='lnprior', format='E', array=lnprior))
    columns.append(Column(name='chi2', format='E', array=chi2))
    columns.append(Column(name='lnlike', format='E', array=lnlike))
    fitstbl = BinTableHDU.from_columns(columns)
    fitstbl.writeto(sampling['output'])
    nwritten = iternum
    print('{2}: Saved to {0} with {1} samples'.format(
            sampling['output'], nwritten*sampling['nwalkers'], ctime()))
    if sampling['thin'] > 1:
        print('(printing every {0}th sample)'.format(sampling['thin']))
    #print('acceptance fraction =', sampler.acceptance_fraction)
    # these two are the same
    #print('autocorrelation length =', sampler.acor)
    #print('autocorrelation time =', sampler.get_autocorr_time())
    return metadata, nwritten


def read_redshift(val1, names, nparams=None):
    # model assumes comoving separations, changing data to accomodate
    # for that
    redshift_index = np.int(where(names == 'z')[0])
    if nparams is None:
        redshift = val1[redshift_index]
    else:
        # val1 is no longer a flat array
        jz = [np.arange(nparams.size)[np.cumsum(nparams) <= redshift_index+1][0]]
        jz.append(redshift_index - nparams[jz[0]])
        redshift = val1[jz[0]][jz[1]]
    return redshift.T


def sampler_ball(names, prior_types, jfree, starting, parameters, nw, ndim):
    """
    This creates a ball around a starting value, taking prior ranges
    into consideration. It takes intervals (max-min)/2 around a
    starting value.
    """
    val1, val2, val3, val4 = parameters[2]
    v1free = val1[where(jfree)]
    v2free = val2[where(jfree)]
    v3free = val3[where(jfree)]
    v4free = val4[where(jfree)]
    names_free = names[where(jfree)]
    prior_free = prior_types[where(jfree)]

    ball = np.zeros((nw, ndim))
    for n, p in enumerate(prior_free):
        ball[:,n] = priors.draw(
            p, (v1free[n],v2free[n]), (v3free[n],v4free[n]), size=nw)
    return ball


def write_hdr(sampling, function, parameters, names, prior_types):
    hdrfile = '.'.join(sampling['output'].split('.')[:-1]) + '.hdr'
    print('Printing header information to', hdrfile)
    #val1, val2, val3, val4 = parameters[2]
    parameters[2] = np.transpose(parameters[2])
    #hdr = open(hdrfile, 'w')
    with open(hdrfile, 'w') as hdr:
        print('Started', ctime(), file=hdr)
        print('datafile', ','.join(sampling['data'][0]), file=hdr)
        print('cols', ','.join([str(c) for c in sampling['data'][1]]),
              file=hdr)
        print('covfile', sampling['covariance'][0], file=hdr)
        print('covcols',
              ','.join([str(c) for c in sampling['covariance'][1]]),
              file=hdr)
        if sampling['exclude'] is not None:
            print('exclude',
                  ','.join([str(c) for c in sampling['exclude']]),
                  file=hdr)
        print('model {0}'.format(function), file=hdr)
        # being lazy for now
        print('observables  {0}'.format(parameters[0]), file=hdr)
        print('ingredients {0}'.format(
            ','.join([key for key, item in parameters[1].items() if item])),
            file=hdr)
        for p, pt, v1, v2, v3, v4 in zip(names, prior_types, *parameters[2].T):
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

        print('nwalkers  {0:5d}'.format(sampling['nwalkers']), file=hdr)
        print('nsteps    {0:5d}'.format(sampling['nsteps']), file=hdr)
        print('nburn     {0:5d}'.format(sampling['nburn']), file=hdr)
        print('thin      {0:5d}'.format(sampling['thin']), file=hdr)

    # back to its original shape
    parameters[2] = np.transpose(parameters[2])

    return hdrfile


