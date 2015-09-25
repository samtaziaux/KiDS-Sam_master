#!/usr/bin/env python
"""
Satellite lensing EMCEE wrapper

"""
import emcee
import numpy
import os
import sys
from astropy.io.fits import BinTableHDU, Column, Header, PrimaryHDU
from itertools import count, izip
from numpy import all as npall, array, append, concatenate, dot, inf, isnan
from numpy import isfinite, log, log10, outer, pi, sqrt, transpose, zeros
from numpy.linalg import det
from os import remove
from os.path import isfile
from time import ctime
import pickle

import sampling_utils

def run_emcee(hm_options, sampling_options, args):
    # load halo model setup
    function, params, param_types, prior_types, \
        val1, val2, val3, val4, hm_functions, \
        starting, meta_names, fits_format = hm_options
    # load MCMC sampler setup
    datafile, datacols, covfile, covcols, exclude_bins, output, \
        sampler, nwalkers, nsteps, nburn, \
        thin, k, threads, sampler_type = sampling_options

    #function = cloud.serialization.cloudpickle.dumps(model)
    #del model
    #print function

    #pickle.dumps(function)
    #print 'pickled'

    if args.demo:
        print ' ** Running demo only **'
    elif os.path.isfile(output):
        msg = 'Warning: output file %s exists. Overwrite? [y/N] ' %output
        answer = raw_input(msg)
        if len(answer) == 0:
            exit()
        if answer[0].lower() != 'y':
            exit()
    if not args.demo:
        print 'Started -', ctime()

    #load data files
    Ndatafiles = len(datafile)
    R, esd = sampling_utils.load_datapoints(datafile, datacols, exclude_bins)
    Nobsbins, Nrbins = esd.shape
    rng_obsbins = xrange(Nobsbins)
    rng_rbins = xrange(Nrbins)
    # load covariance
    cov = sampling_utils.load_covariance(covfile, covcols,
                                         Nobsbins, Nrbins, exclude_bins)
    cov, icov, likenorm, esd_err, cov2d = cov

    # needed for offset central profile
    R, Rrange = sampling_utils.setup_integrand(R, k)
    angles = numpy.linspace(0, 2*numpy.pi, 540)
    val1 = numpy.append(val1, [Rrange, angles])

    # identify fixed and free parameters
    jfixed = (prior_types == 'fixed') | (prior_types == 'read') | \
             (prior_types == 'function')
    jfree = ~jfixed
    ndim = len(val1[(jfree)])
    if len(starting) != ndim:
        msg = 'ERROR: Not all starting points defined for free parameters.'
        print msg
        exit()
    print 'starting =', starting

    # identify the function. Raises an AttributeError if not found
    #function = model.model()
    #sat_profile = params.sat_profile()
    #group_profile = params.group_profile()
    #function = model

    if not args.demo:
        hdrfile = '.'.join(output.split('.')[:-1]) + '.hdr'
        print 'Printing header information to', hdrfile
        hdr = open(hdrfile, 'w')
        print >>hdr, 'Started', ctime()
        print >>hdr, 'datafile', ','.join(datafile)
        print >>hdr, 'cols', ','.join([str(c) for c in datacols])
        print >>hdr, 'covfile', covfile
        print >>hdr, 'covcols', ','.join([str(c) for c in covcols])
        if exclude_bins is not None:
            print >>hdr, 'exclude_bins', ','.join([str(c)
                                                   for c in exclude_bins])
        print >>hdr, 'model %s' %function
        for p, pt, v1, v2, v3, v4 in izip(params, prior_types,
                                        val1, val2, val3, val4):
            try:
                line = '%s  %s  ' %(p, pt)
                line += ','.join(numpy.array(v1, dtype=str))
            except TypeError:
                line = '%s  %s  %s  %s  %s  %s' \
                    %(p, pt, str(v1), str(v2), str(v3), str(v4))
            print >>hdr, line
        print >>hdr, 'nwalkers  {0:5d}'.format(nwalkers)
        print >>hdr, 'nsteps    {0:5d}'.format(nsteps)
        print >>hdr, 'nburn     {0:5d}'.format(nburn)
        print >>hdr, 'thin      {0:5d}'.format(thin)
        hdr.close()

    # are we just running a demo?
    if args.demo:
        import pylab
        from matplotlib import cm
        v1 = val1.copy()
        v1[jfree] = starting
        model = function(v1, R)
        residuals = esd - model[0]
        dof = esd.size - len(v1[jfree])
        chi2 = array([dot(residuals[m], dot(icov[m][n], residuals[n]))
                      for m in rng_obsbins for n in rng_obsbins]).sum()
        print ' ** chi2 = %.2f/%d **' %(chi2, dof)
        fig, axes = pylab.subplots(figsize=(4*Ndatafiles,4), ncols=Ndatafiles)
        #if exclude_bins is not None:
            #alldata = sampling_utils.load_datapoints(datafile, datacols,
                                                     #exclude_bins)
            #Rall, esdall = alldata
            #for ax, Ri
        for ax, Ri, gt, gt_err, f in izip(axes, R, esd, esd_err, model[0]):
            ax.errorbar(Ri[1:], gt, yerr=gt_err, fmt='ko', ms=10)
            ax.plot(Ri[1:], f, 'r-', lw=3)
            ax.set_xscale('log')
        if npall(esd - esd_err > 0):
            for ax in axes:
                ax.set_yscale('log')
        fig.tight_layout(w_pad=0.01)
        pylab.show()
        fig, axes = pylab.subplots(figsize=(8,8), nrows=cov.shape[0],
                                   ncols=cov.shape[0])
        for m, axm in enumerate(axes):
            for n, axmn in enumerate(axm):
                axmn.imshow(cov[m][-n-1][::-1], interpolation='nearest',
                            cmap=cm.CMRmap_r)
        fig.tight_layout()
        pylab.show()
        exit()

    # set up starting point for all walkers
    po = starting * numpy.random.uniform(0.99, 1.01, size=(nwalkers,ndim))
    lnprior = zeros(ndim)
    mshape = meta_names.shape
    # this assumes that all parameters are floats -- can't imagine a
    # different scenario
    metadata = [[] for m in meta_names]
    for j in xrange(len(metadata)):
        for f in fits_format[j]:
            if len(f) == 1:
                metadata[j].append(zeros(nwalkers*nsteps/thin))
            else:
                size = (nwalkers*nsteps/thin, int(f[:-1])-len(exclude_bins))
                metadata[j].append(zeros(size))
    metadata = [array(m) for m in metadata]
    fail_value = []
    for m in metadata:
        shape = list(m.shape)
        shape.remove(max(shape))
        fail_value.append(zeros(shape))
    # the last numbers are data chi2, lnLdata, lnPderived
    for i in xrange(4):
        fail_value.append(9999)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    threads=threads,
                                    args=(R,esd,icov,function,
                                          params,prior_types[jfree],
                                          val1,val2,val3,val4,
                                          jfree,lnprior,likenorm,
                                          rng_obsbins,fail_value,
                                          array,dot,inf,izip,outer,pi))
                                          #isfinite,log,log10
                                          #outer,sqrt,zeros))
    # burn-in
    if nburn > 0:
        pos, prob, state = sampler.run_mcmc(po, nburn)
        sampler.reset()
        print '{0} Burn-in steps finished ({1})'.format(nburn, ctime())
    else:
        pos = po
    # incrementally save output
    chi2 = [zeros(nwalkers*nsteps/thin) for i in xrange(4)]
    nwritten = 0
    for i, result in enumerate(sampler.sample(pos, iterations=nsteps,
                                              thin=thin)):
        # make sure that nwalkers is a factor of this number!
        if i*nwalkers % 10000 == nwalkers:
            out = write_to_fits(output, chi2, sampler, nwalkers, thin,
                                params, jfree, metadata, meta_names,
                                fits_format, i, nwritten, Nobsbins,
                                array, BinTableHDU, Column, ctime, enumerate,
                                isfile, izip, transpose, xrange)
            metadata, nwriten = out

    hdr = open(hdrfile, 'a')
    try:
        print 'acceptance_fraction =', sampler.acceptance_fraction
        print >>hdr, 'acceptance_fraction =',
        for af in sampler.acceptance_fraction:
            print >>hdr, af,
    except ImportError:
        pass
    try:
        print 'acor =', sampler.acor
        print >>hdr, '\nacor =',
        for ac in sampler.acor:
            print >>hdr, ac,
    except ImportError:
        pass
    try:
        print 'acor_time =', sampler.get_autocorr_time()
        print >>hdr, '\nacor_time =',
        for act in sampler.get_autocorr_time():
            print >>hdr, act,
    except AttributeError:
        pass
    print >>hdr, '\nFinished', ctime()
    hdr.close()
    print 'Saved to', hdrfile

    cmd = 'mv {0} {1}'.format(output, output.replace('.fits', '.temp.fits'))
    print cmd
    os.system(cmd)
    print 'Saving everything to {0}...'.format(output)
    print i, nwalkers, nwritten
    write_to_fits(output, chi2, sampler, nwalkers, thin,
                  params, jfree, metadata, meta_names,
                  fits_format, i+1, nwritten, Nobsbins,
                  array, BinTableHDU, Column, ctime, enumerate,
                  isfile, izip, transpose, xrange)
    os.remove(output.replace('.fits', '.temp.fits'))
    print 'Everything saved to {0}!'.format(output)
    return

def lnprob(theta, R, esd, icov, function, params,
           prior_types, val1, val2, val3, val4, jfree, lnprior, likenorm,
           rng_obsbins, fail_value, array, dot, inf, izip, outer, pi):
           #array, dot, inf, izip, isfinite, log, log10, sqrt):
    """
    Probability of a model given the data, i.e., log-likelihood of the data
    given a model, times the prior on the model parameters.

    Parameters
    ----------
        theta
            whatever *free* parameters are received by the model selected.
        R
            projected distances from the satellite
        esd
            Excess surface density at distances R
        esd_err
            Uncertainties on the ESD
        function
            the model used to calculate the likelihood
        prior_types
            one value per parameter in *theta*, {'normal', 'uniform', 'fixed'}
        val1
            depends on each prior_type:
                -normal : the mean of the gaussian
                -uniform : the minimum allowed value
                -fixed : the fixed value
        val2
            depends on each prior_type:
                -normal : the half-width of the gaussian
                -uniform : the maximum allowed value
                -fixed : ignored (but must be there)
        jfree
            indices of the free values
        lnprior
            just a placeholder, should be an array with a length
            equal to theta, so that it doesn't have to be defined every
            time

    """
    _log = log
    v1free = val1[jfree]
    v2free = val2[jfree]
    v3free = val3[jfree]
    v4free = val4[jfree]
    if not isfinite(v1free.sum()):
        return -inf, fail_value
    # satellites cannot be more massive than the group!
    #if theta[params[jfree] == 'Msat'] >= theta[params[jfree] == 'Mgroup']:
        #return -inf, fail_value
    # not normalized yet
    j = (prior_types == 'normal')
    lnprior[j] = array([-(v-v1)**2 / (2*v2**2) - _log(2*pi*v2**2)/2
                        if v3 <= v <= v4 else -inf
                        for v, v1, v2, v3, v4
                        in izip(theta[j], v1free[j], v2free[j],
                                v3free[j], v4free[j])])
    j = (prior_types == 'lognormal')
    lnprior[j] = array([-(log10(v)-v1)**2 / (2*v2**2) - _log(2*pi*v2**2)/2
                        if v3 <= v <= v4 else -inf
                        for v, v1, v2, v3, v4
                        in izip(theta[j], v1free[j], v2free[j],
                                v3free[j], v4free[j])])
    j = (prior_types == 'uniform')
    lnprior[j] = array([0-_log(v2-v1) if v1 <= v <= v2 else -inf
                        for v, v1, v2
                        in izip(theta[j], v1free[j], v2free[j])])
    # note that exp is not normalized
    j = (prior_types == 'exp')
    lnprior[j] = array([v**v1 if v2 <= v <= v3 else -inf
                        for v, v1, v2, v3
                        in izip(theta[j], v1free[j], v2free[j], v3free[j])])
    lnprior_total = lnprior.sum()
    if not isfinite(lnprior_total):
        return -inf, fail_value
    # all other types ('fixed', 'read') should not contribute to the prior
    # run the given model
    v1 = val1.copy()
    v1[jfree] = theta

    model = function(v1, R)
    # no covariance
    #chi2 = (((esd-model[0]) / esd_err) ** 2).sum()
    # full covariance included
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
    return lnlike + model[-3] + lnprior_total, model

def write_to_fits(output, chi2, sampler, nwalkers, thin, params, jfree,
                  metadata, meta_names, fits_format, iternum, nwritten,
                  Nobsbins, array, BinTableHDU, Column, ctime, enumerate,
                  isfile, izip, transpose, xrange):
    nexclude = len(chi2)
    lnprior, lnPderived, chi2, lnlike = chi2
    if isfile(output):
        remove(output)
    chain = transpose(sampler.chain, axes=(2,1,0))
    columns = [Column(name=param, format='E', array=data[:iternum].flatten())
               for param, data in izip(params[jfree], chain)]
    columns.append(Column(name='lnprob', format='E',
                          array=sampler.lnprobability.T[:iternum].flatten()))
    if len(meta_names) > 0:
        # save only the last chunk (starting from t),
        # all others are already in metadata.
        # NOTE that this is only implemented for a model with the
        # same format as fiducial()
        for j, blob in izip(xrange(nwritten, iternum),
                            sampler.blobs[nwritten:]):
            if Nobsbins == 1:
                data = [transpose([b[i] for b in blob])
                        for i in xrange(len(blob[0])-nexclude)]
                for i in xrange(len(data)):
                    if len(data[i].shape) == 2:
                        data[i] = array([b[i] for b in blob])
            else:
                data = [transpose([b[i] for b in blob])
                        for i in xrange(len(blob[0])-nexclude)]
                for i in xrange(len(data)):
                    if len(data[i].shape) == 3:
                        data[i] = transpose([b[i] for b in blob],
                                            axes=(1,0,2))
            # each k is one satellite radial bin
            #if len(data) == 1:
                ## not working but not sure will ever use it
                #metadata[0][0][j*nwalkers:(j+1)*nwalkers] = data[0]
                #if shape[1] == 4:
                    #metadata[0][1][j*nwalkers:(j+1)*nwalkers] = data[1]
                    #metadata[0][2][j*nwalkers:(j+1)*nwalkers] = data[2]
                    #metadata[0][3][j*nwalkers:(j+1)*nwalkers] = data[3]
            ## this is the fiducial case with the three bins
            ## note that all values have three bins!
            #else:
            for k, datum in enumerate(data):
                for i in xrange(len(datum)):
                    metadata[k][i][j*nwalkers:(j+1)*nwalkers] = datum[i]
            lnPderived[j*nwalkers:(j+1)*nwalkers] = array([b[-4]
                                                           for b in blob])
            lnprior[j*nwalkers:(j+1)*nwalkers] = array([b[-3] for b in blob])
            chi2[j*nwalkers:(j+1)*nwalkers] = array([b[-2] for b in blob])
            lnlike[j*nwalkers:(j+1)*nwalkers] = array([b[-1] for b in blob])
        columns.append(Column(name='lnprior', format='E', array=lnprior))
        columns.append(Column(name='lnPderived', format='E',
                              array=lnPderived))
        columns.append(Column(name='chi2', format='E', array=chi2))
        columns.append(Column(name='lnlike', format='E', array=lnlike))
        for n, m, f in izip(meta_names, metadata, fits_format):
            for ni, mi, fi in izip(n, m, f):
                columns.append(Column(name=ni, format=fi, array=mi))
        nwritten = iternum * nwalkers
    fitstbl = BinTableHDU.from_columns(columns)
    fitstbl.writeto(output)
    print 'Saved to {0} with {1} samples'.format(output, iternum*nwalkers),
    if thin > 1:
        print '(printing every {0}th sample)'.format(thin),
    print '- {0}'.format(ctime())
    return metadata, nwritten
