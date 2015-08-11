#!/usr/bin/env python
"""
Satellite lensing MCMC wrapper

Write your own model(s) in models.py and then write a paramfile

this code should be run as

$ ./run.py <data_file> <cols> <covariance_file> <covariance_cols> \
           <parameter_file> <output_file>

"""
import emcee
import numpy
import os
import sys
from numpy import array, inf, isfinite, isnan, log, log10, pi, sqrt
from numpy.linalg import det
from os import remove
from os.path import isfile
from time import ctime

# local
import models
import utils

def main(nwalkers=500, nsteps=600, nburn=0, thin=1, k=5, threads=4,
         sampler_type='ensemble'):
    from itertools import count, izip
    from time import ctime
    from astropy.io.fits import BinTableHDU, Column, Header, PrimaryHDU
    from numpy import all as npall, array, append, concatenate, dot, inf
    from numpy import isfinite, log, log10, outer, sqrt, transpose, zeros
    import numpy
    # local
    import nfw
    # preamble
    fmt_error = 'Usage:\n'
    fmt_error += '    ./run.py <data_file> <data_cols> <cov_file> <cov_col>'
    fmt_error += ' <parameter_file> <output_file>\n'
    fmt_error += '  where <data_cols> are the comma-separated'
    fmt_error += ' columns in <data_file> containing,'
    fmt_error += ' in that order, (R,ESD[,1+K]);\n'
    fmt_error += '  and <cov_file> and <cov_col> are'
    fmt_error += ' the file and column for the covariance'
    fmt_error += ' and, optionally, a comma-separated column for 1+K.'
    argv = sys.argv
    if len(argv) < 6:
        print fmt_error
        exit()
    print 'Started at {0}'.format(ctime())
    # minus the columns, paramfile, and outputfile entries
    Ndatafiles = len(argv) - 6
    datafile = argv[1:1+Ndatafiles]
    datacols = array(argv[1+Ndatafiles].split(','), dtype=int)
    covfile = argv[Ndatafiles+2]
    covcols = array(argv[Ndatafiles+3].split(','), dtype=int)
    paramfile = argv[-2]
    output = argv[-1]
    if output[-5:] != '.fits' and output[-4:] != '.fit':
        output += '.fits'
    if len(datacols) not in (2,3):
        print fmt_error
        exit()
    if os.path.isfile(output):
        msg = 'Warning: output file %s exists. Overwrite? [y/N] ' %output
        answer = raw_input(msg)
        if len(answer) == 0:
            exit()
        if answer[0].lower() != 'y':
            exit()

    if type(datafile) == str:
        R, esd = numpy.loadtxt(datafile, usecols=datacols[:2]).T
        if R[-1] > 500:
            R /= 1000
        if len(datacols) == 3:
            oneplusk = numpy.loadtxt(datafile, usecols=[datacols[2]]).T
            esd /= oneplusk
    else:
        R, esd = numpy.transpose([numpy.loadtxt(df, usecols=datacols[:2])
                                  for df in datafile], axes=(2,0,1))
        if len(datacols) == 3:
            oneplusk = array([numpy.loadtxt(df, usecols=[datacols[2]])
                              for df in datafile])
            esd /= oneplusk
        for i in xrange(len(R)):
            if R[i][-1] > 500:
                R[i] /= 1000
    Nobsbins, Nrbins = esd.shape
    Rshape = numpy.array(R).shape
    rng_obsbins = xrange(Nobsbins)
    rng_rbins = xrange(Nrbins)
    # load covariance
    cov = numpy.loadtxt(covfile, usecols=[covcols[0]])
    if len(covcols) == 2:
        cov /= numpy.loadtxt(covfile, usecols=[covcols[1]])
    # 4-d matrix
    cov = cov.reshape((Nobsbins,Nobsbins,Nrbins,Nrbins))
    prod_detC = array([det(cov[m][n])
                       for m in rng_obsbins for n in rng_obsbins]).prod()
    likenorm = -0.5 * (Nobsbins**2*log(2*pi) + log(prod_detC))
    # switch axes to have the diagonals aligned consistently to make it
    # a 2d array
    cov2d = numpy.transpose(cov, axes=(0,2,1,3))
    cov2d = cov2d.reshape((Nobsbins*Nrbins,Nobsbins*Nrbins))
    esd_err = numpy.sqrt(numpy.diag(cov2d)).reshape((Nobsbins,Nrbins))
    #import pylab
    #print cov2d[0][0], cov2d[0][-1]
    #pylab.imshow(cov2d)
    #pylab.show()
    icov = numpy.linalg.inv(cov2d)
    # reshape back into the desired shape (with the right axes order)
    icov = icov.reshape((Nobsbins,Nrbins,Nobsbins,Nrbins))
    icov = numpy.transpose(icov, axes=(2,0,3,1))
    #icov = numpy.transpose(icov, axes=(0,2,1,3))
    #cov2d = cov2d.reshape((Nobsbins,Nrbins,Nobsbins,Nrbins))
    #cov2d = numpy.transpose(cov2d, axes=(2,0,3,1))
    #fig, axes = pylab.subplots(figsize=(12,12),
                               #ncols=Nobsbins, nrows=Nobsbins)
    #for i, row in enumerate(axes):
        #for j, ax in enumerate(row):
            #ax.imshow(cov[i][j])
            #ax.annotate('%d,%d' %(i, j), xy=(0.5,0.5),
                        #xycoords='axes fraction', color='w')
    #pylab.show()
    #return

    model, sat_profile_name, group_profile_name, \
        params, prior_types, po, val1, val2, val3, val4, \
        meta_names, fits_format = utils.get_priors(paramfile)
    #for x in izip(params, prior_types, po, val1, val2):
        #print x
    print params
    print ''
    # discard Rsat's which don't contribute
    # not doing it for now
    #if Rshape[0] == 1:
        #R = R[0]
        #for i1, p, v in izip(count(), params, val1):
            #if p == 'n_Rsat':
                #j = (v > 0)
                #break
        #for i2, p in enumerate(params):
            #if p == 'Rsat':
                #break
        #val1[i1] = val1[i1][j]
        #val1[i2] = val1[i2][j]
    # values are actually printed in log
    for i1, p in enumerate(params):
        if 'n_Rsat' in p:
            j = (val1[i1] > 0)
            val1[i1][j] = 10**val1[i1][j]
    # these are needed for integration and interpolation and should always
    # be used. k=7 gives a precision better than 1% at all radii
    if Rshape[0] == 1:
        Rrange = numpy.logspace(log10(0.99*R.min()), log10(1.01*R.max()),
                                2**k)
        # this assumes that a value at R=0 will never be provided, which is
        # obviously true in real observations
        R = numpy.append(0, R)
        Rrange = numpy.append(0, Rrange)
    else:
        Rrange = [numpy.logspace(log10(0.99*Ri.min()),
                                 log10(1.01*Ri.max()),
                                 2**k)
                  for Ri in R]
        R = [numpy.append(0, Ri) for Ri in R]
        Rrange = [numpy.append(0, Ri) for Ri in Rrange]
        R = numpy.array(R)
        Rrange = numpy.array(Rrange)
    angles = numpy.linspace(0, 2*numpy.pi, 540)
    val1 = numpy.append(val1, [Rrange, angles])
    #mle = optimize.minimize(
    jfixed = (prior_types == 'fixed') | (prior_types == 'read')
    jfree = ~jfixed
    # identify the function. Raises an AttributeError if not found
    function = getattr(models, model)
    sat_profile = getattr(nfw, sat_profile_name)
    group_profile = getattr(nfw, group_profile_name)
    # print input stuff and configuration here
    #out = open(output, 'w')
    #print >>out, '#', '  '.join(params[jfree])
    #out.close()
    hdrfile = '.'.join(output.split('.')[:-1]) + '.hdr'
    print 'Printing header information to', hdrfile
    hdr = open(hdrfile, 'w')
    print >>hdr, 'Started', ctime()
    print >>hdr, 'datafile', ','.join(datafile)
    print >>hdr, 'cols', argv[1+Ndatafiles]
    print >>hdr, 'covfile', covfile
    print >>hdr, 'covcols', covcols
    print >>hdr, 'model %s' %model
    print >>hdr, 'sat_profile %s' %sat_profile_name
    print >>hdr, 'group_profile %s' %group_profile_name
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
    # run chain
    ndim = len(val1[(jfree)])
    if len(po) != ndim:
        msg = 'ERROR: Not all starting points defined for uniform variables.'
        msg += ' Exiting.'
        print msg
        exit()
    print 'po =', po
    po = po * numpy.random.uniform(0.99, 1.01, size=(nwalkers,ndim))
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
                size = (nwalkers*nsteps/thin, int(f[:-1]))
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


    ## TESTING
    #po = numpy.array([9, 11.5, 12., 0.1,
                      #0.45, 13.52, 13.55, 14.08])
    #log = False
    #t = R
    #n = 20
    #x = numpy.linspace(0, 1, n)
    #if log:
        #Mrange = numpy.logspace(8, 12, n)
    #else:
        #Mrange = numpy.linspace(1e8, 1e12, n)
    ##rtrange = numpy.linspace(0.05, 0.5, 10)
    ##rtrange = numpy.append(numpy.arange(0.02, 0.11, 0.02),
                           ##numpy.arange(0.15, 0.31, 0.05))
    ##rtrange = numpy.append(rtrange, 0.40)
    #rtrange = numpy.linspace(0.1, 1.0, 10)
    #nrt = len(rtrange)
    #print rtrange
    #import pylab
    #from itertools import izip
    #from matplotlib import cm, colors, ticker
    #cmap = pylab.get_cmap('brg')
    #cNorm  = colors.Normalize(vmin=0, vmax=1)
    #scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    #colors = [scalarMap.to_rgba(i) for i in x]
    #chi2 = 999 * numpy.ones(n)
    #Msat_rt = numpy.zeros((n,nrt))
    ##print Msat_rt.shape
    #fig, axes = pylab.subplots(figsize=(30,7), ncols=len(rtrange), nrows=2)
    ##fig = pylab.figure(figsize=(35,7))
    ##axes = [[pylab.subplot(2, nrt+2, i+1) for i in xrange(nrt)],
            ##[pylab.subplot(2, nrt+2, nrt+i+1) for i in xrange(nrt)]]
    #topaxes = []
    #for ax in axes[1]:
        #topaxes.append(ax.twiny())
    #for i, color in izip(count(), colors):
        ##print i
        #logMsat1 = numpy.log10(Mrange[i])
        #po[0] = logMsat1
        #for j, rt in enumerate(rtrange):
            #po[3] = rt
            #lnP, model = lnprob(po, R, esd, icov, function, params,
                                #prior_types[jfree], val1, val2, val3, val4,
                                #jfree, lnprior, likenorm,
                                #sat_profile, group_profile,
                                #rng_obsbins, fail_value,
                                #array, dot, inf, izip, outer, pi)
            #if model[-2] < chi2[j]:
                #chi2[j] = model[-2]
            #Msat_rt[i][j] = model[-5][0]
            #axes[0][j].errorbar(R[0][1:], esd[0], yerr=esd_err[0], fmt='ko')
            #axes[0][j].plot(t[0][1:], model[0][0], color=color, zorder=-1)
            #axes[1][j].plot(Mrange[i], model[-2], 'o', color=color,
                            #ms=12, mec=color)
            #axes[0][j].set_title(r'$r_t=%.2f\,h^{-1}{\rm Mpc}$' %rt)
    #Msat_rt = Msat_rt.T
    ##print Msat_rt
    ##for i in xrange(nrt):
        ##print rtrange[i], Msat_rt[i]
    #axes[0][0].set_ylabel(r'$\Delta\Sigma\,(h\,\mathrm{M_\odot pc^{-2}})$')
    #axes[1][0].set_ylabel(r'$\chi^2$')
    #if log:
        #for i in xrange(nrt):
            #axes[1][i].set_xscale('log')
            #axes[1][i].set_xlabel(r'$M_{\rm sub,200}\,(h^{-1}{\rm M_\odot})$')
    #else:
        #xticklabels = numpy.arange(0, 1.01, 0.2)
        #for i in xrange(nrt):
            #axes[1][i].set_xticklabels(xticklabels)
            #xlabel = r'$M_{\rm sub,200}\,(10^{12}h^{-1}{\rm M_\odot})$'
            #axes[1][i].set_xlabel(xlabel)
            #xlim = (10**Msat_rt[i].min(), 10**Msat_rt[i].max())
            #topaxes[i].set_xlim(*xlim)
            #topaxes[i].xaxis.set_major_locator(ticker.MaxNLocator(4))
            #topaxes[i].set_xlabel(r'$M_{\rm sub}(<r_t)\,h^{-1}{\rm M_\odot}$')
    #for i in xrange(nrt):
        #axes[0][i].set_xscale('log')
        #axes[0][i].set_xlabel(r'$R\,(h^{-1}{\rm Mpc})$')
        #axes[0][i].set_xlim(0.02, 2)
        #axes[0][i].set_ylim(-20, 100)
        #axes[0][i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        #axes[1][i].axhline(chi2[i]+1, ls='--', color='k')
        #axes[1][i].axhline(chi2[i]+4, ls=':', color='k')
        #axes[1][i].yaxis.set_major_locator(ticker.MultipleLocator(1))
        ##topaxes[i].xaxis.set_major_locator(ticker.MultipleLocator(1))
    #fig.tight_layout()
    #pylab.savefig('testplots/chi2_msat.png', format='png')
    #exit()
    ## END TESTING


    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    threads=threads,
                                    args=(R,esd,icov,function,
                                          params,prior_types[jfree],
                                          val1,val2,val3,val4,
                                          jfree,lnprior,likenorm,
                                          sat_profile,group_profile,
                                          rng_obsbins,fail_value,
                                          array,dot,inf,izip,outer,pi))
                                          #isfinite,log,log10
                                          #outer,sqrt,zeros))
    # burn-in
    if nburn > 0:
        #to = time()
        pos, prob, state = sampler.run_mcmc(po, nburn)
        sampler.reset()
        #print 'Burn-in phase finished in %.2f min' %((time() - to) / 60)
        print 'Burn-in phase finished ({0})'.format(ctime())
    else:
        pos = po
    # incrementally save output
    #to = time()
    chi2 = [zeros(nwalkers*nsteps/thin) for i in xrange(4)]
    nwritten = 0
    for i, result in enumerate(sampler.sample(pos, iterations=nsteps,
                                              thin=thin)):
        # make sure that nwalkers is a factor of this number!
        if i*nwalkers % 10000 == nwalkers:
            out = write_to_fits(output, chi2, sampler, nwalkers, thin,
                                params, jfree, metadata, meta_names,
                                fits_format, i, nwritten, Nobsbins,
                                BinTableHDU, Column, ctime, enumerate,
                                isfile, izip, transpose, xrange)
            metadata, nwriten = out
            #print 'nwritten =', nwritten, i
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
                  BinTableHDU, Column, ctime, enumerate,
                  isfile, izip, transpose, xrange)
    print 'Everything saved to {0}!'.format(output)
    return

def lnprob(theta, R, esd, icov, function, params,
           prior_types, val1, val2, val3, val4, jfree, lnprior, likenorm,
           sat_profile, group_profile, rng_obsbins, fail_value,
           array, dot, inf, izip, outer, pi):
           #array, dot, inf, izip, isfinite,
           #log, log10, sqrt):
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
    lnprior[j] = array([-(v - v1)**2 / (2*v2**2) - _log(2*pi*v2**2)/2
                        if v3 <= v <= v4 else -inf
                        for v, v1, v2, v3, v4
                        in izip(theta[j], v1free[j], v2free[j],
                                v3free[j], v4free[j])])
    j = (prior_types == 'lognormal')
    lnprior[j] = array([-(log10(v) - v1)**2 / (2*v2**2) - _log(2*pi*v2**2)/2
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
    model = function(v1, R, sat_profile, group_profile)
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
                  Nobsbins, BinTableHDU, Column, ctime, enumerate,
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
            if len(data) == 1:
                # not working but not sure will ever use it
                metadata[0][0][j*nwalkers:(j+1)*nwalkers] = data[0]
                if shape[1] == 4:
                    metadata[0][1][j*nwalkers:(j+1)*nwalkers] = data[1]
                    metadata[0][2][j*nwalkers:(j+1)*nwalkers] = data[2]
                    metadata[0][3][j*nwalkers:(j+1)*nwalkers] = data[3]
            # this is the fiducial case with the three bins
            # note that all values have three bins!
            else:
                for k, datum in enumerate(data):
                    metadata[k][0][j*nwalkers:(j+1)*nwalkers] = datum[0]
                    metadata[k][1][j*nwalkers:(j+1)*nwalkers] = datum[1]
                    metadata[k][2][j*nwalkers:(j+1)*nwalkers] = datum[2]
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

if __name__ == '__main__':
    main()
