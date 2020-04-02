from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import six

from . import io


def axlabel(return_value):
    # probably want to have units in `setup` as well
    labels = {'esd': r'\Delta\Sigma',
              'kappa': r'\kappa',
              'power': r'\P(k)',
              'sigma': r'\Sigma',
              'xi': r'\xi',
              'wp': r'w_{p}',
              'esd_wp': r'unit'}
    return labels[return_value]


def covariance(R, cov, output=None):
    # still need to add axes labels (complicated by the fact that
    # bins aree usually in log space), and also add a colorbar
    # with a label i.e., this function should probably take `setup`
    # as an (optional) argument
    fig, axes = plt.subplots(
        figsize=(10,8), nrows=cov.shape[0], ncols=cov.shape[0])
    #vmin, vmax = np.percentile(cov, [1,99])
    vmin, vmax = -1.0, 1.0
    if len(R) == 1:
        axes = [[axes]]
    for m, axm in enumerate(axes):
        for n, axmn in enumerate(axm):
            axmn.imshow(cov[m][-n-1], origin='lower', interpolation='nearest',
                        vmin=vmin, vmax=vmax)
            #axmn.imshow(cov[m][-n-1]/np.sqrt(np.outer(np.diag(cov[m][-n-1]),
            #            np.diag(cov[m][-n-1].T))),
            #            origin='lower', interpolation='nearest',
            #            vmin=vmin, vmax=vmax)
    fig.tight_layout(pad=0.4)
    if output:
        save(output, fig, 'covariance')
    else:
        plt.show()
    return


def save(output, fig=None, name='', tight=True, close=True,
         verbose=True, pad=0.4, **kwargs):
    if fig is None:
        fig = plt
    if tight:
        fig.tight_layout(pad=pad, **kwargs)
    plt.savefig(output)
    if close:
        plt.close()
    if verbose:
        if name:
            name = ' {0}'.format(name)
        print('Saved {0} to {1}'.format(name, output))
    return


def signal(R, y, yerr, Nobsbins, model=None, observable='', fig=None, axes=None,
           output=None, **save_kwargs):
    """
    observable should be the name of the thing on the y axis

    how exactly should we decide when the y axis should be
    log scale?
    """
    if Nobsbins == 1:
        R = [R]
        y = [y]
        yerr = [yerr]
    n = len(R)
    if axes is None:
        fig, axes = plt.subplots(figsize=(4*n,4), ncols=n)
    elif fig is None:
        fig = plt
    if observable:
        ylabel = axlabel(observable)
    if model is None:
        model = [[] for _ in R]
    if n == 1:
        axes = [axes]
    for ax, Ri, yi, ei, fi in zip(axes, R, y, yerr, model):
        # this would actually only happen when calling this function
        # from within the model
        if Ri[0] == 0 and len(Ri) == len(yi) + 1:
            Ri = Ri[1:]
        ax.errorbar(Ri, yi, yerr=ei, fmt='ko', ms=10)
        if len(fi) == len(Ri):
            ax.plot(Ri, fi, 'r-', lw=3)
        ax.set_xscale('log')
        # do something fancier later
        ax.set_xlabel('$k$' if observable == 'power' else '$R$')
        ax.set_yscale('log')
    if observable:
        axes[0].set_ylabel(r'${0}$'.format(ylabel))
    #if np.all(y-yerr > 0):
        #for ax in axes:
            #ax.set_yscale('log')
    ylims = np.array([ax.get_ylim() for ax in axes])
    ylim = (ylims[:,0].min(), ylims[:,1].max())
    for ax in axes:
        ax.set_ylim(*ylim)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.tight_layout(pad=0.4)
    if output:
        save(output, fig, 'signal', **save_kwargs)
    else:
        plt.show()
    return fig, axes


def signal_from_files(data_files, cov_file, data_cols=(0,1,4), cov_cols=(4,6),
                      exclude=None, **kwargs):
    """Plot signal from files as produced by kids_ggl

    Load the data and then call ``signal``. You may pass any keyword
    arguments accepted by ``signal``

    Only one observable implemented for now
    """
    if isinstance(data_files, six.string_types):
        data_files = sorted(glob(data_files))
    if '*' in cov_file:
        cov_file = sorted(glob(cov_file))
        assert len(cov_file) == 1, \
            'more than one covariance file provided'
        cov_file = cov_file[0]
    R, y = io.load_datapoints(data_files, data_cols, exclude)
    Nobsbins, Nrbins = R.shape
    cov = io.load_covariance(cov_file, cov_cols, Nobsbins, Nrbins, exclude)
    yerr = cov[3]
    fig, axes = signal(R, y, yerr, **kwargs)
    return fig, axes



