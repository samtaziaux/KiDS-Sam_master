"""KiDS-GGL Debugging utilities"""
from matplotlib import pyplot as plt
import numpy as np

from plottery.plotutils import colorscale, savefig, update_rcParams


update_rcParams()


def plot_profile_mz(R, profile, z, mx=None, z0=1, m0=1e14, xscale='log',
                    yscale='log', ylabel='', output='', close=False, fs=14,
                    figaxes=None):
    """
    Parameters
    ----------
    R must have shape ([N,]M), where N is the number of redshift bins and
        M is the number of radial bins
    profile must have shape (N,P,M) where P is the number of mass bins
    """
    z = np.squeeze(z)
    if len(R.shape) == 1:
        R = np.array([R for i in range(z.size)])
    if mx is None:
        lnm = np.log(m)
    else:
        lnm = np.log(mx)
    logm = np.log10(np.exp(lnm))
    idx_z = np.argmin(np.abs(z-z0))
    idx_m = np.argmin(np.abs(lnm-np.log(m0)))
    if figaxes is None:
        fig, axes = plt.subplots(figsize=(14,5), ncols=2)
    else:
        fig, axes = figaxes
    # varying redshift
    axes[0].set_title('log10(m)={0:.2f}'.format(logm[idx_m]), fontsize=fs)
    colors, cmap = colorscale(z)
    for i in range(z.size):
        axes[0].plot(R[i], profile[i,idx_m], color=colors[i])
    if figaxes is None:
        plt.colorbar(cmap, label='Redshift', ax=axes[0])
    # varying mass
    axes[1].set_title('z={0:.2f}'.format(z[idx_z]), fontsize=fs)
    colors, cmap = colorscale(logm)
    for i in range(lnm.size):
        axes[1].plot(R[idx_z], profile[idx_z,i], color=colors[i])
    if figaxes is None:
        plt.colorbar(cmap, label='log10 m', ax=axes[1])
    for ax in axes:
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if ylabel:
            ax.set_ylabel(ylabel)
#     axes[1].set_x
    if output:
        savefig(output, fig=fig, close=close)
    else:
        fig.tight_layout()
    return fig, axes
