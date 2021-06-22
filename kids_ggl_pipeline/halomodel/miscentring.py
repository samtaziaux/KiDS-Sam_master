import numpy as np


__all__ = ('gamma', 'rayleigh')

def miscentring(name, *args, **kwargs):
    if name == 'gamma':
        return gamma(*args, **kwargs)
    if name == 'rayleigh':
        return rayleigh(*args, **kwargs)
    msg = f'miscentering distribution {name} not implemented. Please' \
          f' specify one of {__all__}'
    raise ValueError(msg)


def gamma(Rmis, tau, Rcl):
    return Rmis/(tau*Rcl[:,None])**2 * np.exp(-Rmis/(tau*Rcl[:,None]))


def rayleigh(Rmis, tau):
    return Rmis/tau**2 * np.exp(-(Rmis/tau)**2)
