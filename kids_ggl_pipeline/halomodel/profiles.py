"""
Profile object definitions

"""
from __future__ import absolute_import, division

from astropy import constants as ct, units as u
from astropy.cosmology import Planck15
from astropy.units import Quantity
import numpy as np
#from numpy.fft import fft, rfft
from scipy.integrate import quad, trapz
from scipy.special import sici

from ..helpers.decorators import array, inMpc


#class Profile(BaseLensing):
class Profile(object):

    def __init__(self, mvir, z, cosmo=Planck15):
        """Initialize a profile object

        Parameters
        ----------
        mvir : float or np.ndarray of floats
            total mass(es) (definition arbitrary)
        z : float or np.ndarray of floats
            redshift(s)
        cosmo : `astropy.cosmology.FLRW` object, optional
            cosmology object
        """
        if isinstance(mvir, Quantity):
            mvir = mvir.to(u.Msun).value
        if not np.iterable(mvir):
            mvir = np.array([mvir])
        self.mvir = mvir
        self.z = self._define_array(z)
        # need this for rho_bg
        #super(Profile, self).__init__(z, cosmo=cosmo)
        # alias
        self.esd = self.excess_surface_density

    @property
    def _one(self):
        if self.__one is None:
            self.__one = u.dimensionless_unscaled
        return self.__one

    ### private methods ###

    def _define_array(self, x):
        if not np.iterable(x):
            return x * np.ones_like(self.mvir)
        return x

    ### methods ###

    #@inMpc
    #@array
    #def sigma_from_xi(self, R):
        #"""Also requires xi!"""
        

    @inMpc
    @array
    def surface_density(self, R, integration_limit=100,
                        Rp=1000, **kwargs):
        """Calculate surface density by numerical integration"""
        if not np.iterable(Rp):
            # using only one Rp simplifies things. Will test
            # whether it makes a difference later.
            Rp = np.logspace(-10, np.log10(integration_limit), Rp) \
                * self.rvir.max()
#         if len(R.shape) == 1:
#             R = np.hypot(*np.meshgrid(Rp, R))
#         else:
#             Rp = np.logspace(-10, np.log10(integration_limit), R.shape[1])
        R = np.hypot(*np.meshgrid(Rp, R[:,0]))
        return 2 * trapz(self.density(R), Rp, axis=1)

    @inMpc
    @array
    def enclosed_surface_density(self, R, integration_limit=100,
                                 Rp=500, **kwargs):
        """
        For <~20 data points this takes order 0.2 s. For 100 data points
        it takes ~1 second. (This for a gNFW)

        Exercise: re-write without list comprehension
        """
        Rp = np.expand_dims(
            [np.logspace(-10, logRi, Rp) for logRi in np.log10(R)], -1)
        return 2 / R**2 \
            * trapz([Rpi*self.surface_density(Rpi, **kwargs)
                     for Rpi in Rp], Rp, axis=1)
        # what's below has the right shape but gives the wrong answer
        # but something like that is probably necessary to speed it up
#         good = 2 / R**2 \
#             * trapz([Rpi*self.surface_density(Rpi, **kwargs)
#                      for Rpi in Rp], Rp, axis=1)
#         print('Rp =', Rp.shape)
#         print('R =', R.shape)
#         sd = self.surface_density(Rp, **kwargs)
#         print('sd =', sd.shape)
#         Rpsd = Rp * sd[:,None]
#         print('Rpsd =', Rpsd.shape)
#         test = 2 / R**2 \
#             * trapz(Rp*self.surface_density(Rp, **kwargs)[:,None], Rp, axis=1)
#         print('good?', np.allclose(test, good), test.shape, good.shape)
#         return test

    def excess_surface_density(self, R, **kwargs):
        """Excess surface density at projected distance(s) R

        The excess surface density or ESD is the lensing observable
        in physical units, and is calculated as:
            ESD(R) = Sigma(<R) - Sigma(R)
        where Sigma(<R) is the average surface density within R and
        Sigma(R) is the surface density at distance R

        Parameters
        ----------
        R : float or array of float
            projected distance(s)
        kwargs : dict, optional
            passed to both `self.enclosed_surface_density` and
            `self.surface_density`
        """
        return self.enclosed_surface_density(R, **kwargs) \
            - self.surface_density(R, **kwargs)

    def fourier(self, R, k, n=10000, **kwargs):
        msg = 'Numerical calculation of the Fourier transform not yet' \
              ' implemented'
        raise ValueError(msg)


class BaseNFW(Profile):

    def __init__(self, mvir, c, z, overdensity=200, background='c',
                 cosmo=Planck15):
        assert background in 'cm', \
            "background must be either 'c' (critical) or 'm' (mean)"
        super(BaseNFW, self).__init__(mvir, z, cosmo=cosmo)
        self._background = background
        self._concentration = self._define_array(c)
        self._overdensity = overdensity
        self._deltac = None
        self._rs = None
        self._rvir = None
        self._sigma_s = None

    ### attributes ###

    @property
    def background(self):
        return self._background

    @property
    def c(self):
        return self._concentration

    @property
    def deltac(self):
        if self._deltac is None:
            self._deltac = (self.overdensity * self.c**3 / 3) \
                / (np.log(1+self.c) - self.c/(1+self.c))
        return self._deltac

    @property
    def overdensity(self):
        return self._overdensity

    @property
    def rs(self):
        if self._rs is None:
            self._rs = self.rvir / self.c
        return self._rs

    @property
    def rvir(self):
        if self._rvir is None:
            self._rvir = \
                (self.mvir / (4*np.pi/3) \
                 / (self.overdensity*self.rho_bg))**(1/3)
        return self._rvir

    @property
    def sigma_s(self):
        if self._sigma_s is None:
            self._sigma_s = self.rs * self.deltac * self.rho_bg
        return self._sigma_s


class gNFW(BaseNFW):

    def __init__(self, mvir, c, alpha, z, **kwargs):
        super(gNFW, self).__init__(mvir, c, z, **kwargs)
        self.alpha = self._define_array(alpha)

    ### main methods ###

    @inMpc
    @array
    def density(self, R):
        return self.deltac * self.rho_bg \
            / ((R/self.rs)**self.alpha * (1+R/self.rs)**(3-self.alpha))


class NFW(BaseNFW):
    """Navarro-Frenk-White profile

    does not yet handle conversion from one overdensity to another
    """

    def __init__(self, mvir, c, z, **kwargs):
        super(NFW, self).__init__(mvir, c, z, **kwargs)

    ### main methods ###

    @inMpc
    @array
    def density(self, R):
        return self.deltac * self.rho_bg / (R/self.rs * (1+R/self.rs)**2)

    @inMpc
    @array
    def surface_density(self, R):
        """Surface density at distance(s) R"""
        x = R / self.rs
        s = np.ones_like(x) / 3
        s[x == 0] = 0
        j = (x > 0) & (x < 1)
        s[j] = (1 - 2*np.arctanh(((1-x[j]) / (1+x[j]))**0.5) \
                    / (1 - x[j]**2)**0.5) \
               / (x[j]**2 - 1)
        j = x > 1
        s[j] = (1 - 2*np.arctan(((x[j]-1) / (1+x[j]))**0.5) \
                    / (x[j]**2 - 1)**0.5) \
               / (x[j]**2 - 1)
        return 2 * self.sigma_s * s

    @inMpc
    @array
    def enclosed_surface_density(self, R):
        """Surface density enclosed within R"""
        x = R / self.rs
        s = np.ones_like(x) + np.log(0.5)
        s[x == 0] = 0
        j = (x > 0) & (x < 1)
        s[j] = (np.log(0.5*x[j]) \
                + 2 * np.arctanh(((1-x[j])/(1+x[j]))**0.5) \
                  / (1 - x[j]**2)**0.5) \
               / x[j]**2
        j = x > 1
        s[j] = (2 * np.arctan(((x[j]-1)/(1+x[j]))**0.5) / (x[j]**2-1)**0.5 \
                + np.log(0.5*x[j])) / x[j]**2
        return 4 * self.sigma_s * s

    @array
    def fourier(self, k):
        uk = np.zeros_like(k)
        ki = k * self.rs
        bs, bc = sici(ki)
        asi, ac = sici((1+self.c)*ki)
        return 4 * np.pi * self.rho_bg * self.deltac * self.rs**3 / self.mvir \
            * (np.sin(ki)*(asi-bs) - (np.sin(self.c*ki) / ((1+self.c)*ki)) \
               + np.cos(ki)*(ac-bc))


class tNFW5(BaseNFW):

    def __init__(self, mvir, c, rt, z, **kwargs):
        super(NFW, self).__init__(mvir, c, z, **kwargs)
        self.rt = rt

    ### methods ###

    @inMpc
    @array
    def density(self, R):
        return

    ### auxiliary methods ###

    @inMpc
    def F(self, R):
        x = R / self.rs
        f = np.ones_like(x)
        f[x < 1] = np.log(1/x[x<1] + (1/x[x<1]**2 - 1)**0.5) \
            / (1 - x[x<1]**2)**0.5
        f[x > 1] = np.arccos(1/x[x>1]) / (x[x>1]**2 - 1)**0.5
        return f

