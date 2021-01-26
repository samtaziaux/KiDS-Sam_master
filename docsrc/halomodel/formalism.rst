======================
 Halo Model Formalism
======================

The *Halo Model* is the formalism resulting from the assumption that all galaxies reside in *haloes*: collasped, spherical dark 
matter structures characterized by a mass *M* at redshift :math:`z`. It is usually calculated in Fourier space for convenience, so 
that observables are calculated in terms of their power spectra. Here we describe the essential elements of the halo model formalism 
implemented in ``kids_ggl``. For more details please refer to `van Uitert et al. 2016 
<https://ui.adsabs.harvard.edu/abs/2016MNRAS.459.3251V/abstract>`_ and `Dvornik et al. 2017 
<https://ui.adsabs.harvard.edu/abs/2017MNRAS.468.3251D/abstract>`_ and `2018
<https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.1240D/abstract>`_. As much as possile, the notation in this page is consistent with 
that used in the default configuration file, which may create some inconsistencies with the notation in those references.

The galaxy-matter power spectrum can then be separated into 
*within-halo* and *between-halo* contributions, commonly referred to as the *1-halo* and *2-halo* terms, respectively:

.. math::
    P_\mathrm{gm}(k,z) = P_\mathrm{gm}^\mathrm{1h}(k,z) + P_\mathrm{gm}^\mathrm{2h}(k,z)

where each term is itself a combination of the contribution of *central* and *satellite* galaxies within these halos:

.. math::
    P_\mathrm{gm}^\mathrm{1h}(k,z) = f_\mathrm{c}\cdot P_\mathrm{cm}^\mathrm{1h}(k,z) + (1-f_\mathrm{c})\cdot P_\mathrm{sm}^\mathrm{1h}(k,z)

    P_\mathrm{gm}^\mathrm{2h}(k,z) = f_\mathrm{c}\cdot P_\mathrm{cm}^\mathrm{2h}(k,z) + (1-f_\mathrm{c})\cdot P_\mathrm{sm}^\mathrm{2h}(k,z)

where :math:`f_\mathrm{c}\equiv \bar n_\mathrm{c}/\bar n_\mathrm{g}` is the fraction of central galaxies in the sample being used.


Power spectra
*************

Let us define

.. math::
    \mathcal{H}_\mathrm{m}(k|M,z) = \frac{M}{\bar\rho_\mathrm{m}}\, \tilde{u}_\mathrm{h}(k|M,z)

..
    \mathcal{H}_\mathrm{c}(k|M,z) = \frac{\langle N_\mathrm{c}|M,z\rangle}{\bar n_\mathrm{c}}\, u_\mathrm{m}(k|M,z) \, \left(1-p_\mathrm{off}+p_\mathrm{off}\,\exp\left[-0.5k^2(r_\mathrm{s}\mathcal{R}_\mathrm{off})^2\right] \right)

.. math::
     \mathcal{H}_\mathrm{c}(k|M,z) = \frac{\langle N_\mathrm{c}|M,z\rangle}{\bar n_\mathrm{c}}

.. math::
    \mathcal{H}_\mathrm{s}(k|M,z) = \frac{\langle N_\mathrm{s}|M,z\rangle}{\bar n_\mathrm{s}}\, \tilde u_\mathrm{s}(k|M,z),

where

* :math:`M` is the halo mass;
* :math:`\bar\rho_\mathrm{m}` is the mean matter density;
* :math:`\langle N_\mathrm{c}|M,z\rangle` and :math:`\langle N_\mathrm{s}|M,z\rangle` are the expected number of central and satellite galaxies in a halo of mass :math:`M` and redshift :math:`z`; and
* :math:`\tilde{u}_\mathrm{h}(k|M,z)` and :math:`\tilde u_\mathrm{s}(k|M,z)` are the Fourier transforms of the spatial distribution of mass and satellite galaxies. (We add a prescription to account for central galaxy miscentring below.).

That is, the various :math:`\mathcal{H}(k|M,z)` terms correspond to the products of the expected number density and the Fourier 
transform of the spatial distribution of each component, and

.. math::
    n_i(z) = \int\mathrm{d}M \,\langle N_i|M,z\rangle n_\mathrm{h}(M,z)

is the number density of galaxies of type :math:`i` (where :math:`i=` ':math:`c`' for centrals or ':math:`s`' for satellites) 
integrated over mass and redshift bin, and :math:`n_\mathrm{h}(M,z)` is the number density of haloes of mass :math:`M` and redshift 
:math:`z`, also known as the *halo mass function*. Theoretically, the total number of galaxies of type :math:`i` in a given redshift 
interval :math:`[z_1,z_2]` is then given by

.. math::
    \bar n_i = \int_{z_1}^{z_2}\mathrm{d}z\,\frac{c\chi^2}{H(z)}\, n_i(z),

where

.. math::
    \mathrm{d}V_\mathrm{C}(z)\equiv\mathrm{d}z\,\frac{c\chi^2}{H(z)}

is the comoving volume element per unit redshift per unit steradian. It is common practice, though, to simply use a single 
effective redshift,

.. math::
    \bar n_i \equiv n_i(z=z_\mathrm{eff}).

Then, the power spectra can be expressed as:

.. math::
    P_{i\mathrm{m}}^\mathrm{1h}(k,z) = \int\mathrm{d}V_\mathrm{C}(z)\int_0^\infty \mathrm{d}M\,n_\mathrm{h}(M,z)\,\mathcal{H}_\mathrm{x}(k,M,z)\,\mathcal{H}_\mathrm{m}(k,M,z)

.. math::
    P_{i\mathrm{m}}^\mathrm{2h}(k,z) = \int\mathrm{d}V_\mathrm{C}(z)P_\mathrm{m}(k,z) \int_0^\infty\,\mathrm{d}M_1\,n_\mathrm{h}(M_1,z)\,b_\mathrm{h}(M_1,z)\,\mathcal{H}_\mathrm{x}(k,M_1,z)

    \int_0^\infty\,\mathrm{d}M_2\,n_\mathrm{h}(M_2,z)\,b_\mathrm{h}(M_2,z)\,\mathcal{H}_\mathrm{m}(k,M_2,z),


and again :math:`i` can be either ':math:`c`' or ':math:`s`'.

In :code:`KiDS-GGL`, integrating over (lens) galaxy redshifts as above can be activated through the parameter :code:`nzlens`, which 
requires the user to provide empirical values for :math:`n(z)` for the lens galaxies, and these empirical values will be used in 
place of the comoving volume integral above. The power spectra are then calculated as

.. math::
    P_{i\mathrm{m}}^\mathrm{1h}(k,z) = \int\mathrm{d}n_\mathrm{lens}(z)\int_0^\infty \mathrm{d}M\,n_\mathrm{h}(M,z)\,\mathcal{H}_\mathrm{x}(k,M,z)\,\mathcal{H}_\mathrm{m}(k,M,z)

.. math::
    P_{i\mathrm{m}}^\mathrm{2h}(k,z) = \int\mathrm{d}n_\mathrm{lens}(z)P_\mathrm{m}(k,z) \int_0^\infty\,\mathrm{d}M_1\,n_\mathrm{h}(M_1,z)\,b_\mathrm{h}(M_1,z)\,\mathcal{H}_\mathrm{x}(k,M_1,z)

    \int_0^\infty\,\mathrm{d}M_2\,n_\mathrm{h}(M_2,z)\,b_\mathrm{h}(M_2,z)\,\mathcal{H}_\mathrm{m}(k,M_2,z),



The Halo Occupation Distribution
********************************

The halo occupation distribution (HOD hereafter) is a commonly used analytical prescription that describes how galaxies populates 
dark matter haloes -- the :math:`\langle N_i|M,z\rangle` above. For reference we describe the version implemented by default in 
``kids_ggl`` below, which is based on the models used in `van Uitert et al. 2016 
<https://ui.adsabs.harvard.edu/abs/2016MNRAS.459.3251V/abstract>`_ and `Dvornik et al. 2017 
<https://ui.adsabs.harvard.edu/abs/2017MNRAS.468.3251D/abstract>`_.

Central galaxies populate haloes following a lognormal distribution in halo mass, independent of redshift:

.. math::
    \langle N_c|M,z \rangle = \frac1{\sqrt{2\pi}\log(10)\,\sigma\,m_0}\exp\left[-\frac{\log_{10}(m_\star/m_0)^2}{2\sigma^2}\right]


*TO BE CONTINUED...*


Finally, there is a "Poisson term" for satellite galaxies that specifies the level of stochasticity in the HOD,

.. math::

    \beta(M) \equiv \frac{\langle N_\mathrm{s}(N_\mathrm{s}-1)|M\rangle}{\langle N_\mathrm{s}|M\rangle^2}

See Eq. (42) of Dvornik et al. (2018) for more details.
