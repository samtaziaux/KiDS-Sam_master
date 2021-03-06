===========
 Procedure
===========

In a (small) nutshell, ESD calculation in ``kids_ggl`` proceeds as follows. The text below is mostly extracted (and somewhat 
updated) from |ref:dvornik17|_.


Since KiDS-450, ``kids_ggl`` uses a global source redshift *distribution*, rather than attempting to estimate redshifts for each 
indificual source galaxy, which in KiDS is calculated by re-weighting the combination of a large number of overlapping redshift 
surveys so it matches the lensing-weighted magnitude distribution of the lensed source sample; we refer to this approach as "DIR 
photo-`z`", and the resulting probability disstribution is labelled :math:`n(z_s)`. The KiDS implementation of this approach is 
described in |ref:hildebrandt17|_.

Given :math:`n(z_s)`, we calculate the critical surface density, :math:`\Sigma_\mathrm{c}`, for every lens-source pair, as

.. math::

    \Sigma_\mathrm{c,ls}^{-1} = \frac{4\pi G}{c^2} \int_0^\infty \mathrm{d}z_l\,p(z_l)\,D(z_l)
    \int_{z_l+\delta_z}^\infty \mathrm{d}z_s\,n(z_s)\,\frac{D(z_l,z_s)}{D(z_s)}

where :math:`D(z_l)`, :math:`D(z_s)`, and :math:`D(z_l,z_s)` are the angular diameter distances to the lens, to the source, and 
between lens and source; :math:`\delta_z` (referred to as ``z_epsilon`` in the configuration file) is necessary to remove 
unlensed objects with :math:`z_s>z_l`, which exist due to photo-z errors, and is set by default at 0.2 (see |ref:dvornik17|_); and
:math:`p(z_l)` is the probability distribution for each lens, modelled as a Gaussian with a (optionally redshift-dependent) width 
that is adjustable through the ``lens_pz_sigma`` parameter in the configuration file. The lensing signal is then calculated as

.. math::

    \Delta\Sigma(R) = \left[\frac{\sum_\mathrm{ls}w_\mathrm{ls}\epsilon_\mathrm{t}(1/\Sigma_\mathrm{c,ls}^{-1})}
    {\sum_\mathrm{ls}w_\mathrm{ls}}\right]
    \,\frac1{1+\mu}

where :math:`\mu` is the weighted average multiplicative bias correction, calibrated with tailored image simulations, that take into 
account all aspects of the selection of lensed sources in the KiDS analysis.


.. include:: ../reference-links.rst
