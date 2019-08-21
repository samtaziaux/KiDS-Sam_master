Release log
===========

**2018 May 7** - Release v1.6.3. Features: - Fixed bug in the Tinker
halo bias function used in the halo model.

**2018 January 19** - Release v1.6.2. Features: - ESD production fully
tested in Python 3 after several bug fixes

**2018 January 12** - Release v1.6.1. Features: - Update to hmf 3.0.3
and newer Python CAMB package - Bug fixes in halo model imports

**2018 January 10** - Release v1.6.0. Features: - The lens files can now
be either FITS or ascii files. - The user may specify the names of the
required columns (ID,RA,Dec[,z]) in the configuration file, instead of
having to modify the input table to comply with GAMA naming conventions
- The ID column is no longer required (one will be created if none is
present in the input table) - Minor bug fixes in the covariance
calculation - Small style changes and code clean-up - The ESD production
can be run on DR2 again

**2017 July 20** - Release v1.5.1. Features: - Bug fix for ID's

**2017 July 19** - Release v1.5.0. Features: - Updates to halo model to
include CAMB and hmf 2.0.5 as defaults. - Added analytical covariance
estimation to halo model (not quite done yet) - Changes to code to make
it Python 3 ready

**2016 November 11** - Release v1.4.0. Features: - Major bug fix in
calculating Sigma\_crit and tangential shear (inproper removal of
sources in front of the lens). - Added unit test. - Solved some problems
with multiprocessing.

**2016 October 6** - Release v1.3.4. Features: - Bug fixes.

**2016 September 30** - Release v1.3.3. Features: - Solved bug when
pipeline stopped in the middle of calculation without any errors.

**2016 September 29** - Release v1.3.2. Features: - Solved some issues
when installing pipeline and running demo config files.

**2016 September 14** - Release v1.3.1. Features: - Solved problems with
multiprocessing.

**2016 September 13** - Release v1.3.0. Features: - More user friendly
output folder and file structure. - Added m-correction to the ESD
production. - Added user selectable epsilon cut (z\_s > z\_l + epsilon).
- Proper paths to catalogues as used in Leiden. - Solved problems when
plotting data. - Conversion from physical units to comoving units when
using halo.py model. - Solved problems with saving the progress of MCMC
chains. - Various bug fixes.

**2016 June 10** - Release v1.2.3. Features: - Bug fixes to
compatibility issues with Numpy and blinding when running bootstrap.

**2016 June 10** - Release v1.2.2. Features: - Bug fixes to filename
generation.

**2016 May 31** - Release v1.2.1. Features: - Bug fixes.

**2016 May 31** - Release v1.2.0 Features: - Option to use The-wiZZ for
obtaining ``n(z)`` of sources. - Addition of miscentering into
``halo.py``, inclusion of simpler HOD and point mass approximation for
stellar contribution to lensing signal. - Fixed issues with halo model
and sampler when using only one bin. - If using angular scales in ESD
production, pipeline returns shears instead of ESD. - KiDS+GAMA matching
algorithm fixed. - Multiple ID files now supported.

**2016 Mar 5** - Release v1.1.2. Features: - Crucial bug fixes.

**2016 Feb 29** - Release v1.1.1. Features: - Fixed demo config file to
avoid confusion and problems when testing the pipeline for the first
time.

**2016 Feb 29** - Release v1.1.0. Features: - The syntax for
``hm_output`` parameters in the configuration file is now simpler and
makes it easier to update the number of observable bins. Details can be
found in ``demo/ggl_demo_nfw_stack.txt``. Configuration files must be
updated accordingly.

-  Cosmological parameters are no longer hard-coded in the halo model
   and can therefore be modified by the user in the config file. This
   also means the user can fit for them in the MCMC.

-  Additional parameters in the halo model that can now be modified from
   the config file:

   -  Conversion from average central stellar mass to average satellite
      stellar mass, ``Ac2s``;
   -  Concentration of the radial distribution of satellite galaxies,
      ``fc_nsat``.

-  General clean up of most modules

**2016 Feb 23** - First release v1.0.0. Features: - Installed with a
single command-line, including all dependencies - User interaction
through a single configuration file - ESD production: - Compatible with
both KiDS-450 (Feb 2016) and KiDS-DR2 (Mar 2015) - Halo model: - Both an
NFW stack and a full halo model (relies on
`hmf <https://github.com/steven-murray/hmf>`__ module for the mass
function) - MCMC sampling: - Uses ``emcee`` to sample parameters

