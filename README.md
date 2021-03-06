# KiDS-GGL
Galaxy-Galaxy Lensing pipeline production

*This Readme contains all the instructions needed to install, set up and run
the KiDS galaxy-galaxy lensing pipeline, which takes current KiDS and GAMA
catalogs and produces (or reads) an ESD and a covariance matrix and runs a
fitting module (i.e., halo model) with a given sampling technique.*

***2018-01-19: As of v1.6.2, the ESD production module is confirmed to be fully compatible with a Python 3 installation. We encourage all users to update their KiDS-GGL pipeline to this version (or newer).***


#### 1. Installation
    
    
**a)** Contact Cristóbal Sifón (cristobal.sifon@pucv.cl) to become a member
       of the KiDS-WL repository and join the KiDS-GGL team

**b)** Download [the latest stable version of the KiDS-GGL pipeline](https://github.com/KiDS-WL/KiDS-GGL/releases/latest) and unpack,

       cd path/to/kids_ggl_folder
       tar xvf KiDS-GGL-<version>.tar.gz

or, if you chose to download the `.zip` file,

        unzip KiDS-GGL-<version>.zip

where for instance `<version>=1.0.0`.

**c)** From within the same folder, run

        python setup.py install [--user]

where the `--user` flag is recommended so as not to require root privileges, and the package will typically be installed in `$HOME/.local/lib`. If instead you want to install it in a non-standard place, then type

        python setup.py install --prefix=path/to/installation/

The commands above will also install some additional packages required by the pipeline: `astropy>=1.1.0` for astronomical utilities (e.g., constants and units), `emcee>=2.1.0` for MCMC, `hmf>=3.0.1` for halo mass function utilities, `mpmath>=0.19` for mathematical utilities, and `numpy>=1.5.0`. Optionally, if you want to use the CAMB transfer functions in your halo models (preffered method), you should also instal CAMB using the following command: 

        pip install --egg camb

**d)** After the setup script has finished, you should have a copy of the `kids_ggl` executable somewhere in your `$PATH`, which means you can run it out-of-the-box from anywhere in your computer. To make sure, type

        which kids_ggl

If you do not see any message it means the place where the executable is is not part of your `$PATH`. To fix this, look through the installation messages where the file was placed (in my Macbook this place is `/Library/Frameworks/Python.framework/Versions/2.7/bin/kids_ggl`) and add the following statement to your `~/.bashrc` or `~/.bash_profile` if your terminal shell is `bash`:

        export PATH=${PATH}:<path_to_kids_ggl_folder>

or the following in your `~/.cshrc` or `~/.tcshrc` if your shell is set to `csh` or `tcsh`:

        setenv PATH ${PATH}:<path_to_kids_ggl_folder>

where, in my case, `<path_to_kids_ggl_folder>=/Library/Frameworks/Python.framework/Versions/3.6/bin`.

**e)** You can run an example by typing (see below for details)

    kids_ggl -c demo/ggl_demo_nfw_stack.txt --sampler --demo

This should show three panels with data points and lines resembling the Early Science satellite galaxy-galaxy lensing results of Sifon et al. (2015) and, after closing it, show the 3x3x14 covariance matrix.

At this point you are also able to import any component of the KiDS-GGL pipeline as `python` modules for use within your own code. Step **c)** above should ensure all the components are accessible by your `python` installation. For instance, to access the different density profiles (currently all based on modifications of the NFW profile) and their associated quantities (such as mass, lensing signal, etc), you may type

    from kids_ggl_pipeline.halomodel import nfw


#### 2. Set up your configuration file.
    
See `demo/ggl_halomodel.config` and `demo/ggl_covariance_demo.config` for guidance. The full halo model requires `kids_ggl>=2.0.0`.


#### 3. Run! 
There are two major things the pipeline can do for you:

**a)** Measure the lensing signal. To do this, type:

        kids_ggl -c <config_file> --esd

**b)** Estimate halo model parameters from the lensing signal. To do this, type:

        kids_ggl -c <config_file> --sampler

The sampler module has a demo option which you should always try before running a particular model on the data. To this end, type

        kids_ggl -c <config_file> --sampler --demo

This option will generate the ESD(s) for your chosen set of initial parameters, print the chi2/dof on screen, overplot the model to the data points and, once you close this plot, will display the full covariance matrix.

Sometimes, after doing the above a couple times, you don't really need to see the covariance all the time. You can run the demo without plotting the covariance with

        kids_ggl -c <config_file> --sampler --demo --no-cov

the `--no-cov` option is ignored if `--demo` is not present.

*New in `v2`:* By default, if the output file exists the pipeline will ask the user whether they want to overwrite it. This can be skipped by setting the `-f` flag in the command line:

        kids_ggl -c <config_file> --sampler -f


### Some suggestions to make the best of this pipeline:

- Make sure you have tested your model using the demo before you set up a full run!
- After running a demo, run an MCMC chain of your model with as few steps as possible to make sure that the output looks the way you want it to look. Fix anything that you can and report any possible bugs.
- Always check how many cores are available in your machine before running in parallel.
- **Contribute!**
 
If you have any questions, please (preferably) raise an issue in `github`, or contact Andrej Dvornik (dvornik@strw.leidenuniv.nl), Margot Brouwer (brouwer@strw.leidenuniv.nl) and/or Cristóbal Sifón (cristobal.sifon@pucv.cl).


Update log:

**2018 May 7** - Release v1.6.3. Features:
- Fixed bug in the Tinker halo bias function used in the halo model. 

**2018 January 19** - Release v1.6.2. Features:
- ESD production fully tested in Python 3 after several bug fixes

**2018 January 12** - Release v1.6.1. Features:
- Update to hmf 3.0.3 and newer Python CAMB package
- Bug fixes in halo model imports

**2018 January 10** - Release v1.6.0. Features:
- The lens files can now be either FITS or ascii files.
- The user may specify the names of the required columns (ID,RA,Dec[,z]) in the configuration file, instead of having to modify the input table to comply with GAMA naming conventions
- The ID column is no longer required (one will be created if none is present in the input table)
- Minor bug fixes in the covariance calculation
- Small style changes and code clean-up
- The ESD production can be run on DR2 again

**2017 July 20** - Release v1.5.1. Features:
- Bug fix for ID's

**2017 July 19** - Release v1.5.0. Features:
- Updates to halo model to include CAMB and hmf 2.0.5 as defaults.
- Added analytical covariance estimation to halo model (not quite done yet)
- Changes to code to make it Python 3 ready

**2016 November 11** - Release v1.4.0. Features:
- Major bug fix in calculating Sigma_crit and tangential shear (inproper removal of sources in front of the lens).
- Added unit test.
- Solved some problems with multiprocessing.

**2016 October 6** - Release v1.3.4. Features:
- Bug fixes.

**2016 September 30** - Release v1.3.3. Features:
- Solved bug when pipeline stopped in the middle of calculation without any errors.

**2016 September 29** - Release v1.3.2. Features:
- Solved some issues when installing pipeline and running demo config files.

**2016 September 14** - Release v1.3.1. Features:
- Solved problems with multiprocessing.
    
**2016 September 13** - Release v1.3.0. Features:
- More user friendly output folder and file structure.
- Added m-correction to the ESD production.
- Added user selectable epsilon cut (z_s > z_l + epsilon).
- Proper paths to catalogues as used in Leiden.
- Solved problems when plotting data.
- Conversion from physical units to comoving units when using halo.py model.
- Solved problems with saving the progress of MCMC chains.
- Various bug fixes.

**2016 June 10** - Release v1.2.3. Features:
- Bug fixes to compatibility issues with Numpy and blinding when running bootstrap.

**2016 June 10** - Release v1.2.2. Features:
- Bug fixes to filename generation.

**2016 May 31** - Release v1.2.1. Features:
 - Bug fixes.

**2016 May 31** - Release v1.2.0 Features:
 - Option to use The-wiZZ for obtaining `n(z)` of sources.
 - Addition of miscentering into `halo.py`, inclusion of simpler HOD and point mass approximation for stellar contribution to lensing signal.
 - Fixed issues with halo model and sampler when using only one bin.
 - If using angular scales in ESD production, pipeline returns shears instead of ESD.
 - KiDS+GAMA matching algorithm fixed.
 - Multiple ID files now supported.

**2016 Mar 5** - Release v1.1.2. Features:
 - Crucial bug fixes.

**2016 Feb 29** - Release v1.1.1. Features:
 - Fixed demo config file to avoid confusion and problems when testing the pipeline for the first time.

**2016 Feb 29** - Release v1.1.0. Features:
 - The syntax for `hm_output` parameters in the configuration file is now simpler and makes it easier to update the number of observable bins. Details can be found in `demo/ggl_demo_nfw_stack.txt`. Configuration files must be updated accordingly.

 - Cosmological parameters are no longer hard-coded in the halo model and can therefore be modified by the user in the config file. This also means the user can fit for them in the MCMC.

 - Additional parameters in the halo model that can now be modified from the config file:
    - Conversion from average central stellar mass to average satellite stellar mass, `Ac2s`;
    - Concentration of the radial distribution of satellite galaxies, `fc_nsat`.

 - General clean up of most modules

**2016 Feb 23** - First release v1.0.0. Features:
- Installed with a single command-line, including all dependencies
- User interaction through a single configuration file
- ESD production:
    - Compatible with both KiDS-450 (Feb 2016) and KiDS-DR2 (Mar 2015)
- Halo model:
    - Both an NFW stack and a full halo model (relies on [hmf](https://github.com/steven-murray/hmf) module for the mass function)
- MCMC sampling:
    - Uses `emcee` to sample parameters

---
