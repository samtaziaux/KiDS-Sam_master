# KiDS-GGL
Galaxy-Galaxy Lensing pipeline production

*This Readme contains all the instructions needed to install, set up and run
the KiDS galaxy-galaxy lensing pipeline, which takes current KiDS and GAMA
catalogs and produces (or reads) an ESD and a covariance matrix and runs a
fitting module (i.e., halo model) with a given sampling technique.*

####1. Installation
    
    
**a)** Contact Cristóbal Sifón (sifon@strw.leidenuniv.nl) to become a member
       of the KiDS-WL repository and join the KiDS-GGL team

**b)** Download [the latest stable version of the KiDS-GGL pipeline](https://github.com/KiDS-WL/KiDS-GGL/releases/latest) and unpack,

       cd path/to/kids_ggl_folder
       tar xvf KiDS-GGL-<version>.tar.gz

or, if you chose to download the `.zip` file,

        unzip KiDS-GGL-<version>.zip

where for instance `<version>=1.0.0`.

**c)** From within the same folder, run

        python setup.py install

If you don't have administrator privileges, or simply want to install it in a non-standard place (e.g., your home directory), then type

        python setup.py install --prefix=path/to/installation/
    
or
    
        python setup.py install --user

Either of these will install some additional packages required by the pipeline: `astropy>=1.1.0` for astronomical utilities (e.g., constants and units), `emcee>=2.1.0` for MCMC, `hmf==1.7.0` for halo mass function utilities, `mpmath>=0.19` for mathematical utilities, and `numpy>=1.5.0`. Optionally, if you want to use the CAMB transfer functions in your halo models (preffered method), you should also instal pycamb with CAMB version from March 2013, using the following command: 

        CAMBURL=http://camb.info/CAMB_Mar13.tar.gz pip install git+git://github.com/steven-murray/pycamb.git --user

**d)** After the setup script has finished, you should have a copy of the `kids_ggl` executable somewhere in your `$PATH`, which means you can run it out-of-the-box from anywhere in your computer. To make sure, type

        which kids_ggl

If you do not see any message it means the place where the executable is is not part of your `$PATH`. To fix this, look through the installation messages where the file was placed (in my Macbook this place is `/Library/Frameworks/Python.framework/Versions/2.7/bin/kids_ggl`) and add the following statement to your `~/.bashrc` or `~/.bash_profile` if your terminal shell is `bash`:

        export PATH=${PATH}:<path_to_kids_ggl_folder>

or the following in your `~/.cshrc` or `~/.tcshrc` if your shell is set to `csh` or `tcsh`:

        setenv PATH ${PATH}:<path_to_kids_ggl_folder>

where, in my case, `<path_to_kids_ggl_folder>=/Library/Frameworks/Python.framework/Versions/2.7/bin`.

**e)** You can run an example by typing (see below for details)

    kids_ggl -c demo/ggl_demo_nfw_stack.txt --sampler --demo

This should show three panels with data points and lines resembling the Early Science satellite galaxy-galaxy lensing results of Sifon et al. (2015) and, after closing it, show the 3x3x14 covariance matrix.

####2. Set up your configuration file.
    
See `demo/ggl_demo_nfw_stack.txt` and `demo/ggl_demo_halo_specific.txt` for guidance.


####3. Run! 
There are two major things the pipeline can do for you:

**a)** Measure the lensing signal. To do this, type:

        kids_ggl -c <config_file> --esd

**b)** Estimate halo model parameters from the lensing signal. To do this, type:

        kids_ggl -c <config_file> --sampler

The sampler module has a demo option which you should always try before running a particular model on the data. To this end, type

        kids_ggl -c <config_file> --sampler --demo

This option will generate the ESD(s) for your chosen set of initial parameters, print the chi2/dof on screen, overplot the model to the data points and, once you close this plot, will display the full covariance matrix.


###Some suggestions to make the best of this pipeline:

- Make sure you have tested your model using the demo before you set up a full run!
- After running a demo, run an MCMC chain of your model with as few steps as possible to make sure that the output looks the way you want it to look. Fix anything that you can and report any possible bugs.
- Always check how many cores are available in your machine before running in parallel.
- **Contribute!**
 
If you have any questions, please contact Andrej Dvornik (`dvornik@strw.leidenuniv.nl`), Margot Brouwer (`brouwer@strw.leidenuniv.nl`) and Cristóbal Sifón (`sifon@strw.leidenuniv.nl`).

---
Update log:

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
