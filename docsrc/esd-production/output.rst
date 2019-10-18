============================
 Output data file structure
============================

The output of the ESD production will always be written to the folder which you specified using the ``ESD_output_folder`` line in your config file. For example: ``/data/users/brouwer/Lensing_results/``.

Within this folder, you will find:
    * Always
        * Text files containing stacked ESD profiles.
        * Text files containing the lens IDs of the stacked lenses.
    * In the case of ``shearcovariance`` or ``shearbootstrap``:
        * A text file containing the full covariance matrix (in the case of shearcovariance and shearbootstrap).
    * In the case of ``shearcatalog`` or ``shearbootstrap``:
        * A fits catalogue containing the unstacked individual shear profiles of all lenses, which can be stacked using the ``shearcatalog`` or ``shearbootstrap`` modes.

Stacked ESD profiles:
*********************

You will find the text files with the stacked ESD profiles, within your ``ESD_output_folder``. This folder contains a sequence of sub-folders which are named according to the configuration of your ESD profile. These subfolders are names as follows:

1. *Lens binning*:
    The bin name followed by the values of the bin edges. If the lenses were not binned, this folder is called ``No_bins``. Example: ``logmstar_8p5_10p3_10p6_10p8_11p0``.

2. *Lens selection and weights*:
    The lens selection parameters followed by the corresponding limit values. If the lens weights were applied, this folder name ends with `lw` followed by the lens weight parameter. Example: ``dist0p1perc_3_inf-Z_0_0p5``.
    
3. *Source selection and cosmology*:
    The specified limits on the KiDS sources, followed by the selected cosmological parameters. Example: ``ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7``.
    
4. *Radial binning*:
    The specified radial bin values (in this case: Nbins, min, max) followed by the selected unit. Example: ``Rbins15_0p03_3_Mpc``.
    
5. *Purpose*:
    The pipeline mode which was used to create the ESD profiles. Example: ``shearcatalogue``, ``shearcovariance`` or ``shearbootstrap``.

Example of the full path: ``/data/users/brouwer/Lensing_results/logmstar_8p5_10p3_10p6_10p8_11p0/dist0p1perc_3_inf-Z_0_0p5/ZB_0p1_1p2-Om_0p2793-Ol_0p7207-Ok_0-h_0p7/Rbins15_0p03_3_Mpc/shearcovariance/``

* Within this folder, you will find text files with the following names:
    * ``<purpose>_bin_[n]_<blind>.txt``: contains the stacked ESD profile for lens bin [n], created using the given mode and blind.  Example: ``shearcovariance_bin_1_A.txt``.
    * ``<purpose>_bin_[n]_<blind>_lensIDs.txt``: contains IDs of all stacked lenses in bin [n]. Example:``shearcovariance_bin_1_lensIDs.txt``
    * ``<purpose>_matrix_<blind>.txt``: contains the full uber-covariance matrix between all radial and lens bins (in the case of ``shearcovariance`` or ``shearbootstrap``). Example: ``shearcovariance_matrix_A.txt``.
    
