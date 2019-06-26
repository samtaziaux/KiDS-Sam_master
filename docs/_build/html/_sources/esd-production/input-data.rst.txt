============
 Input data
============

In order to work, the ESD production module in ``kids_ggl`` requires a number of files beyond the configuration file. These files 
can be found in a few of the servers that are commonly used for KiDS lensing analyses -- see :ref:`servers`.

If you prefer to work from a computer that is not connected to any of these, then you need the following files (*to be added 
somewhere for download - will ping Andrej*):

* Source catalogues: `KiDS-1000`_ *or* `KiDS-450`_.

* `Spectroscopic training data`_

* Multiplicative bias correction: `KiDS-1000`_ *or* `KiDS-450`_


.. _servers:

Working from one of the standard KiDS servers
*********************************************

Below we show how the KiDS data portion of your ESD production configuration file should look like, depending on where you are 
performing the analysis.

Bochum
------


Bonn
----


Edinburgh
---------


Leiden
------

::

    KiDS_path       /disks/shear13/dvornik/KidsCatalogues/K1000/
    KiDS_version    3
    specz_file      /disks/shear13/dvornik/KidsCatalogues/IMSIM_Gall30th_2016-01-14_deepspecz_photoz_1000_4_specweight.cat
    m_corr_file     ???
    kids_columns    SeqNr,ALPHA_J2000,DELTA_J2000,Z_B,model_SNratio,MASK,THELI_NAME,weight,m_cor,e1,e2

For KiDS-450 instead, replace the first line with ::

    KiDS_path       /disks/shear13/dvornik/KidsCatalogues/DR3/
