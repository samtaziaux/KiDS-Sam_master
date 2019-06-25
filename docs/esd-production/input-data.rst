============
 Input data
============

In order to work, the ESD production module in ``kids_ggl`` requires a number of files beyond the configuration file. These files 
can be found in the servers in Leiden, Bonn, and Edinurgh -- see :ref:`servers`.

If you prefer to work from a computer that is not connected to any of these, then you need the following files:

* Source catalogues: `KiDS-1000`_ *or* `KiDS-450`_.

* `Spectroscopic training data`_

* Multiplicative bias correction: `KiDS-1000`_ *or* `KiDS-450`_


.. _servers:

Working from one of the standard KiDS servers
*********************************************


Leiden
------

If you are working from any of the Leiden server machines, this is how the data section of your configuration file should look as 
follows, if you intend to use KiDS-1000: ::

    KiDS_path       /disks/shear13/dvornik/KidsCatalogues/K1000/
    KiDS_version    3
    specz_file      /disks/shear13/dvornik/KidsCatalogues/IMSIM_Gall30th_2016-01-14_deepspecz_photoz_1000_4_specweight.cat
    m_corr_file     ???
    kids_columns    SeqNr,ALPHA_J2000,DELTA_J2000,Z_B,model_SNratio,MASK,THELI_NAME,weight,m_cor,e1,e2

For KiDS-450 instead, replace the first line with ::

    KiDS_path       /disks/shear13/dvornik/KidsCatalogues/DR3/
