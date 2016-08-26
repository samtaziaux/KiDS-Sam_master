# The-wiZZ


INTRODUCTION
------------

The-wiZZ is a clustering redshift recovery analysis tool designed with ease of
use for end users in mind. Simply put, The-wiZZ allows for clustering redshift
estimates for any photometric, unknown sample in a survey by storing all close
pairs between the unknown sample and a target, reference sample into a data
file. Users then query this data file with their specific selection and produce
a recovery. For further details on the method see Schmidt et al. 2013, Menard et
al 2013, Rahman et al. 2015(ab), and Morrison et al (in prep).

The software is composed of two main parts: a pair finder and a pdf maker. The
pair finder does the initial heavy lifting of spatial pair finding and stores
the indices of all closer pairs around the target objects in an output HDF5
data file. Users then query this data file using pdf_maker.py and the indices of
their unknown sample, producing an output recovery.

REQUIREMENTS
------------

The library is designed with as little reliance on nonstandard Python libraries
as possible. It is recommended if using The-wiZZ that you utilize the Anaconda
(https://www.continuum.io/downloads) distribution of Python.

pdf_maker.py requirements:

    astropy (http://www.astropy.org/)
    h5py (http://www.h5py.org/)
    numpy (http://www.numpy.org/)
    scipy (http://www.scipy.org/)
    
pair_maker.py requirements:

    (as above)
    astro-stomp (https://github.com/ryanscranton/astro-stomp)

INSTALLATION
------------

Currently The-wiZZ can be installed from git using the command

    git clone https://github.com/morriscb/The-wiZZ.git <dir_name>

Running pdf_maker.py or pair_maker.py from the created directory will properly
run the code.

TROUBLESHOOTING
---------------

pdf_maker.py
------------

pdf_maker.py should always come with two files, a fits file containing all of
the photometric, unknown galaxies and the HDF5 data file containing all of the
close pairs between the unknown and reference sample. Users should select their
galaxy sample (e.g. galaxies of a certain color, properites, photometric
redshift) from this fits catalog and match the IDS into the HDF5 file using 
pdf_maker.py

For mulit-epoch surveys spanning multiple pointings, large area surveys, or
combining different surveys it is recommended to set the
uknown_stomp_region_name flag to the appropriate column name. Having this
flag not set for such surveys will likely result in unexpected results.

Higher signal to noise is achieved with the flag use_inverse_weighting set.
Setting this mode is recommended.

Care should be taken that the correct index column is being used. It should be
the same as that stored in the pair HDF5 file.

Requesting larger scales for the correlation requires much more computing power.
If the code is taking a significant amount of time (~30 minutes) per sample,
increase the number of processes. (n_processes)

pair_maker.py
-------------

This portion of the code is for experts only. The majority of end users will
only use pdf_maker.py.

This part of the code should be used by surveys interested in using The-wiZZ as
their redshift recovery code. This portion of the code creates the HDF5 data
file of all close pairs that is used in pdf_maker.py. It is recommeneded that
surveys use their full, photometric catalog masked to the same area as the
reference catalog used.

The code uses the spatial pixelization library STOMP for all of it's pair
finding. Those unfamiliar with this libary are recommened to have a look at the
source code header files at https://github.com/ryanscranton/astro-stomp. 

To use pair finder, one much first create a file describing the usmasked area of
the survey, in STOMP this is called a Map. Two utility functions are available
to create these Maps, stomp_adapt_map.py and stomp_map_from_fits.py.
stomp_map_from_fits.py takes in a fits image descripbing the mask and creates an
aproximation of the unmasked area. stomp_adapt_map.py should be used when no
fits mask or weight map exists. It attempts to intuit the mask from an input
catalog of objects. Descriptions of how to use the code are contained in the 
respective python files. It is possible to use STOMP to create a mask from
complex polygons (e.g. ds9 regions, mangle) using code available in the library.
Look to the STOMP::Map and STOMP::Geometry classes for more information.

stomp_mask_catalog.py allows one to mask an input fits catalog to the geomety of
a STOMP Map. It is an extremely useful program as it allows for the creation of
a catalog with the same geometry as that used in pair_maker.py. A catalog
produced from stomp_mask_catalog.py allows the end user to select their sample
from a catalog that has the same geometry as used in the pair finder and thus
all of the average densities will correct. It also allows the ablity to store
the same regionation as used in the pair finder enabling the use of the
"unknown_stomp_region_name" flag in pdf_maker.py. This flag is extremely import
for inhomengious surveys.

The number of regions is an extremely important choice when running The-wiZZ.
The number of regions requested should be a compromise between smoothing the
scale of individual pointings and allowing for the largest physical scale
requested. For instance if you have 1 sq deg. pointings, you'll want to try to
have the regions you request be at most 1 sq deg to smooth pointing to pointing
variations from survey stragegy/data quality variation.

Using unquie indices for the target, reference objects can allow one to combine
the data files produced after the fact, enabling simple paralization for the
pair creation process. Make sure to sum together the total number of randoms
through.

For large unknown catalogs where large is not that large (>100k) it is
sufficient to create at most 10 times the number of randoms.

FAQ
---

Q: Why The-wiZZ?
A: The-wiZZ is designed to take the hard work (i.e. pair finding) and separate
it from the science (redshift recovery) allowing end users to simply select from
a photometric, unknown catalog and match indices into the code and procude high
significance cluster redshift recoveries without the need to re-run a pair
finding/correlation technique every time. The-wiZZ can be thought of as creating
a value added catalog for your survey much like a photometric redshift code.

Q: I'd like to use The-wiZZ for my survey, how do I credit you?
A: There is a helpful section just below here about citing The-wiZZ.

Q: I'm having trouble using STOMP, why don't you use [insert favorite
correlation code].
A: STOMP has a ton of convince functions that make this code possible and is
python wrapped to make it even more convienent. If you would like to use a
library other than STOMP in the code that can be done assuming that the majority
of the functionality is retained. Feel free to contact the maintainers if you
run into problems creatinging inherited methods/modifying the code.

Q: My install of python2.7 is not working with The-wiZZ, can you fix it?
A: The recommended install of python is Anaconda
(https://www.continuum.io/downloads) while I would like to be comptable with
everyone's different installs I can not guarentee full compadiblity. If you
run into an issue with The-wiZZ please add it to the GitHub issue tracker and I
will attempt to address it.

Q: My recovery doesn't look right it either has an incorrect normalization or
just looks weird.
A: Make sure that the photometric, unknown catalog is masked to the same
geometry as was used to create the HDF5 data file using pair_maker.py. For
multi-epoch surveys, usage of the flag "unknown_stomp_region_name" in
pdf_maker.py should be considered a default mode.

Q: Does your software account for galaxy bias in the recovery.
A: No. The recoveries produced by The-wiZZ contain no correction for either the
bias of the unknown sample or the reference sample. Users will have their
preference on which method of bias mitigation is "correct" and are encouraged to
use which ever method suits their needs. The-wiZZ will can be used easily enough
with redshift, color pre-selections as in Rahman et al. 2016(ab).

Q: You call the outputs "PDFs" but they are not normalized to one and sometimes
have negative values, what gives?
A: Clustering redshift recovery returns an estimate of the over-density as a
function of redshift that is then normalized into a PDF. Because it is an
estimate that is measured from data, noise can cause points in the recovery
to be negative but consistent with zero. There are also some unknown galaxy
selections that can anti-correlate with the reference sample at given redshifts.
This problem could be solved with an appropreate weighting scheme on the
reference catalog. The PDFs returned by The-wiZZ are unomalized because everyone
has their favorite technique to do this (e.g. spline integral, trapzoid sum,
rectangular sum). The choice is left to the user.

CITING The-wiZZ
---------------

Papers utilizing The-wiZZ should provide a link back to this repository. It is
also requested that users cite Morrison et al. (in prep), once the paper is
available. The other cites at mentioned at the start of this README are highly
recommended as citations as well.

MAINTAINERS
-----------

Current:
 * Christopher Morison (AIfA) - https://github.com/morriscb