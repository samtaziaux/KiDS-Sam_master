### General settings ###

# Source and lens catalogues
KiDS_path           ../../KidsCatalogues/LF_cat_DR2_v2             # Path to folder that contains the KiDS catalogs (optional - supersedes KiDS_version).
GAMA_path           ../../brouwer/MergedCatalogues/GAMACatalogue_1.0.fits  # Path the GAMA catalog (optional - supersedes GAMA_version).
# lens_catalog      /data2/brouwer/MergedCatalogues/troughs.fits            # Path to the fits file with your custom lens catalogue (supersedes GAMA catalog),
                                                                            #     containing columns for: 'ID', 'RA', 'DEC', optional: 'Z' (if Runit = *pc), used lens paramaters.

# cosmology
Om                  0.315                                       # Omega matter.
Ol                  0.685                                       # Omega Lambda. (Not yet working, please use: 1-Om)
Ok                  0                                           # Omega curvature. (Not yet working, please use: 0)
h                   1.0                                         # Reduced Hubble constant. (h = H0/(100 km/s/Mpc))


### ESD Production ###

# Algorithm
ESD_output_folder     /disks/shear10/brouwer_veersemeer/pipeline_results            # Path to ESD output folder.
ESD_output_filename   None                             # This flag will be appended to the filenames (optional).
#ESD_purpose          shearcovariance                  # One of {'shearcatalog', 'shearcovariance', 'shearbootstrap',
 ESD_purpose          randomcatalog                    #     'randomcatalog', 'randombootstrap'}.
#Rbins                1                                # One of the standard options (1-5, see help file), (Not yet working) or
#Rbins                binlimits/16bins.txt             #     either a file with the radial binning or a
 Rbins                10,20,2000                       #     comma-separated list with (Rmin,Rmax,Nbins).
Runit                 kpc                              # One of these physical {pc, kpc, Mpc} or sky {arcsec, arcmin, deg} coordinates.
ncores                5                                # Any number of cores your machine can use to run the code.

# Lens selection
lensID_file          None                                                                                 # Path to text file with chosen lens IDs (optional).

 lens_weights        None
#lens_weights        mstarweight   /data2/brouwer/shearprofile/KiDS-GGL/brouwer/environment_project/results/mstarweight_rank-999-inf.fits      # Weight name and path to fits file with lens weights (optional).
#lens_weights        mstarweight   /data2/brouwer/shearprofile/KiDS-GGL/brouwer/environment_project/results/mstarweight_rank-999-inf_shuffled.fits

 lens_binning       None                                                                                  # Lens parameter for binning, path to fits file (choose "self" for same as lens catalog), and 
#lens_binning       envS4          self        0,1,2,3,4                                                  #     bin edges (at least 3 edges, bin[i-1] <= x < bin[i]), or the number of bins.
#lens_binning       shuffenvR4      /data2/brouwer/MergedCatalogues/shuffled_environment_S4_deltaR.fits       0,1,2,3,4

#lens_limits1      rankBCG        self                                                      -999,2           # Lens parameter for limits, path to fits file (choose "self" for same as lens catalog), and 
#lens_limits2      Nfof           self                                                      5,inf            #      one value (x=lim) or two comma separated limits (lim[min] <= x < lim[max]) between -inf and inf.


# Source selection
 src_limits1             Z_B        0.005,1.2            # Source parameter for limits and one value, or
#src_limits2                                             #     two comma separated limits between -inf and inf.
#kids_blinds             D


### Halo Model ###

model             nfw_stack.fiducial4_auto        # the name of the halo model python function

# Halo model parameters
hm_param          sat_profile     function    nfw.esd
hm_param          host_profile    function    nfw.esd_offset
hm_param          central_profile function    nfw.esd
hm_param          Rsat            read        Rsatfilename   0
hm_param          n_Rsat1         read        Rsatfilename   1
hm_param          n_Rsat2         read        Rsatfilename   2
hm_param          n_Rsat3         read        Rsatfilename   3
hm_param          n_Rsat4         read        Rsatfilename   4
hm_params         fsat            fixed       fsatvalues
hm_param          fc_sat          fixed       1.0
hm_param          Msat1           uniform     1e10     5e12      2e11
hm_param          Msat2           uniform     1e10     5e12      2e11
hm_param          Msat3           uniform     1e10     5e12      2e11
hm_param          Msat4           uniform     1e10     5e12      2e11
#hm_param          fc_cen1         uniform     0       2         0.5
#hm_param          fc_cen2         uniform     0       2         0.5
#hm_param          fc_cen3         uniform     0       2         0.5
#hm_param          fc_cen4         uniform     0       2         0.5
hm_param          fc_cen1         fixed     1.
hm_param          fc_cen2         fixed     1.
hm_param          fc_cen3         fixed     1.
hm_param          fc_cen4         fixed     1.
hm_param          A_2halo1         uniform     0       10         1.0
hm_param          A_2halo2         uniform     0       10         1.0
hm_param          A_2halo3         uniform     0       10         1.0
hm_param          A_2halo4         uniform     0       10         1.0
hm_param          Mcen1           uniform     1e11   1e14      5e12
hm_param          Mcen2           uniform     1e11   1e14      5e12
hm_param          Mcen3           uniform     1e11   1e14      5e12
hm_param          Mcen4           uniform     1e11   1e14      5e12
hm_params         zgal            fixed       zgalvalues
hm_params         Mstar           fixed       Mstarvalues
hm_param          twohalo1          read        twohalofilename   0
hm_param          twohalo2          read        twohalofilename   1
hm_param          twohalo3          read        twohalofilename   2
hm_param          twohalo4          read        twohalofilename   3
#hm_functions     abs,min
#hm_functions     itertools izip
#hm_functions     numpy     array,cumsum,digitize,log,all,transpose,zeros
#hm_functions     nfw       mass_total_sharp
#hm_functions     utils     cm_duffy08,delta,nfw_profile

# these are extras returned by the model; last column are the fits formats (see pyfits docs)
hm_output         esd_total1,esd_total2,esd_total3,esd_total4            10E,10E,10E,10E
hm_output         esd_sat1,esd_sat2,esd_sat3,esd_sat4                    10E,10E,10E,10E
hm_output         esd_host1,esd_host2,esd_host3,esd_host4                10E,10E,10E,10E
hm_output         esd_cen1,esd_cen2,esd_cen3,esd_cen4                    10E,10E,10E,10E
hm_output         esd_2halo1,esd_2halo2,esd_2halo3,esd_2halo4            10E,10E,10E,10E
hm_output         Mavg1,Mavg2,Mavg3,Mavg4                                E,E,E,E


### Parameter Sampling ###

data                datafilename       0,1,4 # files containing the ESD profile.
                                              # Columns are (R,ESD_t[,1+K(R)]). Should be one file
                                              # per lens bin
covariance          covfilename        4,6   # file containing the covariance.
                                              # Columns are (cov[,1+K(R)]).

sampler_output      /disks/shear10/brouwer_veersemeer/mcmc_output/environment_mcmc_output_A2halo.fits            # output filename (FITS format)
sampler              emcee                    # MCMC sampler
nwalkers	         100
nsteps		         500
nburn		         0
thin		         1
k   		         7
threads		         6
sampler_type	     ensemble

