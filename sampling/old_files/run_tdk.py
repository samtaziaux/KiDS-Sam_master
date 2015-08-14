#code to run halo model MCMC on input data file 

import sys, getopt
sys.path.append('/Users/tdk/Desktop/aum-master/install/lib/python2.7/site-packages');
#############################################################
import cosmology as c;
import hod as h;
from math import *;
import numpy as np;
import matplotlib.pyplot as plt
import hod as h
import csv
import emcee
from astropy.io import ascii
#############################################################
def getdblarr(r):
    temp=h.doubleArray(r.size);
    for i in range(r.size):
        temp[i]=r[i];
    return temp;
#############################################################
def getnparr(r,n):
    temp=np.zeros(n);
    for i in range(n):
        temp[i]=r[i];
    return temp;

#############################################################
def lnlike(theta, x, y, yerr):
    Mmin, siglogM, Msat, alpsat = theta

    yerr = np.array(yerr, np.float)
    y = np.array(y, np.float)
    x = np.array(x, np.float)

    #***=parameters relavent for halo modelling (x** means implemented)

    # Om0 : Matter density parameter 
    # Omk : Curvature parameter 
    # w0 : Dark energy equation of state parameter 
    # wa : Dark energy equation of state parameter 
    # Omb : Baryon density parameter 
    # h : Hubble parameter 
    # th : CMB temperature 
    # s8 : sigma8 
    # nspec : power spectrum index 
    #*** ximax : Parameter psi defined in van den Bosch 2013, only relevant for halo model calculation 
    #*** cfac : Constant multiplicative factor for the c-M relation
    p = h.cosmo()
    p.Om0 = 0.307115
    p.w0 = -1
    p.wa = 0
    p.Omk = 0.0
    p.hval = 0.6777
    p.Omb = 0.048206
    p.th = 2.726
    p.s8 = 0.8228
    p.nspec = 0.96
    p.ximax = log10(8.0)
    p.cfac = 1.0

    #x** Mmin : Minimum halo mass in the central HOD 
    #x** siglogM : Scatter in halo masses in the central HOD 
    #x** Msat : Satellite halo occupation mass scale 
    #x** alpsat : Slope of the satellite halo occupation 
    #*** Mcut : Cut off mass scale of the satellite halo occupation 
    # fac : Unused parameter 
    #*** csbycdm : Multiplicative factor for satellite concentrations
    q = h.hodpars()
    q.Mmin = Mmin
    q.siglogM = siglogM
    q.Msat = Msat
    q.alpsat = alpsat
    q.Mcut = 13.5
    q.csbycdm = 1.0
    q.fac = 1.0

    a = h.hod(p, q)

    rp=x 
    wp=np.zeros(rp.size)
    rp_carr=getdblarr(rp)
    wp2_carr=getdblarr(wp)

    #make model
    a.Sigma(0.5, rp.size, rp_carr, wp2_carr)
    model=getnparr(wp2_carr,wp.size)

    inv_sigma2 = 1.0/(yerr*yerr)
    like=-0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))
    print like 
    return like
#############################################################
def lnprior(theta):
    Mmin, siglogM, Msat, alpsat = theta
    if 11. < Mmin < 20.0 and 0.3 < siglogM < 5.0 and 12.0 < Msat < 16.0 and 0.3 < alpsat < 1.5:
        return 0.0
    return -np.inf
#############################################################
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)
#############################################################
def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print 'run.py -i <inputfile> -o <outputfile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'run.py -i <inputfile> -o <outputfile>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print 'Input file is:', inputfile
   print 'Output file is:', outputfile

#############################################################

#read the data
   filename = inputfile
   delimiter = ' '  # import a file with space delimiters                                                                                                                                            
   data = []
   for row in csv.reader(open(filename), delimiter=delimiter):
       data.append(row)

   A=np.array(data)
   xdata=A[:,0]
   ydata=A[:,1]
   yerrr=A[:,3]

#set the fiducial points
   Mmin_t = 13.0
   siglogM_t = 0.5
   Msat_t = 14.0
   alpsat_t = 1.0
   Mcut_t = 13.5
   csbycdm_t = 1.0
   fac_t = 1.0
   
   print xdata 
   print ydata 
   print yerrr
   
#find best fit 
   import scipy.optimize as op
   nll = lambda *args: -lnlike(*args)
   
   result = lnlike([Mmin_t, siglogM_t, Msat_t, alpsat_t],xdata,ydata,yerrr)
   print result
   
#run MCMC chain, first find maximum likelihood value 
   result = op.minimize(nll, [Mmin_t, siglogM_t, Msat_t, alpsat_t], args=(xdata, ydata, yerrr))
   Mmin_ml, siglogM_ml, Msat_ml, alpsat_ml = result["x"]
   print Mmin_ml, siglogM_ml, Msat_ml, alpsat_ml
   
#dimensions and walkers (like seperate MCMC chains)
   ndim, nwalkers = 4, 10
   pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
   sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(xdata, ydata, yerrr),threads=8)
   sampler.run_mcmc(pos, 8000)
   print ndim 
   samples = sampler.chain[:, 100:, :].reshape((-1, ndim)) #3rd number is runin
   
#########################################################################                                                                                                                                                                                               
#plot the result
   import triangle
# Plot it.
   figure = triangle.corner(samples, labels=[r"$M_{min}$", r"$\sigma(\log M)$", r"$M_{sat}$", r"$\alpha_{sat}$"],
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_args={"fontsize": 12})
   figure.gca().annotate("Posterior", xy=(0.5, 1.0), xycoords="figure fraction",
                         xytext=(0, -5), textcoords="offset points",
                         ha="center", va="top")
   figure.savefig("triangle.png")
   
#########################################################################                                                                                                                                                                                               
   means=np.mean(samples, axis=0);
   Mmin_ml=means[0]
   siglogM_ml=means[1]
   Msat_ml=means[2]
   alpsat_ml=means[3]
   
#make the best fit model
   p = h.cosmo()
   p.Om0 = 0.307115
   p.w0 = -1
   p.wa = 0
   p.Omk = 0.0
   p.hval = 0.6777
   p.Omb = 0.048206
   p.th = 2.726
   p.s8 = 0.8228
   p.nspec = 0.96
   p.ximax = log10(8.0)
   p.cfac = 1.0
   
   q = h.hodpars()
   q.Mmin = Mmin_ml
   q.siglogM = siglogM_ml
   q.Msat = Msat_ml
   q.alpsat = alpsat_ml
   q.Mcut = 13.5
   q.csbycdm = 1.0
   q.fac = 1.0
   
   a = h.hod(p, q)
   
   rp=10.0**np.arange(-2,1.1579,0.1579)
   wp=np.zeros(rp.size)
   rp_carr=getdblarr(rp)
   wp2_carr=getdblarr(wp)
   a.Sigma(0.5, rp.size, rp_carr, wp2_carr)
   wp=getnparr(wp2_carr,wp.size)
   
#plot the data and fit
   plt.clf()
   plt.loglog(xdata,ydata, label="data")
   plt.loglog(rp,wp, label="Best Fit")
   plt.ylabel('Sigma')
   plt.xlabel('R')
   plt.axis([1e-3, 1e2, 1e-2, 1e4])
   plt.legend()
   plt.savefig("bestfit.png")
   
#write out files 
   ascii.write([rp, wp], outputfile, names=['rp', 'wp'])
                                
#########################################################################                                                                                                                                                                                                  
if __name__ == "__main__":
    main(sys.argv[1:])
