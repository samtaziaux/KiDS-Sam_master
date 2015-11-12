import numpy as np
import scipy
import scipy.optimize as opt
import astropy.units as u



def NFW_barSigma(R, rs, delta_c, rho_c):
    try:
        n = len(R)
    except TypeError:
        R = [R]
    R = scipy.array(R, dtype=float)
    x = R / rs
    rs *= u.kpc
    s = scipy.ones(len(x))
    s[x == 0] = 0
    s[x == 1] = 1 + scipy.log(0.5)
    j = (x > 0) & (x < 1)
    s[j] = scipy.arctanh(scipy.sqrt((1 - x[j])/(1 + x[j])))
    s[j] = 2 * s[j] / scipy.sqrt(1 - x[j]**2)
    s[j] = (s[j] + scipy.log(x[j] / 2.)) / x[j]**2
    j = (x > 1)
    s[j] = scipy.arctan(scipy.sqrt((x[j] - 1)/(1 + x[j])))
    s[j] = 2 * s[j] / scipy.sqrt(x[j]**2 - 1)
    s[j] = (s[j] + scipy.log(x[j] / 2.)) / x[j]**2
    out = 4 * rs * delta_c * rho_c * s
    return out.to(u.Msun/u.pc**2).value

def NFW_Sigma(R, rs, delta_c, rho_c):
    x = R / rs
    try:
        s = scipy.ones(x.shape)
    except TypeError:
        x = scipy.array([x])
        s = scipy.ones(1)
    s[x == 0] = 0.
    s[x == 1] = 1 / 3.
    j = (x > 0) & (x < 1)
    s[j] = scipy.arctanh(scipy.sqrt((1 - x[j]) / (1 + x[j])))
    s[j] = (1 - 2 * s[j] / scipy.sqrt(1-x[j]**2)) / (x[j]**2 - 1)
    j = (x > 1)
    s[j] = scipy.arctan(scipy.sqrt((x[j] - 1)/(1 + x[j])))
    s[j] = (1 - 2 * s[j] / scipy.sqrt(x[j]**2 - 1)) / (x[j]**2 - 1)
    out = 2 * (rs*u.kpc) * delta_c * rho_c * s
    return out.to(u.Msun/u.pc**2).value

def NFW_esd(R_plus_rhoc, rs, delta_c):
    R, rho_c = R_plus_rhoc
    return NFW_barSigma(R, rs, delta_c, rho_c) - NFW_Sigma(R, rs, delta_c, rho_c)


def import_data(filename):
	
	data = np.loadtxt(filename).T

	bias = data[4]

	data_x = data[0]
	data_y = data[1]/bias

	ymask = (np.isfinite(data_y))

	data_x = data_x[ymask]
	data_y = data_y[ymask]

	errorh = (data[3])/bias # covariance error
	errorl = (data[3])/bias # covariance error
	errorh = errorh[ymask]
	errorl = errorl[ymask]
	
	return data_x, data_y, errorh, errorl, bias

# Load ESD file
filename = '/disks/shear10/brouwer_veersemeer/pipeline_testresults/output_Nobins_oldcatmatch/results_shearcovariance/shearcovariance_IDs710_Z_B0p005-1p2_Rbins10-20-2000kpc_Om0p315_Ol0p685_Ok0_h1_oldcatmatch_D.txt'
data_x, data_y, errorh, errorl, bias = import_data(filename)

R = data_x
signal = data_y
signal_err = errorh
rho_c = 1 * u.Msun / u.pc**3

# load R, signal, signal_err, rho_c
Rc = [R, rho_c]
# for the scale radius and concentration
init = (0.1, 4)
theta, cov = opt.curve_fit(NFW_esd, Rc, signal, init, sigma=signal_err, absolute_sigma=True)


print theta












