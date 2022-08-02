import time, sys, os
import h5py
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from math import factorial
from astropy.io import fits

import matplotlib
matplotlib.use('Agg')
import fsps
import sedpy
import prospect
# re-defining plotting defaults
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'xtick.color': 'k'})
rcParams.update({'ytick.color': 'k'})
rcParams.update({'font.size': 30})
rcParams.update({'axes.linewidth':2})
rcParams.update({'patch.linewidth':2})
plt.rcParams['figure.figsize'] = (10, 5)

from matplotlib import rc
import matplotlib
import re
import prospect
from prospect.models.templates import TemplateLibrary
from prospect.models import priors
from prospect.sources import CSPSpecBasis
from prospect.likelihood import lnlike_spec, lnlike_phot
from prospect.likelihood import chi_spec, chi_phot
from prospect.io import write_results as writer
import gc
import decimal
from astropy.cosmology import WMAP9 as cosmo
from prospect.utils.obsutils import fix_obs
from prospect.models import SedModel
from prospect.fitting import fit_model
from multiprocessing import Pool
from contextlib import closing
import pickle
###############################################################################################
#TS5 = Training Set 5 - Mass, Metallicity, age, tau, dust

prosp_smoothing = True
plotting=False
#plots_dir = '../plots/ts5_s/'
#data_dir = '../data/ts5_s/'
data_dir = '../data/ts5_s_test/'


mags = np.array([20.4582, 19.3343, 18.7087, 15.8958])   #ADD A BUNCH OF MAGNITUDES AND UNCERTAINTIES HERE TO MATCH THE FILTERS
mags_err = np.array([0.1462, 0.08618, 0.06855, 0.0709] )

mags_bright1 = mags - mags_err #ad hoc in earliest plots was 0.3
mags_faint1 = mags + mags_err

snr=20.  #irrelevant parameter here
redshift7= 2.04
cluster_age = float('%.3f'%(cosmo.age(redshift7).value))
print(cluster_age)
ldist = float('%.3f'%(cosmo.luminosity_distance(redshift7).value))
print(ldist)

import sedpy 
def load_obs(snr=snr,ldist=10,**extras):
    
    sdss = ['sdss_{0}0'.format(b) for b in ['r', 'i', 'z']]
    twomass = ['twomass_{0}'.format(b) for b in ['H']]
    filternames = sdss + twomass
    print(filternames)
    obs = {}
    obs["filters"] = sedpy.observate.load_filters(filternames)
    
    mag1 = mags
    mags_bright = mags_bright1
    mags_faint = mags_faint1
    dmag1 = np.abs((10**(-0.4*mags_bright) - 10**(-0.4*mags_faint))*0.5 )
    
    obs["maggies"] = 10**(-0.4*mag1)
    obs["maggies_unc"] = abs(dmag1)
    print(obs["maggies_unc"])
    obs["phot_mask"] = [True,True,True,True]
    obs["phot_wave"] = [f.wave_effective for f in obs["filters"]]
    obs["wavelength"] = None  # this would be a vector of wavelengths in angstroms if we had 
    obs["spectrum"] = None
    obs['unc'] = None  #spectral uncertainties are given here
    obs['mask'] = None
    
    return obs
    
run_params = {}
run_params["snr"] = snr
run_params["rescale_spectrum"] = False
run_params["zcontinuous"] = 1

obs = load_obs()
print(obs)
obs = fix_obs(obs)

#——————————————————————————————————————————————————————————————————————
from prospect.sources import CSPSpecBasis

def load_sps(zcontinuous=1, **extras):
    sps = CSPSpecBasis(zcontinuous=zcontinuous,reserved_params=["sigma_smooth"])
    return sps
sps=load_sps()
#——————————————————————————————————————————————————————————————————————
###CREATE THE MODEL DICTIONARY
from prospect.models.templates import TemplateLibrary
from prospect.models import priors


poly_calibration=0
velocity_smoothing=0
star_formation_complex=0
burst = 0

redshift_max = 5.5
cluster_age_max = float('%.3f'%(cosmo.age(redshift_max).value))

def loguniform(low=0, high=1):
    return 10**(np.random.uniform(low, high))


def build_model(total_mass = 1e11, age_gal = 2., tau_gal = 0.1,\
 tburst_gal = 1., fburst_gal= 0.2, metallicity = -0.1, dust_all=0.3):

    model_dict = TemplateLibrary["parametric_sfh"]

    model_dict["sfh"]["init"] = 4
    model_dict["zred"] = {"N": 1, "isfree": False, "init": 1e-5,"prior" : priors.TopHat(mini=0.0, maxi=0.5),\
                          "units":"redshift, see the python-FSPS documentation"} 
    model_dict["peraa"] = {"N": 1,"isfree": False,"init": True}
    model_dict["add_neb_emission"] ={"N": 1,"isfree": False,"init": True}
    model_dict["add_neb_continuum"] ={"N": 1,"isfree": False,"init": True}
    model_dict["nebemlineinspec"] ={"N": 1,"isfree": False,"init": True}
    model_dict["dust_type"] = {"N": 1, "isfree": False, "init": 2} 
    #—————————————————————————————————

    model_dict["tage"] = {"N": 1, "isfree": True, "init": age_gal,\
                          "prior":priors.TopHat(mini=0.01, maxi=3),"units":"Gyr"}
    model_dict["dust2"] = {"N": 1, "isfree": False, "init": dust_all,\
                          "prior":priors.TopHat(mini=0.01, maxi=1.0),"units":"opacity"}
    model_dict["logzsol"] = {"N": 1, "isfree": False, "init": metallicity,\
                          "prior":priors.TopHat(mini=-2.0, maxi=0.2),"units":"logz/zsolar"}
    model_dict["tau"]["init"] = tau_gal
    model_dict["tau"]["isfree"] = False
    model_dict["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=10)
    model_dict["mass"]["init"] = total_mass
    model_dict["mass"]["prior"] = priors.LogUniform(mini=1e9, maxi=1e15)
    #-------------------------------------------------------------------------------------
    if burst == 1:
        model_dict['tburst'] = {"N": 1, "isfree": True,"init": tburst_gal,
                                "prior": priors.TopHat(mini=cluster_age-0.499, maxi=cluster_age-0.01)}
        model_dict['fburst'] = {"N": 1, "isfree": True,"init": fburst_gal,
                                "prior": priors.TopHat(mini=0., maxi=1.)}
    #-------------------------------------------------------------------------------------
    if velocity_smoothing==2:
        model_dict["smooth"] = {'N': 1, 'isfree': False, 'init': 'vel'}
        model_dict["fft"] =    {'N': 1, 'isfree': False, 'init': True}
        model_dict["sigma_smooth"] ={"N": 1,"isfree": True,"init": 1.1968936e+02,"prior": priors.TopHat(mini=100, maxi=500)}
    #-------------------------------------------------------------------------------------

    model = SedModel(model_dict)

    return model


#for i, num_iter in enumerate(np.arange(10000)):
for i, num_iter in enumerate(np.arange(1000)):

    print()
    print('Iteration number',i)

    #Randomly sample parameters of interest            
    total_mass = loguniform(low=8, high=13)
    age_gal = np.random.uniform(0.01, 4) #in Gyrs
    tau_gal = loguniform(low=-1, high=1) #in log(Gyrs)
    tburst_gal = 0.2 #in Gyrs
    fburst_gal = 0.1 #in fraction units
    metallicity = np.random.uniform(-2.0, 0.2) #in log(Z/Zsol)
    dust_all = np.random.uniform(0.1, 1.0) #dust2 param


    #print('Drawn parameters....')
    print('LogM=',np.log10(total_mass))
    print('Age=',age_gal)
    print('Metallicity=',metallicity)
    print('Tau=',tau_gal)
    #print('Tburst',tburst_gal)
    #print('fburst=',fburst_gal)
    print('Dust=',dust_all)

    model = build_model(total_mass = total_mass, age_gal = age_gal, tau_gal = tau_gal,
        tburst_gal = tburst_gal, fburst_gal= fburst_gal, metallicity = metallicity, dust_all=dust_all)

    theta = model.theta.copy()
    #print("Initial ln_prior_probability: {}".format(model.prior_product(theta)))
    initial_spec, initial_phot, initial_mfrac = model.sed(theta, obs=obs, sps=sps)
    a = 1.0 + model.params.get('zred', 0.0) # cosmological redshifting
    wspec = sps.wavelengths.copy()
    wspec *= a
#################################################################

    f_array = np.asarray([str(i),str(total_mass),str(age_gal),str(tau_gal),\
        str(metallicity),str(dust_all)])
    f=open(data_dir+"table.txt", "a")
    f.write(','.join(f_array))
    f.write("\n")
    f.close()

    cond = (wspec>3e3)&(wspec<1e4)
    w_short, spec_short = wspec[cond], initial_spec[cond]

    if prosp_smoothing:
        from prospect.utils import smoothing
        outwave1 = np.arange(3010.,10000.,50.)
        len(outwave1)
        f_new = smoothing.smoothspec(w_short,spec_short,resolution=350,outwave=outwave1,smoothtype="vel")
        w_short2,spec_short2 = outwave1[1:-1],f_new[1:-1]

    else: w_short2,spec_short2 = w_short[1:-1], spec_short[1:-1]


    np.savetxt(data_dir+'model_spec_%i.txt'%(i),np.array([w_short2,spec_short2]))
    

    if plotting:
        plt.figure(figsize=(12,6))
        plt.loglog(w_short, spec_short, label=r'$\mathrm{Model\ Spectrum}$',
               lw=5, alpha=1.0)
        plt.legend(loc='best', fontsize=20)
        plt.xlabel('Wavelength (\AA)')
        plt.ylabel('Spectrum')
        plt.tight_layout()
        plt.savefig(plots_dir+'model_%i.png'%(i),dpi=300)
        plt.close('all')

#################################################################