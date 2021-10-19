# -*- coding: utf-8 -*-

###############################################################################
#-------------------SECTION ZERO: IMPORTING PACKAGES--------------------------#
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

# importing astropy packages
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch, simple_norm
from astropy.visualization.mpl_normalize import ImageNormalize

# importing photutils packages
from photutils.aperture import SkyCircularAperture
from photutils import CircularAperture, CircularAnnulus
from photutils import DAOStarFinder, aperture_photometry
from photutils.background import Background2D, MedianBackground
from photutils.datasets import load_spitzer_image, load_spitzer_catalog

# misc packages
import calibrimbore as cal 
from copy import deepcopy

# path-type packages
import os
import glob
from glob import glob
from pathlib import Path

# initialising starting directory
code_home_path = "C:/Users/ave41/OneDrive - University of Canterbury/MSc Astronomy/MSc 2021/ASTR480 Research/ASTR480 Code/01 Data Reduction Pipeline/DataReductionPipeline/src"
os.chdir(code_home_path) #from now on, we are in this directory

# importing functions
from drp_funcs import *
from asp_funcs import *

# initialising starting directory
code_home_path = "C:/Users/ave41/OneDrive - University of Canterbury/MSc Astronomy/MSc 2021/ASTR480 Research/ASTR480 Code/02 Data Analysis/Flux-Photometry-Analysis/src"
os.chdir(code_home_path) #from now on, we are in this directory

# importing functions
from flux_and_photometry_funcs import *

###############################################################################
#----------------------SECTION ONE: INITIALISATION----------------------------#
###############################################################################

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++CHANGES++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# reading in image files from Reduced ALERTS folder
images_path = Path("//spcsfs/ave41/astro/ave41/ObsData-2021-02-13/ALERT/Reduced ALERT/WCS Calibrated/Sidereally Stacked Images")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++CHANGES++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# filtering data files
lst_of_images = [str(images_path) +"/" + n 
                  for n in os.listdir(images_path) if (n.endswith('fit')) and n.__contains__('-aligned-')]
image_names = [n for n in os.listdir(images_path) if (n.endswith('fit')) and n.__contains__('-aligned-')]

outputs_path = path_checker(images_path,'Flux and Photometry Outputs')

# put in loop[]
# Read the image
data, hdr = fits.getdata(lst_of_images[0], header=True)

with fits.open(lst_of_images[0], "append") as img_hdul:
    img_hdr1 = img_hdul[0].header
    target_ra = img_hdr1['RA      '].strip(' ')
    target_dec = img_hdr1['DEC     '].strip(' ')
    my_target_coords = [str(target_ra) + " " + str(target_dec)]

#%%
###############################################################################
#--------------------SECTION TWO: BACKGROUND DETECTION------------------------#
###############################################################################

# Estimate the sky background level and its standard deviation
mean, median, std = sigma_clipped_stats(data, sigma=3.0)   
bkg = get_bkg_info(data) 

#%%
###############################################################################
#---------------------SECTION THREE: SOURCE DETECTION-------------------------#
###############################################################################

# Start up the DAOStarFinder object and detect stars
daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)    
sources = daofind(data - median)

#%%
###############################################################################
#------------------------SECTION FOUR: APERTURES------------------------------#
###############################################################################

apertures, positions = postions_and_apertures(sources)

# Set up a set of circular apertures (one for each position) with a radius of 5 pixels and annuli with
# inner and outer radii of 10 and 15 pixels.
apertures = CircularAperture(positions, r=5)
annulus_apertures = CircularAnnulus(positions, r_in=10, r_out=15)

#%%
###############################################################################
#------------------------SECTION FIVE: PHOTOMETRY-----------------------------#
###############################################################################

# Measure the total flux in both the aperture and annulus for each star. 
apers = [apertures, annulus_apertures]
phot_table = aperture_photometry(data - bkg.background, apers, method='subpixel',
                                 subpixels=5)

# Calculate the mean flux per pixel in the annulus
sky_mean = phot_table['aperture_sum_1'] / annulus_apertures.area

# Multiply this by the number of pixels in the aperture and subtract from the aperture flux measurement.
# Put the result in a new column of the table.
aperture_sky_sum = sky_mean * apertures.area
phot_table['flux'] = phot_table['aperture_sum_0'] - aperture_sky_sum

#%%
f = fits.open(lst_of_images[0])
w = WCS(f[0].header)
ra_dec_sources = []
for i in range(len(sources['xcentroid'])):
    sky = w.pixel_to_world(sources['xcentroid'][i],sources['ycentroid'][i])
    sources['xcentroid'][i] = (sky.ra * u.deg).value
    sources['ycentroid'][i] = (sky.dec * u.deg).value

#%% # RYAN'S CODE
#########################################################
R_filter = 'moared.txt' #wavelengths, ?
R_fit = cal.sauron(band=R_filter,system='skymapper',gr_lims=[-.5,0.8],plot=True,cubic_corr=False)

coords = my_target_coords 
c = SkyCoord(coords,unit=(u.hourangle, u.deg))

try:
    sm_sources = cal.get_skymapper_region(c.ra.deg,c.dec.deg)
except:
    sm_sources = cal.get_skymapper_region(c.ra.deg,c.dec.deg,size=0.4*60**2)

ind = (np.isfinite(sm_sources.r.values) & np.isfinite(sm_sources.i.values) 
       & np.isfinite(sm_sources.z.values) & (sm_sources['g'].values < 19) & (sm_sources['g'].values > 13))
sm_sources = sm_sources.iloc[ind]

R_estimates = R_fit.estimate_mag(mags = sm_sources)

sm_sources['MOA_R_est'] = R_estimates

plt.figure()
plt.plot(sm_sources['g']-sm_sources['r'],sm_sources['MOA_R_est']-sm_sources['r'],'.',color="darkviolet")
plt.xlabel("g-r")
plt.ylabel("MOA-R - r")
plt.grid("both")
plt.title("(g-r) vs (MOA-R - r)")

s1 = deepcopy(sm_sources)
s2 = deepcopy(sm_sources)

dra = s1['ra'].values[:,np.newaxis] - s2['ra'].values[np.newaxis,:]
ddec = s1['dec'].values[:,np.newaxis] - s2['dec'].values[np.newaxis,:]

dist = np.sqrt(dra**2 + ddec**2)

min_value = np.nanmin(dist,axis=0)
min_index = np.argmin(dist,axis=0)

#%%
sources['xcentroid'] == sources['xcentroid'][min_index]
sources['ycentroid'] == sources['ycentroid'][min_index]
print(sources)

#%%
dra = sources['xcentroid'][:,np.newaxis] - s2['ra'].values[np.newaxis,:]
ddec = sources['ycentroid'][:,np.newaxis] - s2['dec'].values[np.newaxis,:]
dist = np.sqrt(dra**2 + ddec**2)

min_value = np.nanmin(dist,axis=0)
min_index = np.argmin(dist,axis=0)

t1 = sources[min_index]

print(len(t1) == len(s2))

good_indices = []
for i in range(len(t1)):
    #if t1['xcentroid'][i] +  min_value[i] == s2['ra'].values[i]:
    if np.abs(t1['xcentroid'][0] - s2['ra'].values[0]) >= min_value[i]:
        good_indices.append(i)

print(good_indices)

#%%
new_t1 = t1[good_indices]
new_s2 = s2.iloc[good_indices]

print(len(new_t1)==len(new_s2))
#%%
zp = new_s2['MOA_R_est'].values - new_t1['flux']

plt.hist(zp,bins=100,color="darkviolet")
plt.grid("both")
plt.xlabel("zero points")
plt.ylabel("frequency")
plt.title("Calibrated Zero Points of Sources")
plt.show()

#%%
for i in range(len(new_t1)):
    new_t1['mag'] = zp[i] - 2.5*np.log10(new_t1['flux'])
    
plt.hist(new_t1['mag'],bins=100,color="darkviolet")
plt.grid("both")
plt.xlabel("apparent magnitudes")
plt.ylabel("frequency")
plt.title("Calibrated Apparent Magnitudes of Sources")
plt.show()

plt.plot(new_t1['flux'],new_t1['mag'],'.',color="darkviolet")
plt.grid("both")
plt.xlabel("flux")
plt.ylabel("magnitude")
plt.title("Calibrated Apparent Magnitudes of Sources")
plt.show()

#%%
final_calibrated_mags = new_t1['mag'] - new_s2['MOA_R_est'].values

plt.hist(final_calibrated_mags,bins=50,color="darkviolet")
plt.grid("both")
plt.xlabel("apparent magnitudes")
plt.ylabel("frequency")
plt.title("Final Calibrated Magnitudes of Sources")
plt.show()

#%%