# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:45:57 2021

@author: ave41
"""

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

# filtering data files
image_names = [n for n in os.listdir(images_path) if (n.endswith('fit')) and n.__contains__('-aligned-')]

# lst_of_images = []
# for n in os.listdir(images_path):
#     if (n.endswith('fit')) and n.__contains__('-aligned-'):
#         lst_of_images.append(str(images_path) +"/" + n)

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
    #print(sources['xcentroid'][i],sources['ycentroid'][i])
    #ra_dec_sources.append([(sky.ra * u.deg).value, (sky.dec * u.deg).value])

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
sm_apertures, sm_positions = postions_and_apertures(sm_sources)

# Set up a set of circular apertures (one for each position) with a radius of 5 pixels and annuli with
# inner and outer radii of 10 and 15 pixels.
sm_apertures = CircularAperture(sm_positions, r=5)
sm_annulus_apertures = CircularAnnulus(sm_positions, r_in=10, r_out=15)

# Measure the total flux in both the aperture and annulus for each star. 
sm_apers = [sm_apertures, sm_annulus_apertures]
sm_phot_table = aperture_photometry(s1, sm_apers, method='subpixel',
                                 subpixels=5)

# Calculate the mean flux per pixel in the annulus
sm_sky_mean = sm_phot_table['aperture_sum_1'] / sm_annulus_apertures.area

# Multiply this by the number of pixels in the aperture and subtract from the aperture flux measurement.
# Put the result in a new column of the table.
sm_aperture_sky_sum = sm_sky_mean * sm_apertures.area
sm_phot_table['flux'] = sm_phot_table['aperture_sum_0'] - sm_aperture_sky_sum



#%%

# good_indices = []
# for i in range(len(ra_dec_sources)):
#     if (ra_dec_sources[i][0] in s1['ra']) or (ra_dec_sources[i][1] in s1['dec']):
#         good_indices.append(i)

# good_indices = []
# for i in range(len(sources)):
#     # if (sources['xcentroid'][i] in s1['ra']) or (sources['ycentroid'][i] in s1['dec']):
#     if (sources['xcentroid'][i] <= s1['ra'] + 0.01) or (sources['xcentroid'][i] <= s1['ra'] - 0.01):
#         good_indices.append(i)
# print(good_indices)


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
# s1['flux'] = 10**(- sm_sources['MOA_R_est'] / 2.5 )
# zp = s2['MOA_R_est'].values - s1['flux']
# zp = s2['MOA_R_est'].values - s1['fluxes']
zp = new_s2['MOA_R_est'].values - new_t1['flux']

# zp = s2['MOA_R_est'].values - sources['flux']
# 
#%%
plt.hist(zp,bins=100,color="darkviolet")
plt.grid("both")
plt.xlabel("zero points")
plt.ylabel("frequency")
plt.title("Calibrated Zero Points of Sources")
plt.show()
# sys_mag - sources['MOA_R_est']

#%%

for i in range(len(new_t1)):
    new_t1['mag'] = zp[i] - 2.5*np.log10(new_t1['flux'])
    
plt.hist(new_t1['mag'],bins=100,color="darkviolet")
plt.grid("both")
plt.xlabel("apparent magnitudes")
plt.ylabel("frequency")
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

# setting zero point and adjusting magnitudes
# Magnitude zero point is arbitrary
ZP = 30
phot_table['mag'] = ZP - 2.5*np.log10(sources['flux'])

print(phot_table)
#%%
# getting plots
diagnostic_plots(data,image_names[0],apertures,annulus_apertures,ZP,phot_table,outputs_path)

#%%
# annulus stuff
annulus_masks = annulus_apertures.to_mask(method='center')
plt.imshow(annulus_masks[0], interpolation='nearest')
plt.colorbar()
annulus_data = annulus_masks[0].multiply(data)
mask = annulus_masks[0].data
annulus_data_1d = annulus_data[mask > 0]
print(annulus_data_1d.shape)
_, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
print(median_sigclip)  

# background subtraction stuff
bkg_median = []
for mask in annulus_masks:
    annulus_data = mask.multiply(data)
    annulus_data_1d = annulus_data[mask.data > 0]
    _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
    bkg_median.append(median_sigclip)
bkg_median = np.array(bkg_median)

# Spitzer catalog stuff
hdu = load_spitzer_image()  
data = u.Quantity(hdu.data, unit=hdu.header['BUNIT'])  
wcs = WCS(hdu.header)  
catalog = load_spitzer_catalog() 
positions = SkyCoord(catalog['l'], catalog['b'], frame='galactic')  
aperture = SkyCircularAperture(positions, r=4.8 * u.arcsec) 

# error stuff
error = 0.1 * data

#%%
phot_table = aperture_photometry(data, apertures, error=error, wcs=wcs)
phot_table['annulus_median'] = bkg_median
phot_table['aper_bkg'] = bkg_median * apertures.area
phot_table['aper_sum_bkgsub'] = phot_table['aperture_sum'].value - phot_table['aper_bkg']
factor = (1.2 * u.arcsec) ** 2 / u.pixel
fluxes_catalog = catalog['f4_5']  
converted_aperture_sum = (phot_table['aperture_sum'] * factor).to(u.mJy / u.pixel) 
print(phot_table)

#%%





























