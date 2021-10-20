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
from astropy.time import Time
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
from copy import deepcopy
import calibrimbore as cal 
from astroquery.jplhorizons import Horizons

# path-type packages
import os
import glob
from glob import glob
from pathlib import Path

# initialising starting directory
code_home_path = "C:/Users/ave41/OneDrive - University of Canterbury//ASTR480 Research/ASTR480 Code/01 Data Reduction Pipeline/DataReductionPipeline/src"
os.chdir(code_home_path) #from now on, we are in this directory

# importing functions
from drp_funcs import *
from asp_funcs import *

# initialising starting directory
code_home_path = "C:/Users/ave41/OneDrive - University of Canterbury//ASTR480 Research/ASTR480 Code/02 Data Analysis/Flux-Photometry-Analysis/src"
os.chdir(code_home_path) #from now on, we are in this directory

# importing functions
from flux_cal_funcs import *

###############################################################################
#----------------------SECTION ONE: INITIALISATION----------------------------#
###############################################################################

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++CHANGES++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# reading in image files from Reduced ALERTS folder
images_path = Path("//spcsfs/ave41/astro/ave41/ObsData-2021-02-18/ALERT/Reduced ALERT/WCS Calibrated/Sidereally Stacked Images")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++CHANGES++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

outputs_path = path_checker(images_path,'Flux and Photometry Outputs')

# filtering data files
# for proper image list
# lst_of_images = [str(images_path) +"/" + n 
#                   for n in os.listdir(images_path) if (n.endswith('fits')) and n.__contains__('-aligned-')]
# image_names = [n for n in os.listdir(images_path) if (n.endswith('fits')) and n.__contains__('-aligned-')]

# for temp testing purposes
temp_short_lst = [str(images_path) +"/" + n 
                   for n in os.listdir(images_path) if (n.endswith('fits')) and n.__contains__('-aligned-')]

temp_image_names = [n for n in os.listdir(images_path) if (n.endswith('fits')) and n.__contains__('-aligned-')]
lst_of_images = [temp_short_lst[0]] #temp_short_lst[:2] #
image_names =  [temp_image_names[0]] #temp_image_names[:2]


#%%
obsDates = []
mags_of_comet = []
target_name = 'C/2021 A7'
for i in range(len(lst_of_images)):
# i = 0
    # Read the image
    data, hdr = fits.getdata(lst_of_images[i], header=True)
    with fits.open(lst_of_images[i], "append") as img_hdul:
        img_hdr1 = img_hdul[i].header
        target_ra = img_hdr1['RA      '].strip(' ')
        target_dec = img_hdr1['DEC     '].strip(' ')
        obsDate = '20' + img_hdr1['DATE    '].strip(' ')
        target_location = '474'
        my_target_coords = [str(target_ra) + " " + str(target_dec)]
    #%%
    #--------------------SECTION TWO: BACKGROUND DETECTION------------------------#
    
    # Estimate the sky background level and its standard deviation
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)   
    bkg = get_bkg_info(data) 
    #%%
    #---------------------SECTION THREE: SOURCE DETECTION-------------------------#
    
    # Start up the DAOStarFinder object and detect stars
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)    
    moa_sources = daofind(data - median)
    #%%
    #------------------------SECTION FOUR: APERTURES------------------------------#
    
    apertures, positions = postions_and_apertures(moa_sources)
    
    # Set up a set of circular apertures (one for each position) with a radius 
    # of 5 pixels and annuli with
    # inner and outer radii of 10 and 15 pixels.
    apertures = CircularAperture(positions, r=5)
    annulus_apertures = CircularAnnulus(positions, r_in=10, r_out=15)
    #%%
    #---------------------SECTION FIVE: FLUX CALIBRATION--------------------------#
    
    moa_sources = pixels_to_ra_dec(lst_of_images[i],moa_sources)
    
    # RYAN'S CODE
    R_filter = 'moared.txt' #wavelengths, ?
    R_fit = cal.sauron(band=R_filter,system='skymapper',gr_lims=[-.5,0.8],plot=True,cubic_corr=False)
    
    sm_sources = skymapper_sources(my_target_coords)
    R_estimates = R_fit.estimate_mag(mags = sm_sources)
    sm_sources['MOA_R_est'] = R_estimates
    
    #%%
    def sm_to_moa_transform(sm_sources,moa_sources):
        s2 = deepcopy(sm_sources)
        
        dra = moa_sources['xcentroid'][:,np.newaxis] - s2['ra'].values[np.newaxis,:]
        ddec = moa_sources['ycentroid'][:,np.newaxis] - s2['dec'].values[np.newaxis,:]
        dist = np.sqrt(dra**2 + ddec**2)
        min_value = np.nanmin(dist,axis=0)
        min_index = np.argmin(dist,axis=0)
        
        # finding closest sources to SM sources in MOA sources
        t1 = moa_sources[min_index]
        good_indices = []
        for i in range(len(t1)):
            if np.abs(t1['xcentroid'][0] - s2['ra'].values[0]) >= min_value[i]:
                good_indices.append(i)
        
        # MOA and SM source lists are now filtered to only include the matching sources
        new_t1 = t1[good_indices]
        new_s2 = s2.iloc[good_indices]
        
        
        # !! I THINK THIS IS WHERE SOMETHING IS GOING WRONG !! #
        
        # calculating zero point
        zp = new_s2['MOA_R_est'].values - new_t1['flux']
        # zp = new_t1['flux'] - new_s2['MOA_R_est'].values
        
        # converting flux to mag
        for i in range(len(new_t1)):
            new_t1['mag'] = zp[i] - (2.5 * np.log10(new_t1['flux']))
        
        # final_calibrated_mags = new_s2['MOA_R_est'].values - new_t1['mag'] #initially a -ve but chnaged to a +ve?
        final_calibrated_mags = new_s2['MOA_R_est'].values - new_t1['mag']
        
        return new_t1, new_s2, zp, final_calibrated_mags
    
    new_t1, new_s2, zp, final_calibrated_mags = sm_to_moa_transform(sm_sources,moa_sources)
    
    plt.hist(final_calibrated_mags,bins=50,color="darkviolet")
    plt.grid("both")
    plt.xlabel("apparent magnitudes")
    plt.ylabel("frequency")
    plt.title("Final Calibrated Magnitudes of Sources")
    # plt.savefig(outputs_path/"final_cal_mags-{}.jpeg".format(img_name),dpi=900)
    plt.show()
    #%%
    zp_no_nan = []
    for i in zp:
        if np.isnan(i):
            pass
        else:
          zp_no_nan.append(i)  
    #%%
    # PLOTS
    # plotting_funcs_flux_cal(image_names,sm_sources,zp,new_t1,new_s2,
    #                             final_calibrated_mags,outputs_path)
    plotting_funcs_flux_cal(image_names,sm_sources,zp,new_t1,new_s2,
                                final_calibrated_mags,outputs_path)
    #%%
    #---------------------SECTION SIX: COMET PHOTOMETRY---------------------------#
    
    # do jpl horizons stuff here to find coords of comet
    t = Time(obsDate, format='fits', scale='utc')
    jd_obsDate = t.jd
    
    obj = Horizons(id=target_name, location=target_location,epochs=i)
    eph = obj.ephemerides()
    
    observations = Horizons(id=target_name, location=target_location,epochs=jd_obsDate)
    jpl_data = observations.ephemerides()
    
    # these are the predicted postions of  comet in this img
    comet_actual_ra = jpl_data['RA'][0]
    comet_actual_dec = jpl_data['DEC'][0]
    comet_geocentric_distance = jpl_data['delta'][0] #au
    ## do conversion stuff here if needed
    
    # now need to check if ra/dec in new_t1 and new_s2 match this bad boi
    new_t1_ra_match = np.where(comet_actual_ra == new_t1['xcentroid'])
    new_s2_ra_match = np.where(comet_actual_ra == new_s2['ra'])
    print(new_t1_ra_match == new_s2_ra_match )
    
    # # do same for dec
    
    #%%
    new_t1_ra_match = np.where(comet_actual_ra == new_t1['xcentroid'])
    new_s2_ra_match = np.where(comet_actual_ra == new_s2['ra'])
    print(new_t1_ra_match == new_s2_ra_match)
    
    #%%
    
    s3 = deepcopy(new_s2)
    
    dra = new_t1['xcentroid'][:,np.newaxis] - s3['ra'].values[np.newaxis,:]
    ddec = new_t1['ycentroid'][:,np.newaxis] - s3['dec'].values[np.newaxis,:]
    dist = np.sqrt(dra**2 + ddec**2)
    min_value = np.nanmin(dist,axis=0)
    min_index = np.argmin(dist,axis=0)
    
    t2 = new_t1[min_index]
    
    good_indices = []
    for i in range(len(t2)):
        if np.abs(t2['xcentroid'][0] - s3['ra'].values[0]) >= min_value[i]:
            good_indices.append(i)
        
    new_t2 = t2[good_indices]
    new_s3 = s3.iloc[good_indices]
        
    #%%
    plt.plot(zp_no_nan,'.')
    plt.show()
    print(np.mean(zp_no_nan))
    
    #%%
    
    # calculating arcsec size of comet to use for FWHM/aperture ring size
    epoch_comet_ang_size = comet_ang_size(comet_geocentric_distance,diameter=10000) # arcsec
    fwhm_radius_in_pixels = arcsec_to_pixel(epoch_comet_ang_size)
    
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)   
    bkg = get_bkg_info(data) 
    daofind = DAOStarFinder(fwhm=fwhm_radius_in_pixels*2, threshold=5.*std)    
    sources = daofind(data - median)
    #%%
    moa_sources = pixels_to_ra_dec(lst_of_images[0],sources)
    #%%
    closest_ra_value_to_comet = min(range(len(moa_sources['xcentroid'])), 
                                    key=lambda i: abs(moa_sources['xcentroid'][i]-comet_actual_ra))
    closest_dec_value_to_comet = min(range(len(moa_sources['ycentroid'])), 
                                     key=lambda i: abs(moa_sources['ycentroid'][i]-comet_actual_dec))
    
    
    
    mag_of_comet = moa_sources['mag'][closest_ra_value_to_comet]
    
    mags_of_comet.append(mag_of_comet)
    obsDates.append(obsDate)

#%%

plt.plot(obsDates,mags_of_comet,'.',color="darkviolet")
# plt.plot(mags_of_comet,'.',color="darkviolet")
plt.grid("both")
plt.xlabel("epoch")
plt.ylabel("apparent magnitude")
plt.title("Brightness Variation in Observations of {}".format(target_name))
plt.show()







    