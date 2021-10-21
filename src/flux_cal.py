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
from astropy.io import fits
from astropy import wcs
from astropy.nddata import Cutout2D

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
#%%
all_mags = []
#%%
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++CHANGES++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# reading in image files from Reduced ALERTS folder
images_path = Path("//spcsfs/ave41/astro/ave41/ObsData-2021-02-18/ALERT/Reduced ALERT/WCS Calibrated/Sidereally Stacked Images")
# 
# images_path = Path("//spcsfs/ave41/astro/ave41/ObsData-2021-02-13/ALERT/Reduced ALERT/WCS Calibrated/Sidereally Stacked Images/Images for Flux Calibration")

d = np.genfromtxt('2021-02-18.txt',delimiter=",")
x_pixel_locs = d[0,:]
y_pixel_locs = d[1,:]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++CHANGES++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

outputs_path = path_checker(images_path,'Flux and Photometry Outputs')

# filtering data files
# for proper image list
lst_of_images = [str(images_path) +"/" + n 
                  for n in os.listdir(images_path) if (n.endswith('fits')) and n.__contains__('-aligned-')]
image_names = [n for n in os.listdir(images_path) if (n.endswith('fits')) and n.__contains__('-aligned-')]

# # for temp testing purposes
# temp_short_lst = [str(images_path) +"/" + n 
#                     for n in os.listdir(images_path) if (n.endswith('fits')) and n.__contains__('-aligned-')]

# temp_image_names = [n for n in os.listdir(images_path) if (n.endswith('fits')) and n.__contains__('-aligned-')]
# lst_of_images = [temp_short_lst[0]] #temp_short_lst[:2] #
# image_names =  [temp_image_names[0]] #temp_image_names[:2]



obsDates = []
mags_of_comet = []
target_name = 'C/2021 A7'
flux_errs = []



for i in range(len(lst_of_images)):
    # Read the image
    # i = 2
    x_pixel_loc = x_pixel_locs[i]
    y_pixel_loc = x_pixel_locs[i]
    
    data, hdr = fits.getdata(lst_of_images[i], header=True)
    with fits.open(lst_of_images[i], "append") as img_hdul:
        img_hdr1 = img_hdul[0].header
        target_ra = img_hdr1['RA      '].strip(' ')
        target_dec = img_hdr1['DEC     '].strip(' ')
        exptime = img_hdr1['EXPTIME ']
        obsDate = '20' + img_hdr1['DATE    '].strip(' ')
        target_location = '474'
        my_target_coords = [str(target_ra) + " " + str(target_dec)]
    
    #--------------------SECTION TWO: BACKGROUND DETECTION------------------------#
    
    # Estimate the sky background level and its standard deviation
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)   
    bkg = get_bkg_info(data) 
    
    #---------------------SECTION THREE: SOURCE DETECTION-------------------------#
    
    # Start up the DAOStarFinder object and detect stars
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)    
    moa_sources = daofind(data - median)
    
    #------------------------SECTION FOUR: APERTURES------------------------------#
    
    apertures, positions = postions_and_apertures(moa_sources)
    
    # Set up a set of circular apertures (one for each position) with a radius 
    # of 5 pixels and annuli with
    # inner and outer radii of 10 and 15 pixels.
    apertures = CircularAperture(positions, r=5)
    annulus_apertures = CircularAnnulus(positions, r_in=10, r_out=15)
    
    
    apers = [apertures, annulus_apertures]
    phot_table = aperture_photometry(data - bkg.background, apers, method='subpixel',
                                     subpixels=5)
    
    moa_sources = pixels_to_ra_dec(lst_of_images[i],phot_table)


    
    # Calculate the mean flux per pixel in the annulus
    sky_mean = moa_sources['aperture_sum_1'] / annulus_apertures.area
    
    # Multiply this by the number of pixels in the aperture and subtract from the aperture flux measurement.
    # Put the result in a new column of the table.
    aperture_sky_sum = sky_mean * apertures.area
    moa_sources['flux'] = moa_sources['aperture_sum_0'] - aperture_sky_sum
    
    #%%
    from photutils.utils import calc_total_error
    from photutils.segmentation import SourceCatalog
    from photutils.segmentation import deblend_sources
    from photutils.segmentation import detect_threshold
    from astropy.convolution import Gaussian2DKernel
    from astropy.stats import gaussian_fwhm_to_sigma
    from photutils.segmentation import detect_sources
    
    
    flux_err = []
    effective_gain = exptime
    for k in range(len(moa_sources)):
        error = np.mean(calc_total_error(moa_sources['flux'][k], bkg.background_rms, effective_gain))
        moa_sources['err'] = error
        flux_err.append(error)
    
    #%%
    # threshold = detect_threshold(data, nsigma=2)
    # threshold =2.0 * bkg.background_rms
    # sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
    # kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    # kernel.normalize()
    # segm = detect_sources(data, threshold, npixels=5)#, kernel=kernel
    # segm_deblend = deblend_sources(data, segm, npixels=5, kernel=kernel,
    #                             nlevels=32, contrast=0.001)
    # cat = SourceCatalog(data, segm_deblend, error=error)

    
    
    #---------------------SECTION FIVE: FLUX CALIBRATION--------------------------#
    
    # moa_sources = pixels_to_ra_dec(lst_of_images[0],moa_sources)
    
    # RYAN'S CODE
    R_filter = 'moared.txt' #wavelengths, ?
    R_fit = cal.sauron(band=R_filter,system='skymapper',gr_lims=[-.5,0.8],plot=True,cubic_corr=False)
    
    sm_sources = skymapper_sources(my_target_coords)
    R_estimates = R_fit.estimate_mag(mags = sm_sources)
    sm_sources['MOA_R_est'] = R_estimates
    
    plt.hist(R_estimates,bins=50,color="darkviolet")
    plt.grid("both")
    plt.xlabel("apparent magnitudes")
    plt.ylabel("frequency")
    plt.title("Estimated Magnitudes of Sources in MOA-R")
    # plt.savefig(outputs_path/"est_mags-{}.jpeg".format(img_name),dpi=900)
    plt.show()
     
    
    def sm_to_moa_transform(sm_sources,moa_sources):
        s2 = deepcopy(sm_sources)
        
        # dra = moa_sources['xcentroid'][:,np.newaxis] - s2['ra'].values[np.newaxis,:]
        # ddec = moa_sources['ycentroid'][:,np.newaxis] - s2['dec'].values[np.newaxis,:]
        dra = moa_sources['xcenter'].value[:,np.newaxis] - s2['ra'].values[np.newaxis,:]
        ddec = moa_sources['ycenter'].value[:,np.newaxis] - s2['dec'].values[np.newaxis,:]
        dist = np.sqrt(dra**2 + ddec**2)
        min_value = np.nanmin(dist,axis=0)
        min_index = np.argmin(dist,axis=0)
        
        # finding closest sources to SM sources in MOA sources
        t1 = moa_sources[min_index]
        good_indices = []
        for j in range(len(t1)):
            # try:
                # if np.abs(t1['xcentroid'][0] - s2['ra'].values[0]) >= min_value[i]:
            if np.abs(t1['xcenter'].value[j] - s2['ra'].values[j]) >= 0.9*min_value[j]:
                good_indices.append(j)
            # except:
                # if np.abs(t1['xcenter'].value[i] - s2['ra'].values[i]) >= 0.5*min_value[i]:
                    # good_indices.append(i)
        print(good_indices)
        
        # MOA and SM source lists are now filtered to only include the matching sources
        new_t1 = t1[good_indices]
        new_s2 = s2.iloc[good_indices]
        # print(new_t1['flux'][0])
        # print(new_t1)
        print(new_t1['flux'])
        
        # !! I THINK THIS IS WHERE SOMETHING IS GOING WRONG !! #
        
        m_moa = (-2.5 * np.log10(new_t1['flux'])) + new_s2['MOA_R_est'].values
        # # calculating zero point
        # zp = new_s2['MOA_R_est'].values - new_t1['flux']
        zp = new_s2['MOA_R_est'].values - m_moa
        # zp = new_t1['flux'] - new_s2['MOA_R_est'].values
        print(np.nanmean(zp))
        
        # # converting flux to mag
        for a in range(len(new_t1)):
            new_t1['mag'] = zp[a] - (2.5 * np.log10(new_t1['flux']))
        
        # # final_calibrated_mags = new_s2['MOA_R_est'].values - new_t1['mag'] #initially a -ve but chnaged to a +ve?
        final_calibrated_mags = new_s2['MOA_R_est'].values - new_t1['mag']
        
        # # this code works
        # zp = new_s2['MOA_R_est'].values - new_t1['mag']
        # m_moa = (-2.5 * np.log10(new_t1['flux'].data)) + zp
        # print(m_moa)
        # final_calibrated_mags = m_moa
        
        return new_t1, new_s2, zp, final_calibrated_mags
    
    new_t1, new_s2, zp, final_calibrated_mags = sm_to_moa_transform(sm_sources,moa_sources)
   
    plt.hist(final_calibrated_mags,color="darkviolet")
    plt.grid("both")
    plt.xlabel("apparent magnitudes")
    plt.ylabel("frequency")
    plt.title("Final Calibrated Magnitudes of Sources")
    # plt.savefig(outputs_path/"final_cal_mags-{}.jpeg".format(img_name),dpi=900)
    plt.show()
    
    # PLOTS
    plotting_funcs_flux_cal(image_names[i],sm_sources,zp,new_t1,new_s2,
                                final_calibrated_mags,outputs_path)
    
    #---------------------SECTION SIX: COMET PHOTOMETRY---------------------------#
    
    # do jpl horizons stuff here to find coords of comet
    t = Time(obsDate, format='fits', scale='utc')
    jd_obsDate = t.jd
    
    obj = Horizons(id=target_name, location=target_location,epochs=i)#i
    eph = obj.ephemerides()
    
    observations = Horizons(id=target_name, location=target_location,epochs=jd_obsDate)
    jpl_data = observations.ephemerides()
    
    # these are the predicted postions of  comet in this img
    comet_actual_ra = jpl_data['RA'][0]
    comet_actual_dec = jpl_data['DEC'][0]
    comet_geocentric_distance = jpl_data['delta'][0] #au
    ## do conversion stuff here if needed
    
    # # now need to check if ra/dec in new_t1 and new_s2 match this bad boi
    # new_t1_ra_match = np.where(comet_actual_ra == new_t1['xcenter'].value) #xcentroid
    # new_s2_ra_match = np.where(comet_actual_ra == new_s2['ra'])
    # print(new_t1_ra_match == new_s2_ra_match )
    
    # # # do same for dec

    # new_t1_ra_match = np.where(comet_actual_ra == new_t1['xcenter'].value)
    # new_s2_ra_match = np.where(comet_actual_ra == new_s2['ra'])
    # print(new_t1_ra_match == new_s2_ra_match)
    
    # s3 = deepcopy(new_s2)
    
    # dra = new_t1['xcenter'].value[:,np.newaxis] - s3['ra'].values[np.newaxis,:]
    # ddec = new_t1['ycenter'].value[:,np.newaxis] - s3['dec'].values[np.newaxis,:]
    # dist = np.sqrt(dra**2 + ddec**2)
    # min_value = np.nanmin(dist,axis=0)
    # min_index = np.argmin(dist,axis=0)
    
    # t2 = new_t1[min_index]
    
    # good_indices = []
    # for i in range(len(t2)):
    #     if np.abs(t2['xcenter'].value[0] - s3['ra'].values[0]) >= min_value[i]:
    #         good_indices.append(i)
        
    # new_t2 = t2[good_indices]
    # new_s3 = s3.iloc[good_indices]
    
    
    
    # calculating arcsec size of comet to use for FWHM/aperture ring size
    epoch_comet_ang_radius = comet_ang_size(comet_geocentric_distance,diameter=20000) # arcsec
    fwhm_radius_in_pixels = arcsec_to_pixel(epoch_comet_ang_radius)
    print("epoch_comet_ang_radius: ",epoch_comet_ang_radius)
    print("fwhm_radius_in_pixels: ",fwhm_radius_in_pixels)
    
    # mean, median, std = sigma_clipped_stats(data, sigma=3.0)   
    # bkg = get_bkg_info(data) 
    # daofind = DAOStarFinder(fwhm=fwhm_radius_in_pixels, threshold=5.*std)    
    # sources = daofind(data - median)

#     moa_sources = pixels_to_ra_dec(lst_of_images[0],sources)
    
#     closest_ra_value_to_comet = min(range(len(moa_sources['xcentroid'])), 
#                                     key=lambda i: abs(moa_sources['xcentroid'][i]-comet_actual_ra))
#     closest_dec_value_to_comet = min(range(len(moa_sources['ycentroid'])), 
#                                      key=lambda i: abs(moa_sources['ycentroid'][i]-comet_actual_dec))
    
    
#     print(comet_actual_ra,comet_actual_dec)
#     print(closest_ra_value_to_comet,closest_dec_value_to_comet)
#     moa_mag = moa_sources['mag'][closest_ra_value_to_comet]
    
#     mag_of_comet = moa_mag + np.nanmean(zp)
    
#     mags_of_comet.append(mag_of_comet)
#     obsDates.append(obsDate)


# plt.plot(obsDates,mags_of_comet,'.',color="darkviolet")
# # plt.plot(mags_of_comet,'.',color="darkviolet")
# plt.grid("both")
# plt.xlabel("epoch")
# plt.ylabel("apparent magnitude")
# plt.title("Brightness Variation in Observations of {}".format(target_name))
# plt.show()

    f = fits.open(lst_of_images[i])
    w = wcs.WCS(f[0].header)
    newf = fits.PrimaryHDU()
    
    newf.data = f[0].data[100:-100,100:-100]
    newf.header = f[0].header
    newf.header.update(w[100:-100,100:-100].to_header())
    
    comet_pixel_location_x = x_pixel_loc
    comet_pixel_location_y = y_pixel_loc
    
    position = (comet_pixel_location_x, comet_pixel_location_y)
    shape = (100, 100)
    cutout = Cutout2D(f[0].data, position, shape, wcs=w)
    
     # norm = simple_norm(data, 'sqrt', percent=96) # another way to "stretch" the image display
     #    plt.imshow(data, norm=norm,origin='lower')
    # norm = simple_norm(cutout.data, 'sqrt', percent=96)
    # plt.imshow(cutout.data, norm=norm, origin='lower')
    # plt.colorbar()
    # plt.title("Cutout of {} on {}".format(target_name,obsDate))
    # plt.show()


    f.data = cutout.data
    # Update the FITS header with the cutout WCS
    f[0].header.update(cutout.wcs.to_header())
    # Write the cutout to a new FITS file
    # cutout_filename = 'example_cutout.fits'
    # f.writeto(outputs_path/cutout_filename, overwrite=True)
    
    # cutout_img = '//spcsfs/ave41/astro/ave41/ObsData-2021-02-18/ALERT/Reduced ALERT/WCS Calibrated/Sidereally Stacked Images/Flux and Photometry Outputs/example_cutout.fits'

#%%
    mean, median, std = sigma_clipped_stats(cutout.data, sigma=3.0)   
    bkg = get_bkg_info(cutout.data) 
    daofind = DAOStarFinder(fwhm=fwhm_radius_in_pixels*2, threshold=5.*std)    
    comet_sources = daofind(cutout.data - median)
 
    centre_postion = (50,50)
    apertures = CircularAperture(centre_postion, r=fwhm_radius_in_pixels*2)
    annulus_apertures = CircularAnnulus(centre_postion, r_in=fwhm_radius_in_pixels*3, r_out=fwhm_radius_in_pixels*4)

    norm = simple_norm(cutout.data, 'sqrt', percent=96)
    plt.imshow(cutout.data, norm=norm, origin='lower')
    plt.colorbar()
    apertures.plot(color='deeppink', lw=2)
    annulus_apertures.plot(color='red', lw=2)
    plt.title("Cutout of {} on {}".format(target_name,obsDate))
    plt.savefig(outputs_path/"cutout_with_annulus_{}".format(i))
    plt.show()
    
    
    
    #%%
    # closest_x_value = min(range(len(comet_sources['xcentroid'])), 
    #                             key=lambda i: abs(comet_sources['xcentroid'][i]-comet_pixel_location_x))
    # closest_y_value = min(range(len(comet_sources['ycentroid'])), 
    #                             key=lambda i: abs(comet_sources['ycentroid'][i]-comet_pixel_location_y))
    
    dra = comet_sources['xcentroid'][:,np.newaxis] - comet_pixel_location_x
    ddec = comet_sources['ycentroid'][:,np.newaxis] - comet_pixel_location_y
    dist = np.sqrt(dra**2 + ddec**2)
    min_value = np.nanmin(dist,axis=0)
    min_index = np.argmin(dist,axis=0)
    #%%
    effective_gain = exptime
    for l in range(len(comet_sources)):
        error = np.mean(calc_total_error(comet_sources['flux'][l], bkg.background_rms, effective_gain))
        comet_sources['err'] = error
#%%    
    
    # comet_index = np.argmax(comet_sources['flux'])
    comet_index = min_index
    comet_max_flux = comet_sources['flux'][comet_index]



# closest_ra_value_to_comet = min(range(len(comet_sources['xcentroid'])), 
#                                 key=lambda i: abs(comet_sources['xcentroid'][i]-comet_actual_ra))
# closest_dec_value_to_comet = min(range(len(comet_sources['ycentroid'])), 
#                                   key=lambda i: abs(comet_sources['ycentroid'][i]-comet_actual_dec))

    comet_mag = comet_sources['mag'][comet_index]
    mag_of_comet_cal = comet_mag + np.nanmean(zp)
    print(mag_of_comet_cal)
    
    moa_mag =  (-2.5 * np.log10(comet_sources['flux'][comet_index])) + np.nanmean(zp)
    
    mag_of_comet = moa_mag + np.nanmedian(zp)
    print("mag_of_comet: ",mag_of_comet)
    
    
    plt.plot(mag_of_comet,obsDate)

    obsDates.append(obsDate)
    mags_of_comet.append(mag_of_comet[0])
    flux_errs.append(flux_err)
    
    all_mags.append([obsDates,mags_of_comet,flux_errs])   

plt.plot(obsDates,mags_of_comet)

#%%

# centre_postion = (50,48)
# apertures = CircularAperture(centre_postion, r=fwhm_radius_in_pixels*2)
# annulus_apertures = CircularAnnulus(centre_postion, r_in=fwhm_radius_in_pixels*3, r_out=fwhm_radius_in_pixels*4)

# plt.figure()
# norm = simple_norm(cutout.data, 'sqrt', percent=96) # another way to "stretch" the image display
# plt.imshow(cutout.data, norm=norm,origin='lower')
# plt.colorbar()
# apertures.plot(color='deeppink', lw=2)
# annulus_apertures.plot(color='red', lw=2)
# plt.title("Zoomed-In Apertures for Comet")
# # plt.savefig(outputs_path/"apertures-{}.jpeg".format(image_name),dpi=900)
# plt.show()

#%%
# apers = [apertures, annulus_apertures]
# comet_phot_table = aperture_photometry(cutout.data - bkg.background, apers, method='subpixel',
#                                      subpixels=5)

# comet_sources = pixels_to_ra_dec(lst_of_images[0],comet_phot_table)

# # Calculate the mean flux per pixel in the annulus
# sky_mean = comet_sources['aperture_sum_1'] / annulus_apertures.area

# # Multiply this by the number of pixels in the aperture and subtract from the aperture flux measurement.
# # Put the result in a new column of the table.
# aperture_sky_sum = sky_mean * apertures.area
# comet_sources['flux'] = comet_sources['aperture_sum_0'] - aperture_sky_sum
    
#%%
# comet_sources = pixels_to_ra_dec(cutout_img,comet_sources)
# comet_index = np.argmax(comet_sources['flux'])
# comet_max_flux = comet_sources['flux'][comet_index]

#%%

# # print(comet_actual_ra,comet_actual_dec)
# # moa_mag = comet_sources['mag'][comet_index]
# moa_mag =  (2.5 * np.log10(comet_sources['flux'][comet_index])) + np.nanmean(zp)

# mag_of_comet = moa_mag + np.nanmedian(zp)
# print("mag_of_comet: ",mag_of_comet)

# mags_of_comet.append(mag_of_comet)
# obsDates.append(obsDate)







# for i in all_mags:
#     plt.plot(i[0],i[1],'.')
#     plt.grid("both")
#%%



# fig, ax = plt.subplots(figsize=(15, 15))
# for m in all_mags:
#     ax.plot(m[0],m[1],'.',label=m[0][1][:10])
# ax.grid("both")
# ax.invert_xaxis()
# plt.xticks(rotation=90)
# plt.legend()
# plt.xlabel("epoch")
# plt.ylabel("apparent magnitude")
# plt.title("Light Curve of C/2021 A7")
# plt.savefig("final_lightcurve.jpeg")
    