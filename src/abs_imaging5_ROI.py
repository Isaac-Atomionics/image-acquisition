# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:27:44 2020

@author: Atomionics
"""

'''
Created 22/01/2020, windows 64-bit python 3.7

This script is a test of the Basler ace acA2040-90umNIR camera. In this script the camera is software triggered, and will only take
a picture and save it when you enter in the command line. 
Make sure Spyder has clear all variables on execution turned on, or throws up device is exclusively opened by another client error on 
next runs if there was a timeout error from retrieve result. May still need to clear variables after timeout error.

Update savepath (line 29) to save images in the folder you want. DeviceLinkThroughputLimitMode (line 41) is needed to stop Payload error.

'''

from pypylon import pylon
from pypylon import genicam
from datetime import datetime
import os
from math import pi
from math import sqrt
import cv2
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fmin
from scipy import exp
import scipy.misc as mpl
import csv
#from os import path
import matplotlib.pyplot as plt

def gaus(x,b,a,x0,sigma):
    return b+a*np.exp(-(x-x0)**2/(2*sigma**2))

img = pylon.PylonImage()
tlf = pylon.TlFactory.GetInstance()


''' Set parameters.'''
num_run = 2 #number of run

#Mention below the starting and end points of x and Z ROI. This wil also be used to plot two red lines showing the region
ROI_x_start = 1124
ROI_x_end = 1348
ROI_z_start = 200
ROI_z_end = 700
# Save path to images folder
savepath = 'C:/Users/admin/Desktop/camera_stuff/images/Abs_imaging-2020-12-20/'
folder_date = datetime.now().strftime("%Y-%m-%d")
folder_to_save_files = savepath 

if not os.path.exists(folder_to_save_files):
    os.mkdir(folder_to_save_files)
    
  # return input("Enter \"t\" to start acquisition or \"e\" to exit, and press enter. (t/e) ")
a = "a"
b = "b"
c = "c"
#%% Rubidium properties

amu = 1.66053873e-27    
isotope = 85

if isotope == 85:
    mass = 85*amu
    lam = 780.2e-9
    gam = 6.067
    Isat = 16.69
    scatleng = 0 # Bohr radii
    scattXsection = 8*pi*(scatleng*5.29e-11)**2  # m^2
    threebodyloss = 0  # m^6/s
elif isotope == 87:
    mass = 87*amu
    lam = 780.2e-9
    gam = 6.065
    Isat = 16.69
    scatleng = 0 # Bohr radii
    scattXsection = 8*pi*(scatleng*5.29e-11)**2  #m^2
    threebodyloss = 0  # m^6/s

#%% Supposedly imported from the GUI 

px_size = 5.5e-6
binning = 1
magnif = 0
px = px_size*binning
tof =40e3
I = 0.06*Isat
IoverIs = I/Isat
delta = 0

ROI_sum=0
#%% Constants definition

kB = 1.38e-23
muB = 9.27e-24
h = 6.63e-34

#%% Importation of the 3 images
info=pylon.DeviceInfo()
info.SetSerialNumber("22943480") #22943480, 23103825
# Create an instant camera object with the camera device found first.
cam = pylon.InstantCamera(tlf.CreateFirstDevice())
# Print the model name of the camera.
print("Device:", cam.GetDeviceInfo().GetModelName())


# Need camera open to make changes to acqusition
cam.Open()
# In order to make sure the payload error doesn't appear
cam.DeviceLinkThroughputLimitMode='Off'

# Pixel format of images taken
cam.PixelFormat='Mono8'
#cam.PixelFormat='Mono12'
print('Pixel Format:',cam.PixelFormat.GetValue())
# Acquisition mode set to continuous so will take images while there are still triggers
cam.AcquisitionMode='Continuous'
print('Acquisition Mode:',cam.AcquisitionMode.GetValue())
  
# Set camera Gamma value to 1, so brightness is unchanged
cam.Gamma=1
print('Gamma:',cam.Gamma.GetValue())
# Set black level of images
cam.BlackLevel=0
print('Black Level:',cam.BlackLevel.GetValue())

# Binning of the pixels, to increase camera response to light
# Set horizontal and vertical separately, any combination allowed
cam.BinningHorizontal=1
cam.BinningVertical=1
print('Binning (HxV):',cam.BinningHorizontal.GetValue(), 'x', cam.BinningVertical.GetValue())
# Set gain, range 0-23.59 dB. Gain auto must be turned off
cam.GainAuto='Off'
cam.Gain=1
if cam.GainAuto.GetValue()=='Continuous':
    print('Gain: Auto')
else:
    print('Gain:', cam.Gain.GetValue(),'dB')
    
# Set trigger options; Trigger action, FrameStart takes a single shot
cam.TriggerSelector='FrameStart'
if cam.TriggerSelector.GetValue()=='FrameStart':
    print('Frame Trigger: Single')
elif cam.TriggerSelector.GetValue()=='FrameBurstStart':
    print('Frame Trigger:')
else:
    print()
# Set trigger to on, default (off) is free-mode
cam.TriggerMode='On'
# Set Line 3 or 4 to input for trigger, Line 1 is only input so is not required
#cam.LineSelector='Line3'
#cam.LineMode='Input' 
# Set trigger source
cam.TriggerSource='Line1'
print('Trigger Source:',cam.TriggerSource.GetValue())
# Set edge to trigger on
cam.TriggerActivation='RisingEdge'
print('Trigger Activation:',cam.TriggerActivation.GetValue())

# Set mode for exposure, automatic or a specified time
cam.ExposureAuto='Off'
# When hardwire triggering, the exposure mode must be set to Timed, even with continuous auto
cam.ExposureMode='TriggerWidth'
# Set exposure time, in microseconds, if using Timed mode
#cam.ExposureTime=1500
if cam.ExposureAuto.GetValue()=='Continuous':
    print('Exposure: Auto')
elif cam.ExposureMode.GetValue()=='TriggerWidth':
    print('Exposure: Trigger')
elif cam.ExposureMode.GetValue()=='Timed':
    print('Exposure Time:', cam.ExposureTime.GetValue(),'us')
else:
    print()
# Numbering for images so image always saved even when files already in Images folder
dirInfo = os.listdir(savepath)    
sizeDirInfo = len(dirInfo)

if sizeDirInfo == 0:
    lastImageNum = 1
else:
    lastImageName = dirInfo[(sizeDirInfo-1)]
    lastImageNum = float(lastImageName[15:20]) + 1
    
''' Saving images. '''
try:
    for i in range(num_run):
                print("run no. = ", i)
            # Starts acquisition and camera waits for frame trigger
                cam.AcquisitionStart.Execute()
    
            # Start grabbing of images, unlimited amount, default type is continuous acquisition
                cam.StartGrabbing()
                # RetrieveResult will timeout after a specified time, in ms, so set much larger than the time of a cycle
                with cam.RetrieveResult(100000) as result:
            
                    # Calling AttachGrabResultBuffer creates another reference to the
                    # grab result buffer. This prevents the buffer's reuse for grabbing.
                    img.AttachGrabResultBuffer(result)
                        
                    filename = savepath + folder_date + "-img_%05d_a.png" % (lastImageNum + i)
                    # Save image to
                    img.Save(pylon.ImageFileFormat_Png, filename)
                    
                    print(filename)
                    # In order to make it possible to reuse the grab result for grabbing
                    # again, we have to release the image (effectively emptying the
                    # image object).
                    img.Release()
                    img_at = cv2.imread(filename,0) # With atom
                   
                with cam.RetrieveResult(2000) as result2:   
                    # Calling AttachGrabResultBuffer creates another reference to the
                    # grab result buffer. This prevents the buffer's reuse for grabbing.
                    img.AttachGrabResultBuffer(result2)
                    
                    filename = savepath + folder_date + "-img_%05d_b.png" % (lastImageNum + i)
                    # Save image to
                    img.Save(pylon.ImageFileFormat_Png, filename)
                    print(filename)
                    # In order to make it possible to reuse the grab result for grabbing
                    # again, we have to release the image (effectively emptying the
                    # image object).
                    img.Release()
                    img_las = cv2.imread(filename,0) # Laser alone
                    
                with cam.RetrieveResult(2000) as result3:   
                    # Calling AttachGrabResultBuffer creates another reference to the
                    # grab result buffer. This prevents the buffer's reuse for grabbing.
                    img.AttachGrabResultBuffer(result3)
                    
                    filename = savepath + folder_date + "-img_%05d_c.png" % (lastImageNum + i)
                    # Save image to
                    img.Save(pylon.ImageFileFormat_Png, filename)
                    print(filename)
                    # In order to make it possible to reuse the grab result for grabbing
                    # again, we have to release the image (effectively emptying the
                    # image object).
                    img.Release()
                    img_bck = cv2.imread(filename,0) # Background
                    
                cam.StopGrabbing()
                
                #%% Creation of the 2D array of optical depths
                
                # Substraction of the background
                img_las = img_las - img_bck
                img_at = img_at -img_bck
                ###Correction of the zeros
                imin = min(np.min(img_las[img_las > 0]), np.min(img_at[img_at > 0]))
                img_las[img_las <= 0] = imin
                img_at[img_at <= 0] = imin

                # a = np.asarray(img_las)
                # b = np.asarray(img_at)
                # c = a/((b.astype('float')+1)/256)
                # d = c*(c<255)+255*np.ones(np.shape(c))*(c>255)
                # e =  d.astype('uint8')
                # 
                # OD=e
                #OD = np.divide(img_at,img_las)
                # Calculation of the optical depth
                OD = np.log(np.divide(img_las,img_at))
                print("OD done") 
                # print(OD[1200,700])
                # Negative transformation if gaussian value < background
                # if OD[int(len(OD[:,0])/2),int(len(OD[0,:])/2)] < OD[0,0]:
                #     for i in range(0, len(OD[:,0])):
                #         for j in range(0, len(OD[0,:])):
                #             OD[i,j] = np.max(OD) - OD[i,j]
                        
            #    with np.printoptions(threshold=np.inf):
                  
                #Plot OD on 2,2,1
                fig = plt.figure()
                plt.subplot(2,2,4)
               #plt.title("OD plot - full scale")
               # plt.xlabel("x")
               #plt.ylabel("OD")
                plt.imshow(OD)
               # plt.clim(0,1) # colourbar limit (,)
                plt.colorbar()
                  
                   #%% Fitting along the x axis (integrated over all z values)
                   # Gaussian fitting to get the width
                sum_x = np.sum(OD, axis = 0)
                n = len(sum_x)
                x = np.linspace(0,n,n)
                fit_x,pcov = curve_fit(gaus,x,sum_x,p0=[imin,np.max(sum_x),n/2,1])
                   # Measure of the fitting accuracy for the horizontal cross-section
                err_x = sqrt(abs(pcov[3,3]))/fit_x[3]
                  
                   #%% Fitting along the z axis (integrated over all x values)
                print("OD x  fit done") 
                   # Gaussian fitting to get the width
                sum_z = np.sum(OD, axis = 1)
                n = len(sum_z)
                z = np.linspace(0,n,n)
                fit_z,pcov = curve_fit(gaus,z,sum_z,p0=[imin,np.max(sum_z),n/2,1])
                   # Measure of the fitting accuracy for the vertical cross-section
                err_z = sqrt(abs(pcov[3,3]))/fit_z[3]
                print("OD z  fit done")  
                   #%% Global peak optical depths
                  
                center_x = int(round(fit_x[2]))
                center_z = int(round(fit_z[2]))
                center_x = ROI_x_start + int((ROI_x_end - ROI_x_start)/2)
                center_z = ROI_z_start + int((ROI_z_end - ROI_z_start)/2)
                #    # Cross sections centered in the previous peaks
                cross_x1 = OD[center_z,:]
                cross_x = cross_x1[ROI_x_start:ROI_x_end]
                cross_z1 = OD[:,center_x]
                cross_z = cross_z1[ROI_z_start:ROI_z_end];
                x = np.linspace(0,len(cross_x), len(cross_x))
                z = np.linspace(0,len(cross_z), len(cross_z))
                   # Gaussian fitting of those cross sections
                cross_x_fit,pcov = curve_fit(gaus,x,cross_x,p0=[imin, np.max(cross_x),len(cross_x)/2,1])
                cross_z_fit,pcov = curve_fit(gaus,z,cross_z,p0=[imin, np.max(cross_z),len(cross_z)/2,1])
                print("OD second  fit done")  
                ROI_sum=0
                for row in range(ROI_x_start,ROI_x_end):
                    for col in range(ROI_z_start, ROI_z_end):
                        ROI_sum = OD[row, col] + ROI_sum
                print(ROI_sum)
                
                points_x = [ROI_x_start, ROI_x_end]
                points_z = [ROI_z_start+(ROI_z_end-ROI_z_start)/2, ROI_z_start+(ROI_z_end-ROI_z_start)/2]
                points_z2 = [ROI_z_start, ROI_z_end]
                points_x2= [ROI_x_start + (ROI_x_end-ROI_x_start)/2, ROI_x_start + (ROI_x_end-ROI_x_start)/2]
                
                
                fit_xl = gaus(x, *cross_x_fit)
                plt.subplot(2,2,3)
                plt.plot(x,cross_x,'--')
                plt.plot(x,fit_xl,'--')
                  
                fit_zl = gaus(z, *cross_z_fit)
                plt.subplot(2,2,2)
                plt.plot(z,cross_z,'--')
                plt.plot(z,fit_zl,'--')
 
                print("OD x  fit done")  
                plt.subplot(2,2,1)
                plt.title("OD ROI_Sum = %i"  %ROI_sum,  fontsize=12)
                #plt.ylabel("OD")
                plt.plot(points_x, points_z, linestyle='solid', color='red')
                plt.plot(points_x2, points_z2, linestyle='solid', color='red')
                plt.imshow(OD)
                plt.clim(0,1) # colourbar limit (,)
                plt.colorbar()                
                   #save image
                filename = savepath + folder_date + "-img_%05d_d.jpg" % (lastImageNum + i)
                plt.savefig(filename, dpi=300, bbox_inches='tight' ) 
               # print(filename)
                os.system(filename)
                # #          plt.show()
                plt.close()
                 
                  # Calculation of the peak optical depth
                ODpk_x = cross_x_fit[0] + cross_x_fit[1]
                ODpk_z = cross_z_fit[0] + cross_z_fit[1]
                ODpk = (ODpk_x + ODpk_z)/2
                 
                  #%% Number of atoms
                 
                sigma_x = cross_x_fit[3]*px/2**0.5
                sigma_z = cross_z_fit[3]*px/2**0.5
               
                  #%% Export parameters of interest
                 
                headline = ['Run #', 'filename','sigma_x','sigma_z','ROI_sum', 'sigma_x fit error','sigma_z fit error']
                  # If the csv file does not exist yet, creates it with its header
                if not os.path.exists(savepath + folder_date + "-Data00000.csv"):
                    with open(savepath + folder_date +  '-Data00000.csv', 'x', newline = '') as file:
                        header = csv.writer(file, dialect = 'excel', quoting = csv.QUOTE_NONE)
                        header.writerow(headline)
                  # Gets the run number as the last line number of the csv file
                with open(savepath + folder_date + '-Data00000.csv', 'r') as file:
                    reader = csv.reader(file, dialect = 'excel')
                    rows = list(reader)
                    line_num = len(rows)
                parameters = [line_num,filename, sigma_x,sigma_z, ROI_sum, err_x,err_z]
                  # Adds the new set of parameters following the existing lines
                with open(savepath + folder_date + '-Data00000.csv', 'a', newline = '') as file:
                    writer = csv.writer(file, dialect = 'excel', quoting = csv.QUOTE_NONE)
                    writer.writerow(parameters)


            
    cam.StopGrabbing()
    cam.Close()
    
except genicam.GenericException as e:
        # Error handling.
        print("An exception occurred.", e.GetDescription())
