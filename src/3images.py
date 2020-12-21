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

# Create input to trigger frame acquisition
#def getkey():
  # return input("Enter \"t\" to start acquisition or \"e\" to exit, and press enter. (t/e) ")
a = "a"
b = "b"
c = "c"
''' Set parameters.'''
# Length of loop, maximum images that can be saved
num_run = 100 #number of run
# Save path to images folder
savepath = 'C:/Users/Sanket/Desktop/Acquisition loop/'
folder_date = datetime.now().strftime("%Y-%m-%d")
folder_to_save_files = savepath 

if not os.path.exists(folder_to_save_files):
    os.mkdir(folder_to_save_files)

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

#%% Constants definition

kB = 1.38e-23
muB = 9.27e-24
h = 6.63e-34
sigma_0 = 3*lam**2/(2.0*pi) # in SI
sigma_total = sigma_0/(1 + 2*IoverIs + 4*(delta/gam)**2)
#%% Importation of the 3 images
def acquire_img():
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
    # cam.ExposureMode='TriggerWidth'
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
        
    #Numbering for images so image always saved even when files already in Images folder
    dirInfo = os.listdir(savepath)    
    sizeDirInfo = len(dirInfo)


    if sizeDirInfo == 0:
        lastImageNum = 1
    else:
        lastImageName = dirInfo[(sizeDirInfo-1)]
        # lastImageNum = float(lastImageName[15:20]) + 1

    # Saving images.
    try:
        print("run no. = ", i)
    # Starts acquisition and camera waits for frame trigger
        cam.AcquisitionStart.Execute()

    # Start grabbing of images, unlimited amount, default type is continuous acquisition
        cam.StartGrabbing()
        # RetrieveResult will timeout after a specified time, in ms, so set much larger than the time of a cycle
        with cam.RetrieveResult(1000) as result:
    
            # Calling AttachGrabResultBuffer creates another reference to the
            # grab result buffer. This prevents the buffer's reuse for grabbing.
            print("Image 1 done")
            
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
            print("Image 2 done")

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
            print("Image 3 done")
            
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
        cam.StopGrabbing()
        cam.Close()
          
    except genicam.GenericException as e:
        # Error handling.
            print("An error occured. Restarting...")
            
 
for i in range(num_run):
    acquire_img()
        
        
