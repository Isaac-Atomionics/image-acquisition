from pypylon import pylon
from pypylon import genicam
from datetime import datetime
import os
import cv2
#from os import path

img = pylon.PylonImage()
tlf = pylon.TlFactory.GetInstance()

num_run = 100 #number of runs

a = "a"
b = "b"
c = "c"

# Save path to images folder
savepath = 'C:/Users/admin/Desktop/camera_stuff/images/Abs_imaging-2020-12-20/'
folder_date = datetime.now().strftime("%Y-%m-%d")
folder_to_save_files = savepath 

if not os.path.exists(folder_to_save_files):
    os.mkdir(folder_to_save_files)
    
#%% Importation of the 1 image
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
        
    # Saving images.
    try:
        for i in range(num_run):
        # Starts acquisition and camera waits for frame trigger
            cam.AcquisitionStart.Execute()
    
        # Start grabbing of images, unlimited amount, default type is continuous acquisition
            cam.StartGrabbing()
            # RetrieveResult will timeout after a specified time, in ms, so set much larger than the time of a cycle
            with cam.RetrieveResult(10000000) as result:
        
                # Calling AttachGrabResultBuffer creates another reference to the
                # grab result buffer. This prevents the buffer's reuse for grabbing.
                img.AttachGrabResultBuffer(result)
                    
                filename = savepath + folder_date + "-img_%05d_fl.png" % (lastImageNum + 1)
                # Save image to
                img.Save(pylon.ImageFileFormat_Png, filename)
                
                print(filename)
                # In order to make it possible to reuse the grab result for grabbing
                # again, we have to release the image (effectively emptying the
                # image object).
                img.Release()
                img_fl = cv2.imread(filename,0) # With atom
                                           
        cam.StopGrabbing()
        cam.Close()
        
    except genicam.GenericException as e:
            # Error handling.
            print("An error occured. Restarting...")
            
for i in range(num_run):
    print("run no. = ", i)
    acquire_img()

