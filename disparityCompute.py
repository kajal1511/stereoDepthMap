import numpy as np
import time
from datetime import datetime
import cv2 # import opencv
from cv2 import *
import math
import os
import sys
import pickle
import multiprocessing
from multiprocessing import Process, Pool

def calculateDisparity(framePair):
        
        gray_l = framePair[0]
        gray_r = framePair[1]
        window_size=3
        isSGBM=1
        if(isSGBM):
            # stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=16*num_disp,\
            # blockSize=2*block_size+1)
            stereo = cv2.StereoSGBM_create(
            minDisparity=2,
            numDisparities=64,             # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=3,
            P1=64 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
            P2=80 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        else:
            stereo = cv2.StereoBM_create(numDisparities=16*num_disp,blockSize=2*block_size+1)

        stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

        lmbda = 8000
        sigma = 1.0
        visual_multiplier = 1.0

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)
        # Compute disparity.    
        disparity = stereo.compute(gray_l, gray_r)
        disparity =disparity.astype(np.uint8)
        np.set_printoptions(threshold=np.inf,linewidth=np.inf)

        dispL= stereo.compute(gray_l,gray_r) #.astype(np.float32)/ 16
        dispR= stereoR.compute(gray_r,gray_l)
        disp= dispL
        dispL= np.int16(dispL)
        dispR= np.int16(dispR)

        # Using the WLS filter
        filteredImg = wls_filter.filter(dispL,gray_l,None,dispR)
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        #cv2.imshow('Disparity Map', filteredImg)
        # disp= ((disp.astype(np.float32)/ 16)-min_disp)/num_disp 
        disparity=filteredImg
        disparity = disparity/16 #Scale to (0~255) int16
        disparity = disparity.astype(np.float16)/55.0*255.0
        disparity = disparity.astype(np.uint8)
        return disparity


#Disparity map parameters
min_disp = 2
num_disp = 10
block_size = 10
mapcolor = 2

# set video format.
camera_l = 3
camera_r = 6    
width = 320
height = 240
FPS = 120
median = 3

# exposure = 0.5
#framerate = 120
# pixelformat = 0 # 0 is MJPEG, 1 is YUYV

# Load the dictionary back from the pickle file.
stereoRemap = pickle.load( open( '/home/kajal/Desktop/A/newstereo/calibration_params/stereoRemap.p', 'rb' ) )
mapxL = stereoRemap['mapxL']
mapyL = stereoRemap['mapyL']
mapxR = stereoRemap['mapxR']
mapyR = stereoRemap['mapyR']

widthPixel = width
heightPixel = height

img_l = cv2.VideoCapture(camera_l)
img_r = cv2.VideoCapture(camera_r)
img_l.set(3, width) # Set resolution width
img_l.set(4, height) # Set resolution height
img_r.set(3, width) # Set resolution width
img_r.set(4, height) # Set resolution hight

print(img_l.isOpened() and img_r.isOpened())
if (img_l.isOpened() and img_r.isOpened())==False:
    exit()
dis_btw_cameras=26


while(True):
 
    retR, frameR= img_r.read()
    retL, frameL= img_l.read()
    focal_length=((frameR.shape[0]/2)/math.tan(math.pi/6))

    frame_l =cv2.remap(frameL, mapxL, mapyL, cv2.INTER_LINEAR)
    frame_r =cv2.remap(frameR, mapxR, mapyR, cv2.INTER_LINEAR)
    remapl = cv2.remap(frameL, mapxL, mapyL, cv2.INTER_LINEAR)
    remapr = cv2.remap(frameR, mapxR, mapyR, cv2.INTER_LINEAR)

    isBlend = 0
    isDisparity = 0
    isSGBM = 1
    isDisplay = 1


    for line in range(0, int(frameL.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
        frameR[line*20,:]= (0,0,255)
        frameL[line*20,:]= (0,0,255)
    for line in range(0, int(frameL.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
        frame_l[line*20,:]= (0,0,255)
        frame_r[line*20,:]= (0,0,255)        
    orignal_image= np.hstack((frameL, frameR))
    cv2.imshow('orignal images',orignal_image)

    
    imgTwin = np.hstack((frame_l, frame_r))
    cv2.imshow('frame_l  frame_r',imgTwin)

    gray_l = cv2.cvtColor(remapl,cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(remapr,cv2.COLOR_BGR2GRAY)
    
    fp=[gray_l,gray_r]
    disparity=calculateDisparity(fp)

    depth=(focal_length*dis_btw_cameras)/(disparity)
    print('->',depth,'->')

    isMedian=1
    if isMedian:
        disparity = cv2.medianBlur(disparity, median)

            
    cv2.imshow('Disparity map', disparity)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# p.close()
# p.join()
# endTime = datetime.now()
# print('FPS : ',numFrames/(endTime - startTime).total_seconds())
img_l.release()
img_r.release()
cv2.destroyAllWindows()