import numpy as np # version: '1.14.0'
import cv2 # version: '3.1.0'
import glob
import pickle
import os


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Set root directory.
rootDir ='/home/kajal/Desktop/A/newstereo'
print('\nrootDir = {}\n'.format(rootDir))

# Set parameter directory.
dir_calib_parameter = '/home/kajal/Desktop/A/newstereo/calibration_params'
# Set original frames directory.
dir_original_L = '/home/kajal/Desktop/A/newstereo/calibration_images/left'
dir_original_R = '/home/kajal/Desktop/A/newstereo/calibration_images/right'
# Set calibration process frames directory.
dir_calib_process = '/home/kajal/Desktop/A/newstereo/calibrated_frames'
print (dir_original_L)

# Frames resolution.
widthPixel = 640
heightPixel = 480

# The inner corner numbers of calibration chessboard.
Nx = 9
Ny = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((Nx*Ny,3), np.float32)
objp[:,:2] = np.mgrid[0:Ny,0:Nx].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

# Load original frames.
os.chdir(dir_original_L) # Change dir to the path of left frames.
imagesL = glob.glob('*.png') # Grab all jpg file names.
imagesL.sort() # Sort frame file names.
os.chdir(dir_original_R) # Change dir to the path of right frames.
imagesR = glob.glob('*.png') # Grab all jpg file names.
imagesR.sort() # Sort frame file names.

# Check if the number of images in two folders are same.
if len(imagesL) != len(imagesR):
    print('Error: the image numbers of left and right cameras must be the same!')
    exit()
n = 0
for i in range(len(imagesL)):
    # imgL = cv2.imread(imagesL[i])
    # imgR = cv2.imread(imagesR[i])
    # cv2.imshow("l",imgL)
    os.chdir(dir_original_L) 
    imgL = cv2.imread(imagesL[i])
    os.chdir(dir_original_R)
    imgR = cv2.imread(imagesR[i])
    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    

    # Find the chess board corners for left camera.
    retL, cornersL = cv2.findChessboardCorners(grayL,(Ny,Nx),None)
    # Find the chess board corners for right camera.
    retR, cornersR = cv2.findChessboardCorners(grayR,(Ny,Nx),None)
    
    # If both are found, add object points, image points (after refining them)
    if retL and retR:
        n += 1
        print('n = {}'.format(n))
        objpoints.append(objp)
        
        cornersL2 = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsL.append(cornersL2)
        
        cornersR2 = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR2)
        
        # Draw and display the corners
        # imgL = cv2.drawChessboardCorners(imgL, (Ny,Nx), cornersL2, retL)
        # imgR = cv2.drawChessboardCorners(imgR, (Ny,Nx), cornersR2, retR)
        # imgTwin = np.hstack((imgL, imgR))
        # cv2.imshow(imagesL[i] + '_' + imagesR[i], imgTwin)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # # Save frame pairs.
        os.chdir(dir_calib_process) 
        cv2.imwrite(imagesL[i] + '_cornersl.jpg', imgL)
        cv2.imwrite(imagesR[i] + '_cornersr.jpg', imgR)
        
cv2.waitKey(1) # Wait program to close the last window.
# Calculate the camera matrix, distortion coefficients, rotation and translation vectors etc.
print('Calculating the camera matrix, distortion coefficients, rotation and translation vectors...')

print('yess')
print(imgR.shape[::-1])
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        (320,240),imgR.shape[2],None,None)
print("before 1")
hR,wR= grayR.shape[:2]
OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))

print("before 2")
print(imgL.shape[::-1])
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        (320,240),imgL.shape[2],None,None)
print("before 3")
hL,wL= grayL.shape[:2]
OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

print('Done.\n')

CamParasL = {'mtxL':mtxL, 'distL':distL, 'rvecsL':rvecsL, 'tvecsL':tvecsL}
CamParasR = {'mtxR':mtxR, 'distR':distR, 'rvecsR':rvecsR, 'tvecsR':tvecsR}

# Save calibration parameters.
print('Saving calibration parameters...')
os.chdir(dir_calib_parameter)

pickle.dump(CamParasL, open('CamParasL.p', 'wb') )
pickle.dump(CamParasR, open('CamParasR.p', 'wb') )
print('Done.\n')

# Load the dictionary back from the pickle file.
#CamParasStereo = pickle.load( open( 'CamParasStereo.p', 'rb' ) )
os.chdir(dir_calib_parameter) 
CamParasL = pickle.load( open('CamParasL.p', 'rb' ) )
CamParasR = pickle.load( open('CamParasR.p', 'rb' ) )
CamParasL.keys()
CamParasR.keys()

# Restore calibration parameters from loaded dictionary.
mtxL = CamParasL['mtxL']
distL = CamParasL['distL']
rvecsL = CamParasL['rvecsL']
tvecsL = CamParasL['tvecsL']
mtxR = CamParasR['mtxR']
distR = CamParasR['distR']
rvecsR = CamParasR['rvecsR']
tvecsR = CamParasR['tvecsR']

# Stereo Calibration.
# cv2.stereoCalibrate Calculate the rectify parameters for stereo cameras.
print('Cacluating the rectify parameters for stereo cameras...')
# cv2.stereoRectify Computes the rotation matrices for each camera that(virtually) make both image planes the same plane. The function takes the matrices computed by stereoCalibrate() as imput.
'''
Usage:
rotationMatrixL, rotationMatrixR, projectionMatrixL, projectionMatrixR, disp2depthMappingMatrix, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, (widthPixel,heightPixel), rotationMatrix, translationVector, alpha=1, newImageSize=(0,0))
'''

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          mtxL,
                                                          distL,
                                                          mtxR,
                                                          distR,
                                                          (320,240),
                                                          criteria_stereo,
                                                          flags)

# StereoRectify function
rectify_scale=  1# if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 (320,240), R, T,
                                                 rectify_scale,(1,1))

# initUndistortRectifyMap function
mapxL, mapyL= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             (320,240), cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
mapxR, mapyR= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              (320,240), cv2.CV_16SC2)

print(mapxL.shape)
print(mapyR.shape)

stereoRemap = {'mapxL':mapxL, 'mapyL':mapyL, 'mapxR':mapxR, 'mapyR':mapyR}
disp2depthMappingMatrix={'disp2depthMappingMatrix':Q}

# Save calibration parameters.
print('Saving stereo remap matrices...')
os.chdir(dir_calib_parameter)
pickle.dump(stereoRemap, open('stereoRemap.p', 'wb' ) )
pickle.dump(disp2depthMappingMatrix, open('disp2depth.p', 'wb'))
print('Done.\n')

exit()