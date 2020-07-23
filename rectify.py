import numpy as np # version: '1.14.0'
import cv2 # version: '3.1.0'
import glob
import pickle
import os

# Set root directory.
rootDir = "/home/kajal/Desktop/A/newstereo"
print('\nrootDir = {}\n'.format(rootDir))
# Set parameter directory.
dir_stereoRemap = rootDir+'/calibration_params/stereoRemap.p'
# Set original frames directory.
dir_original_L = rootDir+'/calibration_images/left/'
dir_original_R = rootDir+'/calibration_images/right/'
# Set compare frames directory.
dir_compare_frame_pairs = rootDir+'/output/compare_recti/'
# Set rectified frames directory.
dir_rectified_L = rootDir+'/output/recti_frames/left/'
dir_rectified_R = rootDir+'/output/recti_frames/right/'
dir_original = '/home/kajal/Desktop/A/newstereo/calibrated_frames'

# Load the dictionary back from the pickle file.
stereoRemap = pickle.load( open(dir_stereoRemap, 'rb') )
mapxL = stereoRemap['mapxL']
mapyL = stereoRemap['mapyL']
mapxR = stereoRemap['mapxR']
mapyR = stereoRemap['mapyR']

# /home/kajal/Desktop/A/newstereo/calibrated_frames/chessboard-L0.png_cornersl.jpg
# Load original frames.
os.chdir(dir_original) # Change dir to the path of left frames.
imagesL = glob.glob('*cornersl.jpg') # Grab all jpg file names.
imagesL.sort() # Sort frame file names.
# os.chdir(dir_original_R) # Change dir to the path of right frames.
imagesR = glob.glob('*cornersr.jpg') # Grab all jpg file names.
imagesR.sort() # Sort frame file names.

print(len(imagesR))
# Rectify each frame.
for i in range(len(imagesL)):
    print('i = {}'.format(i))
    print('Rectifying {} and {}...\n'.format(imagesL[i], imagesR[i]))
    os.chdir(dir_original) # Change dir to the path of left frames.
    imgL = cv2.imread(imagesL[i])
    imgR = cv2.imread(imagesR[i])
    hL,  wL = imgL.shape[:2]
    hR,  wR = imgR.shape[:2]
    
    # Remap.
    dstL = cv2.remap(imgL, mapxL, mapyL, cv2.INTER_LINEAR)
    dstR = cv2.remap(imgR, mapxR, mapyR, cv2.INTER_LINEAR)

    for line in range(0, int(imgR.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
        imgR[line*20,:]= (0,0,255)
        imgL[line*20,:]= (0,0,255)
    for line in range(0, int(imgR.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
        dstL[line*20,:]= (0,0,255)
        dstR[line*20,:]= (0,0,255) 

    
    # Combine frame pairs.
    imgTwin = np.hstack((imgL, imgR))
    imgTwin_rect = np.hstack((dstL, dstR))
    compareImg = np.vstack((imgTwin, imgTwin_rect))
    compareImg = cv2.resize(compareImg, (1024,768))
    # Display comparing frame pairs.
    cv2.imshow(imagesL[i]+'_'+imagesR[i], compareImg)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    # Save comparing frame pairs.
    cv2.imwrite(dir_compare_frame_pairs + imagesL[i][:-6] + '_rectified.jpg', compareImg)
    
    # Save rectified frames.
    cv2.imwrite(dir_rectified_L + imagesL[i][:-4] + '_rectified.jpg', dstL)
    cv2.imwrite(dir_rectified_R + imagesL[i][:-4] + '_rectified.jpg', dstR)
    
exit()