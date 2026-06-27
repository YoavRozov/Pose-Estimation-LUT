import cv2

import numpy as np
import os

# Define termination criteria for subpixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define the size and shape of the calibration board (chessboard)
board_size = (5,7) # number of inner corners per row and column
square_size = 25 # size of each square in mm
camera_resolution = (1280, 720)
calibration_image_count = 40

# Create an array of object points corresponding to the chessboard pattern
objp = np.zeros((board_size[0]*board_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2) * square_size

# Create empty arrays to store object points and image points
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Set up the camera capture
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FPS, 120)

image_count = 0

while image_count < calibration_image_count:
    #print(fname)
    # Read the image and convert it to grayscale
    #img = cv.imread(fname)
    rect, img = cap.read()
    #img = cv2.resize(img, camera_resolution)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners on the image
    ret, corners = cv2.findChessboardCorners(gray, board_size,None)

    # If found, add object points and image points to the arrays
    if ret == True:
        objpoints.append(objp)

        # Refine corner locations with subpixel accuracy
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners on the image (optional)
        img = cv2.drawChessboardCorners(img, board_size, corners2,ret)
        cv2.imshow('img',cv2.flip(img, 1))
        cv2.waitKey(1)
        image_count += 1
    

    cv2.imshow('gray', cv2.flip(gray, 1))
    cv2.waitKey(1)

# Free the camera and close any open windows
cap.release()
cv2.destroyAllWindows()

# Calibrate the camera using the object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,
                                                  gray.shape[::-1],None,None)

# Save the camera matrix and distortion coefficients to a file
output_dir = os.path.join(os.path.dirname(__file__), "calibration_output")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.savez(os.path.join(output_dir, "camera_calibration.npz"), mtx=mtx, dist=dist)

print("Camera calibration completed. Camera matrix and distortion coefficients saved to 'camera_calibration.npz'.")

print("Camera:\n\tfx: {},\n\tfy: {},\n\tcx: {},\n\tcy: {}".format(mtx[0,0], mtx[1,1], mtx[0,2], mtx[1,2]))

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)