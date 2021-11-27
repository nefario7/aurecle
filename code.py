import cv2
import numpy as np
import glob 


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#prepare object point of 3D world
#Mesgrid of 8X11 with 30cm grids
chessboard=(7,10)
board_size=30
points2D=[]
points3D_found=[]
points3D=np.zeros((chessboard[0]*chessboard[1],3),np.float32)
k=0
for i in range(10):
    for j in range(7):
        points3D[k,0]=board_size*j
        points3D[k,1]=board_size*i
        k+=1

calibration_img_names=glob.glob('./Calibration/*.jpg')

for image in calibration_img_names:
    img=cv2.imread(image,0)
    ret,corners=cv2.findChessboardCorners(img,(chessboard[1],chessboard[0]),None)
    # print(corners)
    if ret==False:
        print(image)
    
    if ret==True:
        points3D_found.append(points3D)
        pixel_points=cv2.cornerSubPix(img,corners,(11,11),(-1,-1),criteria)
        points2D.append(pixel_points)
        img=cv2.drawChessboardCorners(img,chessboard,pixel_points,ret)
    cv2.imshow('img',img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
