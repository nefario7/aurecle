"""
Interactive script to determine pixel-to-pixel distances
for respective points. Perform ratio calculation to translate
pixel distance to real distance
"""
import cv2
import numpy as np

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        mousePick(x, y)

def mousePick(x, y):
    global pick, img
    pick.append((x,y))
    for i in range(len(pick)):
        img = cv2.circle(img, pick[i], 5, (0, 0, 255), 2)
        img = cv2.putText(img, str(i), (pick[i][0]+10, pick[i][1]-10),\
                          cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 1)
    cv2.imshow("img", img)
    cv2.waitKey(1)
    if len(pick) > 3:
        calcDistance()
        
def calcDistance():
    global pick, lane_width
    lane_width_pix = np.linalg.norm(np.asarray(pick[0]) - np.asarray(pick[1]))
    clearance_height_pix = np.linalg.norm(np.asarray(pick[2]) - np.asarray(pick[3]))
    clearance_height = lane_width*(clearance_height_pix/lane_width_pix)
    print(clearance_height)
    drawLine()

def drawLine():
    global pick, img
    cv2.line(img, (pick[0]), (pick[1]), (0, 255, 0), thickness=2)
    cv2.line(img, (pick[2]), (pick[3]), (0, 255, 0), thickness=2)
    cv2.imshow("img", img)
    cv2.waitKey(1)

def contourDetect():
    global img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr,dst = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # for i in range(1):
    #     dst = cv2.erode(dst, None)
    # for i in range(2):
    #     dst = cv2.dilate(dst, None)
    cv2.imshow("Clean-up", dst)

if __name__ == "__main__":
    img_filename = 'images/fort-pitt.jpg'
    #img_filename = 'images/pre-fort-pitt.jpg'
    #img_filename = 'images/TEST.jpg'
    img = cv2.imread(img_filename) 

    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)

    print("Select 4 points")
    pick = []
    lane_width = 12 #ft
    #cv2.setMouseCallback("img", click)
    contourDetect()

    cv2.waitKey()
    cv2.destroyAllWindows()