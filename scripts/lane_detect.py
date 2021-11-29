import cv2
import numpy as np

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        mousePick(x, y)

def mousePick(x, y):
    global pick, img
    img_copy = img.copy()
    pick.append((x,y))
    for i in range(len(pick)):
        img_copy = cv2.circle(img_copy, pick[i], 5, (0, 0, 255), 2)
        img_copy = cv2.putText(img_copy, str(i), (pick[i][0]+10, pick[i][1]-10),\
                          cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 1)
    cv2.imshow("img", img_copy)
    cv2.waitKey(1)
    if len(pick) > 3:
        drawMask()

def drawMask():
    global pick, mask
    polygon = np.asarray(pick)
    cv2.fillConvexPoly(mask, polygon, 255) 
    cv2.imshow("mask", mask)
    overlayMask()

def overlayMask():
    global img, mask
    img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("mask overlay", img)
    threshold()

def threshold():
    global img
    thres = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
          cv2.THRESH_BINARY,11,7)
    cv2.imshow("threshold", thres)
    img_canny = cv2.Canny(thres, 100, 200)
    cv2.imshow("image_canny", img_canny)
    houghTransform(img_canny, thres)

def houghTransform(img_canny, thres):
    global img
    lines = cv2.HoughLinesP(thres, 1, np.pi/180, 30, maxLineGap=5, minLineLength=100)
    print(lines)
    print(len(lines))

    #draw Hough lines
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.imshow("hough",img)

if __name__ == "__main__":

    img = cv2.cvtColor(cv2.imread('images/fort-pitt.jpg'), cv2.COLOR_BGR2GRAY)
    scale_percent = 50 
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim)

    x, y = img.shape 
    mask = np.zeros((x,y), dtype=np.uint8)

    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    print("Select 4 points")
    pick = []
    cv2.setMouseCallback("img", click)

    cv2.waitKey()
    cv2.destroyAllWindows()