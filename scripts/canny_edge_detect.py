import cv2
import numpy as np

#function to perform Sobel filtering
#*****REFERENCE: https://github.com/adamiao/sobel-filter-tutorial/blob/master/sobel_from_scratch.py*****
def sobel_filter(image, Gx, Gy, threshold=140):
    rows, cols = image.shape
    ksize, _ = Gx.shape
    offset = 2
    image_sobel = np.zeros([rows - offset, cols - offset])
    for i in range(rows - offset): #nested for-loop to apply kernel to applicable pixels 
        for j in range(cols - offset):
            gx = np.sum(np.multiply(Gx, image[i:i + ksize, j:j + ksize])) #x-direction
            gy = np.sum(np.multiply(Gy, image[i:i + ksize, j:j + ksize])) #y-direction
            norm = np.sqrt(gx**2 + gy**2)  
            if (norm > threshold):
                image_sobel[i, j] = 0
            else:
                image_sobel[i, j] = 255
    return image_sobel

#callback for GUI
def callback(x):
    print(x)

if __name__ == "__main__":
    #user input
    #filename = input("Enter the image filename, including the extension (ensure images are in same directory as the script): ")
    image = cv2.imread('images/334.jpg') 

    scale_percent = 50 
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim)
    cv2.imshow('Original Image', image)

    #generate grayscale image
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Apply Sobel filter
    Gx = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

    Gy = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

    #image_sobel = grayscale#sobel_filter(grayscale, Gx, Gy, 120)
    #cv2.imshow('Sobel Filter', image_sobel)
    #cv2.imwrite(filename.rsplit(".", 1)[0] +'-sobel.png', image_sobel)

    #Apply Canny filter
    cv2.namedWindow('Canny Filter')
    cv2.createTrackbar('threshold1', 'Canny Filter', 0, 255, callback) 
    cv2.createTrackbar('threshold2', 'Canny Filter', 0, 255, callback) 

    #Loop for GUI, press esc to end
    while(1):
        threshold_1 = cv2.getTrackbarPos('threshold1', 'Canny Filter')
        threshold_2 = cv2.getTrackbarPos('threshold2', 'Canny Filter')
        #lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
        #lower_color_bounds = np.array([66, 79, 81])
        #upper_color_bounds = np.array([210, 216, 220])
        #mask = cv2.inRange(image, lower_color_bounds, upper_color_bounds)
        #mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        #cv2.imshow('mask', mask)
        #image = image & mask_rgb
        cv2.imshow('thres_image', image)
        image_canny = cv2.Canny(image, threshold_1, threshold_2)
        #image_canny_inv = cv2.bitwise_not(image_canny)
        cv2.imshow('Canny Filter', image_canny)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    #cv2.imwrite(filename.rsplit(".", 1)[0] +'-canny.png', image_canny_inv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()