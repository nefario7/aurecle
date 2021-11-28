
import cv2
import numpy as np
import sys 
import os


'''
[29 29 29]
[163 163 163]


[  0 105  51]
[106 164 177]
'''

def Lab_Segmentation(image,L_lower, L_upper, a_lower, a_upper, b_lower, b_upper):
    lowerRange= np.array([L_lower, a_lower, b_lower] , dtype="uint8")
    upperRange= np.array([L_upper, a_upper, b_upper], dtype="uint8")
    mask = image[:].copy()
    print(lowerRange)
    print(upperRange)
    # imageLab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    imageRange = cv2.inRange(image,lowerRange, upperRange)
    
    mask[:,:,0] = imageRange
    mask[:,:,1] = imageRange
    mask[:,:,2] = imageRange
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    faceLab = cv2.bitwise_and(image,mask)

    return faceLab




def nothing(x):
    pass

def main(argv):


    image = cv2.imread('sample/ret_image_1.png')
    
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    cv2.namedWindow("Output")

    cv2.createTrackbar("L_lower", "Output", 0,255,nothing )
    cv2.createTrackbar("L_upper", "Output", 0,255,nothing )

    cv2.createTrackbar("a_lower", "Output", 0,255,nothing )
    cv2.createTrackbar("a_upper", "Output", 0,255,nothing )

    cv2.createTrackbar("b_lower", "Output", 0,255,nothing )
    cv2.createTrackbar("b_upper", "Output", 0,255,nothing )


    while(True):
        # Get the user params 

        L_lower = cv2.getTrackbarPos("L_lower", "Output")
        L_upper = cv2.getTrackbarPos("L_upper", "Output")

        a_lower = cv2.getTrackbarPos("a_lower", "Output")
        a_upper = cv2.getTrackbarPos("a_upper", "Output")

        b_lower = cv2.getTrackbarPos("b_lower", "Output")
        b_upper = cv2.getTrackbarPos("b_upper", "Output")


        segmented_lab = Lab_Segmentation(lab_image, L_lower, L_upper, a_lower, a_upper, b_lower, b_upper)
        # Display the image 
        cv2.imshow("Output", segmented_lab)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    # Print the user params 


if __name__ == '__main__':
    main(sys.argv[1:])


# image = cv2.GaussianBlur(image,(5,5),0)

# edges = cv2.Canny(gray, 50, 200)
# cv2.imwrite('pitt_lab.jpeg', image)