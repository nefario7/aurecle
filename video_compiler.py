import cv2 
import os 
path = 'processing/output_from_main/'
frame_count = len(os.listdir(path)) 
frame_1 = cv2.imread(path+"/1-overlay.jpg")
height = frame_1.shape[0]
width = frame_1.shape[1]
out = cv2.VideoWriter('output/output_moving_avg.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height))
for i in range(frame_count):
    filename = str(i+1) + str('-overlay.jpg')
    filepath = path + filename
    frame = cv2.imread(filepath)
    out.write(frame)
out.release()
cv2.destroyAllWindows()