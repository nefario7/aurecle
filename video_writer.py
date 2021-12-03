import cv2 
import os 
# ! Add the directory here 
path = r"lab_thresh_frames"
frame_count = len(os.listdir(path))
print(path+"/lab_thresh_1.png")
frame_1 = cv2.imread(path+"/lab_thresh_1.png")
height = frame_1.shape[0]
width = frame_1.shape[1]
out = cv2.VideoWriter(path+'/output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height))
for i in range(frame_count):
    filename = "/lab_thresh_"+str(i+1) + str('.png')
    filepath = path + filename
    print(filepath)
    frame = cv2.imread(filepath)
    cv2.imshow("frame", frame)
    out.write(frame)

out.release()
cv2.destroyAllWindows()
